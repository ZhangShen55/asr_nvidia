"""跨音频公平的有界批调度；取消等待成员不取消共享物理执行。"""
from __future__ import annotations

import asyncio
import copy
import math
import time
import uuid
import wave
import io
from collections import OrderedDict, deque
from dataclasses import dataclass
from core.async_utils import await_completion


def detached_exception(exc):
    """仅清理副本；原异常可能仍被活动worker持有，不能clear其frame。"""
    result = copy.copy(exc)
    result.__traceback__ = result.__cause__ = result.__context__ = None
    result.__suppress_context__ = True
    return result


@dataclass(eq=False)
class BatchMember:
    audio: bytes
    language: str | None
    request_id: str
    segment_index: int | None
    frames: int
    queued_at: float
    future: asyncio.Future
    started: bool = False
    cancelled: bool = False


def audio_frames(data):
    with wave.open(io.BytesIO(data), 'rb') as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (16000, 1, 2, 'NONE'):
            raise ValueError('Qwen 段音频必须为16kHz单声道PCM16')
        # 锁定processor先补至至少8000样本，STFT有效帧floor(samples/160)。
        return max(1, max(wav.getnframes(), 8000)//160)


def padded_audio_frames(frames):
    """锁定前端n_window=50，完整single特征补至2*n_window=100帧倍数。"""
    return math.ceil(frames/100)*100


class BatchScheduler:
    def __init__(self, execute, *, workers, batch_size, wait_ms, audio_frame_budget,
                 token_budget, max_new_tokens, capacity, duration_ratio=4.0):
        if any(type(v) is not int or v < 1 for v in
               (workers, batch_size, audio_frame_budget, token_budget, max_new_tokens, capacity)):
            raise ValueError('Qwen batch容量与预算必须为正整数')
        if not math.isfinite(wait_ms) or wait_ms < 0:
            raise ValueError('Qwen batch等待必须为非负有限数')
        if isinstance(duration_ratio, bool) or not isinstance(duration_ratio, (int, float)) or not math.isfinite(duration_ratio) or not 1 <= duration_ratio <= 64:
            raise ValueError('Qwen batch最大时长比必须为1~64有限数')
        self.duration_ratio = float(duration_ratio)
        self.execute = execute
        self.workers = workers
        self.batch_size = batch_size
        self.wait_s = wait_ms/1000
        self.audio_frame_budget = audio_frame_budget
        self.token_budget = token_budget
        self.max_new_tokens = max_new_tokens
        self.capacity = capacity
        self._slots = asyncio.Semaphore(capacity)
        self._queues = OrderedDict()
        self._changed = asyncio.Event()
        self._tasks = []
        self._closing = False
        self._batch_sequence = 0
        self._active_entries = {}
        self._recent = deque(maxlen=64)
        self._sizes = {}
        self._running = 0
        self._inflight = 0
        self._publish_state()

    def _fits(self, frames, count):
        # 完整single特征的100帧padding也计入预算；实际shape单独报告。
        frames = padded_audio_frames(frames)
        return (frames*count <= self.audio_frame_budget and
                (math.ceil(frames/8)+self.max_new_tokens+256)*count <= self.token_budget)

    async def submit(self, audio, language, *, request_id=None, segment_index=None):
        frames = audio_frames(audio)
        if not self._fits(frames, 1):
            raise ValueError('单段超过Qwen批输入预算')
        await self._slots.acquire()
        member = None
        try:
            if self._closing:
                raise RuntimeError('Qwen batch scheduler closing')
            if not self._tasks:
                self._tasks = [asyncio.create_task(self._worker()) for _ in range(self.workers)]
            future = asyncio.get_running_loop().create_future()
            member = BatchMember(audio, language, str(request_id or uuid.uuid4()), segment_index,
                                 frames, time.monotonic(), future)
            self._queues.setdefault(member.request_id, deque()).append(member)
            self._inflight += 1
            self._publish_state()
            self._changed.set()
            try:
                return await asyncio.shield(future)
            except asyncio.CancelledError as cancelled:
                member.cancelled = True
                if not member.started:
                    queue = self._queues.get(member.request_id)
                    if queue is not None and member in queue:
                        queue.remove(member)
                        if not queue:
                            self._queues.pop(member.request_id, None)
                    future.cancel()
                else:
                    # 等共享批次真正收尾；其它成员不能因这一成员取消而被中断。
                    try:
                        await await_completion(future)
                    except BaseException:
                        pass
                raise cancelled
        finally:
            if member is not None:
                self._inflight -= 1
            self._slots.release()
            self._publish_state()
            self._changed.set()

    def _take_batch(self):
        batch = []
        max_frames = 0
        reason = 'tail_or_wait_deadline'
        # 全局最老pending head先锚定，避免batch恰为请求数倍数时轮转回同一请求。
        oldest_key = min(self._queues, key=lambda key: self._queues[key][0].queued_at)
        self._queues.move_to_end(oldest_key, last=False)
        # 最老成员锚定本批，保证旧长/短段不会被持续绕过。
        # 后续从各有界队列寻找可兼容成员，避免不兼容队首挡住可执行尾段。
        min_frames = next(iter(self._queues.values()))[0].frames
        visits = 0
        while self._queues and len(batch) < self.batch_size:
            key, queue = self._queues.popitem(last=False)
            chosen = None
            blocked_by_length = False
            for member in queue:
                candidate_frames = max(max_frames, member.frames)
                candidate_min = min(min_frames, member.frames)
                compatible = candidate_frames/candidate_min <= self.duration_ratio
                if (compatible or not batch) and self._fits(candidate_frames, len(batch)+1):
                    chosen = member
                    break
                blocked_by_length |= not compatible
            if chosen is not None:
                queue.remove(chosen)
                chosen.started = True
                batch.append(chosen)
                max_frames = max(max_frames, chosen.frames)
                min_frames = min(min_frames, chosen.frames)
                visits = 0
            else:
                visits += 1
                reason = 'duration_ratio' if blocked_by_length else 'input_budget'
            if queue:
                self._queues[key] = queue
            if visits >= len(self._queues) and visits:
                break
        if len(batch) == self.batch_size:
            reason = 'batch_size'
        return batch, reason, max_frames

    async def _worker(self):
        # 各副本独立补位；其它worker完成不能延长本worker的固定deadline。
        last_completion = None
        while True:
            self._changed.clear()
            if not self._queues:
                if self._closing:
                    return
                await self._changed.wait()
                continue
            oldest = min(q[0].queued_at for q in self._queues.values())
            waiting = sum(map(len, self._queues.values()))
            deadline = oldest + self.wait_s
            if last_completion is not None:
                # 旧尾成员虽已等过入队窗口，上一批消费者还需要释放名额并切PCM。
                # 留一次固定补位窗口；新arrival只唤醒检查，不重置此deadline。
                deadline = max(deadline, last_completion + self.wait_s)
            remaining = deadline - time.monotonic()
            if waiting < self.batch_size and remaining > 0 and not self._closing:
                try:
                    await asyncio.wait_for(self._changed.wait(), remaining)
                except asyncio.TimeoutError:
                    pass
                continue
            batch, reason, frames = self._take_batch()
            if not batch:
                raise RuntimeError('Qwen scheduler budget invariant violated')
            self._batch_sequence += 1
            batch_id = self._batch_sequence
            started = time.monotonic()
            entry = {'batch_id': batch_id, 'size': len(batch), 'reason': reason,
                     'max_raw_audio_frames': frames,
                     'estimated_padded_audio_frames': padded_audio_frames(frames)*len(batch),
                     'estimated_padded_tokens': (math.ceil(padded_audio_frames(frames)/8)+self.max_new_tokens+256)*len(batch),
                     'max_wait_ms': (started-min(m.queued_at for m in batch))*1000,
                     'members': [{'request_id': m.request_id, 'segment_index': m.segment_index} for m in batch]}
            self._active_entries[batch_id] = entry
            self._running += len(batch)
            self._publish_state()
            try:
                values = await self.execute(batch, batch_id)
                if len(values) != len(batch):
                    raise RuntimeError('Qwen batch result cardinality mismatch')
                for member, value in zip(batch, values):
                    if not member.future.done():
                        if isinstance(value, Exception):
                            member.future.set_exception(detached_exception(value))
                        else:
                            member.future.set_result(value)
                entry['status'] = 'completed'
            except asyncio.CancelledError:
                entry["status"] = "cancelled"
                for member in batch:
                    if not member.future.done():
                        member.future.cancel()
                return
            except Exception as exc:
                entry['status'] = 'failed'
                entry['error_type'] = type(exc).__name__
                for member in batch:
                    if not member.future.done():
                        if isinstance(exc, asyncio.CancelledError):
                            member.future.cancel()
                        else:
                            member.future.set_exception(detached_exception(exc))
            finally:
                entry['execution_wall_ms'] = (time.monotonic()-started)*1000
                self._active_entries.pop(batch_id, None)
                self._recent.append(entry)
                self._sizes[entry["size"]] = self._sizes.get(entry["size"], 0)+1
                self._running -= len(batch)
                self._publish_state()
                last_completion = time.monotonic()

    def record_dispatch(self, batch_id, members):
        if batch_id in self._active_entries:
            entry = self._active_entries[batch_id]
            entry.setdefault('selected_size', entry['size'])
            entry['actual_dispatched_size'] = len(members)
            entry['size'] = len(members)
            entry['cancelled_before_dispatch'] = entry['selected_size'] - len(members)
            entry['dispatched_members'] = [{'request_id': m.request_id, 'segment_index': m.segment_index}
                                           for m in members]
            if members:
                attempts = entry.setdefault('dispatch_attempts', [])
                attempts.append({'attempt': len(attempts) + 1, 'size': len(members),
                                 'input_members': list(entry['dispatched_members']),
                                 'members': list(entry['dispatched_members']), 'status': 'started'})

    def record_retry_cancellations(self, batch_id, members, retry_number):
        if batch_id in self._active_entries:
            entry = self._active_entries[batch_id]
            entry['cancelled_before_retry'] = len(members)
            entry['retry_cancelled_members'] = [
                {'request_id': m.request_id, 'segment_index': m.segment_index} for m in members]
            entry['last_retry_number'] = retry_number
            entry['retry_skipped_all_cancelled'] = len(members) == entry.get('selected_size', entry['size'])

    def record_runtime_instance(self, batch_id, identity):
        entry = self._active_entries.get(batch_id)
        if entry is None:
            return
        entry["runtime_instance"] = copy.deepcopy(identity)
        attempts = entry.get("dispatch_attempts", [])
        if attempts:
            attempts[len(attempts) - 1]["runtime_instance"] = copy.deepcopy(identity)

    def record_runtime_timings(self, batch_id, timings, *, status='completed', error_type=None):
        if batch_id in self._active_entries:
            entry = self._active_entries[batch_id]
            attempts = entry.get('dispatch_attempts', [])
            attempt = attempts[-1] if attempts else None
            if attempt is not None:
                attempt['status'] = status
                attempt['error_type'] = error_type
                attempt['runtime_timings'] = copy.deepcopy(timings)
            entry["runtime_timings"] = copy.deepcopy(timings)
            if timings is not None and 'valid_member_indices' in timings:
                # 只能索引本attempt输入，不能使用已由上次valid筛选过的summary。
                dispatched = attempt['input_members'] if attempt is not None else entry.get('dispatched_members', [])
                indices = timings['valid_member_indices']
                actual = [dispatched[i] for i in indices]
                entry['invalid_before_dispatch'] = len(dispatched) - len(indices)
                entry['actual_dispatched_size'] = len(indices)
                entry['size'] = len(indices)
                entry['dispatched_members'] = actual
                if attempt is not None:
                    attempt['actual_dispatched_size'] = len(indices)
                    attempt['members'] = list(actual)

    def _publish_state(self):
        now = time.monotonic()
        self._state = {"_updated_at": now,'configured_batch_size': self.batch_size, 'capacity': self.capacity,
                'waiting': sum(map(len, self._queues.values())), 'running_members': self._running,
                'inflight': self._inflight, 'batch_count': self._batch_sequence,
                'batch_sizes': dict(self._sizes), 'recent_batches': list(self._recent),
                'oldest_wait_ms': max([0]+[(now-q[0].queued_at)*1000 for q in self._queues.values()]),
                'budget_source': 'independent_single_100frame_padding_and_8to1_tokens_plus_256_prompt_reserve',
                'batch_max_duration_ratio': self.duration_ratio,
                'audio_frame_budget': self.audio_frame_budget, 'token_budget': self.token_budget,
                'wait_ms': self.wait_s*1000, 'closing': self._closing}

    def snapshot(self, *, recent_limit=None):
        # 发布时整体换指针；已完成的recent记录不再修改。先限窗口再deepcopy，
        # HTTP读小窗口不会把64批完整shape/member历史都复制进事件循环。
        published = self._state
        if recent_limit is None:
            state = copy.deepcopy(published)
        else:
            if type(recent_limit) is not int or not 1 <= recent_limit <= self._recent.maxlen:
                raise ValueError("recent_limit必须为1到历史容量之间的整数")
            available = published['recent_batches']
            recent = available[max(0, len(available) - recent_limit):]
            selected = dict(published, recent_batches=recent)
            available_ids = [row['batch_id'] for row in available]
            recent_ids = [row['batch_id'] for row in recent]
            def id_range(ids):
                return [min(ids), max(ids)] if ids else None
            selected['history'] = {
                'scope': 'bounded_complete_records_no_member_or_timing_projection',
                'history_limit': recent_limit,
                'stored_limit': self._recent.maxlen,
                'available_count': len(available), 'published_count': len(recent),
                'available_batch_id_range': id_range(available_ids),
                'published_batch_id_range': id_range(recent_ids),
                'truncated': len(available) > len(recent),
                'omitted_available_count': len(available) - len(recent),
                'older_than_available_count': max(0, sum(published['batch_sizes'].values()) - len(available)),
                'published_noncontiguous': bool(recent_ids and max(recent_ids) - min(recent_ids) + 1 != len(set(recent_ids))),
                'gap_detection': 'consumer must compare batch IDs across responses; this is not a lossless event stream',
                'full_snapshot_source': 'direct QwenAsrClient.snapshot() / BatchScheduler.snapshot() defaults retain full stored history',
            }
            state = copy.deepcopy(selected)
        updated = state.pop('_updated_at')
        if state['waiting']:
            state['oldest_wait_ms'] += (time.monotonic()-updated)*1000
        return state

    async def close(self, error=None):
        self._closing = True
        # 尚未开始的成员立即退出；在运行批成员仍等物理执行收尾。
        for queue in self._queues.values():
            for member in queue:
                if not member.future.done():
                    member.future.set_exception(error or RuntimeError('Qwen scheduler closing'))
        self._queues.clear()
        self._publish_state()
        self._changed.set()
        if self._tasks:
            await await_completion(asyncio.gather(*self._tasks))
