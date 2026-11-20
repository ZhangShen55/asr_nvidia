"""Bounded round-robin admission; each sequence completes without a batch barrier."""
from __future__ import annotations

import asyncio
import copy
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass

from core.async_utils import await_completion
from core.qwen_batch_scheduler import audio_frames, padded_audio_frames, detached_exception


@dataclass(eq=False)
class SequenceMember:
    audio: bytes
    language: str | None
    request_id: str
    segment_index: str
    future: asyncio.Future
    queued_at: float
    started: bool = False
    cancelled: bool = False


class ContinuousScheduler:
    def __init__(self, execute, *, capacity, audio_frame_budget, token_budget, max_new_tokens):
        for value in (capacity, audio_frame_budget, token_budget, max_new_tokens):
            if type(value) is not int or value < 1:
                raise ValueError('continuous budgets must be positive integers')
        self.execute = execute
        self.capacity = capacity
        self.audio_frame_budget = audio_frame_budget
        self.token_budget = token_budget
        self.max_new_tokens = max_new_tokens
        self._slots = asyncio.Semaphore(capacity)
        self._queues = OrderedDict()
        self._changed = asyncio.Event()
        self._pump_task = None
        self._active = {}
        self._members = {}
        self._recent = deque(maxlen=64)
        self._closing = False
        self._completed = 0
        self._peak = 0
        self._sequence = 0

    async def submit(self, audio, language, *, request_id=None, segment_index=None):
        frames = padded_audio_frames(audio_frames(audio))
        if (frames > self.audio_frame_budget or
                (int((frames+7)//8)+self.max_new_tokens+256) > self.token_budget):
            raise ValueError('single segment exceeds continuous input budget')
        key = (str(request_id or uuid.uuid4()), str(segment_index if segment_index is not None else uuid.uuid4()))
        await self._slots.acquire()
        member = None
        try:
            if self._closing:
                raise RuntimeError('continuous scheduler closing')
            if key in self._members:
                raise ValueError('duplicate in-flight segment ID')
            member = SequenceMember(audio, language, *key, asyncio.get_running_loop().create_future(), time.monotonic())
            self._members[key] = member
            self._queues.setdefault(member.request_id, deque()).append(member)
            if self._pump_task is None:
                self._pump_task = asyncio.create_task(self._pump())
            self._changed.set()
            try:
                return await asyncio.shield(member.future)
            except asyncio.CancelledError as cancelled:
                member.cancelled = True
                # Execution observes cancelled and sends a backend CANCEL; its owned
                # future must finish before this caller can release its credit.
                if not member.started:
                    queue = self._queues.get(member.request_id)
                    if queue is not None and member in queue:
                        queue.remove(member)
                        if not queue:
                            del self._queues[member.request_id]
                    member.future.cancel()
                else:
                    try:
                        await await_completion(member.future)
                    except BaseException:
                        pass
                raise cancelled
        finally:
            if member is not None:
                self._members.pop(key, None)
            self._slots.release()
            self._changed.set()

    async def _pump(self):
        while True:
            self._changed.clear()
            # Give concurrent producers one event-loop turn to publish ready work.
            # This is not a timed batch wait and never resets after completion.
            await asyncio.sleep(0)
            while self._queues and len(self._active) < self.capacity:
                request, queue = self._queues.popitem(last=False)
                member = queue.popleft()
                if queue:
                    self._queues[request] = queue
                member.started = True
                self._sequence += 1
                key = (member.request_id, member.segment_index)
                task = asyncio.create_task(self._run(member, key, self._sequence))
                self._active[key] = task
                self._peak = max(self._peak, len(self._active))
            if self._closing and not self._active and not self._queues:
                return
            await self._changed.wait()

    async def _run(self, member, key, sequence):
        started = time.monotonic()
        entry = dict(batch_id=sequence, size=1, request_id=member.request_id,
                     segment_index=member.segment_index, mode='continuous_sequence',
                     queue_wall_ms=(started-member.queued_at)*1000)
        try:
            value = await self.execute(member)
            entry['status'] = 'completed'
            if not member.future.done():
                member.future.set_result(value)
        except asyncio.CancelledError:
            entry['status'] = 'cancelled'
            if not member.future.done():
                member.future.cancel()
        except Exception as exc:
            entry.update(status='failed', error_type=type(exc).__name__)
            if not member.future.done():
                member.future.set_exception(detached_exception(exc))
        finally:
            entry['execution_wall_ms'] = (time.monotonic()-started)*1000
            self._recent.append(entry)
            self._completed += 1
            self._active.pop(key, None)
            self._changed.set()

    def snapshot(self, *, recent_limit=None):
        if recent_limit is not None and (type(recent_limit) is not int or not 1 <= recent_limit <= 64):
            raise ValueError('recent_limit must be between 1 and 64')
        recent = list(self._recent)
        return copy.deepcopy(dict(mode='continuous', configured_batch_size=None,
            capacity=self.capacity, waiting=sum(map(len, self._queues.values())),
            running_members=len(self._active), inflight=len(self._members),
            peak_inflight=self._peak, batch_count=self._completed, batch_sizes={},
            recent_batches=recent[-recent_limit:] if recent_limit else recent,
            closing=self._closing, wait_ms=0,
            physical_batch_source='worker continuous execution steps; sequence records are not physical batches'))

    async def close(self, error=None):
        self._closing = True
        for queue in self._queues.values():
            for member in queue:
                if not member.future.done():
                    member.future.set_exception(detached_exception(error or RuntimeError('scheduler closing')))
        self._queues.clear()
        self._changed.set()
        if self._pump_task is not None:
            await await_completion(self._pump_task)
