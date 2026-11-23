"""v3 链路的并发控制：分类音频准入 + 段级/全局请求信号量。"""

from __future__ import annotations

import asyncio
import copy
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from core.async_utils import await_completion
from core.config import settings


class QueueFullError(Exception):
    code = 4013

    def __init__(self, *, active=None, capacity=None, reason="total_full", snapshot=None):
        self.active = active
        self.capacity = capacity
        self.reason = reason
        self.snapshot = snapshot or {}
        if reason == "waiting_full":
            self.message = "服务繁忙，请稍后再试"
        elif None in (active, capacity):
            self.message = "处理队列已满，请稍后再试"
        else:
            self.message = f"处理队列已满，请稍后再试（当前音频并发：{active}/{capacity}）"
        super().__init__(self.message)


class SpkEmoQueueFullError(QueueFullError):
    code = 4017

    def __init__(self, *, active, capacity, snapshot=None):
        super().__init__(active=active, capacity=capacity,
                         reason="spk_emo_full", snapshot=snapshot)
        self.message = f"spk_emo队列已满，请稍后再试（当前增强音频并发：{active}/{capacity}）"
        self.args = (self.message,)


class QueueTimeoutError(Exception):
    code = 4010
    message = "请求排队超时，请稍后再试"
    reason = "wait_timeout"

    def __init__(self, *, snapshot=None):
        self.snapshot = snapshot or {}
        super().__init__(self.message)


@dataclass(frozen=True, eq=False)
class AudioLease:
    """以对象身份标识名额；调用方不能修改取得时的分类。"""

    category: str
    snapshot: dict
    request_id: str | None = None
    owner_index: int | None = None


@dataclass(eq=False)
class _AudioWaiter:
    category: str
    sequence: int
    deadline: float
    future: asyncio.Future
    state: str = "waiting"
    lease: AudioLease | None = None
    request_id: str | None = None


class AudioSlotManager:
    """总音频与 B 音频原子准入；共享有界等待队列，最早可处理请求优先。"""

    def __init__(self, capacity: int, wait_timeout_s: float = 0.0,
                 queue_max_size: int = 64, *, spk_emo_capacity: int = 4, clock=None):
        if type(spk_emo_capacity) is not int or spk_emo_capacity <= 0:
            raise ValueError("audio_spk_emo_concurrency 必须是正整数")
        self.capacity = max(1, int(capacity))
        self.spk_emo_capacity = spk_emo_capacity
        self.wait_timeout_s = max(0.0, float(wait_timeout_s))
        self.queue_max_size = max(0, int(queue_max_size))
        self._clock = clock or time.monotonic
        self._active = 0
        self._active_b = 0
        self._queues = {"A": deque(), "B": deque()}
        self._sequence = 0
        self._leases: set[AudioLease] = set()
        self._lock = asyncio.Lock()
        self._owner_pool = None
        self._publish()

    def bind_owner_pool(self, client) -> None:
        """No I/O: admission and owner assignment subsequently share this lock."""
        if getattr(client, 'generation_mode', None) != 'continuous':
            return
        if self._owner_pool is not None and self._owner_pool is not client:
            old = self._owner_pool
            task = getattr(old, '_close_task', None)
            idle = not (self._active or self._active_b or self.waiting or self._leases or self._lock.locked())
            closed = (task is not None and task.done() and not task.cancelled()
                      and task.exception() is None
                      and getattr(old, '_shutdown_state', {}).get('phase') == 'closed')
            if not idle or not closed:
                raise RuntimeError('audio manager cannot replace a live or closing owner pool')
            previous = old.owner_capacity_snapshot()
            reaped = all(owner.get('model_worker', {}).get('resource_reaped') is True
                         and (owner.get('model_worker', {}).get('worker_alive') is False
                              or (owner.get('model_worker', {}).get('worker_pid') is None
                                  and owner.get('model_worker', {}).get('worker_alive') is None))
                         for owner in previous['owners'])
            if previous['leased'] or not reaped:
                raise RuntimeError('old owner pool resources are not confirmed reaped')
        if client.owner_capacity_snapshot()['configured'] != self.capacity:
            raise ValueError('audio capacity must equal configured Qwen owners')
        self._owner_pool = client
        self._publish()

    def _owner_capacity(self):
        return self._owner_pool.owner_capacity_snapshot() if self._owner_pool is not None else None

    @property
    def active(self) -> int:
        return self._active

    @property
    def active_b(self) -> int:
        return self._active_b

    @property
    def waiting(self) -> int:
        return sum(len(queue) for queue in self._queues.values())

    @property
    def waiting_b(self) -> int:
        return len(self._queues["B"])

    def _state(self) -> dict:
        state = {
            "audioConcurrency": {"configured_total": self.capacity,
                                 "active_total": self._active, "waiting_total": self.waiting},
            "spkEmoConcurrency": {"configured": self.spk_emo_capacity,
                                  "active": self._active_b, "waiting": self.waiting_b},
        }
        owners = self._owner_capacity()
        if owners is not None:
            state['audioConcurrency'].update(effective_total=owners['healthy'],
                healthy_owners=owners['healthy'], leased_owners=owners['leased'],
                available_owners=owners['available'])
            state['spkEmoConcurrency']['effective'] = min(self.spk_emo_capacity, owners['healthy'])
        return state

    def _publish(self) -> None:
        # 一次替换整份快照，供同步状态接口（可能在线程中）读取。
        self._snapshot = self._state()

    def snapshot(self) -> dict:
        state = copy.deepcopy(self._snapshot)
        owners = self._owner_capacity()
        if owners is not None:
            state['audioConcurrency'].update(effective_total=owners['healthy'],
                healthy_owners=owners['healthy'], leased_owners=owners['leased'],
                available_owners=owners['available'])
            state['spkEmoConcurrency']['effective'] = min(self.spk_emo_capacity, owners['healthy'])
        return state

    def _eligible(self, category: str) -> bool:
        owners = self._owner_capacity()
        total = self.capacity if owners is None else owners['healthy']
        b_capacity = min(self.spk_emo_capacity, total)
        # A dead owner's closing lease still counts active. It must not prevent
        # a different healthy/free owner from being granted within configured N.
        return (self._active < self.capacity and (owners is None or owners['available'] > 0)
                and (category == "A" or self._active_b < b_capacity))

    def _grant(self, category: str, request_id=None) -> AudioLease | None:
        owner_index = None
        if self._owner_pool is not None:
            if request_id is None:
                raise ValueError('continuous audio admission requires request_id')
            owner_index = self._owner_pool.try_acquire_owner(str(request_id))
            if owner_index is None:
                return None
        self._active += 1
        self._active_b += int(category == "B")
        lease = AudioLease(category, self._state(), str(request_id) if request_id is not None else None,
                           owner_index)
        self._leases.add(lease)
        return lease

    def _release(self, lease: AudioLease) -> None:
        if lease in self._leases:
            if self._owner_pool is not None and lease.request_id is not None:
                self._owner_pool.release_owner_nowait(lease.request_id)
            self._leases.remove(lease)
            self._active -= 1
            self._active_b -= int(lease.category == "B")

    def _dispatch(self) -> None:
        now = self._clock()
        for queue in self._queues.values():
            for waiter in tuple(queue):
                if waiter.future.cancelled() or waiter.deadline <= now:
                    queue.remove(waiter)
                    waiter.state = "expired"
                    if not waiter.future.done():
                        waiter.future.set_exception(QueueTimeoutError(snapshot=self._state()))
        owners = self._owner_capacity()
        if owners is not None and not owners['healthy']:
            from core.qwen_asr_client import QwenAsrUnavailableError
            for queue in self._queues.values():
                while queue:
                    waiter = queue.popleft()
                    waiter.state = 'expired'
                    if not waiter.future.done():
                        waiter.future.set_exception(QwenAsrUnavailableError())
            self._publish()
            return
        while self._active < self.capacity:
            candidates = [queue[0] for category, queue in self._queues.items()
                          if queue and self._eligible(category)]
            if not candidates:
                break
            waiter = min(candidates, key=lambda entry: entry.sequence)
            self._queues[waiter.category].popleft()
            waiter.lease = self._grant(waiter.category, waiter.request_id)
            if waiter.lease is None:
                self._queues[waiter.category].appendleft(waiter)
                break
            waiter.state = "granted"
            waiter.future.set_result(waiter.lease)
        self._publish()

    async def acquire(self, category: str = "A", *, request_id=None) -> AudioLease:
        if category not in self._queues:
            raise ValueError("音频分类必须是 A 或 B")
        if self._owner_pool is not None:
            if request_id is None:
                raise ValueError('continuous audio admission requires request_id')
            # Lazy discovery must never hold admission while awaiting model IO.
            await self._owner_pool.ensure_owner_pool()
        async with self._lock:
            self._dispatch()
            # Discovery happened before this lock. All owners can fail while a
            # new caller waits for it; that caller must get 4011 just like the
            # queued waiters, rather than a misleading capacity-zero 4013.
            owners = self._owner_capacity()
            if owners is not None and not owners['healthy']:
                from core.qwen_asr_client import QwenAsrUnavailableError
                raise QwenAsrUnavailableError()
            if self._eligible(category) and not self._queues[category]:
                lease = self._grant(category, request_id)
                if lease is not None:
                    self._publish()
                    return lease
            if self.wait_timeout_s <= 0:
                owners = self._owner_capacity()
                total = self.capacity if owners is None else owners['healthy']
                if self._active >= self.capacity or (owners is not None and not owners['available']):
                    raise QueueFullError(active=self._active, capacity=total, snapshot=self._state())
                raise SpkEmoQueueFullError(active=self._active_b, capacity=min(self.spk_emo_capacity,total),
                                          snapshot=self._state())
            if self.waiting >= self.queue_max_size:
                raise QueueFullError(reason="waiting_full", snapshot=self._state())
            self._sequence += 1
            waiter = _AudioWaiter(category, self._sequence, self._clock() + self.wait_timeout_s,
                                  asyncio.get_running_loop().create_future(), request_id=request_id)
            self._queues[category].append(waiter)
            self._publish()
        try:
            # Python 3.10 的 wait_for 在授予与取消竞争时可能吞掉取消，
            # 且 Task.cancelling() 尚不存在。wait 不取消通知对象，并正常传播外层取消。
            done, _ = await asyncio.wait({waiter.future},
                                         timeout=max(0.0, waiter.deadline - self._clock()))
            if not done:
                raise asyncio.TimeoutError
            lease = waiter.future.result()
            return lease
        except asyncio.TimeoutError:
            snapshot = await await_completion(self._withdraw(waiter))
            raise QueueTimeoutError(snapshot=snapshot) from None
        except BaseException:
            await await_completion(self._withdraw(waiter))
            raise

    async def _withdraw(self, waiter: _AudioWaiter) -> dict:
        async with self._lock:
            if waiter.state == "waiting":
                self._queues[waiter.category].remove(waiter)
            elif waiter.state == "granted":
                self._release(waiter.lease)
            waiter.state = "withdrawn"
            if not waiter.future.done():
                waiter.future.cancel()
            elif not waiter.future.cancelled():
                waiter.future.exception()  # 超时通知与取消竞争时也消费异常。
            self._dispatch()
            return self.snapshot()

    async def release(self, lease: AudioLease) -> None:
        if self._owner_pool is not None and lease.request_id is not None and lease in self._leases:
            await await_completion(self._owner_pool.prepare_owner_release(lease.request_id))
        await await_completion(self._release_and_dispatch(lease))

    async def _release_and_dispatch(self, lease: AudioLease) -> None:
        async with self._lock:
            self._release(lease)
            self._dispatch()


class RequestConcurrency:
    """段级（单音频）与全局两级请求信号量。"""

    def __init__(
        self,
        segment_concurrency: Optional[int] = None,
        request_concurrency: Optional[int] = None,
    ):
        self.segment_concurrency = (
            settings.qwen_segment_concurrency if segment_concurrency is None else segment_concurrency
        )
        self.request_concurrency = (
            settings.qwen_request_concurrency if request_concurrency is None else request_concurrency
        )
        self._global = asyncio.Semaphore(max(1, self.request_concurrency))

    def new_segment_semaphore(self) -> asyncio.Semaphore:
        """为单个音频创建段级信号量。"""
        return asyncio.Semaphore(max(1, self.segment_concurrency))

    async def acquire(self, segment_semaphore: asyncio.Semaphore) -> None:
        """先拿段级名额，再拿全局名额。"""
        await segment_semaphore.acquire()
        try:
            await self._global.acquire()
        except BaseException:
            segment_semaphore.release()
            raise

    async def release(self, segment_semaphore: asyncio.Semaphore) -> None:
        self._global.release()
        segment_semaphore.release()

    async def acquire_global(self) -> None:
        """只占全局请求名额。

        段级名额由调用方提前持有时使用：惰性裁剪需要与「在途段数」同窗口限流，
        否则所有分段会在裁剪阶段同时落盘，临时文件峰值又回到与分段总数成正比。
        """
        await self._global.acquire()

    async def release_global(self) -> None:
        self._global.release()


_slot_manager: Optional[AudioSlotManager] = None
_request_concurrency: Optional[RequestConcurrency] = None


def get_slot_manager() -> AudioSlotManager:
    global _slot_manager
    if _slot_manager is None:
        _slot_manager = AudioSlotManager(
            capacity=settings.qwen_audio_concurrency,
            spk_emo_capacity=settings.qwen_audio_spk_emo_concurrency,
            wait_timeout_s=settings.qwen_queue_wait_timeout_s,
            queue_max_size=settings.qwen_queue_max_size,
        )
    return _slot_manager


def get_request_concurrency() -> RequestConcurrency:
    global _request_concurrency
    if _request_concurrency is None:
        _request_concurrency = RequestConcurrency()
    return _request_concurrency


def reset_for_tests() -> None:
    """测试用：重置进程级单例。"""
    global _slot_manager, _request_concurrency
    _slot_manager = None
    _request_concurrency = None


__all__ = [
    "AudioSlotManager",
    "AudioLease",
    "SpkEmoQueueFullError",
    "QueueFullError",
    "QueueTimeoutError",
    "RequestConcurrency",
    "get_request_concurrency",
    "get_slot_manager",
    "reset_for_tests",
]
