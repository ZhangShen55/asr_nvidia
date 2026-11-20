"""按活动段与PCM字节预算限制流水线，不把增强等待当作Qwen名额。"""
from __future__ import annotations

import asyncio
import copy
import logging
from collections import deque
from dataclasses import dataclass
import time

from core.async_utils import await_completion
from core.config import settings


@dataclass(frozen=True, eq=False)
class PendingLease:
    size: int


class PendingSegmentBudget:
    def __init__(self, limit: int, byte_limit: int):
        self.limit = limit
        self.byte_limit = byte_limit
        self._bytes = 0
        self._leases = set()
        self._waiters = deque()
        self._condition = asyncio.Condition()
        self._peak = 0
        self._peak_bytes = 0
        self._publish()

    def _publish(self):
        self._snapshot = {"limit": self.limit, "byte_limit": self.byte_limit,
                          "active": len(self._leases), "audio_bytes": self._bytes,
                          "waiting": len(self._waiters), "peak": self._peak,
                          "peak_audio_bytes": self._peak_bytes}

    def snapshot(self):
        return dict(self._snapshot)

    async def acquire(self, size: int) -> PendingLease:
        if size < 0 or size > self.byte_limit:
            raise ValueError("单段PCM预算超过pending_audio_bytes_limit，不能无限等待")
        token = object()
        async with self._condition:
            self._waiters.append(token)
            self._publish()
            try:
                await self._condition.wait_for(lambda: self._waiters[0] is token
                    and len(self._leases) < self.limit and self._bytes + size <= self.byte_limit)
                self._waiters.popleft()
                lease = PendingLease(size)
                self._leases.add(lease)
                self._bytes += size
                self._peak = max(self._peak, len(self._leases))
                self._peak_bytes = max(self._peak_bytes, self._bytes)
                self._publish()
                self._condition.notify_all()
                return lease
            except BaseException:
                if token in self._waiters:
                    self._waiters.remove(token)
                self._publish()
                self._condition.notify_all()
                raise

    async def release(self, lease: PendingLease):
        async with self._condition:
            if lease in self._leases:
                self._leases.remove(lease)
                self._bytes -= lease.size
                self._publish()
                self._condition.notify_all()


_budget = None
_budget_loop = None


def get_pending_budget():
    global _budget, _budget_loop
    loop = asyncio.get_running_loop()
    if _budget is None or _budget_loop is not loop:
        _budget = PendingSegmentBudget(settings.qwen_pending_segment_limit,
                                       settings.qwen_pending_audio_bytes_limit)
        _budget_loop = loop
    return _budget


def pending_snapshot():
    return _budget.snapshot() if _budget is not None else {"active": 0, "audio_bytes": 0, "waiting": 0}


async def run_bounded_segments(segments, process, worker_count, *, fail_fast=False):
    """只为有限worker创建协程；最终文本/embedding属于响应，不计为在途PCM缓存。"""
    outcomes = [None] * len(segments)
    first_failure = None
    iterator = iter(enumerate(segments))

    async def worker():
        nonlocal first_failure
        for index, segment in iterator:
            try:
                outcomes[index] = await process(index, segment)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # 响应只需错误类型/code/message；traceback会把已归还预算的PCM持到整音频结束。
                # 不修改原异常/活动frame，避免影响仍收尾的worker或设备所有权。
                error = copy.copy(exc)
                error.__traceback__ = error.__cause__ = error.__context__ = None
                error.__suppress_context__ = True
                outcomes[index] = error
                if fail_fast:
                    if first_failure is None:
                        first_failure = (index, error)
                    raise

    tasks = [asyncio.create_task(worker()) for _ in range(min(len(segments), max(1, worker_count)))]
    try:
        await asyncio.gather(*tasks)
    except Exception:
        if not fail_fast:
            raise
        for task in tasks:
            task.cancel()
        # Preserve the first inference failure as the response cause; cancellations
        # of sibling workers are owned cleanup, not additional inference failures.
        await await_completion(asyncio.gather(*tasks, return_exceptions=True))
        # A sibling cancellation must never replace the first real inference error.
        if first_failure is not None:
            index, error = first_failure
            outcomes = [None] * len(segments)
            outcomes[index] = error
            logging.getLogger(__name__).warning(
                'continuous segment failure index=%d type=%s sibling_cancelled_workers=%d',
                index, type(error).__name__, sum(task.cancelled() for task in tasks))
    except BaseException:
        # 停止继续消费尚未开始的段；等待已启动线程/设备收尾后才退出完整请求。
        for task in tasks:
            task.cancel()
        try:
            await await_completion(asyncio.gather(*tasks, return_exceptions=True))
        except BaseException:
            pass
        raise
    return outcomes


async def gather_owned_branches(awaitables):
    """取消一段时取消未开始分支，并等所有已启动分支收尾再归还PCM预算。"""
    tasks = [asyncio.create_task(item) for item in awaitables]
    try:
        return await asyncio.gather(*tasks, return_exceptions=True)
    except BaseException:
        for task in tasks:
            task.cancel()
        try:
            await await_completion(asyncio.gather(*tasks, return_exceptions=True))
        except BaseException:
            pass
        raise
