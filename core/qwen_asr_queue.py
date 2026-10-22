"""v2 链路的并发控制：音频处理槽位 + 段级/全局请求信号量。"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from core.config import settings

logger = logging.getLogger(__name__)


class QueueFullError(Exception):
    """处理队列已满。"""

    code = 4013
    message = "处理队列已满，请稍后再试"


class QueueTimeoutError(Exception):
    """排队等待超时。"""

    code = 4010
    message = "请求排队超时，请稍后再试"


class AudioSlotManager:
    """同时处理的音频数量控制。

    - ``wait_timeout_s`` 为 0 时不做等待，第 N+1 个请求立即返回队列已满；
    - 大于 0 时最多等待该时长，超时返回排队超时；等待队列容量超限同样返回队列已满。
    """

    def __init__(
        self,
        capacity: int,
        wait_timeout_s: float = 0.0,
        queue_max_size: int = 64,
    ):
        self.capacity = max(1, int(capacity))
        self.wait_timeout_s = max(0.0, float(wait_timeout_s))
        self.queue_max_size = max(0, int(queue_max_size))
        self._active = 0
        self._waiting = 0
        self._condition = asyncio.Condition()

    @property
    def active(self) -> int:
        return self._active

    @property
    def waiting(self) -> int:
        return self._waiting

    async def acquire(self) -> None:
        async with self._condition:
            if self._active < self.capacity and self._waiting == 0:
                self._active += 1
                return

            if self._waiting >= self.queue_max_size:
                logger.warning(f"[并发] 等待队列已满（{self._waiting}/{self.queue_max_size}）")
                raise QueueFullError()

            if self.wait_timeout_s <= 0:
                logger.warning(
                    f"[并发] 处理队列已满（{self._active}/{self.capacity}），直接拒绝"
                )
                raise QueueFullError()

            self._waiting += 1
            deadline = time.monotonic() + self.wait_timeout_s
            try:
                while True:
                    if self._active < self.capacity:
                        self._waiting -= 1
                        self._active += 1
                        return
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        logger.warning("[并发] 排队等待超时")
                        raise QueueTimeoutError()
                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        raise QueueTimeoutError() from None
            except BaseException:
                # 统一在此处回收等待名额，避免超时分支重复扣减
                self._waiting -= 1
                raise

    async def release(self) -> None:
        async with self._condition:
            if self._active > 0:
                self._active -= 1
            self._condition.notify_all()


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


_slot_manager: Optional[AudioSlotManager] = None
_request_concurrency: Optional[RequestConcurrency] = None


def get_slot_manager() -> AudioSlotManager:
    global _slot_manager
    if _slot_manager is None:
        _slot_manager = AudioSlotManager(
            capacity=settings.qwen_audio_concurrency,
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
    "QueueFullError",
    "QueueTimeoutError",
    "RequestConcurrency",
    "get_request_concurrency",
    "get_slot_manager",
    "reset_for_tests",
]
