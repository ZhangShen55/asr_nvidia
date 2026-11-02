import asyncio
import unittest
from unittest.mock import patch

from core.qwen_asr_queue import AudioSlotManager, QueueFullError, QueueTimeoutError


class WaitingAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def queued(self, manager, count):
        for _ in range(30):
            if manager.waiting == count:
                return
            await asyncio.sleep(0)
        self.fail(f"expected {count} waiters, got {manager.waiting}")

    async def test_a_bypasses_blocked_b_even_when_shared_waiting_queue_is_full(self):
        manager = AudioSlotManager(3, wait_timeout_s=10, queue_max_size=1, spk_emo_capacity=1)
        held = await manager.acquire("B")
        blocked = asyncio.create_task(manager.acquire("B"))
        await self.queued(manager, 1)
        self.assertEqual((manager.active, manager.active_b, manager.waiting_b), (1, 1, 1))
        a = await manager.acquire("A")
        self.assertFalse(blocked.done())
        with self.assertRaises(QueueFullError) as caught:
            await manager.acquire("B")
        self.assertEqual(caught.exception.reason, "waiting_full")
        self.assertEqual(caught.exception.code, 4013)
        await manager.release(held)
        b = await blocked
        await manager.release(a)
        await manager.release(b)
        self.assertEqual((manager.active, manager.active_b, manager.waiting), (0, 0, 0))

    async def test_earliest_eligible_head_wins_and_same_class_is_fifo(self):
        manager = AudioSlotManager(1, wait_timeout_s=10, spk_emo_capacity=1)
        held = await manager.acquire("A")
        requests = []
        for category in ["B", "A", "B", "A"]:
            requests.append(asyncio.create_task(manager.acquire(category)))
            await self.queued(manager, len(requests))
        for index, request in enumerate(requests):
            await manager.release(held)
            held = await request
            self.assertFalse(any(later.done() for later in requests[index+1:]))
        await manager.release(held)
        self.assertEqual(manager.waiting, 0)

    async def test_zero_waiting_capacity_still_allows_immediate_admission(self):
        manager = AudioSlotManager(1, wait_timeout_s=10, queue_max_size=0)
        held = await manager.acquire("B")
        with self.assertRaises(QueueFullError) as caught:
            await manager.acquire("A")
        self.assertEqual(caught.exception.reason, "waiting_full")
        self.assertEqual(manager.waiting, 0)
        await manager.release(held)
        await manager.release(await manager.acquire("A"))

    async def test_cancel_waiting_requests_of_either_class(self):
        for category in ["A", "B"]:
            manager = AudioSlotManager(1, wait_timeout_s=10)
            held = await manager.acquire("A")
            pending = asyncio.create_task(manager.acquire(category))
            await self.queued(manager, 1)
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
            self.assertEqual((manager.active, manager.active_b, manager.waiting), (1, 0, 0))
            await manager.release(held)

    async def test_grant_then_cancel_before_owner_receives_lease(self):
        manager = AudioSlotManager(1, wait_timeout_s=10)
        held = await manager.acquire("A")
        pending = asyncio.create_task(manager.acquire("B"))
        await self.queued(manager, 1)
        # 在授予通知到达、acquire 返回调用方之前精确注入取消。
        manager._queues["B"][0].future.add_done_callback(lambda future: pending.cancel())
        await manager.release(held)
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertEqual((manager.active, manager.active_b, manager.waiting), (0, 0, 0))

    async def test_expired_waiter_is_not_granted_when_release_races_deadline(self):
        clock = [100.0]
        manager = AudioSlotManager(1, wait_timeout_s=10, clock=lambda: clock[0])
        held = await manager.acquire("A")
        pending = asyncio.create_task(manager.acquire("B"))
        await self.queued(manager, 1)
        clock[0] = 110.0
        await manager.release(held)
        with self.assertRaises(QueueTimeoutError):
            await pending
        self.assertEqual((manager.active, manager.active_b, manager.waiting), (0, 0, 0))

    async def test_granted_waiter_is_rolled_back_if_timeout_wins_delivery(self):
        manager = AudioSlotManager(1, wait_timeout_s=10)
        held = await manager.acquire("A")

        async def timeout_after_grant(awaitable, timeout):
            await manager.release(held)
            self.assertEqual(manager.active_b, 1)
            raise asyncio.TimeoutError

        # 精确模拟 wait_for 截止已触发，但管理器刚好完成授予的顺序。
        with patch("core.qwen_asr_queue.asyncio.wait_for", side_effect=timeout_after_grant):
            with self.assertRaises(QueueTimeoutError):
                await manager.acquire("B")
        await manager.release(await manager.acquire("B"))
        self.assertEqual(manager.active, 0)

    async def test_repeated_cancellation_cannot_interrupt_waiter_cleanup(self):
        manager = AudioSlotManager(1, wait_timeout_s=10)
        held = await manager.acquire("B")
        pending = asyncio.create_task(manager.acquire("B"))
        await self.queued(manager, 1)
        async with manager._lock:
            pending.cancel()
            await asyncio.sleep(0)
            pending.cancel()
            await asyncio.sleep(0)
            self.assertFalse(pending.done())
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertEqual(manager.waiting, 0)
        await manager.release(held)
        self.assertEqual((manager.active, manager.active_b), (0, 0))
