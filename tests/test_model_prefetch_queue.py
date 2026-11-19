"""共享预取容量、及时补位和取消归属；仅CPU桩，不加载任何模型。"""
import asyncio
import unittest
from unittest.mock import patch
from core.models import ModelPool, ModelUnavailableError, local_model_status
from core.async_utils import await_completion


async def reached(predicate):
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError('controlled scheduling did not reach expected state')


class PrefetchQueueTests(unittest.IsolatedAsyncioTestCase):
    def pool(self, count=1, depth=1, fail_first=False):
        pool = ModelPool('speech_campplus', 'unused', devices_getter=lambda: ['cpu'] * count,
                         prefetch_depth_getter=lambda: depth)
        calls = 0
        def build(device):
            nonlocal calls
            calls += 1
            if fail_first and calls == 1:
                raise RuntimeError('controlled failure')
            return object(), device, 0.0
        pool._build_instance = build
        return pool

    async def test_eight_by_five_means_forty_waiting_plus_eight_running(self):
        pool = self.pool(8, 5)
        # 本用例测排队容量；预先完成异步加载，避免把加载调度误当队列失败。
        await pool.ensure_loaded()
        entered = []; finish = asyncio.Event()
        async def work(index):
            async with pool.lease():
                entered.append(index)
                await finish.wait()
        tasks = [asyncio.create_task(work(i)) for i in range(60)]
        try:
            await reached(lambda: len(entered) == 8 and pool.waiting == 40 and len(pool._admission_waiters) == 12)
            pool.refresh_status(); status = local_model_status[pool.name]
            self.assertEqual(status['waiting_capacity'], 40)
            self.assertEqual(status['busy'], 8)
            self.assertEqual(status['backpressure_waiting'], 12)
            self.assertEqual(status['actual_batch_size'], 1)
            self.assertGreaterEqual(status['oldest_wait_ms'], 0)
            finish.set(); await asyncio.gather(*tasks)
            self.assertEqual(len(set(entered)), 60)
            self.assertEqual(pool.waiting, 0)
            self.assertFalse(pool._admission_waiters)
        finally:
            finish.set(); await asyncio.gather(*tasks, return_exceptions=True); await pool.close()

    async def test_depth_zero_is_fifo_rendezvous_not_unbounded_queue(self):
        pool = self.pool(depth=0); order=[]; gate=asyncio.Event()
        async def work(i):
            async with pool.lease():
                order.append(i)
                if i == 0: await gate.wait()
        tasks=[asyncio.create_task(work(i)) for i in range(12)]
        try:
            await reached(lambda: len(order)==1 and len(pool._admission_waiters)==11)
            self.assertEqual(pool.waiting, 0)
            self.assertEqual(pool.waiting_capacity, 0)
            gate.set(); await asyncio.gather(*tasks)
            self.assertEqual(order, list(range(12)))
        finally:
            gate.set(); await asyncio.gather(*tasks, return_exceptions=True); await pool.close()

    async def test_shared_work_uses_free_instance_before_long_instance_finishes(self):
        pool=self.pool(2, 1); gates=[asyncio.Event(), asyncio.Event()]; order=[]; assigned={}
        async def work(i):
            async with pool.lease() as (model, device):
                assigned[i]=model; order.append(i)
                if i<2: await gates[i].wait()
        tasks=[asyncio.create_task(work(i)) for i in range(8)]
        try:
            await reached(lambda: len(order)==2 and pool.waiting==2)
            gates[1].set()
            await reached(lambda: len(order)==8)
            self.assertFalse(tasks[0].done())
            self.assertTrue(all(assigned[i] is assigned[1] for i in range(2,8)))
            self.assertGreater(pool.handoff_count, 0)
            self.assertEqual(order, list(range(8)))
        finally:
            for gate in gates: gate.set()
            await asyncio.gather(*tasks, return_exceptions=True); await pool.close()

    async def test_queue_and_backpressure_cancellation_leave_no_lease(self):
        pool=self.pool(depth=1); gate=asyncio.Event(); entered=[]
        async def work(i):
            async with pool.lease():
                entered.append(i)
                if i==0: await gate.wait()
        tasks=[asyncio.create_task(work(i)) for i in range(4)]
        try:
            await reached(lambda: entered==[0] and pool.waiting==1 and len(pool._admission_waiters)==2)
            for index in [1,2]:
                tasks[index].cancel()
                with self.assertRaises(asyncio.CancelledError): await tasks[index]
            gate.set(); await asyncio.gather(tasks[0],tasks[3])
            self.assertEqual(entered,[0,3]); self.assertEqual(pool.waiting,0)
            self.assertFalse(pool._admission_waiters)
            self.assertFalse(any(lock.locked() for lock in pool.locks))
        finally:
            gate.set(); await asyncio.gather(*tasks,return_exceptions=True); await pool.close()

    async def test_repeated_cancel_retains_running_lease_until_owned_work_completes(self):
        pool=self.pool(depth=0); started=asyncio.Event(); done=asyncio.Event(); next_started=asyncio.Event()
        async def owned():
            started.set(); await done.wait()
        async def first():
            async with pool.lease(): await await_completion(owned())
        async def second():
            async with pool.lease(): next_started.set()
        first_task=asyncio.create_task(first()); await started.wait()
        second_task=asyncio.create_task(second())
        try:
            first_task.cancel(); await asyncio.sleep(0); first_task.cancel(); await asyncio.sleep(0)
            self.assertTrue(pool.locks[0].locked()); self.assertFalse(next_started.is_set())
            done.set()
            with self.assertRaises(asyncio.CancelledError): await first_task
            await second_task
            self.assertFalse(pool.locks[0].locked())
        finally:
            done.set(); await asyncio.gather(first_task,second_task,return_exceptions=True); await pool.close()

    async def test_failed_instance_does_not_contribute_waiting_capacity(self):
        pool=self.pool(3, 5, fail_first=True)
        try:
            await pool.ensure_loaded(); self.assertEqual(pool.waiting_capacity,10)
            self.assertEqual(local_model_status[pool.name]['failed_instances'],1)
        finally: await pool.close()

    async def test_close_wakes_backpressured_producers_and_drains_active(self):
        pool=self.pool(depth=0); gate=asyncio.Event(); active=asyncio.Event()
        async def first():
            async with pool.lease(): active.set(); await gate.wait()
        async def second():
            async with pool.lease(): self.fail('closing pool reissued lease')
        a=asyncio.create_task(first()); await active.wait(); b=asyncio.create_task(second())
        await reached(lambda: len(pool._admission_waiters)==1)
        closing=asyncio.create_task(pool.close())
        try:
            await reached(lambda: b.done())
            with self.assertRaises(ModelUnavailableError): await b
            self.assertFalse(closing.done()); gate.set(); await a; await closing
            self.assertFalse(pool._admission_waiters)
        finally:
            gate.set(); await asyncio.gather(a,b,closing,return_exceptions=True)
