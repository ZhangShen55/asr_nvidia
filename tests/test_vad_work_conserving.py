"""精确驱动真实VAD池消息与调度；不spawn、不加载模型、不访问NPU。"""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from core.vad_pool import VadPool, _Worker, _MSG_IDLE, _MSG_STARTED, _MSG_FINISHED


class TaskQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


async def scheduled(predicate):
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError('controlled VAD schedule did not reach expected state')


def controlled_pool(count, healthy=None):
    healthy = count if healthy is None else healthy
    pool = VadPool(devices=['cpu'] * count, model_dir='unused')
    pool._started = True
    pool._refresh_vad_status = Mock()
    for index in range(count):
        pool._workers[index] = _Worker(
            worker_id=index,
            process=SimpleNamespace(is_alive=lambda: True),
            stop_event=Mock(), task_queue=TaskQueue(), device='cpu')
        pool._on_message(_MSG_IDLE, -1, index,
                         'ok' if index < healthy else 'error',
                         {'allocated_mb': 0.0} if index < healthy else 'controlled warmup failure')
    return pool


def finish(pool, worker_id):
    worker = pool._workers[worker_id]
    task_id = worker.busy_task_id
    if task_id is not None:
        pool._on_message(_MSG_STARTED, task_id, worker_id, 'ok', None)
        pool._on_message(_MSG_FINISHED, task_id, worker_id, 'ok',
                         [{'key': f'full-audio-{task_id}', 'value': [[0, 1000]]}])
    return task_id


async def drain(pool, tasks, downstream=None):
    if downstream is not None:
        downstream.set()
    for _ in range(20):
        for worker_id in pool._workers:
            finish(pool, worker_id)
        await asyncio.sleep(0)
        if not pool._tasks:
            break
    for task in tasks:
        if not task.done() and pool._tasks:
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    for pending in pool._tasks.values():
        pending.finished.cancel()
    pool._tasks.clear()
    pool._queue.clear()


class VadWorkConservingTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_of_four_finished_immediately_dispatches_fifth_without_downstream(self):
        pool = controlled_pool(4)
        downstream = asyncio.Event()
        vad_done = []
        async def full_request(index):
            result = await pool.detect(f'complete-source-{index}.wav')
            vad_done.append(index)
            await downstream.wait()  # 代表仍在Qwen/CAM/emo流程，尚未完成整音频。
            return result
        tasks = [asyncio.create_task(full_request(index)) for index in range(8)]
        try:
            await scheduled(lambda: pool.stats()['busy'] == 4 and pool.stats()['waiting'] == 4)
            self.assertEqual([w.busy_task_id for w in pool._workers.values()], [0,1,2,3])
            self.assertEqual([task_id for task_id, _ in pool._queue], [4,5,6,7])
            # 精确FINISHED回调内就补位，尚未让上层协程获得一次调度机会。
            self.assertEqual(finish(pool, 2), 2)
            self.assertEqual(pool._workers[2].task_queue.items,
                             [(2, 'complete-source-2.wav'), (4, 'complete-source-4.wav')])
            self.assertEqual(pool._workers[2].busy_task_id, 4)
            self.assertEqual([pool._workers[i].busy_task_id for i in [0,1,3]], [0,1,3])
            self.assertEqual(pool.stats()['busy'], 4)
            self.assertEqual(pool.stats()['waiting'], 3)
            self.assertFalse(downstream.is_set())
            self.assertFalse(tasks[2].done())
            await scheduled(lambda: vad_done == [2])
            self.assertFalse(tasks[2].done())
            # 同一个先空闲worker可继续接力第六条，无需等待最初其余三条。
            self.assertEqual(finish(pool, 2), 4)
            self.assertEqual(pool._workers[2].busy_task_id, 5)
            self.assertEqual([task_id for task_id, _ in pool._queue], [6,7])
        finally:
            await drain(pool, tasks, downstream)
        self.assertTrue(all(task.done() and not task.cancelled() for task in tasks))
        self.assertEqual(pool.stats()['waiting'], 0)
        self.assertEqual(pool.stats()['busy'], 0)
        self.assertEqual(sorted(vad_done), list(range(8)))

    async def test_eight_full_audios_have_eight_actual_busy_workers(self):
        pool = controlled_pool(8)
        tasks = [asyncio.create_task(pool.detect(f'complete-{index}.wav')) for index in range(8)]
        try:
            await scheduled(lambda: len(pool._tasks) == 8)
            stats = pool.stats()
            self.assertEqual(stats['configured_instances'], 8)
            self.assertEqual(stats['loaded_instances'], 8)
            self.assertEqual(stats['busy'], 8)
            self.assertEqual(stats['available_instances'], 0)
            self.assertEqual(stats['waiting'], 0)
            self.assertEqual(len({w.busy_task_id for w in pool._workers.values()}), 8)
            self.assertTrue(all(len(w.task_queue.items) == 1 for w in pool._workers.values()))
        finally:
            await drain(pool, tasks)
        self.assertEqual(pool.stats()['busy'], 0)
        self.assertEqual(pool._tasks, {})

    async def test_failed_warmup_workers_are_not_dispatched_or_counted_healthy(self):
        pool = controlled_pool(8, healthy=4)
        tasks = [asyncio.create_task(pool.detect(f'complete-{index}.wav')) for index in range(8)]
        try:
            await scheduled(lambda: len(pool._tasks) == 8)
            self.assertEqual(pool.stats()['configured_instances'], 8)
            self.assertEqual(pool.stats()['loaded_instances'], 4)
            self.assertEqual(pool.stats()['failed_instances'], 4)
            self.assertEqual(pool.stats()['busy'], 4)
            self.assertEqual(pool.stats()['waiting'], 4)
            self.assertTrue(all(not pool._workers[i].task_queue.items for i in range(4,8)))
            finish(pool, 0)
            self.assertEqual(pool._workers[0].busy_task_id, 4)
            self.assertTrue(all(not pool._workers[i].task_queue.items for i in range(4,8)))
        finally:
            await drain(pool, tasks)
        self.assertEqual(pool.stats()['waiting'], 0)

    async def test_cancelled_queued_full_audios_are_removed_before_next_dispatch(self):
        pool = controlled_pool(4)
        tasks = [asyncio.create_task(pool.detect(f'complete-{index}.wav')) for index in range(8)]
        try:
            await scheduled(lambda: pool.stats()['waiting'] == 4)
            for index in [4,6]:
                tasks[index].cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await tasks[index]
            self.assertEqual(pool.stats()['waiting'], 2)
            self.assertEqual([task_id for task_id, _ in pool._queue], [5,7])
            self.assertNotIn(4, pool._tasks)
            self.assertNotIn(6, pool._tasks)
            finish(pool, 0)
            self.assertEqual(pool._workers[0].busy_task_id, 5)
            finish(pool, 1)
            self.assertEqual(pool._workers[1].busy_task_id, 7)
            dispatched = [task_id for w in pool._workers.values() for task_id, _ in w.task_queue.items]
            self.assertNotIn(4, dispatched)
            self.assertNotIn(6, dispatched)
        finally:
            await drain(pool, tasks)
        self.assertEqual(pool.stats()['busy'], 0)
        self.assertEqual(pool.stats()['waiting'], 0)
        self.assertFalse(pool._tasks)
