import asyncio
import unittest
from unittest.mock import Mock, patch

from core.vad_pool import VadPool, VadPoolError


FAKE_BUILDER = "tests.fake_vad:build_fake_model"
FAKE_RUNNER = "tests.fake_vad:run_vad_task"


def make_pool(devices):
    return VadPool(
        devices=devices,
        model_dir="./model/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        builder_spec=FAKE_BUILDER,
        runner_spec=FAKE_RUNNER,
        start_timeout_s=10,
    )


class StaticVadPoolTests(unittest.TestCase):
    def test_device_array_is_fixed_and_fifo_waits(self):
        async def scenario():
            pool = make_pool(["cpu", "cpu"])
            try:
                await pool.start()
                self.assertEqual(pool.stats()["configured_instances"], 2)
                self.assertEqual(pool.stats()["loaded_instances"], 2)
                tasks = [asyncio.create_task(pool.detect(f"sleep:0.2")) for _ in range(3)]
                await asyncio.sleep(0.05)
                self.assertGreaterEqual(pool.stats()["waiting"], 1)
                await asyncio.gather(*tasks)
                self.assertEqual(pool.stats()["waiting"], 0)
            finally:
                await pool.stop()

        asyncio.run(scenario())

    def test_cancelled_waiting_request_is_removed(self):
        async def scenario():
            pool = make_pool(["cpu"])
            try:
                await pool.start()
                first = asyncio.create_task(pool.detect("sleep:0.5"))
                await asyncio.sleep(0.1)
                second = asyncio.create_task(pool.detect("ok"))
                await asyncio.sleep(0.05)
                self.assertEqual(pool.stats()["waiting"], 1)
                second.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await second
                self.assertEqual(pool.stats()["waiting"], 0)
                await first
            finally:
                await pool.stop()

        asyncio.run(scenario())

    def test_stop_drains_late_large_results_and_closes_spawn_resources(self):
        async def scenario():
            pool = make_pool(["cpu"])
            task = None
            try:
                await pool.start()
                worker = next(iter(pool._workers.values()))
                process = worker.process
                task = asyncio.create_task(pool.detect("large:0.5"))
                await asyncio.sleep(0.1)
                with patch.object(process, "terminate", wraps=process.terminate) as terminate:
                    await pool.stop(timeout_s=2)
                    terminate.assert_not_called()
                self.assertTrue(process._closed)
                self.assertIsNone(worker.task_queue)
                self.assertIsNone(worker.stop_event)
                self.assertIsNone(pool._result_queue)
                self.assertIsNone(pool._reader_thread)
                try:
                    await task
                except VadPoolError:
                    pass  # stop rejects an in-flight request after draining IPC.
            finally:
                await pool.stop()
                if task is not None and not task.done():
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task

        asyncio.run(scenario())

    def test_partial_spawn_failure_closes_previously_started_workers(self):
        async def scenario():
            pool = make_pool(["cpu", "cpu"])
            original = pool._spawn_worker
            workers = []

            def spawn(device):
                if workers:
                    raise RuntimeError("synthetic second spawn failure")
                worker = original(device)
                workers.append(worker)
                return worker

            with patch.object(pool, "_spawn_worker", side_effect=spawn):
                with self.assertRaisesRegex(RuntimeError, "second spawn failure"):
                    await pool.start()
            self.assertFalse(pool._workers)
            self.assertTrue(workers[0].process._closed)
            self.assertIsNone(pool._result_queue)
            self.assertIsNone(pool._reader_thread)

        asyncio.run(scenario())

    def test_forced_worker_cleanup_is_failure_and_other_workers_still_close(self):
        from types import SimpleNamespace

        class Process:
            def __init__(self, pid, forced=False, needs_kill=False):
                self.pid, self.alive, self.needs_kill = pid, forced, needs_kill
                self.exitcode, self._closed, self.actions = (None if forced else 0), False, []
            def join(self, timeout):
                pass
            def is_alive(self):
                return self.alive
            def terminate(self):
                self.actions.append("terminate")
                if not self.needs_kill:
                    self.alive, self.exitcode = False, -15
            def kill(self):
                self.actions.append("kill")
                self.alive, self.exitcode = False, -9
            def close(self):
                self._closed = True

        async def scenario(needs_kill):
            pool = make_pool(["cpu", "cpu"])
            workers = [SimpleNamespace(worker_id=index, device="cpu", process=process,
                                       stop_event=Mock(), task_queue=Mock())
                       for index, process in enumerate((Process(100001, forced=True, needs_kill=needs_kill), Process(100002)))]
            queues = [worker.task_queue for worker in workers]
            pool._workers = {worker.worker_id: worker for worker in workers}
            with self.assertLogs("core.vad_pool", level="INFO") as captured:
                with self.assertRaisesRegex(VadPoolError, "关闭超时后被强制回收"):
                    await pool.stop(timeout_s=0.01)
            self.assertFalse(pool._workers)
            self.assertTrue(all(worker.process._closed for worker in workers))
            self.assertTrue(all(worker.task_queue is None and worker.stop_event is None for worker in workers))
            for queue in queues:
                queue.close.assert_called_once()
                queue.join_thread.assert_called_once()
            forced, graceful = pool._shutdown_evidence
            self.assertEqual(forced["pid"], 100001)
            self.assertEqual(forced["device"], "cpu")
            self.assertTrue(forced["forced"])
            self.assertTrue(forced["cleanup_complete"])
            self.assertEqual(forced["forced_actions"], ["terminate", "kill"] if needs_kill else ["terminate"])
            self.assertEqual(forced["exitcode"], -9 if needs_kill else -15)
            self.assertFalse(graceful["forced"])
            self.assertEqual(graceful["exitcode"], 0)
            self.assertTrue(any("VAD_WORKER_SHUTDOWN" in line for line in captured.output))
            # A second empty stop cannot erase the previous timeout evidence.
            with self.assertRaises(VadPoolError):
                await pool.stop()
            self.assertEqual(len(pool._shutdown_evidence), 2)

        for needs_kill in (False, True):
            with self.subTest(needs_kill=needs_kill):
                asyncio.run(scenario(needs_kill))

    def test_nonzero_shutdown_exit_is_failure_but_crash_reaping_is_explicit(self):
        from types import SimpleNamespace

        def worker(exitcode):
            process = Mock()
            process.pid = 100003
            process.exitcode = exitcode
            process.is_alive.return_value = False
            return SimpleNamespace(worker_id=0, device="cpu", process=process,
                                   stop_event=Mock(), task_queue=Mock())

        async def scenario():
            pool = make_pool(["cpu"])
            item = worker(1)
            pool._workers = {0: item}
            with self.assertLogs("core.vad_pool", level="INFO"):
                with self.assertRaisesRegex(VadPoolError, "非零退出"):
                    await pool.stop()
            item.process.close.assert_called_once()
            self.assertFalse(pool._workers)
            self.assertIsNone(item.task_queue)
            self.assertFalse(pool._shutdown_evidence[0]["forced"])
            self.assertEqual(pool._shutdown_evidence[0]["exitcode"], 1)
            self.assertTrue(pool._shutdown_evidence[0]["cleanup_complete"])

        asyncio.run(scenario())
        # A previously reported operational crash has a separate lifecycle
        # reason and can be reaped before replacement without faking shutdown.
        pool = make_pool(["cpu"])
        item = worker(17)
        pool._workers = {0: item}
        pool._stop_worker(item, timeout_s=0, reason="reap_crashed")
        self.assertEqual(pool._shutdown_evidence[0]["reason"], "reap_crashed")
        self.assertEqual(pool._shutdown_evidence[0]["exitcode"], 17)
        self.assertFalse(pool._workers)

    def test_empty_device_array_is_rejected(self):
        async def scenario():
            pool = make_pool([])
            with self.assertRaises(VadPoolError):
                await pool.start()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
