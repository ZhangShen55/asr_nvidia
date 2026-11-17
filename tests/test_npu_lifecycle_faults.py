"""Lifecycle fault windows with CPU threads and NPU-shaped fakes; no hardware."""
import asyncio
import contextlib
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from core.async_utils import run_in_thread
from core.local_inference import InstanceFailure, TorchNpuRunner
from core.models import ModelPool, ModelUnavailableError, local_model_status


class CpuRunner:
    effective_backend = "pytorch_cpu"
    actual_precision = "fp32"

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class LifecycleFaultWindowsTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_load_lock_waiter_does_not_cancel_or_duplicate_loading_owner(self):
        runner = CpuRunner()
        started, complete = threading.Event(), threading.Event()
        pool = ModelPool("speech_campplus", "unused", devices_getter=lambda: ["cpu"])

        def build(_device):
            started.set()
            if not complete.wait(5):
                raise TimeoutError("controlled load completion")
            return runner, "cpu", 0.0

        with patch.object(pool, "_build_instance", side_effect=build) as factory:
            owner = asyncio.create_task(pool.ensure_loaded())
            waiter = None
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                waiter = asyncio.create_task(pool.ensure_loaded())
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                self.assertFalse(waiter.done())
                waiter.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await waiter
                self.assertFalse(owner.done())
                self.assertFalse(runner.closed)
                factory.assert_called_once_with("cpu")
                self.assertEqual(pool.models, [])
                self.assertEqual(pool.waiting, 0)
                complete.set()
                await owner
                self.assertEqual(pool.models, [runner])
                factory.assert_called_once()
            finally:
                complete.set()
                await asyncio.gather(owner, *([waiter] if waiter else []), return_exceptions=True)
                await pool.close()
        self.assertTrue(runner.closed)
        self.assertEqual(pool.models, [])

    async def test_device_to_cpu_copy_fault_waits_for_copy_and_sync_then_quarantines(self):
        # Exercise the real TorchNpuRunner result-read/error/finally path; only
        # its NPU API, network/features and embedding are fakes.
        started, complete = threading.Event(), threading.Event()
        events = []

        class Api:
            conv = SimpleNamespace(allow_hf32=True)
            matmul = SimpleNamespace(allow_hf32=False)
            def set_device(self, device):
                events.append(("select", device))
            def device(self, device):
                self_test.assertEqual(device, "npu:7")
                return contextlib.nullcontext()
            def synchronize(self, device):
                events.append(("sync", device))

        class Embedding:
            def detach(self): return self
            def float(self): return self
            def cpu(self):
                started.set()
                if not complete.wait(5):
                    raise TimeoutError("controlled copy completion")
                events.append(("copy_failed", "npu:7"))
                raise RuntimeError("ACL device-to-host result copy failed")

        self_test = self
        runner = TorchNpuRunner.__new__(TorchNpuRunner)
        runner.api, runner.device, runner.name = Api(), "npu:7", "speech_campplus"
        runner.model = Mock(return_value=Embedding())
        features = SimpleNamespace(to=lambda device: object())
        pool = ModelPool("speech_campplus", "unused", devices_getter=lambda: ["cpu"])
        with patch.object(pool, "_build_instance", return_value=(runner, "npu:7", 0.0)), \
                patch("core.local_inference.campplus_features", return_value=features):
            async def infer():
                async with pool.lease():
                    return await run_in_thread(runner.generate, input=np.zeros(16000, dtype=np.float32), fs=16000)

            async def queued_request():
                async with pool.lease():
                    self.fail("failed copy instance was reissued")

            call = asyncio.create_task(infer())
            waiter = None
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                waiter = asyncio.create_task(queued_request())
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                self.assertFalse(call.done())
                self.assertTrue(pool.locks[0].locked())
                self.assertEqual(pool.waiting, 1)
                self.assertFalse(any(event == "sync" for event, _ in events))
                complete.set()
                with self.assertRaises(InstanceFailure):
                    await call
                with self.assertRaises(ModelUnavailableError):
                    await waiter
                self.assertLess(events.index(("copy_failed", "npu:7")), events.index(("sync", "npu:7")))
                self.assertIsNone(pool.models[0])
                self.assertIsNone(runner.model)
                self.assertFalse(pool.locks[0].locked())
                self.assertEqual(pool.waiting, 0)
                self.assertEqual(local_model_status[pool.name]["busy"], 0)
                self.assertEqual(local_model_status[pool.name]["failed_instances"], 1)
            finally:
                complete.set()
                await asyncio.gather(call, *([waiter] if waiter else []), return_exceptions=True)
                await pool.close()

    async def test_release_condition_wait_repeated_cancel_keeps_lease_until_release_finishes(self):
        runner = CpuRunner()
        pool = ModelPool("emotion2vec", "unused", devices_getter=lambda: ["cpu"])
        with patch.object(pool, "_build_instance", return_value=(runner, "cpu", 0.0)):
            await pool.ensure_loaded()
        entered, finish_body = asyncio.Event(), asyncio.Event()

        async def request():
            async with pool.lease():
                entered.set()
                await finish_body.wait()

        call = asyncio.create_task(request())
        closing = None
        condition_held = False
        try:
            await entered.wait()
            await pool._condition.acquire()
            condition_held = True
            finish_body.set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            # Body is finished, but release cannot yet acquire its condition.
            call.cancel()
            await asyncio.sleep(0)
            call.cancel()
            await asyncio.sleep(0)
            self.assertFalse(call.done())
            self.assertTrue(pool.locks[0].locked())
            self.assertEqual(local_model_status[pool.name]["busy"], 1)
            closing = asyncio.create_task(pool.close())
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertFalse(closing.done())
            self.assertFalse(runner.closed)
            pool._condition.release()
            condition_held = False
            with self.assertRaises(asyncio.CancelledError):
                await call
            await closing
            self.assertTrue(runner.closed)
            self.assertEqual(pool.models, [])
            self.assertEqual(pool.waiting, 0)
            self.assertEqual(local_model_status[pool.name]["busy"], 0)
            self.assertEqual(local_model_status[pool.name]["waiting"], 0)
        finally:
            finish_body.set()
            if condition_held:
                pool._condition.release()
            await asyncio.gather(call, *([closing] if closing else []), return_exceptions=True)
            await pool.close()
