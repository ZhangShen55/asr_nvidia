import asyncio
import contextlib
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from core.async_utils import run_in_thread
from core.local_inference import InferenceRunner, InstanceFailure, backend_for_device, generate_local
from core.models import ModelPool, ModelUnavailableError, _resolve_device, local_model_status


class FakeRunner(InferenceRunner):
    def __init__(self, *args, **kwargs):
        self.effective_backend = "torch_npu"
        self.actual_precision = "fp32"
        self.closed = False

    def generate(self, **kwargs):
        return [{"scores": [1.0]}]

    def close(self):
        self.closed = True


class BackendSelectionTests(unittest.TestCase):
    def test_npu_constructor_isolates_cuda_and_tensorrt(self):
        from core.local_inference import TorchNpuRunner

        api = Mock()
        api.device.return_value = contextlib.nullcontext()
        import torch
        model = torch.nn.Linear(2, 2)
        model.to = Mock(return_value=model)
        with patch("core.local_inference.npu_api", return_value=api), \
                patch("core.local_inference.load_native_network", return_value=model), \
                patch("torch.cuda.device", side_effect=AssertionError("CUDA called")), \
                patch("torch.cuda.is_available", side_effect=AssertionError("CUDA queried")):
            runner = TorchNpuRunner("speech_campplus", "/model/cam", "npu:1")
            self.assertEqual(runner.effective_backend, "torch_npu")
            self.assertEqual(runner.actual_precision, "fp32")
            model.to.assert_called_once_with("npu:1")
            runner.close()

    def test_devices_are_cpu_or_npu_and_cuda_is_rejected(self):
        self.assertEqual(backend_for_device("npu:7"), "torch_npu")
        self.assertEqual(backend_for_device("cpu"), "pytorch_cpu")
        for value in ("cuda:0", "npu:-1", "npu:x", "gpu:0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                backend_for_device(value)

    def test_resolve_npu_never_queries_cuda(self):
        api = Mock()
        api.is_available.return_value = True
        api.device_count.return_value = 3
        with patch("core.models.npu_api", return_value=api), \
                patch("torch.cuda.is_available", side_effect=AssertionError("CUDA called")):
            self.assertEqual(_resolve_device("npu:2"), "npu:2")

    def test_runner_manages_own_context_and_precision(self):
        with patch("torch.amp.autocast", side_effect=AssertionError("outer AMP called")), \
                patch("torch.cuda.device", side_effect=AssertionError("outer CUDA called")):
            self.assertEqual(generate_local(FakeRunner(), "npu:7", input=None)[0]["scores"], [1.0])

    def test_input_validation(self):
        from core.local_inference import waveform_tensor

        for value in (np.zeros((1, 10), np.float32), np.zeros(0, np.float32),
                      np.array([np.nan], np.float32), np.zeros(10, np.int16)):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    waveform_tensor(value, 16000)


class BackendPoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_four_cpu_npu_combinations_preserve_route_outputs(self):
        import torch
        from api.routes import asr_qwen_common as route
        from core.config import settings

        class Model:
            def generate(self, **kwargs):
                return [{"spk_embedding": np.ones(192, np.float32),
                         "labels": ["开心/happy", "中立/neutral"], "scores": [0.9, 0.1]}]

        class Npu(FakeRunner):
            def generate(self, **kwargs):
                return Model().generate(**kwargs)

        for cam_device, emo_device in [("cpu", "cpu"), ("npu:7", "cpu"), ("cpu", "npu:7"), ("npu:7", "npu:7")]:
            with self.subTest(cam=cam_device, emo=emo_device), \
                    patch("core.models._resolve_device", side_effect=lambda d: d), \
                    patch("core.models.AutoModel", side_effect=lambda **kw: Model()), \
                    patch("core.models.TorchNpuRunner", side_effect=lambda *a: Npu()), \
                    patch("core.models.device_allocated_mb", return_value=0.0), \
                    patch.dict(settings._cfg, {"features": {"open_v2_role": True, "open_v2_emotion": True}}):
                pools = {name: ModelPool(name, "/model/test", devices_getter=lambda d=device: [d])
                         for name, device in [("speech_campplus", cam_device), ("emotion2vec", emo_device)]}
                with patch.object(route, "get_model_pool", side_effect=lambda name: pools[name]):
                    embedding, emotion = await asyncio.gather(
                        route._speaker_embedding(np.zeros(16000, np.float32), 16000),
                        route._segment_emotion(np.zeros(16000, np.float32), 16000, 1.0))
                self.assertEqual(embedding.shape, (192,))
                self.assertEqual(embedding.dtype, torch.float32)
                self.assertEqual(emotion, "积极")
                for name, device in [("speech_campplus", cam_device), ("emotion2vec", emo_device)]:
                    self.assertEqual(pools[name].models[0].effective_backend, backend_for_device(device))
                    self.assertFalse(pools[name].locks[0].locked())
                    await pools[name].close()

    async def test_mixed_pool_has_independent_cpu_npu_backends(self):
        with patch("core.models.device_allocated_mb", return_value=0.0), patch("core.models._resolve_device", side_effect=lambda d: d), \
                patch("core.models.AutoModel", return_value=Mock()), \
                patch("core.models.device_allocated_mb", return_value=0.0), \
                patch("core.models.TorchNpuRunner", side_effect=lambda *a: FakeRunner()) as factory:
            pool = ModelPool("speech_campplus", "/model/cam", devices_getter=lambda:
                             ["cpu", "npu:7", "cpu", "npu:7"])
            await pool.ensure_loaded()
            self.assertEqual(factory.call_count, 2)
            self.assertEqual(factory.call_args.args, ("speech_campplus", "/model/cam", "npu:7"))
            details = local_model_status["speech_campplus"]["instance_details"]
            self.assertEqual([d["effective_backend"] for d in details],
                             ["pytorch_cpu", "torch_npu", "pytorch_cpu", "torch_npu"])
            self.assertIsNone(local_model_status["speech_campplus"]["runtime_hbm_mb"])
            await pool.close()

    async def test_unloaded_status_does_not_load_models(self):
        with patch("core.models.AutoModel", side_effect=AssertionError("loaded")):
            pool = ModelPool("emotion2vec", "/model/emo", devices_getter=lambda: ["npu:0"])
            pool.refresh_status()
            detail = local_model_status["emotion2vec"]["instance_details"][0]
            self.assertEqual(detail["configured_backend"], "torch_npu")
            self.assertEqual(detail["health"], "not_loaded")
            self.assertIsNone(detail["effective_backend"])

    async def test_npu_failure_does_not_fall_back_and_other_instances_survive(self):
        with patch("core.models.device_allocated_mb", return_value=0.0), patch("core.models._resolve_device", side_effect=lambda d: d), \
                patch("core.models.AutoModel", side_effect=AssertionError("fallback")), \
                patch("core.models.TorchNpuRunner", side_effect=[RuntimeError("bad NPU weight"), FakeRunner()]):
            pool = ModelPool("speech_campplus", "/model/cam", devices_getter=lambda: ["npu:7"] * 2)
            await pool.ensure_loaded()
            self.assertIsNone(pool.models[0])
            async with pool.lease() as (runner, _):
                self.assertIs(runner, pool.models[1])
            self.assertIn("bad NPU weight", pool.failures[0])
            await pool.close()

    async def test_broken_instance_wakes_waiters_with_unavailable(self):
        runner = FakeRunner()
        with patch("core.models.device_allocated_mb", return_value=0.0), patch("core.models._resolve_device", return_value="npu:7"), \
                patch("core.models.TorchNpuRunner", return_value=runner):
            pool = ModelPool("emotion2vec", "/model/emo", devices_getter=lambda: ["npu:7"])

            async def waiter():
                async with pool.lease():
                    self.fail("broken instance leased")

            with self.assertRaises(InstanceFailure):
                async with pool.lease():
                    waiting = asyncio.create_task(waiter())
                    await asyncio.sleep(0.01)
                    raise InstanceFailure("NPU context lost")
            with self.assertRaises(ModelUnavailableError) as error:
                await asyncio.wait_for(waiting, 2)
            self.assertEqual(error.exception.code, 4016)
            self.assertTrue(runner.closed)
            self.assertEqual(pool.waiting, 0)
            self.assertFalse(pool.locks[0].locked())
            self.assertEqual(local_model_status["emotion2vec"]["failed_instances"], 1)
            await pool.close()

    async def test_repeated_cancel_holds_lease_until_delayed_device_complete(self):
        submitted, complete = threading.Event(), threading.Event()
        runner = FakeRunner()
        with patch("core.models.device_allocated_mb", return_value=0.0), patch("core.models._resolve_device", return_value="npu:7"), \
                patch("core.models.TorchNpuRunner", return_value=runner):
            pool = ModelPool("speech_campplus", "/model/cam", devices_getter=lambda: ["npu:7"])

            def device_work():
                submitted.set()  # 模拟 enqueue 已返回，设备仍未完成。
                if not complete.wait(5):
                    raise TimeoutError("test device completion")

            async def request():
                async with pool.lease():
                    await run_in_thread(device_work)

            task = asyncio.create_task(request())
            await asyncio.to_thread(submitted.wait, 2)
            task.cancel()
            await asyncio.sleep(0.01)
            task.cancel()
            await asyncio.sleep(0.01)
            self.assertFalse(task.done())
            self.assertTrue(pool.locks[0].locked())
            closing = asyncio.create_task(pool.close())
            await asyncio.sleep(0.01)
            self.assertFalse(closing.done())
            self.assertFalse(runner.closed)
            complete.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(closing, 2)
            self.assertTrue(runner.closed)
            self.assertEqual(pool.models, [])

    async def test_cancel_during_load_registers_and_later_closes_resource(self):
        started, finish = threading.Event(), threading.Event()
        runner = FakeRunner()

        def build(*args):
            started.set()
            if not finish.wait(5):
                raise TimeoutError("test loading")
            return runner

        with patch("core.models.device_allocated_mb", return_value=0.0), patch("core.models._resolve_device", return_value="npu:7"), \
                patch("core.models.TorchNpuRunner", side_effect=build):
            pool = ModelPool("speech_campplus", "/model/cam", devices_getter=lambda: ["npu:7"])
            loading = asyncio.create_task(pool.ensure_loaded())
            await asyncio.to_thread(started.wait, 2)
            loading.cancel()
            await asyncio.sleep(0.01)
            self.assertFalse(loading.done())
            finish.set()
            with self.assertRaises(asyncio.CancelledError):
                await loading
            self.assertIs(pool.models[0], runner)
            await pool.close()
            self.assertTrue(runner.closed)

    async def test_close_failure_keeps_ownership_rejects_leases_and_can_retry(self):
        first, second = FakeRunner(), FakeRunner()
        attempts = []

        def close_first():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("controlled close failure")
            first.closed = True

        first.close = close_first
        second.close = Mock(wraps=second.close)
        pool = ModelPool("speech_campplus", "/model/cam", devices_getter=lambda: ["cpu", "cpu"])
        with patch.object(pool, "_build_instance", side_effect=[(first, "cpu", 1.0), (second, "cpu", 2.0)]):
            await pool.ensure_loaded()
        with self.assertRaisesRegex(RuntimeError, "instance#0.*controlled close failure"):
            await pool.close()
        self.assertIs(pool.models[0], first)
        self.assertIsNone(pool.models[1])
        self.assertTrue(second.closed)
        self.assertTrue(pool._closing)
        self.assertEqual(local_model_status[pool.name]["available_instances"], 0)
        self.assertEqual(local_model_status[pool.name]["pending_close_instances"], 1)
        self.assertEqual(local_model_status[pool.name]["instance_details"][0]["health"], "close_failed")
        with self.assertRaises(ModelUnavailableError) as error:
            await pool.ensure_loaded()
        self.assertEqual(error.exception.code, 4015)
        await pool.close()
        self.assertTrue(first.closed)
        self.assertEqual(len(attempts), 2)
        second.close.assert_called_once()
        self.assertEqual(pool.models, [])
        self.assertFalse(pool._closing)

    async def test_failed_inference_cleanup_is_quarantined_until_close_retry(self):
        runner = FakeRunner()
        attempts = []

        def close():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("cleanup failed")
            runner.closed = True

        runner.close = close
        pool = ModelPool("emotion2vec", "/model/emo", devices_getter=lambda: ["cpu"])
        with patch.object(pool, "_build_instance", return_value=(runner, "cpu", 1.0)):
            await pool.ensure_loaded()
        with self.assertRaises(InstanceFailure):
            async with pool.lease():
                raise InstanceFailure("device lost")
        self.assertIsNone(pool.models[0])
        self.assertIs(pool._unclosed_instances[0], runner)
        with self.assertRaises(ModelUnavailableError):
            async with pool.lease():
                self.fail("quarantined instance reused")
        await pool.close()
        self.assertTrue(runner.closed)
        self.assertFalse(pool._unclosed_instances)

    async def test_repeated_cancel_cannot_interrupt_other_instance_closes(self):
        first, second = FakeRunner(), FakeRunner()
        started, release = threading.Event(), threading.Event()

        def blocking_close():
            started.set()
            if not release.wait(5):
                raise TimeoutError("controlled close wait")
            first.closed = True

        first.close = blocking_close
        pool = ModelPool("speech_campplus", "/model/cam", devices_getter=lambda: ["cpu", "cpu"])
        with patch.object(pool, "_build_instance", side_effect=[(first, "cpu", 0.0), (second, "cpu", 0.0)]):
            await pool.ensure_loaded()
        closing = asyncio.create_task(pool.close())
        self.assertTrue(await asyncio.to_thread(started.wait, 2))
        closing.cancel()
        await asyncio.sleep(0.01)
        closing.cancel()
        await asyncio.sleep(0.01)
        self.assertFalse(closing.done())
        self.assertIs(pool.models[0], first)
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await closing
        self.assertTrue(first.closed)
        self.assertTrue(second.closed)
        self.assertEqual(pool.models, [])
