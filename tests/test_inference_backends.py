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
        self.effective_backend = "tensorrt"
        self.actual_precision = "fp16"
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
        model = Mock()
        model.to.return_value = model
        with patch("core.local_inference.npu_api", return_value=api), \
                patch("core.local_inference.load_native_network", return_value=model), \
                patch("torch.cuda.device", side_effect=AssertionError("CUDA called")), \
                patch("torch.cuda.is_available", side_effect=AssertionError("CUDA queried")):
            runner = TorchNpuRunner("speech_campplus", "/model/cam", "npu:1")
            self.assertEqual(runner.effective_backend, "torch_npu")
            self.assertEqual(runner.actual_precision, "fp32")
            model.to.assert_called_once_with("npu:1")
            runner.close()

    def test_device_overrides_switch(self):
        for enabled in (False, True):
            self.assertEqual(backend_for_device("npu:2", enabled), "torch_npu")
            self.assertEqual(backend_for_device("cpu", enabled), "pytorch_cpu")
        self.assertEqual(backend_for_device("cuda:1", True), "tensorrt")
        self.assertEqual(backend_for_device("cuda:1", False), "pytorch_cuda")

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
            self.assertEqual(generate_local(FakeRunner(), "cuda:0", input=None)[0]["scores"], [1.0])

    def test_input_validation(self):
        from core.local_inference import waveform_tensor

        for value in (np.zeros((1, 10), np.float32), np.zeros(0, np.float32),
                      np.array([np.nan], np.float32), np.zeros(10, np.int16)):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    waveform_tensor(value, 16000)


class BackendPoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_four_switch_combinations_preserve_route_outputs(self):
        import torch
        from api.routes import asr_qwen_common as route
        from core.config import settings

        class Model:
            def generate(self, **kwargs):
                return [{"spk_embedding": np.ones(192, np.float32),
                         "labels": ["开心/happy", "中立/neutral"], "scores": [0.9, 0.1]}]

        class Trt(FakeRunner):
            def generate(self, **kwargs):
                return Model().generate(**kwargs)

        for cam_enabled, emo_enabled in [(False, False), (True, False), (False, True), (True, True)]:
            with self.subTest(cam=cam_enabled, emo=emo_enabled):
                with patch("core.models._resolve_device", return_value="cuda:0"), \
                        patch("core.models.AutoModel", side_effect=lambda **kw: Model()), \
                        patch("core.tensorrt_runner.TensorRTRunner", side_effect=lambda *a: Trt()), \
                        patch("torch.cuda.device", side_effect=lambda *a: contextlib.nullcontext()), \
                        patch("torch.cuda.synchronize"), \
                        patch("torch.amp.autocast", side_effect=lambda *a, **kw: contextlib.nullcontext()), \
                        patch.dict(settings._cfg, {"features": {"open_v2_role": True, "open_v2_emotion": True}}):
                    pools = {name: ModelPool(name, "/model/test", devices_getter=lambda: ["cuda:0"],
                              tensorrt_getter=lambda enabled=enabled: enabled, fp16=True)
                             for name, enabled in [("speech_campplus", cam_enabled), ("emotion2vec", emo_enabled)]}
                    with patch.object(route, "get_model_pool", side_effect=lambda name: pools[name]):
                        embedding, emotion = await asyncio.gather(
                            route._speaker_embedding(np.zeros(16000, np.float32), 16000),
                            route._segment_emotion(np.zeros(16000, np.float32), 16000, 1.0))
                    self.assertEqual(embedding.shape, (192,))
                    self.assertEqual(embedding.dtype, torch.float32)
                    self.assertEqual(emotion, "积极")
                    for name, enabled in [("speech_campplus", cam_enabled), ("emotion2vec", emo_enabled)]:
                        self.assertEqual(pools[name].models[0].effective_backend,
                                         "tensorrt" if enabled else "pytorch_cuda")
                        self.assertFalse(pools[name].locks[0].locked())
                        await pools[name].close()

    async def test_mixed_pool_and_npu_ignore_engine(self):
        npu = FakeRunner()
        npu.effective_backend = "torch_npu"
        with patch("core.models._resolve_device", side_effect=lambda d: d), \
                patch("core.models.AutoModel", return_value=Mock()), \
                patch("core.models.TorchNpuRunner", return_value=npu) as npu_factory, \
                patch("core.tensorrt_runner.TensorRTRunner", side_effect=lambda *a: FakeRunner()) as trt:
            pool = ModelPool("speech_campplus", "/model/cam", devices_getter=lambda:
                             ["cuda:0", "npu:1", "cpu", "cuda:0"], tensorrt_getter=lambda: True)
            await pool.ensure_loaded()
            self.assertEqual(trt.call_count, 2)
            self.assertEqual(npu_factory.call_args.args, ("speech_campplus", "/model/cam", "npu:1"))
            details = local_model_status["speech_campplus"]["instance_details"]
            self.assertEqual([d["effective_backend"] for d in details],
                             ["tensorrt", "torch_npu", "pytorch_cpu", "tensorrt"])
            self.assertIsNone(local_model_status["speech_campplus"]["allocated_mb"])
            await pool.close()

    async def test_unloaded_status_does_not_load_models(self):
        with patch("core.models.AutoModel", side_effect=AssertionError("loaded")):
            pool = ModelPool("emotion2vec", "/model/emo", devices_getter=lambda: ["npu:0"],
                             tensorrt_getter=lambda: True)
            pool.refresh_status()
            detail = local_model_status["emotion2vec"]["instance_details"][0]
            self.assertEqual(detail["configured_backend"], "torch_npu")
            self.assertEqual(detail["health"], "not_loaded")
            self.assertIsNone(detail["effective_backend"])

    async def test_trt_failure_does_not_fall_back_and_other_instances_survive(self):
        with patch("core.models._resolve_device", side_effect=lambda d: d), \
                patch("core.models.AutoModel", side_effect=AssertionError("fallback")), \
                patch("core.tensorrt_runner.TensorRTRunner", side_effect=[RuntimeError("bad engine"), FakeRunner()]):
            pool = ModelPool("speech_campplus", "/model/cam", devices_getter=lambda: ["cuda:0"] * 2,
                             tensorrt_getter=lambda: True)
            await pool.ensure_loaded()
            self.assertIsNone(pool.models[0])
            async with pool.lease() as (runner, _):
                self.assertIs(runner, pool.models[1])
            self.assertIn("bad engine", pool.failures[0])
            await pool.close()

    async def test_broken_instance_wakes_waiters_with_unavailable(self):
        runner = FakeRunner()
        with patch("core.models._resolve_device", return_value="cuda:0"), \
                patch("core.tensorrt_runner.TensorRTRunner", return_value=runner):
            pool = ModelPool("emotion2vec", "/model/emo", devices_getter=lambda: ["cuda:0"],
                             tensorrt_getter=lambda: True)

            async def waiter():
                async with pool.lease():
                    self.fail("broken instance leased")

            with self.assertRaises(InstanceFailure):
                async with pool.lease():
                    waiting = asyncio.create_task(waiter())
                    await asyncio.sleep(0.01)
                    raise InstanceFailure("CUDA context lost")
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
        with patch("core.models._resolve_device", return_value="cuda:0"), \
                patch("core.tensorrt_runner.TensorRTRunner", return_value=runner):
            pool = ModelPool("speech_campplus", "/model/cam", devices_getter=lambda: ["cuda:0"],
                             tensorrt_getter=lambda: True)

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

        with patch("core.models._resolve_device", return_value="cuda:0"), \
                patch("core.tensorrt_runner.TensorRTRunner", side_effect=build):
            pool = ModelPool("speech_campplus", "/model/cam", devices_getter=lambda: ["cuda:0"],
                             tensorrt_getter=lambda: True)
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
