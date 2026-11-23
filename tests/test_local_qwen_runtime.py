"""本地Qwen的资源所有权与失败语义；不加载模型或初始化NPU。"""
import asyncio
import io
import threading
import tempfile
import sys
import unittest
import wave
from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import torch

from core import qwen_asr_client as client_module
from core.qwen_asr_client import (
    QwenAsrClient, QwenAsrCallError, QwenAsrUnavailableError,
    QwenAsrUnsupportedLanguageError, get_qwen_status,
)
from core.qwen_asr_runtime import (
    LocalQwenRuntime, LocalModelUnavailable, LocalUnsupportedLanguage,
    decode_wav_bytes, move_inputs,
)


def wav_bytes(rate=16000, channels=1, values=None):
    data = io.BytesIO()
    with wave.open(data, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(np.asarray(values if values is not None else [0, 1024, -32768, 32767], dtype="<i2").tobytes())
    return data.getvalue()


class InputBoundaryTests(unittest.TestCase):
    def test_pcm_samples_are_preserved_without_resampling(self):
        out = decode_wav_bytes(wav_bytes())
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_equal(out, np.asarray([0, 1024, -32768, 32767], np.float32) / 32768)
        with self.assertRaises(ValueError):
            decode_wav_bytes(wav_bytes(rate=8000))
        with self.assertRaises((ValueError, wave.Error, EOFError)):
            decode_wav_bytes(b"bad")

    def test_move_inputs_preserves_integer_and_boolean_dtypes(self):
        integer = torch.tensor([[1, 10000, 123456]], dtype=torch.int64)
        mask = torch.tensor([[True, False, True]])
        out = move_inputs({"input_ids": integer, "mask": mask, "features": torch.ones(2)},
                          device="cpu", dtype=torch.bfloat16)
        self.assertEqual(out["features"].dtype, torch.bfloat16)
        self.assertEqual(out["input_ids"].dtype, torch.int64)
        self.assertEqual(out["mask"].dtype, torch.bool)
        self.assertTrue(torch.equal(out["input_ids"], integer))
        self.assertTrue(torch.equal(out["mask"], mask))

    def test_construct_and_snapshot_do_not_import_or_load_models(self):
        runtime = LocalQwenRuntime("/missing", "npu:7")
        self.assertFalse(runtime.snapshot()["loaded"])
        self.assertEqual(runtime.snapshot()["memory_allocated_mb"], 0)
        with self.assertRaises(LocalModelUnavailable):
            runtime.ensure_loaded()
        self.assertFalse(runtime.snapshot()["loaded"])
        self.assertIn("/missing", runtime.snapshot()["failure"])

    def test_global_status_does_not_construct_client(self):
        with patch.object(client_module, "_client_instance", None), \
                patch.object(client_module, "QwenAsrClient") as constructor:
            status = get_qwen_status()
        constructor.assert_not_called()
        self.assertFalse(status["loaded"])
        self.assertEqual(status["memory_allocated_mb"], 0)


class RuntimeOutputTests(unittest.TestCase):
    def test_initial_load_sets_process_precision_before_target_context(self):
        events = []
        def scope(device):
            events.append(("scope", device))
            return nullcontext()
        api = SimpleNamespace(device=scope, set_device=lambda d: events.append(("device", d)),
                              synchronize=lambda d: events.append(("sync", d)),
                              is_available=lambda: True, device_count=lambda: 8,
                              memory_allocated=lambda d: 0, memory_reserved=lambda d: 0,
                              conv=SimpleNamespace(allow_hf32=True), matmul=SimpleNamespace(allow_hf32=True))
        model = SimpleNamespace(dtype=torch.bfloat16)
        model.named_parameters = lambda: [("weight", torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16)))]
        model.to = lambda d: model
        model.eval = lambda: model
        processor = object()
        transformers = SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor),
            Qwen3ASRForConditionalGeneration=SimpleNamespace(from_pretrained=lambda *args, **kwargs: model))
        with tempfile.TemporaryDirectory() as directory, patch.object(torch, "npu", api, create=True), \
                patch.dict(sys.modules, {"torch_npu": SimpleNamespace(), "transformers": transformers}):
            runtime = LocalQwenRuntime(directory, "npu:7")
            runtime.ensure_loaded()
            count = len(events)
            runtime.ensure_loaded()
        self.assertEqual(len(events), count)
        self.assertEqual(events[:2], [("device", "npu:7"), ("scope", "npu:7")])
        self.assertFalse(api.conv.allow_hf32)
        self.assertFalse(api.matmul.allow_hf32)
        self.assertEqual(runtime.snapshot()["precision_policy"], {
            "mode": "fp32_ops_hf32_disabled", "conv_allow_hf32": False, "matmul_allow_hf32": False})

    def make_runtime(self, text):
        runtime = LocalQwenRuntime("/unused", "npu:7")
        events = []
        def scope(device):
            events.append(("scope", device))
            return nullcontext()
        runtime.api = SimpleNamespace(device=scope, set_device=lambda d: events.append(("device", d)),
                                      synchronize=lambda d: events.append(("sync", d)))
        runtime.model = SimpleNamespace(dtype=torch.bfloat16, generate=lambda **kwargs: torch.tensor([[1, 2, 100, 101]]))
        inputs = {"input_ids": torch.tensor([[1, 2]]), "input_features": torch.ones(1, 2)}
        processor = SimpleNamespace()
        processor.apply_transcription_request = lambda **kwargs: inputs
        def decode(ids, **kwargs):
            self.assertTrue(torch.equal(ids, torch.tensor([[100, 101]])))
            self.assertFalse(kwargs["skip_special_tokens"])
            return [text]
        processor.batch_decode = decode
        runtime.processor = processor
        return runtime, inputs, events

    def test_prompt_tokens_removed_and_language_prefix_preserved(self):
        runtime, inputs, events = self.make_runtime("language English<asr_text>hello<|im_end|>")
        with patch("core.qwen_asr_runtime.move_inputs", return_value=inputs):
            self.assertEqual(runtime.transcribe(wav_bytes(), "en"), "language English<asr_text>hello")
        self.assertEqual(events[:2], [("device", "npu:7"), ("scope", "npu:7")])
        self.assertEqual(events[-1], ("sync", "npu:7"))

    def test_missing_prefix_is_execution_failure_not_fake_no_speech(self):
        runtime, inputs, events = self.make_runtime("unstructured model output")
        with patch("core.qwen_asr_runtime.move_inputs", return_value=inputs):
            with self.assertRaisesRegex(RuntimeError, "语种前缀"):
                runtime.transcribe(wav_bytes())
        self.assertEqual(events[-1], ("sync", "npu:7"))

    def test_processor_unsupported_language_is_distinct_and_synchronized(self):
        runtime, inputs, events = self.make_runtime("")
        def reject(**kwargs):
            raise ValueError("Unsupported language: xx")
        runtime.processor.apply_transcription_request = reject
        with self.assertRaises(LocalUnsupportedLanguage):
            runtime.transcribe(wav_bytes(), "xx")
        self.assertEqual(events[-1], ("sync", "npu:7"))


class FakeRuntime:
    def __init__(self, *args):
        self.loaded = False
        self.failure = None
        self.calls = []
        self.loads = 0
        self.closed = False
        self.error = None
        self.block = False
        self.started = threading.Event()
        self.release = threading.Event()
        self.active = 0
        self.peak = 0

    def ensure_loaded(self):
        if not self.loaded:
            self.loaded = True
            self.loads += 1

    def transcribe(self, data, language):
        self.ensure_loaded()
        self.calls.append(language)
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started.set()
        try:
            if self.block and not self.release.wait(3):
                raise RuntimeError("test release deadline")
            if self.error:
                raise self.error
            return "language English<asr_text>hello"
        finally:
            self.active -= 1

    def snapshot(self):
        return {"loaded": self.loaded, "loading": False, "failure": self.failure,
                "memory_allocated_mb": 1.0 if self.loaded else 0.0,
                "memory_reserved_mb": 1.0 if self.loaded else 0.0}

    def close(self):
        if self.active:
            raise RuntimeError("closed while busy")
        self.closed = True
        self.loaded = False


async def until(predicate):
    for _ in range(300):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition deadline")


class LocalClientTests(unittest.IsolatedAsyncioTestCase):
    def make_client(self, **kwargs):
        kwargs.setdefault("generation_mode", "legacy")
        kwargs.setdefault("local_instance_count", 1)
        kwargs.setdefault("max_inference_batch_size", 1)
        return QwenAsrClient("http://unused", retry_count=0, execution_mode="thread", runtime_factory=FakeRuntime, **kwargs)

    async def test_language_and_discovery_without_http(self):
        client = self.make_client()
        self.assertFalse(client.snapshot()["loaded"])
        self.assertEqual(await client.discover_model(), "Qwen3-ASR-1.7B-hf")
        await client.transcribe_bytes(wav_bytes(), "auto")
        await client.transcribe_bytes(wav_bytes(), " en ")
        await client.transcribe_bytes(wav_bytes(), "")
        self.assertEqual(client._runtimes[0].calls, [None, "en", None])
        self.assertEqual(client._runtimes[0].loads, 1)
        await client.aclose()

    async def test_failure_codes_do_not_become_no_speech(self):
        for error, kind, code in [
            (LocalModelUnavailable("missing"), QwenAsrUnavailableError, 4011),
            (LocalUnsupportedLanguage("language unsupported"), QwenAsrUnsupportedLanguageError, 4009),
            (RuntimeError("device failed"), QwenAsrCallError, 4012),
        ]:
            client = self.make_client()
            client._runtimes[0].error = error
            with self.assertRaises(kind) as caught:
                await client.transcribe_bytes(wav_bytes())
            self.assertEqual(caught.exception.code, code)
            self.assertEqual(client.snapshot()["busy"], 0)
            self.assertEqual(client.snapshot()["waiting"], 0)
            await client.aclose()

    async def test_single_instance_serializes_and_cancelled_waiter_does_not_execute(self):
        client = self.make_client()
        runtime = client._runtimes[0]
        runtime.block = True
        first = asyncio.create_task(client.transcribe_bytes(wav_bytes()))
        await until(runtime.started.is_set)
        second = asyncio.create_task(client.transcribe_bytes(wav_bytes(), "en"))
        await until(lambda: client.snapshot()["waiting"] == 1)
        second.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await second
        self.assertEqual(runtime.calls, [None])
        runtime.release.set()
        await first
        self.assertEqual(runtime.peak, 1)
        await client.aclose()

    async def test_cancel_and_repeated_cancel_wait_for_actual_work(self):
        client = self.make_client()
        runtime = client._runtimes[0]
        runtime.block = True
        call = asyncio.create_task(client.transcribe_bytes(wav_bytes()))
        await until(runtime.started.is_set)
        call.cancel()
        await asyncio.sleep(0.01)
        call.cancel()
        await asyncio.sleep(0.01)
        self.assertFalse(call.done())
        self.assertEqual(client.snapshot()["busy"], 1)
        runtime.release.set()
        with self.assertRaises(asyncio.CancelledError):
            await call
        self.assertEqual(client.snapshot()["busy"], 0)
        self.assertEqual(client._available.qsize(), 1)
        await client.aclose()

    async def test_cancellation_at_worker_completion_is_not_swallowed(self):
        client = self.make_client()
        create_task = asyncio.create_task
        def inject(coroutine, **kwargs):
            task = create_task(coroutine, **kwargs)
            if getattr(getattr(coroutine, "cr_code", None), "co_name", None) == "to_thread":
                owner = asyncio.current_task()
                task.add_done_callback(lambda _future: owner.cancel())
            return task
        with patch("core.qwen_asr_client.asyncio.create_task", side_effect=inject):
            call = create_task(client.transcribe_bytes(wav_bytes()))
            with self.assertRaises(asyncio.CancelledError):
                await call
        self.assertEqual(client.snapshot()["busy"], 0)
        self.assertEqual(client._available.qsize(), 1)
        await client.aclose()

    async def test_timeout_waits_for_original_work_before_reporting_error(self):
        client = self.make_client(request_timeout_s=0.01)
        runtime = client._runtimes[0]
        runtime.block = True
        call = asyncio.create_task(client.transcribe_bytes(wav_bytes()))
        await until(runtime.started.is_set)
        await asyncio.sleep(0.03)
        self.assertFalse(call.done())
        self.assertEqual(client.snapshot()["busy"], 1)
        runtime.release.set()
        with self.assertRaises(QwenAsrCallError):
            await call
        self.assertEqual(client._available.qsize(), 1)
        await client.aclose()

    async def test_close_and_its_repeated_cancellation_wait_for_work(self):
        client = self.make_client()
        runtime = client._runtimes[0]
        runtime.block = True
        call = asyncio.create_task(client.transcribe_bytes(wav_bytes()))
        await until(runtime.started.is_set)
        close = asyncio.create_task(client.aclose())
        await until(lambda: client.snapshot()["closing"])
        close.cancel()
        await asyncio.sleep(0.01)
        close.cancel()
        self.assertFalse(close.done())
        self.assertFalse(runtime.closed)
        with self.assertRaises(QwenAsrUnavailableError):
            await client.transcribe_bytes(wav_bytes())
        runtime.release.set()
        await call
        with self.assertRaises(asyncio.CancelledError):
            await close
        self.assertTrue(runtime.closed)
        await client.aclose()

    async def test_two_instances_have_independent_execution_ownership(self):
        client = self.make_client(local_instance_count=2)
        for runtime in client._runtimes:
            runtime.block = True
        first = asyncio.create_task(client.transcribe_bytes(wav_bytes()))
        second = asyncio.create_task(client.transcribe_bytes(wav_bytes(), "en"))
        await until(lambda: all(x.started.is_set() for x in client._runtimes))
        self.assertEqual(client.snapshot()["busy"], 2)
        for runtime in client._runtimes:
            self.assertEqual(runtime.peak, 1)
            runtime.release.set()
        await asyncio.gather(first, second)
        await client.aclose()

    async def test_failed_instance_not_reissued_and_all_failed_returns_unavailable(self):
        client = self.make_client(local_instance_count=2)
        client._runtimes[0].failure = "load failed"
        await client.transcribe_bytes(wav_bytes())
        self.assertEqual(client._runtimes[0].calls, [])
        self.assertEqual(client._runtimes[1].calls, [None])
        self.assertEqual(client.snapshot()["capacity"], 1)
        client._runtimes[1].failure = "device failed"
        with self.assertRaises(QwenAsrUnavailableError):
            await asyncio.wait_for(client.transcribe_bytes(wav_bytes()), 0.2)
        await client.aclose()

    async def test_close_wakes_waiting_requests_without_starting_them(self):
        client = self.make_client()
        runtime = client._runtimes[0]
        runtime.block = True
        call = asyncio.create_task(client.transcribe_bytes(wav_bytes()))
        await until(runtime.started.is_set)
        waiter = asyncio.create_task(client.transcribe_bytes(wav_bytes(), "zh"))
        await until(lambda: client.snapshot()["waiting"] == 1)
        close = asyncio.create_task(client.aclose())
        with self.assertRaises(QwenAsrUnavailableError):
            await asyncio.wait_for(waiter, 0.3)
        self.assertEqual(runtime.calls, [None])
        self.assertFalse(close.done())
        runtime.release.set()
        await call
        await close
        self.assertTrue(runtime.closed)
        self.assertEqual(client.snapshot()["waiting"], 0)

    async def test_shutdown_attempts_each_runtime_and_keeps_failure_visible(self):
        client = self.make_client(local_instance_count=2)
        client._runtimes[0].close = lambda: (_ for _ in ()).throw(RuntimeError("close failed"))
        with self.assertRaises(RuntimeError):
            await client.aclose()
        self.assertTrue(client._runtimes[1].closed)
        self.assertTrue(client.snapshot()["closing"])


if __name__ == "__main__":
    unittest.main()
