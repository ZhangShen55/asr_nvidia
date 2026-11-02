import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import torch

from api.routes import asr_common, asr_qwen_common as pipeline, asr_v2 as route
from core.config import settings
from core.models import ModelPool
from core.qwen_asr_queue import AudioSlotManager
from core.vad import VadSegment
from entity.data import AsrRequestParams


class AudioLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.features = patch.dict(settings._cfg, {"features": {"open_v2_role": True, "open_v2_emotion": True}})
        self.features.start()
        self.addCleanup(self.features.stop)

    def request(self, **flags):
        return AsrRequestParams(audioFile=SimpleNamespace(filename="a.wav", read=AsyncMock(return_value=b"x"*2048)),
                                wordTimestamps=False, **flags)

    async def cancel_and_check(self, task, started, manager, *, b_count):
        await asyncio.wait_for(started.wait(), 3)
        task.cancel()
        for _ in range(5):
            await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.assertEqual((manager.active, manager.active_b), (1, b_count))
        task.cancel()  # 重复取消不能跳过正在进行的必要收尾。

    async def test_cancel_during_context_load_waits_for_thread_then_cleans_partial_file(self):
        request = self.request(showSpk=True)
        manager = AudioSlotManager(1)
        loop = asyncio.get_running_loop()
        started, finish = asyncio.Event(), threading.Event()
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory)/"partial.wav"

            def write(*args, **kwargs):
                audio_path.write_bytes(b"audio")
                return str(audio_path)

            def load(path):
                loop.call_soon_threadsafe(started.set)
                if not finish.wait(3):
                    raise RuntimeError("test thread timed out")
                return torch.zeros(1, 16000), 16000

            with patch.object(route, "get_slot_manager", return_value=manager), \
                    patch.object(asr_common, "preprocess_audio", new=AsyncMock(return_value=b"x"*2048)), \
                    patch.object(asr_common, "write_audio_bytes_to_temp_file", side_effect=write), \
                    patch.object(asr_common, "load_audio_tensor", side_effect=load), \
                    patch.object(route, "run_qwen_asr", new_callable=AsyncMock) as run:
                task = asyncio.create_task(route.api_asr_v2(request))
                try:
                    await self.cancel_and_check(task, started, manager, b_count=1)
                    self.assertTrue(audio_path.exists())
                finally:
                    finish.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                run.assert_not_awaited()
                self.assertFalse(audio_path.exists())
                self.assertEqual((manager.active, manager.active_b), (0, 0))

    async def test_cancel_during_model_inference_holds_model_and_audio_leases(self):
        request = self.request(showSpk=True)
        manager = AudioSlotManager(1)
        loop = asyncio.get_running_loop()
        started, finish = asyncio.Event(), threading.Event()

        class Model:
            def generate(self, **kwargs):
                loop.call_soon_threadsafe(started.set)
                if not finish.wait(3):
                    raise RuntimeError("test inference timed out")
                return [{"spk_embedding": np.ones(3, dtype=np.float32)}]

        pool = ModelPool("speech_campplus", "/fake", devices_getter=lambda: ["cpu"])

        async def run(ctx):
            await pipeline._speaker_embedding(np.zeros(16000, dtype=np.float32), 16000)
            return {"segments": []}

        ctx = SimpleNamespace(request=request, tmp_paths=[], filename="a.wav")
        with patch.object(route, "get_slot_manager", return_value=manager), \
                patch.object(route, "prepare_asr_context", new=AsyncMock(return_value=(None, ctx))), \
                patch.object(route, "run_qwen_asr", side_effect=run), \
                patch.object(pipeline, "get_model_pool", return_value=pool), \
                patch("core.models.AutoModel", return_value=Model()):
            task = asyncio.create_task(route.api_asr_v2(request))
            try:
                await self.cancel_and_check(task, started, manager, b_count=1)
                self.assertTrue(pool.locks[0].locked())
            finally:
                finish.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertFalse(pool.locks[0].locked())
            self.assertEqual((manager.active, manager.active_b), (0, 0))
            await pool.close()

    async def test_cancel_during_compat_clip_creation_cleans_file_created_after_cancel(self):
        request = self.request()
        manager = AudioSlotManager(1)
        loop = asyncio.get_running_loop()
        started, finish = asyncio.Event(), threading.Event()
        with tempfile.TemporaryDirectory() as directory:
            clip_path = Path(directory)/"clip.wav"
            ctx = SimpleNamespace(request=request, tmp_paths=[], tmp_path="/fake.wav", filename="a.wav",
                                  audio_tensor=None, sample_rate=0)

            def extract(*args):
                loop.call_soon_threadsafe(started.set)
                if not finish.wait(3):
                    raise RuntimeError("test cut timed out")
                clip_path.write_bytes(b"clip")
                return str(clip_path)

            with patch.object(route, "get_slot_manager", return_value=manager), \
                    patch.object(route, "prepare_asr_context", new=AsyncMock(return_value=(None, ctx))), \
                    patch.object(pipeline, "detect_segments", new=AsyncMock(return_value=[VadSegment(0, 3)])), \
                    patch.object(pipeline, "extract_audio_clip", side_effect=extract), \
                    patch.object(pipeline, "get_qwen_client", new_callable=AsyncMock) as qwen:
                task = asyncio.create_task(route.api_asr_v2(request))
                try:
                    await self.cancel_and_check(task, started, manager, b_count=0)
                finally:
                    finish.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertFalse(clip_path.exists())
                qwen.assert_not_awaited()
                self.assertEqual(manager.active, 0)

    async def test_cancel_during_vad_waits_for_worker_before_file_cleanup(self):
        request = self.request()
        manager = AudioSlotManager(1)
        started, finish = asyncio.Event(), asyncio.Event()
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory)/"vad.wav"
            audio_path.write_bytes(b"audio")
            ctx = SimpleNamespace(request=request, tmp_paths=[str(audio_path)], tmp_path=str(audio_path),
                                  filename="a.wav", audio_tensor=None, sample_rate=0)

            async def detect(path):
                started.set()
                await finish.wait()
                self.assertTrue(audio_path.exists())
                return []

            with patch.object(route, "get_slot_manager", return_value=manager), \
                    patch.object(route, "prepare_asr_context", new=AsyncMock(return_value=(None, ctx))), \
                    patch.object(pipeline, "detect_segments", side_effect=detect):
                task = asyncio.create_task(route.api_asr_v2(request))
                try:
                    await self.cancel_and_check(task, started, manager, b_count=0)
                    self.assertTrue(audio_path.exists())
                finally:
                    finish.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertFalse(audio_path.exists())
                self.assertEqual(manager.active, 0)
