"""内存 clip 生命周期与兼容裁剪路径的严格清理。"""

from __future__ import annotations

import asyncio
import os
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import numpy as np
import unittest

from api.routes.asr_qwen_common import run_qwen_asr
from core.vad import VadSegment
from entity.data import AsrRequestParams


def make_request(**overrides):
    params = {
        "audioFile": object(),
        "language": "auto",
        "wordTimestamps": False,
        "hotWords": [],
        "showSpk": False,
        "openPanel": False,
        "showEmotion": False,
        "showSpeed": False,
        "showRoleIdentify": False,
    }
    params.update(overrides)
    return AsrRequestParams(**params)


@dataclass
class FakeCtx:
    request: AsrRequestParams
    tmp_path: str = "/tmp/fake_source.wav"
    tmp_paths: list = None
    audio_total_s: float = 12.0
    load_audio_time_ms: float = 3.5
    audio_tensor: object = None
    sample_rate: int = 0

    def __post_init__(self):
        if self.tmp_paths is None:
            self.tmp_paths = []


class SegmentClipLifecycleTests(unittest.TestCase):
    SEGMENTS = (
        VadSegment(start_s=0.0, end_s=3.0),
        VadSegment(start_s=3.2, end_s=6.5),
    )

    def setUp(self):
        self.created = []

    def _make_clip(self, _src, _start, _duration):
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        handle.write(b"0" * 2048)
        handle.close()
        self.created.append(handle.name)
        return handle.name

    def _run(self, ctx, transcribe, emotion=None, raise_after_cut=False):
        async def _fake_detect(_path):
            return list(self.SEGMENTS)

        class _FakeClient:
            async def transcribe_bytes(self, audio_bytes, language=None, filename=None):
                return await transcribe(audio_bytes, language)

        async def _fake_get_client():
            return _FakeClient()

        async def _fake_emotion(*_args, **_kwargs):
            return "积极" if emotion is None else emotion

        with ExitStack() as stack:
            stack.enter_context(
                patch("api.routes.asr_qwen_common.detect_segments", _fake_detect)
            )
            stack.enter_context(
                patch("api.routes.asr_qwen_common.extract_audio_clip", self._make_clip)
            )
            stack.enter_context(
                patch(
                    "api.routes.asr_qwen_common.read_audio_array",
                    lambda _p: (np.zeros(1600, dtype="float32"), 16000),
                )
            )
            stack.enter_context(
                patch("api.routes.asr_qwen_common.get_qwen_client", _fake_get_client)
            )
            # 单测不加载真实本地模型
            stack.enter_context(
                patch(
                    "api.routes.asr_qwen_common.load_v2_models_if_needed",
                    new=AsyncMock(),
                )
            )
            if raise_after_cut:
                def _explode(*_args, **_kwargs):
                    raise RuntimeError("模拟链路异常")

                stack.enter_context(
                    patch(
                        "api.routes.asr_qwen_common.normalize_segments",
                        _explode,
                    )
                )
            if emotion is not None:
                stack.enter_context(
                    patch(
                        "api.routes.asr_qwen_common._segment_emotion",
                        _fake_emotion,
                    )
                )
            return asyncio.run(run_qwen_asr(ctx))

    def test_success_uploads_memory_bytes_and_deletes_compat_clip_before_upload(self):
        observed = []

        async def transcribe(audio_bytes, _language):
            observed.append(audio_bytes[:4])
            return "language Chinese<asr_text>你好"

        ctx = FakeCtx(request=make_request())
        payload = self._run(ctx, transcribe)

        self.assertEqual(payload["segments"][0]["segment_text"], "你好")
        self.assertEqual(observed, [b"RIFF", b"RIFF"])
        for path in self.created:
            self.assertFalse(os.path.exists(path), "内存上传前兼容段文件应删除")
        self.assertNotIn(self.created[0], ctx.tmp_paths)

    def test_transcription_failure_still_deletes_clip(self):
        async def transcribe(_audio_bytes, _language):
            raise RuntimeError("模拟 Qwen 失败")

        ctx = FakeCtx(request=make_request())
        payload = self._run(ctx, transcribe)
        self.assertEqual(payload["code"], 4012)
        for path in self.created:
            self.assertFalse(os.path.exists(path), "转写失败时段文件也应删除")

    def test_overall_exception_is_cleaned_by_finally(self):
        async def transcribe(_audio_bytes, _language):
            return "language Chinese<asr_text>你好"

        ctx = FakeCtx(request=make_request())
        with self.assertRaises(RuntimeError):
            self._run(ctx, transcribe, raise_after_cut=True)
        for path in self.created:
            self.assertFalse(os.path.exists(path), "整体异常时段文件应被兜底清理")

    def test_emotion_runs_in_parallel_without_relying_on_clip(self):
        """showEmotion=true 时，段文件已删除也不影响情绪结果。"""

        async def transcribe(_audio_bytes, _language):
            await asyncio.sleep(0.05)
            return "language Chinese<asr_text>你好"

        ctx = FakeCtx(request=make_request(showEmotion=True))
        payload = self._run(ctx, transcribe, emotion="积极")

        self.assertEqual(payload["segments"][0]["emotion"], "积极")
        for path in self.created:
            self.assertFalse(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
