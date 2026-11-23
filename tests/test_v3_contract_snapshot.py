"""固定业务输入的v3协议快照，避免把迁移后的接口漂移误当数值误差。"""
import asyncio
import copy
import json
import unittest
from contextlib import ExitStack
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import asr_qwen_common as pipeline
from api.routes import asr_v3 as route
from core.config import settings
from core.qwen_asr_client import QwenAsrUnavailableError, QwenAsrCallError, QwenAsrUnsupportedLanguageError, normalize_language_param
from core.qwen_asr_queue import AudioSlotManager, RequestConcurrency
from core.vad import VadSegment
from entity.data import AsrRequestParams

FIXTURE = Path(__file__).parent / "fixtures" / "v3_contract.json"


def fixture():
    return json.loads(FIXTURE.read_text())


def request(**overrides):
    values = {"audioFile": SimpleNamespace(filename="contract.wav"), "wordTimestamps": False,
              "hotWords": [], "language": "auto", "showSpk": False, "openPanel": False,
              "showEmotion": False, "showSpeed": False, "showRoleIdentify": False}
    values.update(overrides)
    return AsrRequestParams(**values)


class ControlledClock:
    """主链路从10秒到20秒；中间所有设备/后处理耗时为零，验证旧计时边界。"""
    def __init__(self):
        self.started = False

    def __call__(self):
        if not self.started:
            self.started = True
            return 10.0
        return 20.0


async def run_contract(flags=None, error=None, segments=None, raw_texts=None):
    flags = flags or {}
    params = request(**flags)
    context = SimpleNamespace(request=params, audio_tensor=torch.zeros((1, 48000)),
                              sample_rate=16000, audio_total_s=3.0, tmp_paths=[], tmp_path="unused.wav",
                              task_id="contract-task", filename="contract.wav", load_audio_time_ms=12.345)
    rows = segments if segments is not None else [VadSegment(0.0, 1.0), VadSegment(1.5, 3.0)]
    texts = raw_texts or ["language Chinese<asr_text>，今天学习。", "language English<asr_text>Hello class."]
    calls = []
    class Client:
        async def transcribe_bytes(self, data, language=None, filename="segment.wav"):
            if error:
                raise error
            assert isinstance(data, bytes) and data.startswith(b"RIFF")
            index = int(filename.rsplit("-", 1)[1].split(".", 1)[0])
            calls.append((index, language))
            return texts[index]
    load = AsyncMock()
    speaker = AsyncMock(return_value=torch.ones(192))
    emotion = AsyncMock(return_value="积极")
    with ExitStack() as stack:
        stack.enter_context(patch.dict(settings._cfg, {"features": {"open_v2_role": True, "open_v2_emotion": True},
                                                        "itn": {"enabled": False}, "speech_rate": {"rate_factor": 0.4}}))
        stack.enter_context(patch.object(pipeline, "time", SimpleNamespace(perf_counter=ControlledClock())))
        stack.enter_context(patch.object(pipeline, "load_v3_models_if_needed", load))
        stack.enter_context(patch.object(pipeline, "detect_segments", AsyncMock(return_value=rows)))
        stack.enter_context(patch.object(pipeline, "get_qwen_client", AsyncMock(return_value=Client())))
        stack.enter_context(patch.object(pipeline, "get_request_concurrency", return_value=RequestConcurrency(2, 2)))
        stack.enter_context(patch.object(pipeline, "_speaker_embedding", speaker))
        stack.enter_context(patch.object(pipeline, "_segment_emotion", emotion))
        stack.enter_context(patch.object(pipeline, "cluster_speakers", side_effect=lambda outcomes: {o.index: "spk_0" for o in outcomes}))
        stack.enter_context(patch.object(pipeline, "_apply_role_identify"))
        stack.enter_context(patch.object(pipeline, "update_stat"))
        stack.enter_context(patch.object(pipeline, "update_fail_task"))
        result = await pipeline.run_qwen_asr(context)
    return result, load, speaker, emotion, calls


class V3HttpDefaultsTests(unittest.TestCase):
    def test_form_defaults_and_public_route_contract(self):
        captured = []
        app = FastAPI()
        app.include_router(route.router)
        async def prepare(params, unique_tmp=False):
            captured.append(params)
            return None, SimpleNamespace(request=params, tmp_paths=[], task_id="contract-defaults", filename="contract.wav")
        with ExitStack() as stack:
            stack.enter_context(patch.object(route, "get_slot_manager", return_value=AudioSlotManager(1)))
            stack.enter_context(patch.object(route, "prepare_asr_context", side_effect=prepare))
            stack.enter_context(patch.object(route, "run_qwen_asr", AsyncMock(return_value={"segments": []})))
            for name in ("add_processing_task", "remove_processing_task", "record_request"):
                stack.enter_context(patch.object(route, name))
            with TestClient(app) as client:
                result = client.post("/v3/seacraft_asr", files={"audioFile": ("contract.wav", b"fixture")})
                self.assertEqual(result.status_code, 200)
                for language in ("auto", ""):
                    result = client.post("/v3/seacraft_asr", data={"language": language},
                                         files={"audioFile": ("contract.wav", b"fixture")})
                    self.assertEqual(result.status_code, 200)
                self.assertEqual(client.post("/v2/seacraft_asr").status_code, 404)
                self.assertEqual(client.post("/v3/seacraft_asr").status_code, 422)
        expected = fixture()["request_defaults"]
        self.assertEqual({name: getattr(captured[0], name) for name in expected}, expected)
        self.assertEqual([normalize_language_param(row.language) for row in captured], [None, None, None])
        schema = next(x for x in app.openapi()["components"]["schemas"].values()
                      if "audioFile" in x.get("properties", {}))
        self.assertEqual(set(schema["properties"]), set(fixture()["request_fields"]))
        self.assertEqual(schema["required"], ["audioFile"])


class V3PipelineSnapshotTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_body_matches_frozen_business_snapshot(self):
        result, *_ = await run_contract()
        self.assertEqual(result, fixture()["response"])
        self.assertEqual(result["gpu_time_ms"], "10000.00")
        self.assertEqual(result["load_audio_time_ms"], "12.35")
        for segment in result["segments"]:
            self.assertRegex(segment["bg"], r"^\d+\.\d{2}$")
            self.assertRegex(segment["ed"], r"^\d+\.\d{2}$")

    async def test_all_eight_enhancement_combinations_preserve_shape_and_loading(self):
        baseline = fixture()["response"]
        for spk, emo, role in product((False, True), repeat=3):
            with self.subTest(showSpk=spk, showEmotion=emo, showRoleIdentify=role):
                result, load, speaker, emotion, calls = await run_contract(
                    {"showSpk": spk, "showEmotion": emo, "showRoleIdentify": role})
                need_role = spk or role
                load.assert_awaited_once_with(need_role=need_role, need_emotion=emo)
                self.assertEqual(speaker.await_count, 2 if need_role else 0)
                self.assertEqual(emotion.await_count, 2 if emo else 0)
                self.assertEqual(sorted(calls), [(0, "auto"), (1, "auto")])
                expected = copy.deepcopy(baseline)
                for row in expected["segments"]:
                    row["role"] = "spk_0" if need_role else None
                    row["emotion"] = "积极" if emo else None
                self.assertEqual(result, expected)

    async def test_local_model_errors_keep_existing_payloads(self):
        for error in [QwenAsrUnavailableError(), QwenAsrCallError(), QwenAsrUnsupportedLanguageError()]:
            with self.subTest(code=error.code):
                # This freezes error payload mapping, not processor validation.
                # Empty/omitted/auto language remains auto-detection, never 4009.
                flags = {"language": "xx"} if error.code == 4009 else None
                result, *_ = await run_contract(flags=flags, error=error)
                self.assertEqual(result, fixture()["errors"][str(error.code)])
        result, *_ = await run_contract(segments=[])
        self.assertEqual(result, fixture()["errors"]["4008"])

    async def test_no_speech_and_punctuation_only_keep_existing_filtering(self):
        no_speech, *_ = await run_contract(raw_texts=["language None<asr_text>", "language None<asr_text>"])
        self.assertEqual(no_speech, fixture()["errors"]["4008"])
        punctuation, *_ = await run_contract(raw_texts=["language Chinese<asr_text>，。！？", "language Chinese<asr_text>！"])
        # 记录现行边界行为：标点卫生删除最终所有段后仍返回空成功结构，不借迁移改语义。
        self.assertEqual(punctuation, fixture()["punctuation_only_response"])


if __name__ == "__main__":
    unittest.main()


# 本文件冻结旧准入/序列化/错误映射，不用真实owner池；continuous接线由multi_api单独覆盖。
import pytest


@pytest.fixture(autouse=True)
def explicit_legacy_contract(monkeypatch):
    monkeypatch.setattr(type(settings), 'qwen_generation_mode', property(lambda self: 'legacy'))
