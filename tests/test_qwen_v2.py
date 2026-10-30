"""v2（Qwen3-ASR）链路的单元测试：语言透传、语种占比、响应结构、并发与排队。"""

import asyncio
import unittest
from contextlib import ExitStack
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import torch

from core.qwen_asr_client import (
    QwenAsrCallError,
    QwenAsrUnsupportedLanguageError,
    normalize_language_param,
)
from core.qwen_asr_queue import (
    AudioSlotManager,
    QueueFullError,
    QueueTimeoutError,
    RequestConcurrency,
)
from entity.data import AsrRequestParams
from utils.qwen_response import build_language_ratio, parse_qwen_text


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


def fake_read_audio_array(_path):
    """桩：返回一段静音的 16k 单声道音频。"""
    import numpy as np

    return np.zeros(16000, dtype="float32"), 16000


class LanguagePassthroughTests(unittest.TestCase):
    def test_auto_and_empty_are_not_sent(self):
        self.assertIsNone(normalize_language_param("auto"))
        self.assertIsNone(normalize_language_param("AUTO"))
        self.assertIsNone(normalize_language_param(""))
        self.assertIsNone(normalize_language_param(None))

    def test_concrete_language_is_forwarded(self):
        self.assertEqual(normalize_language_param("en"), "en")
        self.assertEqual(normalize_language_param(" zh "), "zh")


class PrefixParsingTests(unittest.TestCase):
    def test_single_prefix_is_stripped(self):
        result = parse_qwen_text("language Chinese<asr_text>你好")
        self.assertEqual(result.text, "你好")
        self.assertEqual(result.language, "Chinese")
        self.assertTrue(result.has_speech)

    def test_multiple_prefixes_are_concatenated(self):
        raw = "language Chinese<asr_text>第一段 language English<asr_text>second part"
        result = parse_qwen_text(raw)
        self.assertEqual(result.text, "第一段 second part")
        self.assertEqual(result.language, "Chinese")

    def test_none_language_has_no_speech(self):
        result = parse_qwen_text("language None<asr_text>")
        self.assertEqual(result.text, "")
        self.assertFalse(result.has_speech)

    def test_empty_input(self):
        result = parse_qwen_text("")
        self.assertEqual(result.text, "")
        self.assertIsNone(result.language)

    def test_language_ratio_percentages(self):
        ratio = build_language_ratio(
            [("Chinese", "中文字符"), ("English", "ab")]
        )
        self.assertAlmostEqual(ratio["zh"], 66.7, places=1)
        self.assertAlmostEqual(ratio["en"], 33.3, places=1)
        self.assertAlmostEqual(sum(ratio.values()), 100.0, places=1)

    def test_language_ratio_empty(self):
        self.assertEqual(build_language_ratio([]), {})
        self.assertEqual(build_language_ratio([("Chinese", "   ")]), {})


class SlotManagerTests(unittest.TestCase):
    def test_n_plus_one_request_is_rejected_immediately(self):
        async def _run():
            manager = AudioSlotManager(capacity=2, wait_timeout_s=0)
            await manager.acquire()
            await manager.acquire()
            with self.assertRaises(QueueFullError):
                await manager.acquire()
            await manager.release()
            await manager.acquire()
            self.assertEqual(manager.active, 2)

        asyncio.run(_run())

    def test_wait_timeout_raises_queue_timeout(self):
        async def _run():
            manager = AudioSlotManager(capacity=1, wait_timeout_s=0.1)
            await manager.acquire()
            with self.assertRaises(QueueTimeoutError):
                await manager.acquire()

        asyncio.run(_run())

    def test_waiting_count_is_restored_after_timeout(self):
        """排队超时后等待名额必须归还，否则后续请求会被误判为队列已满。"""

        async def _run():
            manager = AudioSlotManager(capacity=1, wait_timeout_s=0.1)
            await manager.acquire()
            with self.assertRaises(QueueTimeoutError):
                await manager.acquire()
            self.assertEqual(manager.waiting, 0)

            # 归还名额后仍可正常进入等待
            manager.wait_timeout_s = 5
            task = asyncio.create_task(manager.acquire())
            await asyncio.sleep(0.05)
            self.assertEqual(manager.waiting, 1)
            await manager.release()
            await asyncio.wait_for(task, timeout=2)
            self.assertEqual(manager.waiting, 0)

        asyncio.run(_run())

    def test_waiter_gets_slot_after_release(self):
        async def _run():
            manager = AudioSlotManager(capacity=1, wait_timeout_s=5)
            await manager.acquire()

            async def _waiter():
                await manager.acquire()
                return "acquired"

            task = asyncio.create_task(_waiter())
            await asyncio.sleep(0.05)
            await manager.release()
            self.assertEqual(await asyncio.wait_for(task, timeout=2), "acquired")
            self.assertEqual(manager.waiting, 0)

        asyncio.run(_run())


class SpeakerClusteringTests(unittest.TestCase):
    """聚类实现的正确性与性能回归。"""

    @staticmethod
    def _reference_clusters(similarity, limit):
        """旧版朴素实现，作为等价性参照。"""
        clusters = [[index] for index in range(len(similarity))]
        while len(clusters) > 1:
            best_score = -1.0
            best_pair = (-1, -1)
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    score = float(similarity[np.ix_(clusters[i], clusters[j])].mean())
                    if score > best_score:
                        best_score = score
                        best_pair = (i, j)
            if best_pair[0] < 0 or best_score < limit:
                break
            i, j = best_pair
            clusters[i] = clusters[i] + clusters[j]
            clusters.pop(j)
        return clusters

    def test_matches_reference_implementation(self):
        from api.routes.asr_qwen_common import _average_linkage_clusters

        rng = np.random.default_rng(20261023)
        for _ in range(20):
            n = int(rng.integers(4, 40))
            centers = rng.normal(size=(3, 8))
            labels = rng.integers(0, 3, size=n)
            vectors = centers[labels] + rng.normal(scale=0.15, size=(n, 8))
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / norms
            similarity = vectors @ vectors.T

            reference = self._reference_clusters(similarity, 0.3)
            actual = _average_linkage_clusters(similarity, 0.3)

            def partition(clusters):
                return sorted(sorted(c) for c in clusters)

            self.assertEqual(partition(reference), partition(actual))

    def test_single_and_empty_inputs(self):
        from api.routes.asr_qwen_common import _average_linkage_clusters

        self.assertEqual(_average_linkage_clusters(np.zeros((0, 0)), 0.3), [])
        self.assertEqual(_average_linkage_clusters(np.ones((1, 1)), 0.3), [[0]])

    def test_large_input_is_fast(self):
        """395 段的聚类必须在秒级完成（旧实现需要约 100 秒）。"""
        import time

        from api.routes.asr_qwen_common import _average_linkage_clusters

        rng = np.random.default_rng(7)
        centers = rng.normal(size=(4, 192))
        labels = rng.integers(0, 4, size=395)
        vectors = centers[labels] + rng.normal(scale=0.2, size=(395, 192))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        similarity = vectors @ vectors.T

        started = time.perf_counter()
        clusters = _average_linkage_clusters(similarity, 0.3)
        elapsed = time.perf_counter() - started

        self.assertEqual(len(clusters), 4)
        self.assertLess(elapsed, 5.0, f"聚类耗时 {elapsed:.2f}s，超过预期")


class ShortSegmentSpeakerTests(unittest.TestCase):
    """短片段（<1 秒）不跑 CAM++，但说话人标签仍按时间就近继承。"""

    @staticmethod
    def _make_outcome(index, start_s, duration_s, embedding=None):
        from api.routes.asr_qwen_common import SegmentOutcome
        from core.vad import VadSegment

        return SegmentOutcome(
            index=index,
            segment=VadSegment(start_s=start_s, end_s=start_s + duration_s),
            embedding=embedding,
        )

    def test_short_segment_inherits_nearest_speaker(self):
        from api.routes.asr_qwen_common import cluster_speakers

        embedding = torch.ones(192)
        outcomes = [
            self._make_outcome(0, 0.0, 3.0, embedding),
            self._make_outcome(1, 3.2, 0.6, None),   # 短片段：没有嵌入
            self._make_outcome(2, 6.0, 3.0, embedding),
        ]

        labels = cluster_speakers(outcomes, threshold=0.3)

        self.assertEqual(labels[0], "spk_0")
        self.assertEqual(labels[2], "spk_0")
        self.assertEqual(labels[1], "spk_0", "短片段应继承时间上最近的说话人")


class RequestConcurrencyTests(unittest.TestCase):
    def test_segment_and_global_limits(self):
        async def _run():
            concurrency = RequestConcurrency(segment_concurrency=2, request_concurrency=3)
            sem_a = concurrency.new_segment_semaphore()
            sem_b = concurrency.new_segment_semaphore()

            await concurrency.acquire(sem_a)
            await concurrency.acquire(sem_a)
            await concurrency.acquire(sem_b)
            self.assertEqual(sem_a._value, 0)
            self.assertEqual(sem_b._value, 1)

            async def _extra():
                await concurrency.acquire(sem_b)
                return True

            task = asyncio.create_task(_extra())
            await asyncio.sleep(0.05)
            self.assertFalse(task.done())
            await concurrency.release(sem_a)
            await concurrency.release(sem_a)
            await concurrency.release(sem_b)
            self.assertTrue(await asyncio.wait_for(task, timeout=2))

        asyncio.run(_run())


@dataclass
class FakeCtx:
    request: AsrRequestParams
    tmp_path: str = "/tmp/fake.wav"
    tmp_paths: list = None
    audio_total_s: float = 12.0
    load_audio_time_ms: float = 3.5
    audio_tensor: object = None
    sample_rate: int = 0

    def __post_init__(self):
        if self.tmp_paths is None:
            self.tmp_paths = []


class V2RouteTests(unittest.TestCase):
    def _client(self):
        from main import create_app

        return TestClient(create_app())

    def test_route_is_registered(self):
        client = self._client()
        paths = client.get("/openapi.json").json()["paths"]
        self.assertNotIn("/v1.1.8/seacraft_asr", paths)
        self.assertIn("/v2/seacraft_asr", paths)

    @patch("api.routes.asr_v2.get_slot_manager")
    def test_queue_full_returns_4013(self, mock_slots):
        manager = SimpleNamespace()
        manager.acquire = AsyncMock(
            side_effect=QueueFullError(
                active=8, capacity=8, waiting=0, queue_max_size=64
            )
        )
        manager.release = AsyncMock()
        mock_slots.return_value = manager

        client = self._client()
        response = client.post(
            "/v2/seacraft_asr",
            files={"audioFile": ("a.wav", b"0" * 2048, "audio/wav")},
        )
        self.assertEqual(response.json()["code"], 4013)
        self.assertEqual(
            response.json()["msg"],
            "处理队列已满，请稍后再试（当前音频并发：8/8，等待中：0/64）",
        )

    @patch("api.routes.asr_v2.get_slot_manager")
    def test_queue_timeout_returns_4010(self, mock_slots):
        manager = SimpleNamespace()
        manager.acquire = AsyncMock(side_effect=QueueTimeoutError())
        manager.release = AsyncMock()
        mock_slots.return_value = manager

        client = self._client()
        response = client.post(
            "/v2/seacraft_asr",
            files={"audioFile": ("a.wav", b"0" * 2048, "audio/wav")},
        )
        self.assertEqual(response.json()["code"], 4010)

    @patch("api.routes.asr_v2.run_qwen_asr")
    @patch("api.routes.asr_v2.prepare_asr_context")
    def test_success_response_contains_expected_fields(self, mock_prepare, mock_run):
        ctx = FakeCtx(request=make_request())
        mock_prepare.return_value = (None, ctx)

        async def _run(_ctx):
            return {
                "language": "Chinese",
                "segments": [],
                "text": "",
                "speed_info": [],
                "load_audio_time_ms": "1.00",
                "gpu_time_ms": "2.00",
                "language_ratio": {"zh": 100.0},
            }

        mock_run.side_effect = _run

        client = self._client()
        response = client.post(
            "/v2/seacraft_asr",
            files={"audioFile": ("a.wav", b"0" * 2048, "audio/wav")},
        )
        payload = response.json()
        self.assertIn("language_ratio", payload)
        self.assertIn("segments", payload)
        self.assertIn("speed_info", payload)


class RepetitionGuardPipelineTests(unittest.TestCase):
    """复读修复在 v2 链路上的端到端行为（打桩 Qwen，行为确定）。"""

    def _run(self, qwen_text, segments, **request_overrides):
        from api.routes.asr_qwen_common import run_qwen_asr
        from core.vad import VadSegment

        ctx = FakeCtx(request=make_request(**request_overrides))

        async def _fake_detect(_path):
            return list(segments)

        def _fake_clip(_src, _start, _duration):
            return "/tmp/fake_clip.wav"

        class _FakeClient:
            async def transcribe(self, path, language=None):
                return f"language Chinese<asr_text>{qwen_text}"

        async def _fake_get_client():
            return _FakeClient()

        with ExitStack() as stack:
            stack.enter_context(
                patch("api.routes.asr_qwen_common.detect_segments", _fake_detect)
            )
            stack.enter_context(
                patch("api.routes.asr_qwen_common.extract_audio_clip", _fake_clip)
            )
            stack.enter_context(
                patch(
                    "api.routes.asr_qwen_common.read_audio_array",
                    fake_read_audio_array,
                )
            )
            stack.enter_context(
                patch("api.routes.asr_qwen_common.get_qwen_client", _fake_get_client)
            )
            return asyncio.run(run_qwen_asr(ctx))

    def test_hallucination_segment_is_collapsed_in_response(self):
        from core.vad import VadSegment

        hallucination = "最小S零，" + "这个，" * 900
        result = self._run(hallucination, [VadSegment(0.0, 6.0)])
        text = result["segments"][0]["segment_text"]
        self.assertLess(len(text), 120)
        self.assertTrue(text.startswith("最小S零"))

    def test_normal_segment_is_unchanged_in_response(self):
        from core.vad import VadSegment

        text = "今天我们讲一下功率变换器的基本原理。"
        result = self._run(text, [VadSegment(0.0, 5.0)])
        self.assertEqual(result["segments"][0]["segment_text"], text)

    def test_overlong_without_repeat_structure_is_preserved_in_response(self):
        from core.vad import VadSegment

        text = "".join(chr(0x4E00 + i) for i in range(300))
        result = self._run(text, [VadSegment(0.0, 5.0)])
        self.assertEqual(result["segments"][0]["segment_text"], text)

    def test_role_and_emotion_unaffected_by_repair(self):
        from core.vad import VadSegment

        hallucination = "最小S零，" + "这个，" * 900

        async def _speaker(_array, _sr):
            return torch.zeros(192)

        async def _emotion(_array, _sr, _duration):
            return "开心"

        from api.routes.asr_qwen_common import run_qwen_asr

        ctx = FakeCtx(
            request=make_request(showSpk=True, showEmotion=True)
        )

        async def _fake_detect(_path):
            return [VadSegment(0.0, 6.0)]

        def _fake_clip(_src, _start, _duration):
            return "/tmp/fake_clip.wav"

        class _FakeClient:
            async def transcribe(self, path, language=None):
                return f"language Chinese<asr_text>{hallucination}"

        async def _fake_get_client():
            return _FakeClient()

        async def _noop_load(*_args, **_kwargs):
            return None

        with ExitStack() as stack:
            stack.enter_context(
                patch("api.routes.asr_qwen_common.detect_segments", _fake_detect)
            )
            stack.enter_context(
                patch("api.routes.asr_qwen_common.extract_audio_clip", _fake_clip)
            )
            stack.enter_context(
                patch(
                    "api.routes.asr_qwen_common.read_audio_array",
                    fake_read_audio_array,
                )
            )
            stack.enter_context(
                patch("api.routes.asr_qwen_common.get_qwen_client", _fake_get_client)
            )
            stack.enter_context(
                patch("api.routes.asr_qwen_common.load_v2_models_if_needed", _noop_load)
            )
            stack.enter_context(
                patch("api.routes.asr_qwen_common._speaker_embedding", _speaker)
            )
            stack.enter_context(
                patch("api.routes.asr_qwen_common._segment_emotion", _emotion)
            )
            result = asyncio.run(run_qwen_asr(ctx))

        segment = result["segments"][0]
        self.assertLess(len(segment["segment_text"]), 120)
        self.assertIsNotNone(segment["role"])
        self.assertEqual(segment["emotion"], "开心")


class ResponseStructureTests(unittest.TestCase):
    def test_v2_response_structure(self):
        from api.routes.asr_qwen_common import run_qwen_asr
        from core.vad import VadSegment

        ctx = FakeCtx(request=make_request())

        async def _fake_detect(_path):
            return [VadSegment(0.0, 2.0), VadSegment(2.5, 5.0)]

        def _fake_clip(_src, _start, _duration):
            return "/tmp/fake_clip.wav"

        class _FakeClient:
            def __init__(self):
                self.calls = []

            async def transcribe(self, path, language=None):
                self.calls.append((path, language))
                return "language Chinese<asr_text>你好"

        fake_client = _FakeClient()

        async def _fake_get_client():
            return fake_client

        with ExitStack() as stack:
            stack.enter_context(
                patch("api.routes.asr_qwen_common.detect_segments", _fake_detect)
            )
            stack.enter_context(
                patch("api.routes.asr_qwen_common.extract_audio_clip", _fake_clip)
            )
            stack.enter_context(
                patch(
                    "api.routes.asr_qwen_common.read_audio_array",
                    fake_read_audio_array,
                )
            )
            stack.enter_context(
                patch("api.routes.asr_qwen_common.get_qwen_client", _fake_get_client)
            )
            result = asyncio.run(run_qwen_asr(ctx))

        self.assertEqual(result["language"], "Chinese")
        self.assertEqual(result["language_ratio"], {"zh": 100.0})
        self.assertEqual(len(result["segments"]), 2)
        self.assertEqual(
            set(result["segments"][0].keys()),
            {"segment_text", "bg", "ed", "speed", "segment_words", "role", "emotion"},
        )
        self.assertEqual(result["segments"][0]["segment_text"], "你好")
        self.assertEqual(result["segments"][0]["segment_words"], [])
        self.assertIsNone(result["segments"][0]["role"])
        self.assertIsNone(result["segments"][0]["emotion"])
        self.assertEqual(result["segments"][0]["bg"], "0.00")
        self.assertEqual(result["segments"][0]["ed"], "2.00")
        # auto 语言不应下发给 Qwen
        self.assertEqual(fake_client.calls[0][1], "auto")

    def test_unsupported_language_maps_to_4009(self):
        from api.routes.asr_qwen_common import run_qwen_asr
        from core.vad import VadSegment

        ctx = FakeCtx(request=make_request(language="xx"))

        async def _fake_detect(_path):
            return [VadSegment(0.0, 2.0)]

        def _fake_clip(_src, _start, _duration):
            return "/tmp/fake_clip.wav"

        class _FakeClient:
            async def transcribe(self, path, language=None):
                raise QwenAsrUnsupportedLanguageError()

        async def _fake_get_client():
            return _FakeClient()

        with ExitStack() as stack:
            stack.enter_context(
                patch("api.routes.asr_qwen_common.detect_segments", _fake_detect)
            )
            stack.enter_context(
                patch("api.routes.asr_qwen_common.extract_audio_clip", _fake_clip)
            )
            stack.enter_context(
                patch(
                    "api.routes.asr_qwen_common.read_audio_array",
                    fake_read_audio_array,
                )
            )
            stack.enter_context(
                patch("api.routes.asr_qwen_common.get_qwen_client", _fake_get_client)
            )
            result = asyncio.run(run_qwen_asr(ctx))

        self.assertEqual(result["code"], 4009)

    def test_no_speech_returns_4008(self):
        from api.routes.asr_qwen_common import run_qwen_asr

        ctx = FakeCtx(request=make_request())

        async def _fake_detect(_path):
            return []

        with patch("api.routes.asr_qwen_common.detect_segments", _fake_detect):
            result = asyncio.run(run_qwen_asr(ctx))
        self.assertEqual(result["code"], 4008)


class QwenClientErrorMappingTests(unittest.TestCase):
    def test_error_codes(self):
        self.assertEqual(QwenAsrCallError().code, 4012)
        self.assertEqual(QwenAsrUnsupportedLanguageError().code, 4009)


class FaultInjectionTests(unittest.TestCase):
    """异常注入：角色/情绪失败必须回退为 null，且不影响转写结果。"""

    def _run_chain(self, ctx, client, speaker=None, emotion=None, reader=None):
        from api.routes.asr_qwen_common import run_qwen_asr
        from core.vad import VadSegment

        async def _fake_detect(_path):
            return [VadSegment(0.0, 2.0), VadSegment(2.5, 5.0)]

        def _fake_clip(_src, start, _duration, *args, **kwargs):
            # 用起始时间区分片段，便于注入「单段损坏」
            return f"/tmp/clip_{start}.wav"

        async def _fake_get_client():
            return client

        async def _noop_load(**_kwargs):
            return None

        with ExitStack() as stack:
            stack.enter_context(
                patch("api.routes.asr_qwen_common.load_v2_models_if_needed", _noop_load)
            )
            stack.enter_context(
                patch("api.routes.asr_qwen_common.detect_segments", _fake_detect)
            )
            stack.enter_context(
                patch("api.routes.asr_qwen_common.extract_audio_clip", _fake_clip)
            )
            stack.enter_context(
                patch(
                    "api.routes.asr_qwen_common.extract_audio_clip_from_tensor",
                    _fake_clip,
                )
            )
            stack.enter_context(
                patch(
                    "api.routes.asr_qwen_common.read_audio_array",
                    reader or fake_read_audio_array,
                )
            )
            stack.enter_context(
                patch("api.routes.asr_qwen_common.get_qwen_client", _fake_get_client)
            )
            if speaker is not None:
                stack.enter_context(
                    patch("api.routes.asr_qwen_common._speaker_embedding", speaker)
                )
            if emotion is not None:
                stack.enter_context(
                    patch("api.routes.asr_qwen_common._segment_emotion", emotion)
                )
            return asyncio.run(run_qwen_asr(ctx))

    def test_emotion_failure_falls_back_to_null(self):
        ctx = FakeCtx(request=make_request(showSpk=True, showEmotion=True))

        class _FakeClient:
            async def transcribe(self, path, language=None):
                return "language Chinese<asr_text>你好"

        async def _boom_speaker(audio_array, sample_rate):
            return torch.zeros(8)

        async def _boom_emotion(audio_array, sample_rate, duration_s):
            raise RuntimeError("emotion2vec 推理失败")

        result = self._run_chain(
            ctx, _FakeClient(), speaker=_boom_speaker, emotion=_boom_emotion
        )
        self.assertEqual(len(result["segments"]), 2)
        self.assertEqual(result["text"], "你好 你好")
        for segment in result["segments"]:
            self.assertIsNone(segment["emotion"])
            self.assertIsNotNone(segment["role"])

    def test_speaker_failure_falls_back_to_null(self):
        ctx = FakeCtx(request=make_request(showSpk=True, showEmotion=True))

        class _FakeClient:
            async def transcribe(self, path, language=None):
                return "language Chinese<asr_text>你好"

        async def _boom_speaker(audio_array, sample_rate):
            raise RuntimeError("CAM++ 推理失败")

        async def _ok_emotion(audio_array, sample_rate, duration_s):
            return "平淡"

        result = self._run_chain(
            ctx, _FakeClient(), speaker=_boom_speaker, emotion=_ok_emotion
        )
        self.assertEqual(len(result["segments"]), 2)
        for segment in result["segments"]:
            self.assertIsNone(segment["role"])
            self.assertEqual(segment["emotion"], "平淡")

    def test_corrupted_segment_only_affects_that_segment(self):
        """单段音频损坏时，该段的 role/emotion 为 null，其余段与转写照常。"""
        import numpy as np

        ctx = FakeCtx(request=make_request(showSpk=True, showEmotion=True))

        class _FakeClient:
            async def transcribe(self, path, language=None):
                return "language Chinese<asr_text>你好"

        def _reader(path):
            # 把片段起始时间编码进数组，便于识别「哪一段损坏」
            start = float(path.split("_")[-1].replace(".wav", ""))
            return np.full(1600, start, dtype="float32"), 16000

        async def _speaker(audio_array, sample_rate):
            if float(audio_array[0]) == 0.0:
                raise RuntimeError("音频损坏")
            return torch.zeros(8)

        async def _emotion(audio_array, sample_rate, duration_s):
            return "平淡"

        result = self._run_chain(
            ctx, _FakeClient(), speaker=_speaker, emotion=_emotion, reader=_reader
        )
        self.assertEqual(len(result["segments"]), 2)
        self.assertEqual(result["text"], "你好 你好")
        # 第一段（起始 0.0）损坏：role 为 null；第二段正常
        self.assertIsNone(result["segments"][0]["role"])
        self.assertIsNotNone(result["segments"][1]["role"])

    def test_qwen_failure_returns_4012(self):
        ctx = FakeCtx(request=make_request(showSpk=True))

        class _FakeClient:
            async def transcribe(self, path, language=None):
                raise QwenAsrCallError()

        async def _speaker(audio_array, sample_rate):
            return torch.zeros(8)

        result = self._run_chain(ctx, _FakeClient(), speaker=_speaker)
        self.assertEqual(result["code"], 4012)


if __name__ == "__main__":
    unittest.main()
