from dataclasses import dataclass
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from entity.data import AsrRequestParams


SUCCESS_KEYS = {
    "language",
    "segments",
    "text",
    "speed_info",
    "load_audio_time_ms",
    "gpu_time_ms",
}


@dataclass
class FakeWord:
    start: float
    end: float
    word: str


@dataclass
class FakeSegment:
    start: float
    end: float
    text: str
    words: list = None


class FakeWhisper:
    def __init__(self, segments):
        self._segments = list(segments)
        self.calls = []

    def transcribe(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return iter(self._segments), SimpleNamespace(language="fr")


def make_request(
    *,
    language="fr",
    word_timestamps=False,
    show_spk=False,
    show_emotion=False,
    show_role_identify=False,
):
    return AsrRequestParams(
        audioFile=object(),
        language=language,
        wordTimestamps=word_timestamps,
        hotWords=[],
        showSpk=show_spk,
        openPanel=False,
        showEmotion=show_emotion,
        showSpeed=False,
        showRoleIdentify=show_role_identify,
    )


def make_context(request, *, duration=60.0):
    return SimpleNamespace(
        request=request,
        tmp_path="/tmp/fake-french-audio.wav",
        tmp_paths=[],
        audio_total_s=duration,
        task_id="fake-french-task",
        load_audio_time_ms=12.345,
    )


class HttpLanguageFormContractTests(unittest.TestCase):
    @staticmethod
    def _build_client():
        from api.routes import asr_v18 as route

        app = FastAPI()
        app.include_router(route.router)
        return route, TestClient(app)

    def test_explicit_empty_language_is_rejected_before_audio_preparation(self):
        route, client = self._build_client()
        prepare = AsyncMock(return_value=({"msg": "audio read", "code": 4002}, None))

        with patch.object(route, "prepare_asr_context", prepare):
            response = client.post(
                "/v1.1.8/seacraft_asr",
                data={"language": ""},
                files={"audioFile": ("dummy.wav", b"not-read", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"msg": "不支持的语言: ", "code": 4009})
        prepare.assert_not_awaited()

    def test_omitted_language_keeps_auto_default(self):
        route, client = self._build_client()
        captured = {}

        async def _prepare(context):
            captured["language"] = context.language
            return None, make_context(context)

        prepare = AsyncMock(side_effect=_prepare)

        with (
            patch.object(route, "prepare_asr_context", prepare),
            patch.object(
                route,
                "run_paraformer_asr",
                AsyncMock(side_effect=lambda ctx: {"language": ctx.request.language}),
            ),
        ):
            response = client.post(
                "/v1.1.8/seacraft_asr",
                files={"audioFile": ("dummy.wav", b"not-read", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"language": "auto"})
        self.assertEqual(captured["language"], "auto")


class LanguageDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_paraformer_languages_are_normalized_before_dispatch(self):
        from api.routes import asr_v18 as route

        for supplied, expected in ((" Auto ", "auto"), ("ZH", "zh"), ("EN", "en")):
            with self.subTest(language=supplied):
                request = make_request(language=supplied)
                ctx = make_context(request)
                prepare = AsyncMock(return_value=(None, ctx))
                paraformer = AsyncMock(
                    side_effect=lambda prepared: {"language": prepared.request.language}
                )
                whisper_getter = Mock(return_value=FakeWhisper([]))

                with (
                    patch.object(route, "prepare_asr_context", prepare),
                    patch.object(route, "run_paraformer_asr", paraformer),
                    patch.object(route, "get_whisper_model", whisper_getter),
                ):
                    result = await route.api_asr_v18(request)

                self.assertEqual(result, {"language": expected})
                self.assertEqual(request.language, expected)
                prepare.assert_awaited_once_with(request)
                paraformer.assert_awaited_once_with(ctx)
                whisper_getter.assert_not_called()

    async def test_unknown_and_empty_languages_are_rejected_before_audio_preparation(self):
        from api.routes import asr_v18 as route

        for supplied, normalized in ((" Klingon ", "klingon"), ("", "")):
            with self.subTest(language=supplied):
                request = make_request(language=supplied)
                prepare = AsyncMock(return_value=(None, make_context(request)))
                paraformer = AsyncMock(return_value={"unexpected": True})

                with (
                    patch.object(route, "prepare_asr_context", prepare),
                    patch.object(route, "run_paraformer_asr", paraformer),
                ):
                    result = await route.api_asr_v18(request)

                self.assertEqual(
                    result,
                    {"msg": f"不支持的语言: {normalized}", "code": 4009},
                )
                prepare.assert_not_awaited()
                paraformer.assert_not_awaited()

    async def test_disabled_french_is_rejected_before_audio_preparation(self):
        from api.routes import asr_v18 as route

        request = make_request(language=" FR ")
        prepare = AsyncMock(return_value=(None, make_context(request)))
        paraformer = AsyncMock(return_value={"path": "paraformer"})
        whisper_getter = Mock(return_value=FakeWhisper([]))

        with (
            patch.object(route, "settings", SimpleNamespace(open_mul_lang=False)),
            patch.object(route, "prepare_asr_context", prepare),
            patch.object(route, "run_paraformer_asr", paraformer),
            patch.object(route, "get_whisper_model", whisper_getter),
        ):
            result = await route.api_asr_v18(request)

        self.assertEqual(
            result,
            {"msg": "未开启小语种识别或模型未就绪", "code": 4003},
        )
        self.assertEqual(request.language, "fr")
        prepare.assert_not_awaited()
        whisper_getter.assert_not_called()

    async def test_unready_french_is_rejected_before_audio_preparation(self):
        from api.routes import asr_v18 as route

        request = make_request(language="fr")
        prepare = AsyncMock(return_value=(None, make_context(request)))
        paraformer = AsyncMock(return_value={"path": "paraformer"})
        whisper_getter = Mock(return_value=None)

        with (
            patch.object(route, "settings", SimpleNamespace(open_mul_lang=True)),
            patch.object(route, "prepare_asr_context", prepare),
            patch.object(route, "run_paraformer_asr", paraformer),
            patch.object(route, "get_whisper_model", whisper_getter),
        ):
            result = await route.api_asr_v18(request)

        self.assertEqual(
            result,
            {"msg": "未开启小语种识别或模型未就绪", "code": 4003},
        )
        prepare.assert_not_awaited()
        whisper_getter.assert_called_once_with()


class WhisperResponseContractTests(unittest.IsolatedAsyncioTestCase):
    async def _call_french(self, request, segments, *, duration=60.0):
        from api.routes import asr_v18 as route

        ctx = make_context(request, duration=duration)
        model = FakeWhisper(segments)
        prepare = AsyncMock(return_value=(None, ctx))
        paraformer = AsyncMock(return_value={"path": "paraformer"})

        with (
            patch.object(route, "settings", SimpleNamespace(open_mul_lang=True)),
            patch.object(route, "prepare_asr_context", prepare),
            patch.object(route, "run_paraformer_asr", paraformer),
            patch.object(route, "get_whisper_model", Mock(return_value=model)),
        ):
            result = await route.api_asr_v18(request)

        return result, model, prepare, paraformer

    async def test_french_uses_whisper_and_preserves_success_top_level_fields(self):
        request = make_request(language=" FR ")
        segment = FakeSegment(0.0, 2.0, " bonjour le monde ")

        result, model, prepare, paraformer = await self._call_french(request, [segment])

        self.assertEqual(set(result), SUCCESS_KEYS)
        self.assertEqual(result["language"], "fr")
        self.assertEqual(result["text"], "bonjour le monde")
        self.assertNotIn("code", result)
        prepare.assert_awaited_once_with(request)
        paraformer.assert_not_awaited()
        self.assertEqual(len(model.calls), 1)
        args, kwargs = model.calls[0]
        self.assertEqual(args, ("/tmp/fake-french-audio.wav",))
        self.assertEqual(
            kwargs,
            {"language": "fr", "beam_size": 5, "word_timestamps": True},
        )

    async def test_requested_role_and_emotion_are_null_without_enhancement_models(self):
        request = make_request(
            show_spk=True,
            show_emotion=True,
            show_role_identify=True,
        )
        segment = FakeSegment(0.0, 2.0, "bonjour")

        result, _model, _prepare, _paraformer = await self._call_french(request, [segment])

        self.assertIsNone(result["segments"][0]["role"])
        self.assertIsNone(result["segments"][0]["emotion"])

    async def test_word_timestamps_are_returned_when_requested(self):
        request = make_request(language="fr", word_timestamps=True)
        segment = FakeSegment(
            0.0,
            2.0,
            "bonjour le monde",
            words=[FakeWord(0.0, 0.5, " bonjour")],
        )

        result, _model, _prepare, _paraformer = await self._call_french(request, [segment])

        self.assertEqual(
            result["segments"][0]["segment_words"],
            [{"bg": "0.00", "ed": "0.50", "word_text": "bonjour"}],
        )

    async def test_empty_transcription_returns_no_voice_error(self):
        request = make_request(language="fr")

        result, _model, _prepare, _paraformer = await self._call_french(request, [])

        self.assertEqual(
            result,
            {"msg": "音频文件为空或未检测到任何人声", "code": 4008},
        )


if __name__ == "__main__":
    unittest.main()
