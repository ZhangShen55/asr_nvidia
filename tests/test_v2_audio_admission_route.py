import asyncio
import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import asr_v3 as route
from core.config import settings
from core import request_telemetry as telemetry
from core.qwen_asr_queue import AudioSlotManager, QueueFullError
from entity.data import AsrRequestParams


def make_request(**flags):
    return AsrRequestParams(audioFile=SimpleNamespace(filename="a.wav"), wordTimestamps=False, **flags)


def context(request):
    return SimpleNamespace(request=request, tmp_paths=[], task_id="test", filename="a.wav")


class CapabilityRouteTests(unittest.TestCase):
    def test_http_feature_matrix_precedes_full_capacity_without_processing(self):
        app = FastAPI()
        app.include_router(route.router)
        client = TestClient(app)
        for role_open, emotion_open, spk, emo, identify in product((False, True), repeat=5):
            role_disabled = (spk or identify) and not role_open
            emotion_disabled = emo and not emotion_open
            expected = 4018 if role_disabled and emotion_disabled else (
                4015 if role_disabled else 4016 if emotion_disabled else 4013)
            slots = SimpleNamespace(acquire=AsyncMock(side_effect=QueueFullError(active=8, capacity=8)),
                                    release=AsyncMock(), snapshot=AudioSlotManager(8).snapshot)
            with self.subTest(role_open=role_open, emotion_open=emotion_open, spk=spk, emo=emo, identify=identify), \
                    patch.dict(settings._cfg, {"features": {"open_v2_role": role_open, "open_v2_emotion": emotion_open}}), \
                    patch.object(route, "get_slot_manager", return_value=slots), \
                    patch.object(route, "prepare_asr_context", new_callable=AsyncMock) as prepare, \
                    patch.object(route, "run_qwen_asr", new_callable=AsyncMock) as run, \
                    patch("core.models.ModelPool.ensure_loaded", new_callable=AsyncMock) as load, \
                    patch("api.routes.asr_qwen_common.detect_segments", new_callable=AsyncMock) as vad, \
                    patch("api.routes.asr_qwen_common.get_qwen_client", new_callable=AsyncMock) as qwen:
                response = client.post("/v3/seacraft_asr", files={"audioFile": ("a.wav", b"x"*2048)},
                                       data={"showSpk": str(spk).lower(), "showEmotion": str(emo).lower(),
                                             "showRoleIdentify": str(identify).lower()})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["code"], expected)
                self.assertEqual(set(response.json()), {"msg", "code"})
                if expected == 4013:
                    slots.acquire.assert_awaited_once_with("B" if spk or emo or identify else "A")
                else:
                    slots.acquire.assert_not_awaited()
                slots.release.assert_not_awaited()
                for callback in [prepare, run, load, vad, qwen]:
                    callback.assert_not_awaited()


class LeaseRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_classification_and_disabled_features_allow_a(self):
        for spk, emo, identify in product((False, True), repeat=3):
            request = make_request(showSpk=spk, showEmotion=emo, showRoleIdentify=identify)
            manager = AudioSlotManager(8, spk_emo_capacity=4)
            expected_b = int(spk or emo or identify)

            async def run(ctx):
                self.assertEqual((manager.active, manager.active_b), (1, expected_b))
                return {"segments": [{"role": None, "emotion": None, "segment_words": []}]}

            features = {"open_v2_role": bool(expected_b), "open_v2_emotion": bool(expected_b)}
            with patch.dict(settings._cfg, {"features": features}), \
                    patch.object(route, "get_slot_manager", return_value=manager), \
                    patch.object(route, "prepare_asr_context", new=AsyncMock(return_value=(None, context(request)))), \
                    patch.object(route, "run_qwen_asr", side_effect=run):
                payload = await route.api_asr_v3(request)
                self.assertIn("segment_words", payload["segments"][0])
                self.assertEqual((manager.active, manager.active_b), (0, 0))

    async def test_every_exit_cleans_before_releasing_exactly_once(self):
        for scenario in ["success", "prepare_error", "prepare_exception", "prepare_cancel", "business_error",
                         "exception", "cancel", "cleanup_error"]:
            manager = AudioSlotManager(1, spk_emo_capacity=1)
            request = make_request(showSpk=True)
            ctx = context(request)
            events = []
            real_release = manager.release

            async def release(lease):
                events.append("release")
                self.assertEqual((manager.active, manager.active_b), (1, 1))
                await real_release(lease)

            def cleanup(paths):
                events.append("cleanup")
                self.assertEqual((manager.active, manager.active_b), (1, 1))
                if scenario == "cleanup_error":
                    raise OSError("cleanup failed")

            prepare = AsyncMock(return_value=(None, ctx))
            run = AsyncMock(return_value={"segments": []})
            if scenario == "prepare_error":
                prepare.return_value = ({"code": 4002, "msg": "prepare failed"}, None)
            elif scenario == "prepare_exception":
                prepare.side_effect = RuntimeError("prepare failed")
            elif scenario == "prepare_cancel":
                prepare.side_effect = asyncio.CancelledError
            elif scenario == "business_error":
                run.return_value = {"code": 4012, "msg": "call failed"}
            elif scenario == "exception":
                run.side_effect = RuntimeError("processing failed")
            elif scenario == "cancel":
                run.side_effect = asyncio.CancelledError
            with self.subTest(scenario=scenario), \
                    patch.dict(settings._cfg, {"features": {"open_v2_role": True, "open_v2_emotion": True}}), \
                    patch.object(route, "get_slot_manager", return_value=manager), \
                    patch.object(route, "prepare_asr_context", new=prepare), \
                    patch.object(route, "run_qwen_asr", new=run), \
                    patch.object(route, "cleanup_temp_files", side_effect=cleanup), \
                    patch.object(manager, "release", side_effect=release) as release_mock:
                if "cancel" in scenario:
                    with self.assertRaises(asyncio.CancelledError):
                        await route.api_asr_v3(request)
                elif scenario == "prepare_exception":
                    with self.assertRaisesRegex(RuntimeError, "prepare failed"):
                        await route.api_asr_v3(request)
                else:
                    payload = await route.api_asr_v3(request)
                    expected_code = {"prepare_error": 4002, "business_error": 4012, "exception": 4005}.get(scenario)
                    self.assertEqual(payload.get("code"), expected_code)
                release_mock.assert_awaited_once()
                self.assertEqual(events, ["release"] if scenario.startswith("prepare_") else ["cleanup", "release"])
                self.assertEqual((manager.active, manager.active_b, manager.waiting), (0, 0, 0))


class AdmissionTraceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        telemetry.reset_for_tests()
        self.addCleanup(telemetry.reset_for_tests)
        features = patch.dict(settings._cfg, {"features": {"open_v2_role": True, "open_v2_emotion": True}})
        features.start()
        self.addCleanup(features.stop)

    def latest(self):
        return telemetry.snapshot()["recent"][-1]

    async def test_rejections_record_distinct_reason_snapshot_and_log(self):
        for wait_s, queue_max, category, code, reason in (
            (0, 64, "A", 4013, "total_full"),
            (0, 64, "B", 4017, "spk_emo_full"),
            (1, 0, "B", 4013, "waiting_full"),
            (0.001, 64, "B", 4010, "wait_timeout"),
        ):
            manager = AudioSlotManager(2, wait_timeout_s=wait_s, queue_max_size=queue_max, spk_emo_capacity=1)
            held = [await manager.acquire("B")]
            if category == "A":
                held.append(await manager.acquire("A"))
            request = make_request(showSpk=category == "B")
            try:
                with patch.object(route, "get_slot_manager", return_value=manager), \
                        patch.object(route, "prepare_asr_context", new_callable=AsyncMock) as prepare, \
                        self.assertLogs(route.logger, level="INFO") as logs:
                    response = await route.api_asr_v3(request)
                self.assertEqual(response["code"], code)
                event = self.latest()
                self.assertEqual(event["admission"]["category"], category)
                self.assertEqual(event["admission"]["reason"], reason)
                self.assertEqual(event["admission"]["snapshot"], manager.snapshot())
                self.assertEqual(event["status"], "failed")
                self.assertGreaterEqual(event["phases"]["audio_wait_ms"], 0)
                self.assertTrue(any("[V3准入]" in log and f"reason={reason}" in log and f"code={code}" in log
                                    for log in logs.output))
                prepare.assert_not_awaited()
            finally:
                for lease in held:
                    await manager.release(lease)

    async def test_early_rejection_resets_phases_from_previous_request(self):
        manager = AudioSlotManager(8)
        request = make_request()

        async def run(ctx):
            from api.routes.asr_qwen_common import reset_phase_stats
            reset_phase_stats().update(qwen_ms=123, spk_ms=456, vad_ms=78)
            return {"segments": []}

        with patch.object(route, "get_slot_manager", return_value=manager), \
                patch.object(route, "prepare_asr_context", new=AsyncMock(return_value=(None, context(request)))), \
                patch.object(route, "run_qwen_asr", side_effect=run):
            await route.api_asr_v3(request)
            self.assertEqual(self.latest()["phases"]["qwen_ms"], 123)
            with patch.dict(settings._cfg, {"features": {"open_v2_role": False, "open_v2_emotion": False}}):
                response = await route.api_asr_v3(make_request(showSpk=True, showEmotion=True))
            self.assertEqual(response["code"], 4018)
            event = self.latest()
            self.assertTrue(all(value == 0 for value in event["phases"].values()))
            self.assertEqual(event["admission"]["reason"], "both_disabled")
            self.assertEqual(event["admission"]["category"], "B")
            self.assertEqual(event["admission"]["snapshot"]["spkEmoConcurrency"]["active"], 0)

    async def test_waiting_cancellation_and_post_admission_failures(self):
        manager = AudioSlotManager(1, wait_timeout_s=5)
        held = await manager.acquire("A")
        with patch.object(route, "get_slot_manager", return_value=manager):
            waiting = asyncio.create_task(route.api_asr_v3(make_request(showEmotion=True)))
            await asyncio.sleep(0)
            self.assertEqual(manager.waiting_b, 1)
            waiting.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiting
        event = self.latest()
        self.assertEqual(event["status"], "cancelled")
        self.assertEqual(event["admission"]["outcome"], "cancelled")
        self.assertEqual(event["admission"]["reason"], "cancelled")
        self.assertEqual(event["admission"]["snapshot"], manager.snapshot())
        self.assertEqual(manager.waiting, 0)
        await manager.release(held)

        request = make_request(showEmotion=True)
        for failure in (RuntimeError("failure"), asyncio.CancelledError()):
            with patch.object(route, "get_slot_manager", return_value=manager), \
                    patch.object(route, "prepare_asr_context", new=AsyncMock(return_value=(None, context(request)))), \
                    patch.object(route, "run_qwen_asr", new=AsyncMock(side_effect=failure)):
                if isinstance(failure, asyncio.CancelledError):
                    with self.assertRaises(asyncio.CancelledError):
                        await route.api_asr_v3(request)
                    expected_status = "cancelled"
                else:
                    self.assertEqual((await route.api_asr_v3(request))["code"], 4005)
                    expected_status = "failed"
            event = self.latest()
            self.assertEqual(event["status"], expected_status)
            self.assertEqual(event["admission"]["outcome"], "admitted")
            self.assertIsNone(event["admission"]["reason"])
            self.assertEqual((manager.active, manager.active_b, manager.waiting), (0, 0, 0))


class MixedAdmissionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_mixed_route_capacity_and_final_recovery(self):
        manager = AudioSlotManager(8, spk_emo_capacity=4)
        finish = asyncio.Event()
        entered = asyncio.Queue()
        tasks = []

        async def run(ctx):
            entered.put_nowait(None)
            await finish.wait()
            return {"segments": [], "language_ratio": {"zh": 100}, "speed_info": []}

        async def prepare(request, **kwargs):
            return None, context(request)

        async def start(category):
            task = asyncio.create_task(route.api_asr_v3(make_request(showEmotion=category == "B")))
            tasks.append(task)
            await asyncio.wait_for(entered.get(), timeout=1)

        with patch.dict(settings._cfg, {"features": {"open_v2_role": True, "open_v2_emotion": True}}), \
                patch.object(route, "get_slot_manager", return_value=manager), \
                patch.object(route, "prepare_asr_context", side_effect=prepare), \
                patch.object(route, "run_qwen_asr", side_effect=run):
            try:
                for category in ["A"] * 3 + ["B"] * 4:
                    await start(category)
                self.assertEqual((manager.active, manager.active_b), (7, 4))
                self.assertEqual((await route.api_asr_v3(make_request(showEmotion=True)))["code"], 4017)
                await start("A")
                self.assertEqual((manager.active, manager.active_b), (8, 4))
                for enhanced in (False, True):
                    self.assertEqual((await route.api_asr_v3(make_request(showEmotion=enhanced)))["code"], 4013)
            finally:
                finish.set()
                responses = await asyncio.gather(*tasks)
        self.assertEqual(len(responses), 8)
        self.assertTrue(all("language_ratio" in response and "speed_info" in response for response in responses))
        self.assertEqual((manager.active, manager.active_b, manager.waiting), (0, 0, 0))
