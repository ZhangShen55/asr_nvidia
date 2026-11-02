import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core import models


class CapabilityValidationTests(unittest.TestCase):
    def test_requested_capabilities_and_feature_switch_matrix(self):
        for role_open, emotion_open, need_role, need_emotion in product((False, True), repeat=4):
            with self.subTest(role_open=role_open, emotion_open=emotion_open,
                              need_role=need_role, need_emotion=need_emotion), \
                    patch.object(models, "settings", SimpleNamespace(
                        open_v2_role=role_open, open_v2_emotion=emotion_open)):
                role_disabled = need_role and not role_open
                emotion_disabled = need_emotion and not emotion_open
                code = 4018 if role_disabled and emotion_disabled else (
                    4015 if role_disabled else 4016 if emotion_disabled else None)
                if code is None:
                    models.validate_v2_capabilities(need_role, need_emotion)
                else:
                    with self.assertRaises(models.ModelUnavailableError) as caught:
                        models.validate_v2_capabilities(need_role, need_emotion)
                    self.assertEqual(caught.exception.code, code)


class DisabledRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_global_pools_never_load_or_lease_even_existing_models(self):
        config = SimpleNamespace(open_v2_role=False, open_v2_emotion=False,
                                 model_device_speech_campplus_devices=["cpu"],
                                 model_device_emotion2vec_devices=["cpu"])
        with patch.object(models, "settings", config), patch.object(models, "AutoModel") as constructor:
            for name, code in [("speech_campplus", 4015), ("emotion2vec", 4016)]:
                pool = models.get_model_pool(name)
                for existing in ([], [object()]):
                    with self.subTest(name=name, loaded=bool(existing)), patch.object(pool, "models", existing):
                        with self.assertRaises(models.ModelUnavailableError) as caught:
                            await pool.ensure_loaded()
                        self.assertEqual(caught.exception.code, code)
                        with self.assertRaises(models.ModelUnavailableError):
                            async with pool.lease():
                                self.fail("disabled model must not be leased")
            constructor.assert_not_called()

    async def test_full_requirements_rejected_before_loading_any_model(self):
        with patch.object(models, "settings", SimpleNamespace(open_v2_role=True, open_v2_emotion=False)), \
                patch.object(models.ModelPool, "ensure_loaded", new_callable=AsyncMock) as load:
            with self.assertRaises(models.ModelUnavailableError) as caught:
                await models.load_v2_models_if_needed(True, True)
            self.assertEqual(caught.exception.code, 4016)
            load.assert_not_awaited()

    async def test_direct_pipeline_and_short_emotion_cannot_bypass_disabled_switches(self):
        from api.routes import asr_qwen_common as pipeline

        request = SimpleNamespace(showSpk=True, showEmotion=True, showRoleIdentify=False)
        with patch.object(models, "settings", SimpleNamespace(open_v2_role=False, open_v2_emotion=False)), \
                patch.object(pipeline, "get_model_pool") as get_pool, \
                patch.object(pipeline, "detect_segments", new_callable=AsyncMock) as vad:
            with self.assertRaises(models.ModelUnavailableError) as caught:
                await pipeline.run_qwen_asr(SimpleNamespace(request=request))
            self.assertEqual(caught.exception.code, 4018)
            with self.assertRaises(models.ModelUnavailableError):
                await pipeline._speaker_embedding(None, 16000)
            with self.assertRaises(models.ModelUnavailableError):
                await pipeline._segment_emotion(None, 16000, 0.5)
            get_pool.assert_not_called()
            vad.assert_not_awaited()
