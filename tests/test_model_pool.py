import asyncio
import unittest
from unittest.mock import patch

from core.models import ModelPool, ModelUnavailableError, local_model_status


class _FakeModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class ModelPoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_loads_configured_instances_and_leases_different_models(self):
        with patch("core.models._resolve_device", return_value="cuda:0"), \
                patch("core.models.device_allocated_mb", return_value=100.0), \
                patch("core.models.AutoModel", side_effect=lambda **kwargs: _FakeModel(**kwargs)):
            pool = ModelPool("speech_campplus", "/model/cam", lambda: "cuda:0", lambda: 2)
            await pool.ensure_loaded()
            self.assertEqual(len(pool.models), 2)
            self.assertEqual(pool.configured_instances, 2)
            async with pool.lease() as first:
                async with pool.lease() as second:
                    self.assertIsNot(first, second)
            pool.refresh_status()
            status = local_model_status["speech_campplus"]
            self.assertEqual(status["instances"], 2)
            self.assertEqual(status["available_instances"], 2)
            await pool.close()

    async def test_failed_instance_is_isolated(self):
        calls = 0

        def make_model(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("simulated OOM")
            return _FakeModel(**kwargs)

        with patch("core.models._resolve_device", return_value="cuda:0"), \
                patch("core.models.device_allocated_mb", return_value=100.0), \
                patch("core.models.AutoModel", side_effect=make_model):
            pool = ModelPool("emotion2vec", "/model/emo", lambda: "cuda:0", lambda: 2, fp16=True)
            await pool.ensure_loaded()
            self.assertEqual(pool.models[0], None)
            self.assertIsNotNone(pool.models[1])
            pool.refresh_status()
            self.assertEqual(local_model_status["emotion2vec"]["failed_instances"], 1)
            async with pool.lease() as model:
                self.assertIs(model, pool.models[1])
                self.assertTrue(model.kwargs["fp16"])
            await pool.close()

    async def test_device_array_is_used_for_each_instance(self):
        with patch("core.models._resolve_device", side_effect=lambda value: value), \
                patch("core.models.device_allocated_mb", return_value=100.0), \
                patch("core.models.AutoModel", side_effect=lambda **kwargs: _FakeModel(**kwargs)):
            pool = ModelPool(
                "speech_campplus", "/model/cam",
                devices_getter=lambda: ["cuda:1", "cuda:2"],
            )
            await pool.ensure_loaded()
            self.assertEqual([model.kwargs["device"] for model in pool.models], ["cuda:1", "cuda:2"])
            async with pool.lease() as leased:
                model, device = leased
                self.assertEqual(device, "cuda:1")
                self.assertIs(model, pool.models[0])
            pool.refresh_status()
            self.assertEqual(local_model_status["speech_campplus"]["devices"], ["cuda:1", "cuda:2"])
            await pool.close()

    async def test_all_failed_instances_fail_fast(self):
        with patch("core.models._resolve_device", return_value="cuda:0"), \
                patch("core.models.AutoModel", side_effect=RuntimeError("simulated failure")):
            pool = ModelPool("speech_campplus", "/model/cam", lambda: "cuda:0", lambda: 1)
            with self.assertRaisesRegex(RuntimeError, "没有可用副本"):
                async with pool.lease():
                    pass

    async def test_empty_device_array_disables_pool_with_explicit_error(self):
        pool = ModelPool(
            "emotion2vec", "/model/emo", devices_getter=lambda: []
        )
        await pool.ensure_loaded()
        self.assertEqual(pool.configured_instances, 0)
        self.assertEqual(pool.models, [])
        with self.assertRaisesRegex(ModelUnavailableError, "emotion2vec 模型未启用") as context:
            async with pool.lease():
                pass
        self.assertEqual(context.exception.code, 4016)
        await pool.close()


if __name__ == "__main__":
    unittest.main()
