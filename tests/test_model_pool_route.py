import contextlib
import unittest
from unittest.mock import patch

import numpy as np
import torch

from api.routes import asr_qwen_common as route


class _FakeModel:
    def __init__(self, kind):
        self.kind = kind

    def generate(self, **kwargs):
        if self.kind == "spk":
            return [{"spk_embedding": np.array([1.0, 2.0, 3.0], dtype=np.float32)}]
        return [{"labels": ["开心/happy", "中立/neutral"], "scores": [0.9, 0.1]}]


class _FakeLease:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device

    async def __aenter__(self):
        return (self.model, self.device) if self.device else self.model

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakePool:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device

    def lease(self):
        return _FakeLease(self.model, self.device)


class ModelPoolRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_speaker_and_emotion_use_pool_and_preserve_contract(self):
        audio = np.zeros(16000, dtype=np.float32)
        with patch.object(route.torch.cuda, "device", return_value=contextlib.nullcontext()), \
                patch.object(route, "get_model_pool", side_effect=lambda name: _FakePool(
                    _FakeModel("spk" if name == "speech_campplus" else "emo")
                )):
            embedding = await route._speaker_embedding(audio, 16000)
            emotion = await route._segment_emotion(audio, 16000, 2.0)
        self.assertIsInstance(embedding, torch.Tensor)
        self.assertEqual(embedding.dtype, torch.float32)
        self.assertEqual(tuple(embedding.shape), (3,))
        self.assertEqual(emotion, "积极")

    async def test_pool_device_is_used_by_inference(self):
        audio = np.zeros(16000, dtype=np.float32)
        devices = []
        original_device = route.torch.cuda.device

        def capture_device(device):
            devices.append(device)
            return contextlib.nullcontext()

        with patch.object(route.torch.cuda, "device", side_effect=capture_device), \
                patch.object(route, "get_model_pool", side_effect=lambda name: _FakePool(
                    _FakeModel("spk" if name == "speech_campplus" else "emo"), "cuda:2"
                )):
            await route._speaker_embedding(audio, 16000)
            await route._segment_emotion(audio, 16000, 2.0)
        self.assertEqual(devices, ["cuda:2", "cuda:2"])

    async def test_pool_failure_is_returned_as_request_level_degradation(self):
        class FailingPool:
            def lease(self):
                raise RuntimeError("没有可用副本")

        audio = np.zeros(16000, dtype=np.float32)
        with patch.object(route, "get_model_pool", return_value=FailingPool()), \
                patch.object(route.torch.cuda, "device", return_value=contextlib.nullcontext()):
            with self.assertRaises(RuntimeError):
                await route._speaker_embedding(audio, 16000)


if __name__ == "__main__":
    unittest.main()
