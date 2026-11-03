"""验证真实 TOML 配置边界；不加载任何模型或加速器依赖。"""
import tempfile
import unittest
from pathlib import Path

from core.config import Settings


class TensorRTConfigTests(unittest.TestCase):
    def config(self, body):
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            f.write(body)
        self.addCleanup(Path(f.name).unlink)
        return Settings(config_path=f.name)

    def test_defaults_and_independent_choices(self):
        cfg = self.config("")
        cfg.validate_model_backends()
        for name in ("speech_campplus", "emotion2vec"):
            self.assertFalse(getattr(cfg, name + "_use_tensorrt"))
            self.assertEqual(getattr(cfg, name + "_tensorrt_precision"), "fp16")
        cfg = self.config('[model_device_pool]\nspeech_campplus_use_tensorrt=true\n'
                          'speech_campplus_tensorrt_precision="fp32"\n')
        cfg.validate_model_backends()
        self.assertTrue(cfg.speech_campplus_use_tensorrt)
        self.assertEqual(cfg.speech_campplus_tensorrt_precision, "fp32")
        self.assertFalse(cfg.emotion2vec_use_tensorrt)

    def test_rejects_non_boolean_and_invalid_precision_with_field(self):
        for name in ("speech_campplus", "emotion2vec"):
            for option, invalid in [("use_tensorrt", ['"false"', "0", "1.0", "[]"]),
                                    ("tensorrt_precision", ['"FP16"', '"int8"', "16", "true"])]:
                key = f"{name}_{option}"
                for value in invalid:
                    with self.subTest(key=key, value=value):
                        cfg = self.config(f"[model_device_pool]\n{key}={value}\n")
                        with self.assertRaisesRegex(ValueError, key):
                            cfg.validate_model_backends()

    def test_repeated_mixed_devices_preserve_order_and_empty_pool(self):
        cfg = self.config('[model_device_pool]\nspeech_campplus_devices='
                          '["cuda:0", "cuda:0", "npu:1", "cpu"]\nemotion2vec_devices=[]\n')
        cfg.validate_model_backends()
        self.assertEqual(cfg.model_device_speech_campplus_devices,
                         ["cuda:0", "cuda:0", "npu:1", "cpu"])
        self.assertEqual(cfg.model_device_emotion2vec_devices, [])

    def test_vad_device_contract_does_not_accept_npu(self):
        cfg = self.config('[qwen_asr]\nvad_device=["npu:0"]\n')
        with self.assertRaises(ValueError):
            _ = cfg.vad_device

    def test_templates_have_disabled_switches_and_fp16(self):
        local = Settings(config_path="config.toml")
        local.validate_model_backends()
        self.assertFalse(local.speech_campplus_use_tensorrt)
        self.assertFalse(local.emotion2vec_use_tensorrt)
        self.assertEqual(local.speech_campplus_tensorrt_precision, "fp16")
        self.assertEqual(local.emotion2vec_tensorrt_precision, "fp16")

        docker = Settings(config_path="docker/config.toml")
        docker.validate_model_backends()
        self.assertTrue(docker.speech_campplus_use_tensorrt)
        self.assertTrue(docker.emotion2vec_use_tensorrt)
        self.assertEqual(docker.speech_campplus_tensorrt_precision, "fp16")
        self.assertEqual(docker.emotion2vec_tensorrt_precision, "fp16")
