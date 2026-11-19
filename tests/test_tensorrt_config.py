"""旧 NVIDIA 配置迁移与新 CPU/NPU 配置边界，不加载硬件。"""
import tempfile
import unittest
from pathlib import Path
from core.config import Settings


class NpuConfigTests(unittest.TestCase):
    def config(self, body):
        with tempfile.NamedTemporaryFile(suffix=".toml", mode="w", delete=False) as f:
            f.write(body)
        self.addCleanup(Path(f.name).unlink)
        return Settings(config_path=f.name)

    def test_legacy_tensorrt_keys_are_ignored_with_warning_even_invalid_old_types(self):
        cfg = self.config('[model_device_pool]\nspeech_campplus_use_tensorrt="old"\n'
                          'emotion2vec_tensorrt_precision=17\n')
        with self.assertLogs('core.config', level='WARNING') as logs:
            cfg.validate_model_backends()
        self.assertTrue(any('已忽略废弃 TensorRT' in line for line in logs.output))
        self.assertFalse(hasattr(cfg, 'speech_campplus_use_tensorrt'))

    def test_repeated_mixed_devices_preserve_order_and_empty_pool(self):
        cfg = self.config('[model_device_pool]\nspeech_campplus_devices='
                          '["npu:7","npu:7","cpu"]\nemotion2vec_devices=[]\n')
        cfg.validate_model_backends()
        self.assertEqual(cfg.model_device_speech_campplus_devices, ['npu:7','npu:7','cpu'])
        self.assertEqual(cfg.model_device_emotion2vec_devices, [])

    def test_vad_accepts_npu_but_rejects_cuda_and_invalid_devices(self):
        self.assertEqual(self.config('[vad]\nvad_device=["cpu","npu:7"]\n').vad_device,
                         ['cpu','npu:7'])
        for value in ['"cuda:0"','"npu:-1"','"npu:x"','true','7']:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, r'\[vad\].vad_device\[0\]'):
                _ = self.config(f'[vad]\nvad_device=[{value}]\n').vad_device

    def test_local_qwen_config_has_strict_types(self):
        cfg = self.config('')
        self.assertEqual(cfg.qwen_local_device, 'npu:7')
        self.assertEqual(cfg.qwen_local_dtype, 'bfloat16')
        self.assertEqual(cfg.qwen_local_instance_count, 1)
        for key, value in [('local_device','"cuda:0"'),('local_device','"cpu"'),
                           ('local_dtype','[]'),('local_instance_count','true'),
                           ('max_new_tokens','0'),('local_model_dir','""')]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.config(f'[qwen_asr]\n{key}={value}\n').validate_model_backends()

    def test_templates_use_npu_and_have_no_active_tensorrt_keys(self):
        for name in ['config.toml','docker/config.toml']:
            cfg = Settings(config_path=str(Path(__file__).resolve().parents[1] / name))
            cfg.validate_model_backends()
            self.assertFalse(any('tensorrt' in k for k in cfg._cfg.get('model_device_pool', {})))
            self.assertTrue(all(x == 'cpu' or x.startswith('npu:') for x in cfg.vad_device))
