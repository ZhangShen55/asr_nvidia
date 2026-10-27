import logging
import tempfile
import unittest
from pathlib import Path

from core.config import Settings


class ConfigContractTests(unittest.TestCase):
    def _settings(self, text):
        handle = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
        handle.write(text)
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return Settings(config_path=handle.name)

    def test_nested_log_path_has_priority(self):
        settings = self._settings('[log]\nlog_path = "nested.log"\nlog_max_days = 7\n')
        self.assertEqual(settings.log_path, "nested.log")

    def test_legacy_log_path_falls_back_with_warning(self):
        settings = self._settings('log_path = "legacy.log"\n')
        with self.assertLogs("core.config", level=logging.WARNING) as logs:
            self.assertEqual(settings.log_path, "legacy.log")
        self.assertTrue(any("迁移" in message for message in logs.output))

    def test_device_integer_is_migrated_to_cuda_string(self):
        settings = self._settings('[model_device]\nemotion2vec = 1\nspeech_campplus = "cuda:2"\n')
        self.assertEqual(settings.model_device_emotion2vec, "cuda:1")
        self.assertEqual(settings.model_device_speech_campplus, "cuda:2")

    def test_invalid_device_and_threshold_are_rejected(self):
        settings = self._settings(
            '[model_device]\nemotion2vec = "gpu0"\n'
            '[qwen_asr]\nvad_speech_noise_thres = 2\n'
        )
        with self.assertRaises(ValueError):
            _ = settings.model_device_emotion2vec
        with self.assertRaises(ValueError):
            _ = settings.qwen_vad_speech_noise_thres

    def test_local_model_pool_defaults_and_valid_counts(self):
        settings = self._settings('[model_device]\n')
        self.assertEqual(settings.local_model_pool_speech_campplus_instances, 1)
        self.assertEqual(settings.local_model_pool_emotion2vec_instances, 1)
        self.assertEqual(settings.instance_count, 1)

        settings = self._settings(
            '[model_device]\n'
            '[model_device.local_model_pool]\n'
            'speech_campplus = 2\n'
            'emotion2vec = 4\n'
        )
        self.assertEqual(settings.model_pool_speech_campplus_instances, 2)
        self.assertEqual(settings.model_pool_emotion2vec_instances, 4)

    def test_local_model_pool_counts_must_be_positive_integers(self):
        for value in (0, -1, 1.5, '"2"', 'true'):
            settings = self._settings(
                '[model_device.local_model_pool]\n'
                f'speech_campplus = {value}\n'
            )
            with self.assertRaises(ValueError):
                _ = settings.local_model_pool_speech_campplus_instances

    def test_model_device_rejects_multi_gpu_values(self):
        for value in ('["cuda:0", "cuda:1"]', '{gpu = "cuda:0"}'):
            settings = self._settings(f'[model_device]\nspeech_campplus = {value}\n')
            with self.assertRaises(ValueError):
                _ = settings.model_device_speech_campplus

    def test_local_model_pool_device_arrays_define_instances(self):
        settings = self._settings(
            '[model_device.local_model_pool]\n'
            'speech_campplus_devices = ["cuda:1", "cuda:2"]\n'
            'emotion2vec_devices = ["cuda:1"]\n'
        )
        self.assertEqual(settings.model_device_speech_campplus_devices, ["cuda:1", "cuda:2"])
        self.assertEqual(settings.local_model_pool_emotion2vec_devices, ["cuda:1"])
        self.assertEqual(settings.local_model_pool_speech_campplus_instances, 2)
        self.assertEqual(settings.local_model_pool_emotion2vec_instances, 1)
        self.assertEqual(settings.instance_count, 1)

    def test_local_model_pool_device_arrays_allow_empty_and_validate_values(self):
        settings = self._settings(
            '[model_device.local_model_pool]\n'
            'speech_campplus_devices = []\n'
            'emotion2vec_devices = []\n'
        )
        self.assertEqual(settings.model_device_speech_campplus_devices, [])
        self.assertEqual(settings.model_device_emotion2vec_devices, [])
        self.assertEqual(settings.local_model_pool_speech_campplus_instances, 0)
        self.assertEqual(settings.local_model_pool_emotion2vec_instances, 0)

        invalid_values = ('["gpu:0"]', '["cuda:x"]', '["cuda:0", ""]', "[0]")
        for value in invalid_values:
            settings = self._settings(
                '[model_device.local_model_pool]\n'
                f'speech_campplus_devices = {value}\n'
            )
            with self.assertRaises(ValueError):
                _ = settings.model_device_speech_campplus_devices

    def test_low_volume_defaults_and_boundaries(self):
        settings = self._settings('[qwen_asr]\n')
        self.assertEqual(settings.qwen_vad_decibel_thres, -100.0)
        self.assertEqual(settings.qwen_vad_snr_thres, -100.0)
        self.assertEqual(settings.qwen_vad_low_volume_rms_dbfs_thres, -100.0)
        self.assertFalse(settings.qwen_vad_low_volume_filter_enabled)
        settings = self._settings(
            '[qwen_asr]\n'
            'vad_decibel_thres = -80\n'
            'vad_snr_thres = 5\n'
            'vad_low_volume_rms_dbfs_thres = -42.5\n'
            'vad_low_volume_filter_enabled = true\n'
        )
        self.assertEqual(settings.vad_decibel_thres, -80.0)
        self.assertEqual(settings.vad_snr_thres, 5.0)
        self.assertEqual(settings.vad_low_volume_rms_dbfs_thres, -42.5)
        self.assertTrue(settings.vad_low_volume_filter_enabled)

    def test_vad_queue_and_call_timeout_defaults_are_independent(self):
        settings = self._settings('[qwen_asr]\n')
        self.assertEqual(settings.qwen_vad_queue_timeout_s, 120.0)
        self.assertEqual(settings.qwen_vad_call_timeout_s, 120.0)
        settings = self._settings(
            '[qwen_asr]\nvad_queue_timeout_s = 30\nvad_call_timeout_s = 600\n'
        )
        self.assertEqual(settings.qwen_vad_queue_timeout_s, 30.0)
        self.assertEqual(settings.qwen_vad_call_timeout_s, 600.0)

    def test_low_volume_invalid_values_are_rejected(self):
        for key, value in (("vad_decibel_thres", 1), ("vad_snr_thres", 101), ("vad_low_volume_rms_dbfs_thres", -101)):
            settings = self._settings(f'[qwen_asr]\n{key} = {value}\n')
            with self.assertRaises(ValueError):
                getattr(settings, f"qwen_{key}")


if __name__ == "__main__":
    unittest.main()
