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
        settings = self._settings("[log]\nlog_path = \"nested.log\"\n")
        self.assertEqual(settings.log_path, "nested.log")

    def test_legacy_model_device_is_ignored_with_warning(self):
        settings = self._settings('[model_device]\nemotion2vec = "cuda:1"\n')
        with self.assertLogs("core.config", level=logging.WARNING) as logs:
            self.assertEqual(settings.model_device_emotion2vec_devices, [])
        self.assertTrue(any("model_device_pool" in message for message in logs.output))

    def test_device_arrays_define_instances_and_order(self):
        settings = self._settings(
            '[qwen_asr]\nvad_device = ["cpu", "cuda:0"]\n'
            '[model_device_pool]\n'
            'speech_campplus_devices = ["cuda:1", "cuda:1", "cuda:2"]\n'
            'emotion2vec_devices = ["cuda:0"]\n'
        )
        self.assertEqual(settings.vad_device, ["cpu", "cuda:0"])
        self.assertEqual(settings.model_device_speech_campplus_devices, ["cuda:1", "cuda:1", "cuda:2"])
        self.assertEqual(settings.model_device_emotion2vec_devices, ["cuda:0"])
        self.assertEqual(settings.local_model_pool_speech_campplus_instances, 3)
        self.assertEqual(settings.local_model_pool_emotion2vec_instances, 1)

    def test_empty_arrays_disable_models(self):
        settings = self._settings(
            '[qwen_asr]\nvad_device = []\n'
            '[model_device_pool]\nspeech_campplus_devices = []\nemotion2vec_devices = []\n'
        )
        self.assertEqual(settings.vad_device, [])
        self.assertEqual(settings.model_device_speech_campplus_devices, [])
        self.assertEqual(settings.model_device_emotion2vec_devices, [])

    def test_invalid_device_arrays_are_rejected(self):
        settings = self._settings('[model_device_pool]\nspeech_campplus_devices = ["gpu:0"]\n')
        with self.assertRaises(ValueError):
            _ = settings.model_device_speech_campplus_devices
        settings = self._settings('[qwen_asr]\nvad_device = ["cuda:x"]\n')
        with self.assertRaises(ValueError):
            _ = settings.vad_device

    def test_low_volume_defaults_and_boundaries(self):
        settings = self._settings("[qwen_asr]\n")
        self.assertEqual(settings.qwen_vad_decibel_thres, -100.0)
        self.assertEqual(settings.qwen_vad_snr_thres, -100.0)
        self.assertEqual(settings.qwen_vad_low_volume_rms_dbfs_thres, -100.0)
        self.assertFalse(settings.qwen_vad_low_volume_filter_enabled)
        settings = self._settings(
            "[qwen_asr]\n"
            "vad_decibel_thres = -80\n"
            "vad_snr_thres = 5\n"
            "vad_low_volume_rms_dbfs_thres = -42.5\n"
            "vad_low_volume_filter_enabled = true\n"
        )
        self.assertEqual(settings.vad_decibel_thres, -80.0)
        self.assertEqual(settings.vad_snr_thres, 5.0)
        self.assertEqual(settings.vad_low_volume_rms_dbfs_thres, -42.5)
        self.assertTrue(settings.vad_low_volume_filter_enabled)


if __name__ == "__main__":
    unittest.main()
