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


if __name__ == "__main__":
    unittest.main()
