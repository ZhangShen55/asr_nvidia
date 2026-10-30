import os
import tempfile
import unittest
from pathlib import Path

from api.routes.asr_common import cleanup_stale_service_cache, cleanup_temp_files
from utils.audio_utils import SERVICE_CACHE_DIR, write_audio_bytes_to_temp_file


class CacheLifecycleTests(unittest.TestCase):
    def tearDown(self):
        root = Path(SERVICE_CACHE_DIR)
        if root.is_dir():
            for path in root.glob("seacraft-asr-*"):
                if path.is_file():
                    path.unlink()

    def test_same_name_uploads_are_isolated_and_cleanup_is_idempotent(self):
        first = write_audio_bytes_to_temp_file(b"one", "lesson.wav", unique=True)
        second = write_audio_bytes_to_temp_file(b"two", "lesson.wav", unique=True)
        self.assertNotEqual(first, second)
        self.assertTrue(os.path.exists(first))
        self.assertTrue(os.path.exists(second))
        cleanup_temp_files([first])
        cleanup_temp_files([first])
        self.assertFalse(os.path.exists(first))
        self.assertTrue(os.path.exists(second))
        cleanup_temp_files([second])
        self.assertFalse(os.path.exists(second))

    def test_startup_scan_only_removes_service_prefix(self):
        root = Path(SERVICE_CACHE_DIR)
        root.mkdir(parents=True, exist_ok=True)
        owned = root / "seacraft-asr-stale.wav"
        other = root / "other-service.wav"
        owned.write_bytes(b"owned")
        other.write_bytes(b"other")
        cleanup_stale_service_cache()
        self.assertFalse(owned.exists())
        self.assertTrue(other.exists())
        other.unlink()


if __name__ == "__main__":
    unittest.main()
