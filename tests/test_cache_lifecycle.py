import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from api.routes.asr_common import cleanup_stale_service_cache, cleanup_temp_files
from utils.audio_utils import write_audio_bytes_to_temp_file


class CacheLifecycleTests(unittest.TestCase):
    def setUp(self):
        # 即使外部启动器没有设置TMPDIR，单测也只操作本用例的私有缓存。
        directory = tempfile.TemporaryDirectory(prefix="asr-cache-unit-")
        self.addCleanup(directory.cleanup)
        self.cacheRoot = Path(directory.name)
        for target in ("utils.audio_utils.SERVICE_CACHE_DIR",
                       "api.routes.asr_common.SERVICE_CACHE_DIR"):
            patcher = patch(target, str(self.cacheRoot))
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_same_name_uploads_are_isolated_and_cleanup_is_idempotent(self):
        first = write_audio_bytes_to_temp_file(b"one", "lesson.wav", unique=True)
        second = write_audio_bytes_to_temp_file(b"two", "lesson.wav", unique=True)
        self.assertNotEqual(first, second)
        self.assertEqual(Path(first).parent, self.cacheRoot)
        self.assertEqual(Path(second).parent, self.cacheRoot)
        self.assertTrue(os.path.exists(first))
        self.assertTrue(os.path.exists(second))
        cleanup_temp_files([first])
        cleanup_temp_files([first])
        self.assertFalse(os.path.exists(first))
        self.assertTrue(os.path.exists(second))
        cleanup_temp_files([second])
        self.assertFalse(os.path.exists(second))

    def test_startup_scan_only_removes_service_prefix(self):
        root = self.cacheRoot
        root.mkdir(parents=True, exist_ok=True)
        owned = root / "seacraft-asr-stale.wav"
        other = root / "other-service.wav"
        owned.write_bytes(b"owned")
        other.write_bytes(b"other")
        cleanup_stale_service_cache()
        self.assertFalse(owned.exists())
        self.assertTrue(other.exists())
        other.unlink()


    def test_private_cleanup_preserves_other_namespace_sentinel(self):
        # PRIVATE模拟的另一个服务；不在真实/tmp/seacraft_asr创建或删除任何文件。
        with tempfile.TemporaryDirectory(prefix="asr-foreign-cache-") as other_dir:
            sentinel = Path(other_dir) / "seacraft-asr-v3-inflight.wav"
            sentinel.write_bytes(b"foreign in-flight request")
            owned = self.cacheRoot / "seacraft-asr-own.wav"
            owned.write_bytes(b"own stale request")
            cleanup_stale_service_cache()
            self.assertFalse(owned.exists())
            self.assertEqual(sentinel.read_bytes(), b"foreign in-flight request")


if __name__ == "__main__":
    unittest.main()
