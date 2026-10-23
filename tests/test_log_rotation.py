"""日志按天 + 按大小双轮转与过期清理的单元测试。"""

import logging
import os
import tempfile
import time
import unittest
from datetime import date, timedelta

from core.logging import RotatingSizeAndDayFileHandler


class LogRotationTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.base = os.path.join(self._dir.name, "asr_service.log")
        self.addCleanup(self._dir.cleanup)

    def _make_handler(self, **kwargs):
        handler = RotatingSizeAndDayFileHandler(self.base, **kwargs)
        self.addCleanup(handler.close)
        return handler

    def _emit(self, handler, message):
        handler.emit(
            logging.LogRecord("test", logging.INFO, __file__, 1, message, None, None)
        )

    def test_size_limit_rotates_to_indexed_file(self):
        handler = self._make_handler(max_size_mb=1, max_days=7)
        payload = "x" * 4096
        for _ in range(300):
            self._emit(handler, payload)

        self.assertTrue(os.path.exists(f"{self.base}.1"), "应切出 .1 文件")
        self.assertLessEqual(
            os.path.getsize(self.base), 1024 * 1024 + 4096, "当前文件不应明显超过上限"
        )

    def test_day_change_rotates_to_dated_file(self):
        handler = self._make_handler(max_size_mb=100, max_days=7)
        self._emit(handler, "第一条")
        yesterday = date.today() - timedelta(days=1)
        handler._current_date = yesterday
        self._emit(handler, "第二条")

        self.assertTrue(
            os.path.exists(f"{self.base}.{yesterday.isoformat()}"),
            "应切出带日期的文件",
        )
        with open(self.base, "r", encoding="utf-8") as handle:
            self.assertIn("第二条", handle.read())

    def test_expired_files_are_removed_on_startup(self):
        expired = f"{self.base}.2020-01-01"
        with open(expired, "w", encoding="utf-8") as handle:
            handle.write("旧日志")
        old = time.time() - 30 * 24 * 3600
        os.utime(expired, (old, old))

        self._make_handler(max_size_mb=100, max_days=7)

        self.assertFalse(os.path.exists(expired), "超过保留期的日志应被删除")

    def test_fresh_files_are_kept(self):
        keep = f"{self.base}.2020-01-01"
        with open(keep, "w", encoding="utf-8") as handle:
            handle.write("较新的日志")

        self._make_handler(max_size_mb=100, max_days=7)

        self.assertTrue(os.path.exists(keep), "未超期的日志不应被删除")

    def test_default_limits(self):
        handler = self._make_handler()
        self.assertEqual(handler.max_days, 7)
        self.assertEqual(handler.max_size_bytes, 100 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
