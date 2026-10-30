import io
import logging
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from core.logging import _ConsoleHttpSuccessFilter
from core.progress import AudioProgress


class ProgressLoggingTests(unittest.TestCase):
    def test_non_tty_progress_counts_success_and_failure(self):
        progress = AudioProgress(3, "lesson", stream=io.StringIO())
        progress.update()
        progress.update(failed=True)
        progress.update()
        self.assertEqual(progress.snapshot()["completed"], 3)
        self.assertEqual(progress.snapshot()["failed"], 1)
        self.assertFalse(progress.snapshot()["enabled"])
        progress.close()

    def test_only_success_http_info_is_filtered(self):
        rule = _ConsoleHttpSuccessFilter()
        success = logging.LogRecord(
            "httpx", logging.INFO, __file__, 1, 'HTTP Request: POST /v1 "HTTP/1.1 200 OK"', (), None
        )
        failure = logging.LogRecord(
            "httpx", logging.WARNING, __file__, 1, 'HTTP Request: POST /v1 "HTTP/1.1 500 Internal Server Error"', (), None
        )
        business = logging.LogRecord("api", logging.INFO, __file__, 1, "业务耗时", (), None)
        self.assertFalse(rule.filter(success))
        self.assertTrue(rule.filter(failure))
        self.assertTrue(rule.filter(business))

    def test_eight_audio_progress_counters_are_isolated(self):
        progresses = [AudioProgress(i + 1, f"audio-{i}", stream=io.StringIO()) for i in range(8)]

        def finish(progress):
            for index in range(progress.total):
                progress.update(failed=(index == progress.total - 1 and progress.total % 2 == 0))
            snapshot = progress.snapshot()
            progress.close()
            return snapshot

        with ThreadPoolExecutor(max_workers=8) as executor:
            snapshots = list(executor.map(finish, progresses))

        self.assertEqual([item["completed"] for item in snapshots], list(range(1, 9)))
        self.assertEqual([item["total"] for item in snapshots], list(range(1, 9)))
        self.assertEqual([item["failed"] for item in snapshots], [0, 1, 0, 1, 0, 1, 0, 1])

    def test_non_tty_has_no_ansi_and_missing_tqdm_degrades(self):
        stream = io.StringIO()
        with patch("core.progress._tqdm", None):
            progress = AudioProgress(1, "audio", stream=stream)
            progress.update(failed=True)
            progress.close()
        self.assertFalse(progress.enabled)
        self.assertNotRegex(stream.getvalue(), r"\x1b\[[0-9;?]*[ -/]*[@-~]")

    def test_console_filter_does_not_change_file_handler_records(self):
        records = []
        file_handler = logging.Handler()
        file_handler.emit = records.append
        console_handler = logging.Handler()
        console_handler.addFilter(_ConsoleHttpSuccessFilter())

        success = logging.LogRecord(
            "httpx", logging.INFO, __file__, 1,
            'HTTP Request: POST /v1 "HTTP/1.1 200 OK"', (), None,
        )
        console_handler.handle(success)
        file_handler.handle(success)

        self.assertEqual(len(records), 1)


if __name__ == "__main__":
    unittest.main()
