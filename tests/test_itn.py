import unittest
from unittest.mock import patch

from utils.itn import ItnConfig, normalize_segments


def _segments(*texts):
    return [{"segment_text": text, "bg": "0.00", "ed": "1.00"} for text in texts]


class NormalizeSegmentsTests(unittest.TestCase):
    def test_chinese_segments_are_normalized(self):
        segments = _segments("我有三个苹果和一百块")

        normalized, text, stats = normalize_segments(
            segments, "zh", ItnConfig(enabled=True)
        )

        self.assertEqual(normalized[0]["segment_text"], "我有三个苹果和100块")
        self.assertEqual(text, "我有三个苹果和100块")
        self.assertTrue(stats["enabled"])

    def test_unsupported_language_is_skipped(self):
        segments = _segments("bonjour le monde")

        normalized, text, stats = normalize_segments(
            segments, "fr", ItnConfig(enabled=True)
        )

        self.assertEqual(normalized[0]["segment_text"], "bonjour le monde")
        self.assertEqual(text, "bonjour le monde")
        self.assertEqual(stats["skipped"], "unsupported_language")

    def test_disabled_config_keeps_raw_text(self):
        segments = _segments("我有三个苹果和一百块")

        normalized, text, stats = normalize_segments(
            segments, "zh", ItnConfig(enabled=False)
        )

        self.assertEqual(normalized[0]["segment_text"], "我有三个苹果和一百块")
        self.assertEqual(text, "我有三个苹果和一百块")
        self.assertFalse(stats["enabled"])

    def test_missing_dependency_falls_back_to_raw_text(self):
        segments = _segments("我有三个苹果和一百块")
        config = ItnConfig(enabled=True, failure_mode="passthrough")

        with patch(
            "utils.itn._get_normalizer",
            side_effect=RuntimeError("pynini missing"),
        ):
            normalized, text, stats = normalize_segments(segments, "zh", config)

        self.assertEqual(normalized[0]["segment_text"], "我有三个苹果和一百块")
        self.assertEqual(text, "我有三个苹果和一百块")
        self.assertEqual(stats["failure_stage"], "initialize")


if __name__ == "__main__":
    unittest.main()
