import unittest

from utils.text_postprocess import (
    is_single_interjection,
    remove_single_interjection_segments,
    replace_english_interjections,
)
from utils.qwen_response import build_language_ratio


class TextPostprocessTests(unittest.TestCase):
    def test_single_interjection_and_punctuation(self):
        for text in ("啊", "嗯。", "！哦！", " 呃 "):
            self.assertTrue(is_single_interjection(text))
        self.assertFalse(is_single_interjection("嗯嗯"))
        self.assertFalse(is_single_interjection("嗯，我认为"))
        self.assertFalse(is_single_interjection(""))

    def test_remove_preserves_metadata_and_valid_text(self):
        segments = [
            {"segment_text": "嗯。", "bg": "1.00", "role": "teacher"},
            {"segment_text": "嗯，我认为", "bg": "2.00", "role": "student"},
        ]
        result = remove_single_interjection_segments(segments)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["bg"], "2.00")
        self.assertEqual(result[0]["role"], "student")

    def test_english_replacement_is_boundary_limited(self):
        self.assertEqual(replace_english_interjections("I，嗯，think ah"), "I，um，think ah")
        self.assertEqual(replace_english_interjections("嗯 oh 呃"), "um oh uh")
        self.assertEqual(replace_english_interjections("嗯嗯和啊呀"), "嗯嗯和啊呀")
        self.assertEqual(replace_english_interjections("嗯，我认为"), "um，我认为")

    def test_language_ratio_aggregates_aliases(self):
        self.assertEqual(build_language_ratio([("English", "ab"), ("en", "cd")]), {"en": 100.0})


if __name__ == "__main__":
    unittest.main()
