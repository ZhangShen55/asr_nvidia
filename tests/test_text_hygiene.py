"""标点卫生规则与语种占比简写的单元测试。"""

import unittest

from utils.itn import is_punctuation_only, sanitize_segments, strip_leading_punctuation
from utils.qwen_response import abbreviate_language, build_language_ratio


def _segments(*texts):
    return [{"segment_text": text, "bg": "0.00", "ed": "1.00"} for text in texts]


class PunctuationOnlyTests(unittest.TestCase):
    def test_single_punctuation_is_punctuation_only(self):
        self.assertTrue(is_punctuation_only("。"))
        self.assertTrue(is_punctuation_only("！"))
        self.assertTrue(is_punctuation_only("，、"))

    def test_text_with_word_is_not_punctuation_only(self):
        self.assertFalse(is_punctuation_only("你好"))
        self.assertFalse(is_punctuation_only("。你好"))
        self.assertFalse(is_punctuation_only(""))


class LeadingPunctuationTests(unittest.TestCase):
    def test_single_leading_punctuation_is_removed(self):
        self.assertEqual(strip_leading_punctuation("，就。"), "就。")

    def test_consecutive_leading_punctuation_is_fully_removed(self):
        self.assertEqual(strip_leading_punctuation("，，就。"), "就。")
        self.assertEqual(strip_leading_punctuation("，。！？好"), "好")

    def test_text_without_leading_punctuation_is_unchanged(self):
        self.assertEqual(strip_leading_punctuation("你好，世界"), "你好，世界")


class SanitizeSegmentsTests(unittest.TestCase):
    def test_pure_punctuation_segment_is_dropped(self):
        segments, text = sanitize_segments(_segments("你好", "。", "再见"))
        self.assertEqual(len(segments), 2)
        self.assertEqual([s["segment_text"] for s in segments], ["你好", "再见"])
        self.assertEqual(text, "你好 再见")

    def test_leading_punctuation_is_stripped(self):
        segments, text = sanitize_segments(_segments("，，就。"))
        self.assertEqual(segments[0]["segment_text"], "就。")
        self.assertEqual(text, "就。")

    def test_normal_text_is_untouched(self):
        segments, text = sanitize_segments(_segments("我有三个苹果和一百块"))
        self.assertEqual(segments[0]["segment_text"], "我有三个苹果和一百块")
        self.assertEqual(text, "我有三个苹果和一百块")

    def test_other_fields_are_preserved(self):
        segments, _ = sanitize_segments(
            [{"segment_text": "，你好", "bg": "1.00", "ed": "2.00", "role": "spk_0"}]
        )
        self.assertEqual(segments[0]["bg"], "1.00")
        self.assertEqual(segments[0]["role"], "spk_0")


class LanguageAbbreviationTests(unittest.TestCase):
    def test_known_languages_are_abbreviated(self):
        self.assertEqual(abbreviate_language("Chinese"), "zh")
        self.assertEqual(abbreviate_language("English"), "en")
        self.assertEqual(abbreviate_language("Cantonese"), "yue")

    def test_unknown_language_is_passed_through(self):
        self.assertEqual(abbreviate_language("Klingon"), "Klingon")

    def test_ratio_uses_abbreviated_keys(self):
        ratio = build_language_ratio([("Chinese", "中文字符"), ("English", "ab")])
        self.assertEqual(set(ratio.keys()), {"zh", "en"})
        self.assertAlmostEqual(sum(ratio.values()), 100.0, places=1)

    def test_ratio_single_language_is_100(self):
        ratio = build_language_ratio([("Chinese", "中文字符")])
        self.assertEqual(ratio, {"zh": 100.0})

    def test_ratio_without_valid_text_is_empty(self):
        self.assertEqual(build_language_ratio([]), {})
        self.assertEqual(build_language_ratio([("Chinese", "   ")]), {})

    def test_unknown_language_key_is_kept(self):
        ratio = build_language_ratio([("Klingon", "abc")])
        self.assertEqual(ratio, {"Klingon": 100.0})


if __name__ == "__main__":
    unittest.main()
