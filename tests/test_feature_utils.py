import unittest

from utils.feature_utils import calculate_speech_rate, count_content_words


class CountContentWordsTests(unittest.TestCase):
    def test_chinese_characters_counted_per_character(self):
        self.assertEqual(count_content_words("我有三个苹果"), 6)

    def test_french_words_with_accents_are_counted(self):
        self.assertEqual(count_content_words("café très chaud"), 3)

    def test_english_contractions_count_as_single_word(self):
        self.assertEqual(count_content_words("don't stop"), 2)

    def test_punctuation_is_ignored(self):
        self.assertEqual(count_content_words("hello, world!"), 2)

    def test_empty_text_counts_zero(self):
        self.assertEqual(count_content_words(""), 0)


class SpeechRateTests(unittest.TestCase):
    def test_french_segment_has_non_zero_speed(self):
        speed = calculate_speech_rate("café très chaud", 0.0, 60.0, 0.4)

        self.assertGreater(speed, 0)


if __name__ == "__main__":
    unittest.main()
