"""复读（幻觉循环）检测与修复的单元测试。"""

import unittest

from utils.repetition import (
    collapse_repetition,
    find_repeat_loop,
    max_reasonable_chars,
    repair_repetition,
)


class MaxReasonableCharsTests(unittest.TestCase):
    def test_zero_duration_returns_absolute_minimum(self):
        self.assertEqual(max_reasonable_chars(0), 120)

    def test_negative_duration_returns_absolute_minimum(self):
        self.assertEqual(max_reasonable_chars(-3), 120)

    def test_short_duration_uses_absolute_minimum(self):
        self.assertEqual(max_reasonable_chars(1), 120)

    def test_scales_with_duration(self):
        self.assertEqual(max_reasonable_chars(10), 400)
        self.assertEqual(max_reasonable_chars(60), 2400)


class FindRepeatLoopTests(unittest.TestCase):
    def test_empty_text_returns_none(self):
        self.assertIsNone(find_repeat_loop(""))

    def test_normal_text_returns_none(self):
        self.assertIsNone(find_repeat_loop("今天我们讲一下功率变换器的基本原理。"))

    def test_detects_long_loop(self):
        found = find_repeat_loop("最小S零，" + "这个" * 200)
        self.assertIsNotNone(found)
        unit, size, times = found
        self.assertGreaterEqual(size, 4)
        self.assertGreaterEqual(times, 3)

    def test_ignores_unit_shorter_than_min_unit(self):
        # 重复单元短于 min_unit=4 时不判定为循环
        self.assertIsNone(find_repeat_loop("好好好"))


class CollapseRepetitionTests(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(collapse_repetition(""), "")

    def test_collapses_repeated_unit(self):
        collapsed = collapse_repetition("最小S零，" + "这个，" * 300)
        self.assertLess(len(collapsed), 50)

    def test_collapses_three_times_repeated_unit(self):
        # collapse_repetition 本身是激进的：4 字单元连续 3 次即折叠。
        # 业务链路上由 repair_repetition 的长度判据兜底（见下方用例）。
        text = "温度载荷，温度载荷，温度载荷，"
        self.assertEqual(collapse_repetition(text), "温度载荷，")

    def test_long_english_sentence_unchanged(self):
        text = (
            "The transformer converts alternating current into another "
            "voltage with a different current and the same frequency."
        )
        self.assertEqual(collapse_repetition(text), text)


class RepairRepetitionTests(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(repair_repetition("", 5.0), ("", False))

    def test_short_normal_text_unchanged(self):
        self.assertEqual(
            repair_repetition("今天讲一下开关电源。", 5.0),
            ("今天讲一下开关电源。", False),
        )

    def test_normal_length_text_unchanged(self):
        # 长度正常时，即便存在叠词也原样返回
        text = "这个这个这个这个" * 3
        self.assertEqual(repair_repetition(text, 10.0), (text, False))

    def test_hallucination_is_collapsed(self):
        hallucination = "最小S零，" + "这个，" * 900
        fixed, changed = repair_repetition(hallucination, 6.0)
        self.assertTrue(changed)
        self.assertLess(len(fixed), 120)
        self.assertTrue(fixed.startswith("最小S零"))

    def test_overlong_without_repeat_structure_is_preserved(self):
        # 长度异常但没有复读结构：不截断、不改写，只告警
        text = "".join(chr(0x4E00 + i) for i in range(300))
        fixed, changed = repair_repetition(text, 5.0)
        self.assertFalse(changed)
        self.assertEqual(fixed, text)


if __name__ == "__main__":
    unittest.main()
