import unittest
import asyncio

from core.vad import VadSegment
from scripts.vad_reference import (
    interval_hit,
    overlap_ratio,
    parse_reference_annotations,
    scan_thresholds,
)


class VadReferenceTests(unittest.TestCase):
    def test_reference_fixture_has_expected_markers(self):
        segments = parse_reference_annotations()
        self.assertEqual(len(segments), 658)
        self.assertEqual(sum(item["small_noise"] for item in segments), 46)

    def test_interval_hit_uses_fifty_percent_marked_overlap(self):
        marked = (10.0, 12.0)
        self.assertTrue(interval_hit([VadSegment(10.0, 12.0)], marked))
        self.assertTrue(interval_hit([VadSegment(11.0, 12.5)], marked))
        self.assertFalse(interval_hit([VadSegment(11.1, 12.0)], marked))
        self.assertEqual(overlap_ratio(VadSegment(11.0, 12.0), marked), 0.5)

    def test_fake_threshold_scan_report_has_contract_fields(self):
        class FakePool:
            def __init__(self, vad_kwargs):
                self.threshold = vad_kwargs["speech_noise_thres"]

            async def detect(self, _path):
                return [{"value": [[9.81 * 1000, 10.42 * 1000]]}]

            async def stop(self):
                return None

        report = asyncio.run(
            scan_thresholds(
                thresholds=(0.3, 0.6, 0.8),
                pool_factory=FakePool,
            )
        )
        self.assertEqual(report["annotation_count"], 46)
        self.assertEqual([item["threshold"] for item in report["thresholds"]], [0.3, 0.6, 0.8])
        for item in report["thresholds"]:
            self.assertIn("marked_suppression_rate", item)
            self.assertIn("unmarked_retention_rate", item)
            self.assertIn("filtered_segment_count", item)
            self.assertIn("qwen_request_count", item)
            self.assertIn("elapsed_s", item)


if __name__ == "__main__":
    unittest.main()
