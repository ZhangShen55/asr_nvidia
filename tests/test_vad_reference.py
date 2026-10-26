import unittest
import asyncio

from core.vad import VadSegment
from scripts.vad_reference import (
    _audio_diagnostics,
    assign_frame_segments,
    collect_vad_frame_diagnostics,
    evaluate_duration_fallback,
    evaluate_sound_rule_candidates,
    evaluate_threshold,
    generate_threshold_candidates,
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

    def test_audio_diagnostics_has_marked_details_and_percentiles(self):
        report = _audio_diagnostics(
            __import__("pathlib").Path("test/audio_file/理论计算机科学导论_时间am9.55_191803.wav"),
            parse_reference_annotations(),
        )
        self.assertEqual(report["marked_count"], 46)
        self.assertEqual(len(report["marked_intervals"]), 46)
        self.assertEqual(set(report["marked_rms_dbfs"]), {"p10", "p50", "p90", "p99"})
        self.assertGreater(report["audio_duration_s"], 3000)
        self.assertIn("context", report["marked_intervals"][0])

    def test_frame_mapping_and_candidate_search(self):
        mapped = assign_frame_segments(
            [{"frame_index": 0, "posterior_delta": 0.2, "log_energy": -5}],
            [VadSegment(0.0, 0.02)],
        )
        self.assertEqual(mapped[0]["segment_index"], 0)
        candidates = generate_threshold_candidates([-50, -45, -40], [-20, -15, -10], fine_step=1)
        self.assertTrue(candidates["all"])
        self.assertTrue(all(-80 <= value <= 0 for value in candidates["all"]))

    def test_frame_diagnostics_normalizes_model_field_names(self):
        report = collect_vad_frame_diagnostics(
            [{"score_delta": 0.3, "decibel": -12, "frame_state": "speech"}],
            [VadSegment(0.0, 0.01)],
        )
        self.assertEqual(report["frame_count"], 1)
        self.assertEqual(report["frames"][0]["segment_index"], 0)
        self.assertEqual(report["posterior_delta"]["p50"], 0.3)
        self.assertEqual(report["log_energy"]["p50"], -12.0)

    def test_threshold_evaluation_rejects_segment_hit(self):
        marked = [{"bg": 1.0, "ed": 1.2, "rms_dbfs": -50.0}]
        unmarked = [{"bg": 2.0, "ed": 3.0, "rms_dbfs": -10.0}]
        metrics = evaluate_threshold(
            -40.0,
            marked,
            unmarked,
            [VadSegment(1.0, 1.2)],
        )
        self.assertEqual(metrics["marked_suppression_rate"], 1.0)
        self.assertEqual(metrics["marked_segment_hit_rate"], 1.0)
        self.assertFalse(metrics["passes_targets"])

    def test_sound_rule_reports_no_solution_before_duration(self):
        records = [
            {"small_noise": True, "rms_dbfs": -30, "snr_db": 5, "low_energy_frame_ratio": 0.2},
            {"small_noise": False, "rms_dbfs": -30, "snr_db": 5, "low_energy_frame_ratio": 0.2},
        ]
        result = evaluate_sound_rule_candidates(records, rms_thresholds=(-30,), snr_thresholds=(5,))
        self.assertEqual(result["status"], "无可行声音组合规则")
        fallback = evaluate_duration_fallback(records, (0.5,), sound_rule_result=result)
        self.assertEqual(fallback["status"], "evaluated")
        self.assertTrue(fallback["duration_filter_used"])


if __name__ == "__main__":
    unittest.main()
