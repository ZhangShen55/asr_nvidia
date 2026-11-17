import unittest
import asyncio
import tempfile
import wave
from pathlib import Path

import numpy as np

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
    def setUp(self):
        # 可追溯的合成样本：4 秒、16 kHz、单声道 PCM16 的确定性 220Hz 正弦。
        # 4 段中 2 段音量低；这里的手工标记只验证工具解析，并非真实课堂标注。
        directory = tempfile.TemporaryDirectory(prefix="vad-synthetic-")
        self.addCleanup(directory.cleanup)
        self.audio_path = Path(directory.name) / "synthetic-tones.wav"
        self.annotation_path = Path(directory.name) / "synthetic-annotations.json"
        audio = np.zeros(4 * 16000, dtype=np.float64)
        for start, end, amplitude in ((0.0, 0.5, 0.3), (1.0, 1.2, 0.003), (2.0, 2.5, 0.2), (3.0, 3.2, 0.005)):
            indices = np.arange(round(start * 16000), round(end * 16000))
            audio[indices] = amplitude * np.sin(2 * np.pi * 220 * indices / 16000)
        with wave.open(str(self.audio_path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(np.round(audio * 32767).astype("<i2").tobytes())
        # 包含人工文档支持的独立“细小声音”标记、说明行与缺失逗号修复。
        self.annotation_path.write_text("""{
    "source": "synthetic deterministic PCM fixture; no real speech labels",
    "segments": [
        {
            "segment_text": "synthetic loud A",
            "bg": 0.0,
            "ed": 0.5
        },
        {
细小声音
            "segment_text": "synthetic quiet A"
            "bg": 1.0,
            "ed": 1.2
        },
        {
转写错误应该是 synthetic note: not a speech ground truth
            "segment_text": "synthetic loud B",
            "bg": 2.0,
            "ed": 2.5
        },
        {
细小声音
            "segment_text": "synthetic quiet B",
            "bg": 3.0,
            "ed": 3.2
        }
    ]
}
""", encoding="utf-8")

    def test_synthetic_reference_has_explicit_manual_markers(self):
        segments = parse_reference_annotations(self.annotation_path)
        self.assertEqual(len(segments), 4)
        self.assertEqual([item["small_noise"] for item in segments], [False, True, False, True])
        self.assertEqual(segments[1]["segment_text"], "synthetic quiet A")
        self.assertEqual((segments[1]["bg"], segments[1]["ed"]), (1.0, 1.2))

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
                return [{"value": [[0, 500], [2000, 2500]]}]

            async def stop(self):
                return None

        report = asyncio.run(
            scan_thresholds(
                audio_path=self.audio_path,
                annotation_path=self.annotation_path,
                thresholds=(0.3, 0.6, 0.8),
                pool_factory=FakePool,
            )
        )
        self.assertEqual(report["annotation_count"], 2)
        self.assertEqual(report["diagnostics"]["marked_count"], 2)
        self.assertEqual(report["diagnostics"]["unmarked_count"], 2)
        self.assertEqual([item["threshold"] for item in report["thresholds"]], [0.3, 0.6, 0.8])
        for item in report["thresholds"]:
            self.assertIn("marked_suppression_rate", item)
            self.assertIn("unmarked_retention_rate", item)
            self.assertIn("filtered_segment_count", item)
            self.assertIn("qwen_request_count", item)
            self.assertIn("elapsed_s", item)
            self.assertEqual(item["filtered_segment_count"], 2)
            self.assertEqual(item["qwen_request_count"], 2)
            self.assertEqual(item["marked_suppression_rate"], 1.0)
            self.assertEqual(item["unmarked_retention_rate"], 1.0)

    def test_audio_diagnostics_has_marked_details_and_percentiles(self):
        report = _audio_diagnostics(
            self.audio_path,
            parse_reference_annotations(self.annotation_path),
        )
        self.assertEqual(report["marked_count"], 2)
        self.assertEqual(len(report["marked_intervals"]), 2)
        self.assertEqual(set(report["marked_rms_dbfs"]), {"p10", "p50", "p90", "p99"})
        self.assertEqual(report["audio_duration_s"], 4.0)
        self.assertEqual(report["sample_rate"], 16000)
        self.assertLess(report["marked_rms_dbfs"]["p90"], report["unmarked_rms_dbfs"]["p10"])
        marked = report["marked_intervals"][0]
        self.assertAlmostEqual(marked["duration_s"], 0.2)
        self.assertEqual(marked["context"], {"previous_bg": 0.0, "previous_ed": 0.5, "next_bg": 2.0, "next_ed": 2.5})

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
