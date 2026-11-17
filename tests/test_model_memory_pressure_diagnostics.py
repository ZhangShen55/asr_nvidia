import json
import tempfile
import unittest
from pathlib import Path

from scripts.model_memory_pressure_diagnostics import _stage_plan, resource_analysis, target_error, display_name, post_audio, Sampler
from core.resource_diagnostics import SafetyPolicy
from unittest.mock import patch, Mock
import os


class ModelMemoryPressureDiagnosticsTests(unittest.TestCase):
    def test_plan_skips_vad_only_and_records_matrix(self):
        plan = _stage_plan(16)
        self.assertFalse(any(stage["show_spk"] is False and stage["show_emotion"] is False for stage in plan))
        self.assertEqual([stage["audio_concurrency"] for stage in plan[-3:]], [2, 4, 8])
        self.assertTrue(all(stage["segment_concurrency"] == 16 for stage in plan))

    def test_resource_analysis_calculates_pss_cache_and_npu_evidence(self):
        samples = [
            {
                "timestamp": "2026-10-31T00:00:00+00:00",
                "backend": {"memory": {"pss_bytes": 100, "rss_bytes": 200}},
                "cache": {"files": 2, "bytes": 10},
                "npu": {"npus": [{"index": 7, "utilization_aicore": 40}, {"index": 0, "utilization_aicore": 100}]},
                "status": {"localModels": {"speech_campplus": {"waiting": 1}}},
            },
            {
                "timestamp": "2026-10-31T00:00:01+00:00",
                "backend": {"memory": {"pss_bytes": 300, "rss_bytes": 500}},
                "cache": {"files": 1, "bytes": 4},
                "npu": {"npus": [{"index": 7, "utilization_aicore": 90}]},
                "status": {"localModels": {"speech_campplus": {"waiting": 2}}},
            },
        ]
        result = resource_analysis(samples)
        self.assertEqual(result["backend_pss_slope_bytes_per_s"], 200.0)
        self.assertEqual(result["cache_delta"], {"files": -1, "bytes": -6})
        self.assertEqual(result["npu"]["utilization_aicore_max_percent"], 90)
        self.assertEqual(result["model_pool_waiting"]["speech_campplus"]["waiting"], 2)

    def test_target_pid_must_match_status(self):
        self.assertIsNone(target_error({"resourceDiagnostics": {"backend": {"pid": 123}}}, 123))
        self.assertIsNotNone(target_error({"resourceDiagnostics": {"backend": {"pid": 456}}}, 123))
        self.assertIsNotNone(target_error({}, 123))

    def test_legacy_filename_is_displayable_and_upload_uses_ascii(self):
        raw_name = "音频.wav".encode("gb18030")
        name = os.fsdecode(raw_name)
        self.assertEqual(display_name(Path(name)), "音频.wav")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / name
            path.write_bytes(b"audio")
            response = Mock(status_code=200)
            response.json.return_value = {"code": 0, "segments": []}
            with patch("scripts.model_memory_pressure_diagnostics.requests.post", return_value=response) as post:
                result = post_audio("http://127.0.0.1:18085", path, show_spk=True, show_emotion=True, timeout_s=1)
            self.assertTrue(result["ok"])
            upload_name = post.call_args.kwargs["files"]["audioFile"][0]
            self.assertTrue(upload_name.isascii())
            self.assertEqual(result["file"], "音频.wav")

    def test_missing_hbm_baseline_prevents_pressure_requests(self):
        sample = {"host": {"available": True}, "npu": {"available": False}, "health": {"backend": {"ok": True}}}
        with tempfile.TemporaryDirectory() as directory, \
                patch("scripts.model_memory_pressure_diagnostics.get_json", return_value=({}, None)), \
                patch("scripts.model_memory_pressure_diagnostics.collect_snapshot", return_value=sample):
            sampler = Sampler(backend_pid=123, base_url="http://127.0.0.1:18085", cache_dir=Path(directory), out_file=Path(directory) / "samples", policy=SafetyPolicy())
            sampler.start()
            self.assertTrue(sampler.stop_event.is_set())
            self.assertIsNone(sampler.thread)
            self.assertTrue(sampler.red_events)


if __name__ == "__main__":
    unittest.main()
