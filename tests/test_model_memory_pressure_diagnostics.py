import json
import tempfile
import unittest
from pathlib import Path

from scripts.model_memory_pressure_diagnostics import _stage_plan, resource_analysis


class ModelMemoryPressureDiagnosticsTests(unittest.TestCase):
    def test_plan_skips_vad_only_and_records_matrix(self):
        plan = _stage_plan(16)
        self.assertFalse(any(stage["show_spk"] is False and stage["show_emotion"] is False for stage in plan))
        self.assertEqual([stage["audio_concurrency"] for stage in plan[-3:]], [2, 4, 8])
        self.assertTrue(all(stage["segment_concurrency"] == 16 for stage in plan))

    def test_resource_analysis_calculates_pss_cache_and_gpu_evidence(self):
        samples = [
            {
                "timestamp": "2026-10-31T00:00:00+00:00",
                "backend": {"memory": {"pss_bytes": 100, "rss_bytes": 200}},
                "cache": {"files": 2, "bytes": 10},
                "gpu": {"gpus": [{"utilization_gpu": 40}]},
                "status": {"localModels": {"speech_campplus": {"waiting": 1}}},
            },
            {
                "timestamp": "2026-10-31T00:00:01+00:00",
                "backend": {"memory": {"pss_bytes": 300, "rss_bytes": 500}},
                "cache": {"files": 1, "bytes": 4},
                "gpu": {"gpus": [{"utilization_gpu": 90}]},
                "status": {"localModels": {"speech_campplus": {"waiting": 2}}},
            },
        ]
        result = resource_analysis(samples)
        self.assertEqual(result["backend_pss_slope_bytes_per_s"], 200.0)
        self.assertEqual(result["cache_delta"], {"files": -1, "bytes": -6})
        self.assertEqual(result["gpu"]["utilization_max_percent"], 90)
        self.assertEqual(result["model_pool_waiting"]["speech_campplus"]["waiting"], 2)


if __name__ == "__main__":
    unittest.main()
