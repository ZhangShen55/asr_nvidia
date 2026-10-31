import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.resource_diagnostics import (
    BYTES_PER_GIB,
    BYTES_PER_MIB,
    SafetyBaseline,
    SafetyPolicy,
    cache_snapshot,
    evaluate_safety,
    gpu_snapshot,
    host_memory_snapshot,
    status_snapshot,
    vmstat_snapshot,
    write_json_line,
)
from core.request_telemetry import record_request, reset_for_tests, snapshot


class ResourceDiagnosticsTests(unittest.TestCase):
    def test_vmstat_uses_proc_whitespace_format(self):
        with patch(
            "core.resource_diagnostics._read_whitespace_file",
            return_value={"pswpin": "10", "pswpout": "20"},
        ):
            result = vmstat_snapshot()
        self.assertTrue(result["available"])
        self.assertEqual(result["swap_in_pages"], 10)
        self.assertEqual(result["swap_out_pages"], 20)

    def test_missing_host_metrics_are_unavailable(self):
        with patch("core.resource_diagnostics._read_key_value_file", return_value={}):
            result = host_memory_snapshot()
        self.assertFalse(result["available"])
        self.assertIn("reason", result)

    def test_cache_snapshot_counts_only_service_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "seacraft-asr-a.wav").write_bytes(b"123")
            (root / "other-app.tmp").write_bytes(b"123456")
            result = cache_snapshot(root)
        self.assertTrue(result["available"])
        self.assertEqual(result["files"], 1)
        self.assertEqual(result["bytes"], 3)

    def test_gpu_unavailable_is_explicit(self):
        with patch(
            "core.resource_diagnostics.subprocess.run",
            side_effect=FileNotFoundError("nvidia-smi"),
        ):
            result = gpu_snapshot()
        self.assertFalse(result["available"])
        self.assertIn("reason", result)

    def test_safety_policy_detects_memory_swap_health_and_vad(self):
        baseline = SafetyBaseline(
            memory={"available": True, "total_bytes": 32 * BYTES_PER_GIB, "available_bytes": 16 * BYTES_PER_GIB, "swap_used_bytes": 0},
            vmstat={"available": True, "swap_in_pages": 100, "swap_out_pages": 100},
            gpu={"available": True, "gpus": [{"index": 0, "memory_total_mb": 1000, "memory_used_mb": 100}]},
            cache={"available": True, "disk_free_ratio": 0.8, "disk_total_bytes": 1000},
        )
        snapshot_data = {
            "host": {"available": True, "total_bytes": 32 * BYTES_PER_GIB, "available_bytes": 7 * BYTES_PER_GIB, "swap_used_bytes": 300 * BYTES_PER_MIB},
            "vmstat": {"available": True, "swap_in_pages": 101, "swap_out_pages": 101},
            "gpu": {"available": True, "gpus": [{"index": 0, "memory_total_mb": 1000, "memory_used_mb": 950}]},
            "cache": {"available": True, "disk_free_ratio": 0.05, "disk_total_bytes": 1000},
            "services": {"vad": [{"pid": 1}]},
            "health": {},
        }
        decision = evaluate_safety(
            snapshot_data,
            baseline,
            SafetyPolicy(consecutive_swap_samples=1, consecutive_health_failures=1),
            swap_samples=1,
            health_failures=1,
            expected_vad_workers=2,
        )
        self.assertTrue(decision.should_stop)
        self.assertGreaterEqual(len(decision.reasons), 5)

    def test_jsonl_flushes_and_is_replayable(self):
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as handle:
            write_json_line(handle, {"run_id": "run-1", "value": 1})
            handle.seek(0)
            self.assertEqual(json.loads(handle.readline())["run_id"], "run-1")

    def test_request_telemetry_records_phase_percentiles_and_is_bounded_snapshot(self):
        reset_for_tests()
        try:
            for index in range(2):
                record_request(
                    request_id=f"r-{index}",
                    task_id=f"t-{index}",
                    filename="a.wav",
                    started_at=0,
                    ended_at=(index + 1) / 100,
                    phases={"qwen_wait_ms": index + 1, "spk_ms": 4 + index},
                    valid_segments=3,
                )
            result = snapshot(include_events=False)
            self.assertEqual(result["events"], 2)
            self.assertEqual(result["duration_ms"]["p50_ms"], 15.0)
            self.assertEqual(result["phases_ms"]["qwen_wait_ms"]["p95_ms"], 1.95)
            self.assertNotIn("recent", result)
        finally:
            reset_for_tests()

    def test_status_snapshot_marks_health_as_skipped(self):
        result = status_snapshot(backend_pid=999999)
        self.assertIn("health", result)
        self.assertFalse(result["health"]["available"])


if __name__ == "__main__":
    unittest.main()
