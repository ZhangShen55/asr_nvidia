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
    collect_snapshot,
    npu_snapshot,
    parse_npu_smi,
    service_processes,
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

    def test_npu_unavailable_is_explicit(self):
        with patch(
            "core.resource_diagnostics.subprocess.run",
            side_effect=FileNotFoundError("npu-smi"),
        ):
            result = npu_snapshot()
        self.assertFalse(result["available"])
        self.assertIn("reason", result)

    def test_npu_table_uses_exact_hbm_and_process_rows(self):
        table = """
| NPU   Name                | Health        | Power(W) Temp(C) Hugepages-Usage(page) |
| Chip                      | Bus-Id        | AICore(%) Memory-Usage(MB) HBM-Usage(MB) |
| 0     910B3               | OK            | 92.6 35 0 / 0 |
| 0                         | 0000:C1:00.0  | 0 0 / 0 56623/ 65536 |
| 7     910B3               | OK            | 84.0 37 0 / 0 |
| 0                         | 0000:42:00.0  | 26 0 / 0 15362/ 65536 |
| NPU     Chip              | Process id    | Process name | Process memory(MB) |
| 7       0                 | 1601397       | uvicorn      | 11875 |
"""
        result = parse_npu_smi(table)
        self.assertTrue(result["available"])
        card = result["npus"][1]
        self.assertEqual(card["index"], 7)
        self.assertEqual(card["hbm_used_mb"], 15362)
        self.assertEqual(card["hbm_total_mb"], 65536)
        self.assertEqual(card["hbm_free_mb"], 50174)
        self.assertEqual(card["utilization_aicore"], 26)
        self.assertEqual(result["compute_apps"][0]["pid"], 1601397)

    def test_unparseable_hbm_stays_unknown(self):
        result = parse_npu_smi("| 7 910B3 | OK | 84 37 0 / 0 |\n| 0 | 0000:42:00.0 | N/A 0 / 0 N/A |")
        self.assertFalse(result["available"])
        self.assertIsNone(result["npus"][0]["hbm_used_mb"])
        self.assertIsNone(result["npus"][0]["hbm_total_mb"])

    def test_only_target_card_is_used_by_safety_gate(self):
        baseline = SafetyBaseline({}, {}, {}, {})
        data = {"npu": {"available": True, "npus": [
            {"index": 0, "hbm_total_mb": 1000, "hbm_used_mb": 999},
            {"index": 7, "hbm_total_mb": 1000, "hbm_used_mb": 200},
        ]}}
        decision = evaluate_safety(data, baseline)
        self.assertEqual(decision.level, "green")
        data["npu"]["npus"][1]["hbm_used_mb"] = 950
        self.assertTrue(evaluate_safety(data, baseline).should_stop)
        self.assertIn("NPU 7", evaluate_safety(data, baseline).reasons[0])

    def test_missing_target_hbm_is_not_reported_as_safe_zero(self):
        decision = evaluate_safety({"npu": {"available": False}}, SafetyBaseline({}, {}, {}, {}))
        self.assertEqual(decision.level, "yellow")
        self.assertIn("不可用", decision.warnings[0])

    def test_service_processes_only_attributes_target_descendants(self):
        rows = [
            {"pid": 10, "ppid": 1, "cmdline": "uvicorn main:app"},
            {"pid": 11, "ppid": 10, "cmdline": "asr-offline-vad"},
            {"pid": 20, "ppid": 1, "cmdline": "uvicorn unrelated:app"},
            {"pid": 21, "ppid": 20, "cmdline": "asr-offline-vad"},
            {"pid": 22, "ppid": 1, "cmdline": "vllm"},
        ]
        with patch("core.resource_diagnostics._process_listing", return_value=rows):
            result = service_processes(10)
        self.assertEqual([row["pid"] for row in result["backend"]], [10])
        self.assertEqual([row["pid"] for row in result["vad"]], [11])
        self.assertNotIn("qwen", result)

    def test_snapshot_separates_shared_occupancy_and_checks_only_backend_http(self):
        with patch("core.resource_diagnostics.service_processes", return_value={"backend": [{"pid": 10}], "vad": [{"pid": 11}]}), \
                patch("core.resource_diagnostics.process_snapshot", return_value={"available": True, "pid": 10}), \
                patch("core.resource_diagnostics.npu_snapshot", return_value={"available": True, "compute_apps": [{"pid": 10}, {"pid": 99}]}), \
                patch("core.resource_diagnostics.endpoint_health", return_value={"ok": True}) as health:
            result = collect_snapshot(backend_pid=10, include_process_tree=False)
        self.assertEqual([row["pid"] for row in result["npu"]["service_compute_apps"]], [10])
        self.assertIn("process_identity", result["npu"]["service_compute_apps"][0])
        self.assertEqual([row["pid"] for row in result["npu"]["other_compute_apps"]], [99])
        self.assertEqual(health.call_count, 1)
        self.assertNotIn("qwen", result["health"])

    def test_status_defaults_to_own_pid(self):
        with patch("core.resource_diagnostics.os.getpid", return_value=123), \
                patch("core.resource_diagnostics.collect_snapshot", return_value={}) as collect:
            status_snapshot()
        self.assertEqual(collect.call_args.kwargs["backend_pid"], 123)

    def test_safety_policy_detects_memory_swap_health_and_vad(self):
        baseline = SafetyBaseline(
            memory={"available": True, "total_bytes": 32 * BYTES_PER_GIB, "available_bytes": 16 * BYTES_PER_GIB, "swap_used_bytes": 0},
            vmstat={"available": True, "swap_in_pages": 100, "swap_out_pages": 100},
            npu={"available": True, "npus": [{"index": 7, "hbm_total_mb": 1000, "hbm_used_mb": 100}]},
            cache={"available": True, "disk_free_ratio": 0.8, "disk_total_bytes": 1000},
        )
        snapshot_data = {
            "host": {"available": True, "total_bytes": 32 * BYTES_PER_GIB, "available_bytes": 7 * BYTES_PER_GIB, "swap_used_bytes": 300 * BYTES_PER_MIB},
            "vmstat": {"available": True, "swap_in_pages": 101, "swap_out_pages": 101},
            "npu": {"available": True, "npus": [{"index": 7, "hbm_total_mb": 1000, "hbm_used_mb": 950}]},
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
