"""Only owned CPU processes/fakes; never start ASR, torch or an NPU service."""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from scripts.validate_npu_service import close_owned_service, compact_status, listening, run, shutdown_log_evidence

CPU_CLEANUP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


class FakeProcess:
    pid = 987654321
    returncode = None

    def __init__(self, timeout=False, signal_during_wait=False):
        self.timeout = timeout
        self.signal_during_wait = signal_during_wait
        self.waits = 0
        self.signals = []

    def poll(self):
        return self.returncode

    def send_signal(self, signum):
        self.signals.append(signum)

    def wait(self, timeout):
        self.waits += 1
        if self.signal_during_wait:
            # A first and repeated cancellation during finally must be ignored.
            for signum in CPU_CLEANUP_SIGNALS:
                os.kill(os.getpid(), signum)
        if self.waits <= int(self.timeout):
            raise subprocess.TimeoutExpired("owned-test", timeout)
        self.returncode = 0
        return 0


class NpuServiceCleanupTests(unittest.TestCase):
    def fake_cleanup(self, process, directory, **kwargs):
        with patch("scripts.validate_npu_service._owned_group_members", return_value=[]), \
                patch("scripts.validate_npu_service.listening", return_value=False), \
                patch("scripts.validate_npu_service.time.sleep"), \
                patch("scripts.validate_npu_service.os.killpg") as killpg:
            evidence = close_owned_service(process, 12345, Path(directory), observe_s=0, **kwargs)
        return evidence, killpg

    def test_first_and_repeated_signals_during_cleanup_cannot_interrupt_wait(self):
        previous = {sig: signal.getsignal(sig) for sig in CPU_CLEANUP_SIGNALS}
        def unexpected_signal(signum, _frame):
            raise RuntimeError(f"cleanup did not ignore {signum}")
        try:
            # Safe sentinels turn a future missing ignore into an assertion rather than killing pytest.
            for sig in CPU_CLEANUP_SIGNALS:
                signal.signal(sig, unexpected_signal)
            with tempfile.TemporaryDirectory() as directory:
                evidence, _ = self.fake_cleanup(FakeProcess(signal_during_wait=True), directory)
            self.assertFalse(evidence["failed"])
            for sig in CPU_CLEANUP_SIGNALS:
                self.assertEqual(signal.getsignal(sig), unexpected_signal)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def test_shutdown_timeout_is_failure_even_if_escalation_releases_everything(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence, killpg = self.fake_cleanup(FakeProcess(timeout=2), directory)
            timeout = json.loads((Path(directory) / "shutdown-timeout.json").read_text())
        self.assertTrue(evidence["shutdown_timeout"])
        self.assertTrue(evidence["failed"])
        self.assertFalse(evidence["listening"])
        self.assertFalse(evidence["remaining_children"])
        self.assertEqual(timeout["owned_pgid"], FakeProcess.pid)
        self.assertEqual([call.args for call in killpg.call_args_list], [(FakeProcess.pid, signal.SIGTERM), (FakeProcess.pid, signal.SIGKILL)])

    def exercise_own_cpu_server(self, term_exit_code):
        with socket.socket() as candidate:
            candidate.bind(("127.0.0.1", 0))
            port = candidate.getsockname()[1]
        program = """import signal,sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
port,code=map(int,sys.argv[1:3])
def stop(*_): raise SystemExit(code)
signal.signal(signal.SIGTERM,stop)
with HTTPServer(('127.0.0.1',port),SimpleHTTPRequestHandler) as server:
    server.serve_forever(poll_interval=.05)
"""
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.Popen([sys.executable, "-c", program, str(port), str(term_exit_code)],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
            try:
                deadline = time.monotonic() + 5
                while not listening(port) and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(listening(port))
                evidence = close_owned_service(process, port, Path(directory), graceful_timeout=2,
                                               terminate_timeout=1, kill_timeout=1, observe_s=0.02)
                self.assertFalse(listening(port))
                stderr = process.stderr.read().decode(errors="replace") if process.poll() is not None else "parent remains alive"
                self.assertEqual(process.poll(), term_exit_code, stderr)
                self.assertEqual(evidence["remaining_children"], [])
                return evidence
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=3)
                process.stderr.close()

    def test_own_cpu_server_releases_its_port(self):
        evidence = self.exercise_own_cpu_server(0)
        self.assertFalse(evidence["failed"])
        self.assertFalse(evidence["parent_exit_failed"])
        self.assertFalse(evidence["listening"])

    def test_nonzero_parent_exit_fails_even_after_own_cpu_server_releases_port(self):
        evidence = self.exercise_own_cpu_server(7)
        self.assertTrue(evidence["failed"])
        self.assertTrue(evidence["parent_exit_failed"])
        self.assertEqual(evidence["exit_code"], 7)
        self.assertFalse(evidence["listening"])

    def test_partial_start_failure_preserves_original_error_and_runs_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"samples": [
                {"sample_id": f"unit-audio-{index}", "extension": ".wav", "errors": [], "sha256": str(index), "media": {"duration_seconds": index + 100}}
                for index in range(8)]}))
            (root / "config.toml").write_text("segment_concurrency = 1\nrequest_concurrency = 1\n")
            args = argparse.Namespace(port=12345, output=root / "run", manifest=manifest,
                                      concurrency=1, instances=1, mode="short",
                                      models_root=None, app_root=str(root))
            with patch("scripts.validate_npu_service.ROOT", root), \
                    patch("scripts.validate_npu_service.listening", return_value=False), \
                    patch("scripts.validate_npu_service.subprocess.Popen", side_effect=OSError("synthetic partial startup")):
                with self.assertRaisesRegex(OSError, "synthetic partial startup"):
                    run(args)
            evidence = json.loads((args.output / "shutdown-after.json").read_text())
            self.assertIn("OSError: synthetic partial startup", evidence["original_failure"])
            self.assertFalse(evidence["failed"])
            self.assertIsNone(evidence["exit_code"])

    def test_compact_status_keeps_target_card_other_pids_and_process_metrics(self):
        status = {"audioConcurrency": {"active_total": 8}, "spkEmoConcurrency": {"active": 4},
                  "localModels": {"x": {"allocated_mb": None}}, "qwenModel": {"busy": 2}, "vadPool": {"failed_instances": 0},
                  "resourceDiagnostics": {"host": {"available": True, "available_bytes": 123}, "backend": {"pid": 10},
                      "npu": {"available": True, "raw_stdout": "large", "npus": [{"index": 0}, {"index": 7, "hbm_used_mb": 15000}],
                              "service_compute_apps": [{"index": 7, "pid": 10}], "other_compute_apps": [{"index": 0, "pid": 20}, {"index": 7, "pid": 30}]},
                      "processes": {"vad": {"count": 4}}, "cache": {"files": 1, "bytes": 2}, "requestTelemetry": {"recent": ["large"]}}}
        process = Mock()
        process.memory_info.return_value.rss = 321
        process.num_threads.return_value = 4
        process.num_fds.return_value = 10
        process.cpu_times.return_value.user = 12.5
        process.cpu_times.return_value.system = 3.25
        with patch("scripts.validate_npu_service.psutil.Process", return_value=process):
            result = compact_status(status, 10)
        self.assertEqual(result["spkEmoConcurrency"]["active"], 4)
        self.assertEqual(result["audioConcurrency"]["active_total"], 8)
        self.assertEqual(result["resourceDiagnostics"]["npu"]["npus"], [{"index": 7, "hbm_used_mb": 15000}])
        self.assertEqual(result["resourceDiagnostics"]["npu"]["other_compute_apps"], [{"index": 7, "pid": 30}])
        self.assertNotIn("raw_stdout", result["resourceDiagnostics"]["npu"])
        self.assertNotIn("requestTelemetry", result["resourceDiagnostics"])
        self.assertEqual(result["owned_process"]["rss_bytes"], 321)
        self.assertEqual(result["owned_process"]["num_fds"], 10)
        self.assertEqual(result["owned_process"]["cpu_user_s"], 12.5)
        self.assertEqual(result["owned_process"]["cpu_system_s"], 3.25)
        self.assertEqual(result["owned_process"]["cpu_scope"], "main_service_process_excludes_workers_and_shared_processes")


class ShutdownLogEvidenceTests(unittest.TestCase):
    @staticmethod
    def worker(**changes):
        record = {"pid": 701, "device": "npu:7", "graceful_elapsed_s": .01,
                  "forced": False, "actions": [], "exitcode": 0, "cleanup_complete": True}
        record.update(changes)
        return "2026-11-17 INFO core.vad_pool VAD_WORKER_SHUTDOWN " + json.dumps(record)

    def evidence(self, lines):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "service.log"
            path.write_text("\n".join(lines))
            return shutdown_log_evidence(path)

    def test_four_graceful_workers_keep_original_per_worker_audit_fields(self):
        lines = [self.worker(pid=pid, device=device) for pid, device in
                 ((701, "cpu"), (702, "cpu"), (703, "npu:7"), (704, "npu:7"))]
        evidence = self.evidence(lines + ["INFO: Application shutdown complete."])
        self.assertFalse(evidence["worker_shutdown_failed"])
        self.assertFalse(evidence["application_shutdown_failed"])
        self.assertEqual(len(evidence["worker_shutdown"]), 4)
        self.assertEqual([row["pid"] for row in evidence["worker_shutdown"]], [701, 702, 703, 704])
        self.assertEqual(evidence["worker_shutdown"][2]["device"], "npu:7")
        self.assertEqual(evidence["worker_shutdown"][2]["graceful_elapsed_s"], .01)

    def test_force_is_failure_even_if_cleanup_complete_and_worker_exits_zero(self):
        evidence = self.evidence([self.worker(forced=True, actions=["terminate"])])
        self.assertTrue(evidence["worker_shutdown_failed"])
        self.assertTrue(evidence["worker_shutdown"][0]["cleanup_complete"])
        self.assertEqual(evidence["worker_shutdown"][0]["exitcode"], 0)

    def test_nonzero_exit_incomplete_and_malformed_records_fail_closed(self):
        for line in (self.worker(exitcode=9), self.worker(cleanup_complete=False),
                     "INFO VAD_WORKER_SHUTDOWN {broken-json"):
            with self.subTest(line=line):
                evidence = self.evidence([line])
                self.assertTrue(evidence["worker_shutdown_failed"])
        self.assertIn("parse_error", self.evidence(["VAD_WORKER_SHUTDOWN {broken"])["worker_shutdown"][0])

    def test_later_graceful_record_cannot_hide_an_earlier_forced_record(self):
        evidence = self.evidence([self.worker(forced=True, actions=["kill"]), self.worker()])
        self.assertTrue(evidence["worker_shutdown_failed"])
        self.assertEqual(len(evidence["worker_shutdown"]), 2)

    def test_application_failure_markers_fail_even_with_graceful_worker_records(self):
        for marker in ("ERROR: Application shutdown failed. Exiting.", "服务收尾失败", "VAD 池关闭失败"):
            with self.subTest(marker=marker):
                evidence = self.evidence([self.worker(), marker])
                self.assertTrue(evidence["application_shutdown_failed"])
                self.assertFalse(evidence["worker_shutdown_failed"])


class HarnessShutdownVerdictTests(unittest.TestCase):
    def exercise_fake_run(self, root, shutdown_text, *, models_root=None, expected_failure=False):
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"samples": [
            {"sample_id": f"unit-audio-{index}", "extension": ".wav", "errors": [], "sha256": str(index),
             "media": {"duration_seconds": 100 + index}} for index in range(8)]}))
        original_config = ("log_path = \"original.log\"\nsegment_concurrency = 1\n"
                           "request_concurrency = 1\nlocal_instance_count = 1\n"
                           "vad_model_dir = \"model/vad\"\nspk_model_dir = \"model/spk\"\n"
                           "emotion_model_dir = \"model/emo\"\n")
        (root / "config.toml").write_text(original_config)
        args = argparse.Namespace(port=12345, output=root / "run", manifest=manifest,
                                  concurrency=1, instances=1, mode="short",
                                  models_root=models_root, app_root=str(root))
        process = FakeProcess()
        identity = Mock()
        identity.create_time.return_value = 123.
        identity.cmdline.return_value = ["fake-owned-main"]
        def spawn(*_, **kwargs):
            kwargs["stdout"].write(shutdown_text)
            kwargs["stdout"].flush()
            return process
        def closed(*_, **kwargs):
            for signum in CPU_CLEANUP_SIGNALS:
                self.assertEqual(signal.getsignal(signum), signal.SIG_IGN)
            process.returncode = 0
            return {"exit_code": 0, "listening": False, "remaining_children": [],
                    "observed_s": 0, "shutdown_timeout": False, "forced_child_cleanup": False,
                    "cleanup_errors": [], "parent_exit_failed": False,
                    "original_failure": kwargs.get("original_failure"), "failed": False}
        previous_handlers = {sig: signal.getsignal(sig) for sig in CPU_CLEANUP_SIGNALS}
        with patch("scripts.validate_npu_service.ROOT", root), \
                patch("scripts.validate_npu_service.listening", return_value=False), \
                patch("scripts.validate_npu_service.subprocess.Popen", side_effect=spawn) as popen, \
                patch("scripts.validate_npu_service.psutil.Process", return_value=identity), \
                patch("scripts.validate_npu_service.os.getpgid", return_value=process.pid), \
                patch("scripts.validate_npu_service.exercise", new=AsyncMock(return_value=None)), \
                patch("scripts.validate_npu_service.close_owned_service", side_effect=closed) as cleanup, \
                patch.dict(os.environ, {"PYTHONPATH": "/cann/compiler/python"}):
            if expected_failure:
                with self.assertRaisesRegex(RuntimeError, "关闭验收失败"):
                    run(args)
            else:
                self.assertIsNone(run(args))
        cleanup.assert_called_once()
        for signum, handler in previous_handlers.items():
            self.assertEqual(signal.getsignal(signum), handler)
        self.assertEqual((root / "config.toml").read_text(), original_config)
        evidence = json.loads((args.output / "shutdown-after.json").read_text())
        self.assertEqual(evidence["exit_code"], 0)
        self.assertFalse(evidence["listening"])
        self.assertEqual(evidence["remaining_children"], [])
        return evidence, popen.call_args, (args.output / "config.toml").read_text()

    def test_cli_default_namespace_and_unified_main_entry_preserve_cann_pythonpath(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence, launch, _ = self.exercise_fake_run(root, ShutdownLogEvidenceTests.worker())
        self.assertFalse(evidence["failed"])
        self.assertEqual(launch.kwargs["cwd"], root)
        self.assertEqual(launch.args[0][1], "main.py")
        self.assertNotIn("uvicorn", launch.args[0])
        self.assertEqual(launch.kwargs["env"]["PYTHONPATH"], str(root) + os.pathsep + "/cann/compiler/python")

    def test_explicit_models_root_only_rewrites_the_generated_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "read only original models"
            _, _, generated = self.exercise_fake_run(root, ShutdownLogEvidenceTests.worker(), models_root=str(models))
            for key, folder in (("vad_model_dir", "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
                                ("spk_model_dir", "speech_campplus_sv_zh_en_16k-common_advanced"),
                                ("emotion_model_dir", "emotion2vec_plus_large")):
                self.assertIn(key + " = " + json.dumps(str(models / folder)), generated)

    def test_forced_worker_makes_run_fail_even_when_parent_exits_zero_and_releases_port(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence, _, _ = self.exercise_fake_run(Path(directory),
                ShutdownLogEvidenceTests.worker(forced=True, actions=["terminate"]), expected_failure=True)
        self.assertTrue(evidence["failed"])
        self.assertTrue(evidence["worker_shutdown_failed"])
        self.assertFalse(evidence["application_shutdown_failed"])

    def test_application_failure_makes_run_fail_even_when_parent_exits_zero_and_releases_port(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence, _, _ = self.exercise_fake_run(Path(directory),
                ShutdownLogEvidenceTests.worker() + "\nERROR: Application shutdown failed. Exiting.\n", expected_failure=True)
        self.assertTrue(evidence["failed"])
        self.assertTrue(evidence["application_shutdown_failed"])
        self.assertFalse(evidence["worker_shutdown_failed"])


if __name__ == "__main__":
    unittest.main()
