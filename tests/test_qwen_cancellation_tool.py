"""CPU fake exercises the same cancellation harness the root runs on NPU."""
import asyncio
import time

import pytest

from scripts.benchmark_qwen_candidates import Sample
from scripts.validate_qwen_cancellation import Observer, observed_runtime, run_validation


class CpuRuntime:
    def __init__(self, *args):
        self.loaded = False
        self.loading = False
        self.failure = None
    def ensure_loaded(self):
        if not self.loaded:
            self.loading = True
            time.sleep(.08)
            self.loaded = True
            self.loading = False
    def transcribe(self, payload, language=None):
        self.ensure_loaded()
        time.sleep(.06)
        return "language English<asr_text>actual fake worker finished"
    def snapshot(self):
        return {"loaded": self.loaded, "loading": self.loading, "failure": self.failure,
                "device": "cpu-fake", "dtype": "float32", "memory_allocated_mb": 0,
                "memory_reserved_mb": 0}
    def close(self):
        time.sleep(.01)
        self.loaded = False


def clip():
    # Fake runtime consumes bytes directly; real CLI validates/decodes the WAV.
    return Sample("fake-4.1s", "unused.wav", b"RIFFfake", 65600, None, "sourcehash", 0, 65600)


def execute(runtime=CpuRuntime, report=None, checkpoint=None, should_stop=None):
    return asyncio.run(run_validation(clip(), model_dir="unused", runtime_base=runtime,
                                       report=report, checkpoint=checkpoint,
                                       observation_timeout_s=2, request_timeout_s=2,
                                       should_stop=should_stop))


def test_harness_proves_cancel_serialization_and_close_with_cpu_threads():
    snapshots = []
    report = {}
    execute(report=report, checkpoint=lambda: snapshots.append(report["status"]))
    assert report["status"] == "passed"
    assert len(report["phases"]) == 3
    cold, serial, closing = report["phases"]
    assert cold["cancel_outcome"]["status"] == "cancelled"
    assert cold["before_finish"]["busy"] == 1
    assert cold["after_finish"]["busy"] == cold["after_finish"]["waiting"] == 0
    assert serial["during_wait"]["waiting"] == serial["during_wait"]["busy"] == 1
    assert serial["peak_actual_active"] == 1
    assert [row["request"] for row in serial["outputs"]] == ["first", "second"]
    assert all(row["parsed"]["has_speech"] for row in serial["outputs"])
    assert closing["cancel_outcome"]["status"] == "cancelled"
    assert closing["before_finish"]["busy"] == 1
    assert not closing["after_close"]["loaded"]
    observer = report["observer"]
    assert observer["actual_started"] == observer["actual_finished"] == 4
    assert observer["actual_active"] == 0 and observer["peak_actual_active"] == 1
    assert observer["load_started"] and observer["load_finished"]
    assert observer["close_started"] and observer["close_finished"]
    assert len(observer["outputs"]) == 4
    events = [event["event"] for event in observer["events"]]
    assert events.index("runtime_close_started") > max(i for i, event in enumerate(events) if event == "call_finished_after_runtime_sync")
    assert report["cleanup"]["status"] == "passed"
    assert report["cleanup"]["client"]["busy"] == report["cleanup"]["client"]["waiting"] == 0
    assert len(snapshots) >= 5


def test_failure_after_real_worker_finish_still_closes_and_records_error():
    class FailedGeneration(CpuRuntime):
        def transcribe(self, payload, language=None):
            self.ensure_loaded()
            time.sleep(.03)
            raise ValueError("controlled generation failure")
    report = {}
    with pytest.raises(AssertionError, match="inspectable raw"):
        execute(FailedGeneration, report)
    assert report["status"] == "failed"
    assert report["cleanup"]["status"] == "passed"
    assert report["observer"]["actual_active"] == 0
    finished = next(event for event in report["observer"]["events"] if event["event"] == "call_finished_after_runtime_sync")
    assert finished["error"]["error_type"] == "ValueError"
    assert report["observer"]["close_finished"]


def test_external_stop_closes_without_starting_new_owned_work():
    report = {}
    with pytest.raises(InterruptedError):
        execute(report=report, should_stop=lambda: True)
    assert report["cleanup"]["status"] == "passed"
    assert report["observer"]["actual_started"] == report["observer"]["actual_active"] == 0
    assert report["observer"]["close_finished"]


def test_observer_does_not_hide_close_failure():
    class BadClose(CpuRuntime):
        def close(self):
            raise RuntimeError("controlled close failure")
    observer = Observer()
    runtime = observed_runtime(BadClose, observer)("unused", "cpu", "fp32", 256)
    with pytest.raises(RuntimeError, match="close failure"):
        runtime.close()
    assert observer.close_started.is_set() and observer.close_finished.is_set()
