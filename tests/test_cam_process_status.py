"""CAM process attribution uses cached model identities and OS-only samples."""
import copy
from contextlib import ExitStack
from datetime import datetime
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core import status
from core import resource_diagnostics as resources


def worker(pid=200, *, reaped=False, alive=True):
    return {
        "execution_mode": "process", "pool_instance_index": 0, "configured_device": "npu:7",
        "worker_identity": {"pid": pid, "create_time": 100., "start_ticks": "50", "sid": 100},
        "worker_alive": alive, "resource_reaped": reaped, "loaded": not reaped,
        "normal_exit_ok": reaped, "forced_cleanup": False,
        "allocator": {"worker_pid": pid, "worker_rss_mb": None, "memory_allocated_bytes": None,
                      "peak_scope": "last child RPC only; independent of parent and device totals"},
        "last_done": {"rpc": 3, "allocator": {"max_memory_allocated_bytes": 500}},
        "last_audio_bytes": 16000, "last_request_wire_bytes": 16200,
        "last_response_wire_bytes": 900, "parent_rss_mb": 100,
        "IPC_scope": "extra parent/worker copies; not the pipeline logical PCM budget",
    }


def pool(*workers):
    return {"speech_campplus": {"execution_mode": "process", "model_workers": list(workers)}}


def app_row(pid, *, ticks="50", created=100., sid=100, memory=157):
    return {"pid": pid, "memory_mb": memory,
            "process_identity": {"pid": pid, "start_ticks": ticks, "create_time": created, "sid": sid},
            "process_resource": {"available": True, "rss_bytes": 1234, "memory": {"pss_bytes": 900}}}


def test_cam_qwen_parent_external_and_unknown_are_separate_without_mutating_cache():
    cam = worker()
    local = pool(cam)
    qwen = {"model_workers": [{"worker_pid": 300, "worker_create_time": 100., "worker_start_ticks": "50"}]}
    raw = {"npu": {"compute_apps": [app_row(100, memory=0), app_row(200), app_row(300),
                                    app_row(1601397), {"pid": 999}]}}
    original = copy.deepcopy((raw, local, qwen))
    with patch.object(status.os, "getpid", return_value=100):
        result = status._classify_npu_apps(raw, qwen, local)
    assert [r["pid"] for r in result["backend_compute_apps"]] == [100]
    assert [r["pid"] for r in result["qwen_compute_apps"]] == [300]
    assert [r["pid"] for r in result["cam_compute_apps"]] == [200]
    assert result["cam_compute_apps"][0]["memory_mb"] == 157
    assert result["cam_compute_apps"][0]["process_resource"]["rss_bytes"] == 1234
    assert [r["pid"] for r in result["other_compute_apps"]] == [1601397]
    assert [r["pid"] for r in result["unverified_compute_apps"]] == [999]
    assert (raw, local, qwen) == original


@pytest.mark.parametrize("change", ["ticks", "ctime", "sid", "pid", "missing", "reaped", "dead"])
def test_reused_disappeared_or_retired_cam_is_unverified_even_if_in_cached_service_tree(change):
    cam, row = worker(), app_row(200)
    if change == "ticks":
        row["process_identity"]["start_ticks"] = "51"
    elif change == "ctime":
        row["process_identity"]["create_time"] = 101.
    elif change == "sid":
        row["process_identity"]["sid"] = 101
    elif change == "pid":
        row["process_identity"]["pid"] = 201
    elif change == "missing":
        row["process_identity"] = {"available": False}
    elif change == "reaped":
        cam["resource_reaped"] = True
    else:
        cam["worker_alive"] = False
    raw = {"services": {"cam": [{"pid": 200}]}, "npu": {"compute_apps": [row]}}
    with patch.object(status.os, "getpid", return_value=100):
        result = status._classify_npu_apps(raw, {}, pool(cam))
    assert not result["cam_compute_apps"]
    assert not result["other_compute_apps"]
    assert not result["service_compute_apps"]
    assert result["unverified_compute_apps"][0]["service_role"] == "unverified_cam_pid"


def test_ready_identity_and_cached_allocator_are_assembled_once_on_event_loop_without_rpc():
    thread = threading.get_ident()
    cam = worker(alive=None)
    cam["loaded"] = False  # Birth registered before publication/READY still remains observable.
    local = pool(cam)
    local["speech_campplus"]["instance_details"] = [{"memory": copy.deepcopy(cam)}]
    resource = {"available": True, "sampling": {"age_s": 5., "in_progress": True},
                "backend": {"npu_allocator": {"allocated_bytes": 0}}, "npu": {"compute_apps": []}}

    def local_snapshot():
        assert threading.get_ident() == thread
        return local

    stats = {"total_failed_tasks": 0}
    with ExitStack() as stack:
        cached = stack.enter_context(patch.object(status, "get_local_model_status", side_effect=local_snapshot))
        stack.enter_context(patch.object(status, "get_qwen_status", return_value={"model_workers": []}))
        stack.enter_context(patch.object(status, "get_slot_manager", return_value=SimpleNamespace(snapshot=lambda: {})))
        stack.enter_context(patch.object(status, "get_vad_pool", return_value=SimpleNamespace(stats=lambda: {})))
        stack.enter_context(patch.object(status, "get_live_allocator_status", return_value={"allocated_bytes": 0}))
        stack.enter_context(patch.object(status, "request_telemetry_snapshot", return_value={}))
        stack.enter_context(patch.object(status, "pending_snapshot", return_value={}))
        stack.enter_context(patch.object(status, "status_snapshot", side_effect=AssertionError("no synchronous OS sampling")))
        stack.enter_context(patch.object(status, "settings", SimpleNamespace(
            model_device_speech_campplus_execution_mode="process", deployment_profile="npu-910b",
            config_path="fixture", qwen_local_device="npu:7", qwen_asr_enabled=True,
            vad_device=["cpu"], model_device_speech_campplus_devices=["npu:7"],
            model_device_emotion2vec_devices=["npu:7"])))
        payload = status.build_status_payload(stats, datetime.utcnow(), resource_snapshot=resource)
    cached.assert_called_once_with()
    diagnostics = payload["resourceDiagnostics"]
    assert payload["localModels"] is local
    assert diagnostics["cam_model_workers"][0]["allocator"]["memory_allocated_bytes"] is None
    assert diagnostics["backend"]["npu_allocator"]["allocated_bytes"] == 0
    assert diagnostics["cam_model_workers"][0]["last_done"]["allocator"]["max_memory_allocated_bytes"] == 500
    assert diagnostics["sampling"] == {"age_s": 5., "in_progress": True}
    assert diagnostics["cam_model_workers"][0]["last_request_wire_bytes"] == 16200
    process = diagnostics["processes"]["cam_model_worker"]
    assert process["count"] is None and process["unknown_liveness_count"] == 1
    assert process["registered_not_reaped_count"] == 1


def test_unrequested_cam_has_no_worker_and_does_not_invent_resource_zeros():
    with patch.object(status.os, "getpid", return_value=100):
        result = status._classify_npu_apps({"npu": {"compute_apps": [app_row(1601397)]}}, {}, pool())
    assert not result["cam_compute_apps"]
    assert [row["pid"] for row in result["other_compute_apps"]] == [1601397]
    assert status._cam_model_workers(pool()) == []


@pytest.mark.parametrize("birth_pending", [False, True])
def test_real_lazy_cam_and_pool_preserve_prebirth_cache_without_spawning(tmp_path, birth_pending):
    from core.cam_process import CamProcessRunner
    from core import models
    import multiprocessing.process

    local_entry = copy.deepcopy(models.local_model_status["speech_campplus"])
    runner = None
    with ExitStack() as stack:
        start = stack.enter_context(patch.object(
            multiprocessing.process.BaseProcess, "start", side_effect=AssertionError("status spawned a child")))
        load = stack.enter_context(patch.object(
            CamProcessRunner, "ensure_loaded", side_effect=AssertionError("status loaded a model")))
        cfg = SimpleNamespace(model_device_speech_campplus_execution_mode="process",
                              deployment_profile="npu-910b", config_path="fixture",
                              qwen_local_device="npu:7", qwen_asr_enabled=True, vad_device=["cpu"],
                              model_device_speech_campplus_devices=["npu:7"],
                              model_device_emotion2vec_devices=["npu:7"])
        stack.enter_context(patch.object(models, "settings", cfg))
        stack.enter_context(patch.object(status, "settings", cfg))
        try:
            runner = CamProcessRunner(tmp_path, "npu:7")
            model_pool = models.ModelPool("speech_campplus", str(tmp_path), devices_getter=lambda: ["npu:7"])
            # Reproduce the real registry publication before ensure_loaded/slot publication.
            model_pool._process_runners[id(runner)] = {
                "runner": runner, "index": 0, "device": "npu:7", "close_requested": False,
                "close_task": None, "close_completed": False, "close_error": None, "last_state": None,
            }
            if birth_pending:
                # A controlled cached transition, with no actual process.start or model work.
                runner._update(resource_reaped=False, birth_in_progress=True, worker_alive=None)
            model_pool.refresh_status()
            local = {"speech_campplus": copy.deepcopy(models.local_model_status["speech_campplus"])}
            assert local["speech_campplus"]["model_workers"][0]["worker_identity"] is None
            assert local["speech_campplus"]["instance_details"][0]["memory"]["ready"] is None
            stack.enter_context(patch.object(status, "get_local_model_status", return_value=local))
            stack.enter_context(patch.object(status, "get_qwen_status", return_value={"model_workers": []}))
            stack.enter_context(patch.object(status, "get_slot_manager", return_value=SimpleNamespace(snapshot=lambda: {})))
            stack.enter_context(patch.object(status, "get_vad_pool", return_value=SimpleNamespace(stats=lambda: {})))
            stack.enter_context(patch.object(status, "get_live_allocator_status", return_value={"allocated_bytes": 0}))
            stack.enter_context(patch.object(status, "request_telemetry_snapshot", return_value={}))
            stack.enter_context(patch.object(status, "pending_snapshot", return_value={}))
            stack.enter_context(patch.object(status, "status_snapshot", side_effect=AssertionError("synchronous collector")))
            result = status.build_status_payload({"total_failed_tasks": 0}, datetime.utcnow(),
                                                 resource_snapshot={"npu": {"compute_apps": [app_row(1601397)]}})
            diagnostic = result["resourceDiagnostics"]
            cache = diagnostic["cam_model_workers"]
            assert len(cache) == 1 and cache[0]["worker_identity"] is None
            assert cache[0]["ready"] is None and cache[0]["allocator"] is None
            assert not diagnostic["npu"]["cam_compute_apps"]
            assert not any(row.get("service_role") == "cam_model_worker"
                           for row in diagnostic["npu"]["service_compute_apps"])
            observed = diagnostic["processes"]["cam_model_worker"]
            assert observed["pids"] == []
            assert observed["identities"][0]["worker_identity"] is None
            if birth_pending:
                assert observed["count"] is None
                assert observed["unknown_liveness_count"] == observed["registered_not_reaped_count"] == 1
                assert observed["identities"][0]["birth_in_progress"] is True
            else:
                assert observed["count"] == 0
                assert observed["unknown_liveness_count"] == observed["registered_not_reaped_count"] == 0
            start.assert_not_called()
            load.assert_not_called()
        finally:
            if runner is not None:
                runner._update(resource_reaped=True, birth_in_progress=False, worker_alive=False)
                runner.close()
            models.local_model_status["speech_campplus"].clear()
            models.local_model_status["speech_campplus"].update(local_entry)


@pytest.mark.parametrize("changed", [False, True])
def test_os_only_worker_rss_sample_rejects_pid_reuse_during_read(changed):
    before = {"pid": 200, "start_ticks": "50", "create_time": 100., "sid": 100}
    after = dict(before, start_ticks="51", create_time=101.) if changed else before
    os_resource = {"available": True, "pid": 200, "rss_bytes": 1000, "memory": {"pss_bytes": 800}}
    with patch.object(resources, "_process_start_identity", side_effect=[before, after]), \
            patch.object(resources, "process_snapshot", return_value=os_resource) as sampled:
        result = resources._npu_process_sample({"pid": 200, "memory_mb": 157})
    sampled.assert_called_once_with(200)
    assert result["memory_mb"] == 157  # npu-smi HBM remains its distinct sampled scope.
    if changed:
        assert result["process_identity"]["available"] is False
        assert result["process_resource"]["available"] is False
        assert "rss_bytes" not in result["process_resource"]
    else:
        assert result["process_identity"] == before
        assert result["process_resource"]["rss_bytes"] == 1000
        assert result["process_resource"]["memory"]["pss_bytes"] == 800
