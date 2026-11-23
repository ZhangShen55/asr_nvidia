from __future__ import annotations
import asyncio
from copy import deepcopy
from functools import wraps
import json
from types import SimpleNamespace
import pytest
from scripts import benchmark_qwen_multi_instance as controller
from scripts.benchmark_qwen_multi_instance import Safety, summarize, covered_owners, batch


def async_test(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return run


def test_complete_and_censored_statistics_use_different_denominators():
    rows = [dict(status="completed", elapsed_s=10, warning_count=0),
            dict(status="cancelled", elapsed_s=20)]
    value = summarize(rows, 20, "hard_336s")
    assert not value["complete"] and value["all_request_latency"] is None
    assert value["completion_subset_latency"] == {"n": 1, "p50_s": 10, "p95_s": 10}
    rows[1] = dict(status="completed", elapsed_s=20, warning_count=1, discarded=1)
    value = summarize(rows, 20)
    assert value["complete"] and value["all_request_latency"]["p95_s"] == 19.5
    assert value["warning_free"] == 1 and value["discarded_segments"] == 1


@pytest.mark.parametrize("free_mb", [0, 1, 6323, 8191, None])
def test_cancelled_hbm_threshold_has_no_replacement(free_mb):
    status = {"resourceDiagnostics": {"npu": {"npus": [{"index": 7, "available": True, "hbm_free_mb": free_mb}]}}}
    assert Safety().check(status) is None


@pytest.mark.parametrize("status,reason", [
    ({"qwenModel": {"failed_instances": 1}}, "qwen_owner_failure"),
    ({"qwenModel": {"execution_failures": 1}}, "qwen_owner_failure"),
    ({"vadPool": {"failed_instances": 1}}, "vad_worker_failure"),
    ({"localModels": {"emotion2vec": {"failed_instances": 1}}}, "enhancement_worker_failure"),
])
def test_real_failures_remain_stop_conditions(status, reason):
    assert Safety().check(status) == reason


def test_swap_requires_distinct_actual_samples():
    gate = Safety()
    for stamp, amount in [("a", 0), ("b", 1), ("b", 9), ("c", 2)]:
        status = {"resourceDiagnostics": {"timestamp": stamp, "vmstat": {"available": True, "swap_in_bytes": amount, "swap_out_bytes": 0}}}
        assert gate.check(status) is None
    status["resourceDiagnostics"].update(timestamp="d", vmstat={"available": True, "swap_in_bytes": 3, "swap_out_bytes": 0})
    assert gate.check(status) == "sustained_host_swap"


def test_ready_is_not_real_forward_coverage():
    item = {"index": 0, "loaded": True, "model_worker": {"worker_alive": True, "ready_verified": True, "worker_pid": 123, "worker_create_time": 1.0}}
    status = {"qwenModel": {"instances": [item]}}
    assert covered_owners(status) == {}
    item["continuous"] = {"completed": 1}
    assert covered_owners(status)["0"]["pid"] == 123
    item["model_worker"]["worker_alive"] = False
    assert covered_owners(status) == {}


@async_test
async def test_real_client_deadline_cancels_outstanding_requests_and_marks_censored(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"fixture")
    finalized = asyncio.Event()
    class Client:
        async def post(self, *args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                finalized.set()
        async def get(self, *args, **kwargs):
            await asyncio.Event().wait()
    class Process:
        def poll(self): return None
    class Owner:
        process = Process()
        def register(self): pass
    result = await batch(Client(), [{"sample_id": "a1", "path": str(clip)}], "A", tmp_path / "round", Owner(), hard_s=.01)
    assert finalized.is_set()
    assert result["trigger"] == "hard_336s" and result["all_request_latency"] is None
    assert result["records"][0]["status"] == "cancelled"
    assert json.loads((tmp_path / "round/summary.json").read_text())["complete"] is False


@async_test
async def test_partial_completion_waits_only_pending_tasks_without_monitor_spin(tmp_path, monkeypatch):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"fixture")
    finish = asyncio.Event()
    waits, registrations = [], []
    original_wait = asyncio.wait
    async def observed_wait(tasks, **kwargs):
        pending = list(tasks)
        assert pending and all(not task.done() for task in pending)
        waits.append(len(pending))
        if len(pending) == 1:
            finish.set()
        return await original_wait(pending, **kwargs)
    class Client:
        async def post(self, *args, **kwargs):
            if kwargs["files"]["audioFile"][0].endswith("-1.wav"):
                await finish.wait()
            return ProbeResponse()
        async def get(self, *args, **kwargs):
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {})
    owner = SimpleNamespace(process=SimpleNamespace(poll=lambda: None), register=lambda: registrations.append(True))
    monkeypatch.setattr(controller.asyncio, "wait", observed_wait)
    result = await batch(Client(), [{"sample_id": f"a{i}", "path": str(clip)} for i in range(2)],
                         "A", tmp_path / "round", owner, hard_s=5)
    assert result["complete"]
    assert waits == [2, 1]
    assert len(registrations) == 2


@pytest.mark.parametrize("field", ["vadPool", "localModels"])
def test_stale_resource_cache_never_hides_live_worker_failure(field):
    status = {"resourceDiagnostics": {"sampling": {"age_s": 100}}}
    status[field] = {"failed_instances": 1} if field == "vadPool" else {"speech_campplus": {"failed_instances": 1}}
    assert Safety().check(status) == ("vad_worker_failure" if field == "vadPool" else "enhancement_worker_failure")


def test_response_clocks_define_full_wall_and_only_a6_has_200_second_target():
    rows = [dict(status="completed", started=1, response_received=11, elapsed_s=10, warning_count=0),
            dict(status="completed", started=2, response_received=22, elapsed_s=20, warning_count=0)]
    value = summarize(rows, 999)
    assert value["wall_s"] == 21 and value["observation_wall_s"] == 999
    assert value["wall_scope"] == "first_send_to_last_response"
    assert value["target_200s_met"] is None
    assert summarize(rows, 999, target_s=200)["target_200s_met"] is True


def config_bytes():
    return b'''instance_count=1
[qwen_asr]
generation_mode="continuous"
execution_mode="process"
local_instance_count=6
segment_concurrency=16
local_device="npu:7"
audio_spk_emo_concurrency=4
[model_device_pool]
speech_campplus_devices=["npu:7"]
emotion2vec_devices=["npu:7"]
'''


def good_status():
    instances = [{"index": index, "loaded": True, "continuous": {"completed": 1, "active_sequences": 0, "peak_active": 1},
                  "model_worker": {"worker_pid": 100 + index, "worker_create_time": index + 1.0,
                      "worker_alive": True, "ready_verified": True,
                      "worker_entry_module_file": str(controller.ROOT / "core/qwen_asr_process.py"),
                      "launch_config": {"device": "npu:7", "continuous_options": {"max_active": 16}}}}
                 for index in range(6)]
    return {"qwenModel": {"generation_mode": "continuous", "execution_mode": "process", "device": "npu:7",
                "capacity": 6, "instances": instances,
                "owner_pool": {"configured": 6, "healthy": 6, "leased": 0, "segment_capacity_per_owner": 16,
                               "configured_global_segments": 96},
                "batch_scheduler": {"capacity": 96, "running_members": 0, "peak_inflight": 6}},
            "audioConcurrency": {"configured_total": 6, "active_total": 0},
            "spkEmoConcurrency": {"configured": 4, "effective": 4, "active": 0},
            "localModels": {name: {"configured_instances": 1, "instances": 1, "loaded": True,
                                  "called_instances": 1, "failed_instances": 0} for name in ("speech_campplus", "emotion2vec")},
            "resourceDiagnostics": {"segmentPipeline": {"active": 0, "audio_bytes": 0}}}


def test_service_contract_checks_owner_identity_s_global_and_resident_models():
    status = good_status()
    expected = controller.deployment_contract(config_bytes())
    covered = covered_owners(status)
    assert controller.service_validation(status, expected, baseline=covered, idle=True, resident=True)["passed"]
    broken = deepcopy(status)
    broken["qwenModel"]["instances"][0]["model_worker"]["worker_create_time"] += 1
    assert not controller.service_validation(broken, expected, baseline=covered)["passed"]
    broken = deepcopy(status)
    broken["resourceDiagnostics"]["segmentPipeline"]["active"] = 1
    assert not controller.service_validation(broken, expected, idle=True)["passed"]
    broken = deepcopy(status)
    broken["localModels"]["emotion2vec"]["called_instances"] = 0
    assert not controller.service_validation(broken, expected, resident=True)["passed"]


def test_formal_config_cannot_silently_change_six_owner_protocol():
    with pytest.raises(ValueError):
        controller.deployment_contract(config_bytes().replace(b"local_instance_count=6", b"local_instance_count=4"))
    with pytest.raises(ValueError):
        controller.deployment_contract(config_bytes().replace(b'local_device="npu:7"', b'local_device="npu:6"'))


def enhancement_fixture():
    before, after = {"localModels": {}}, {"localModels": {}}
    rows = []
    stages = []
    for name in ("speech_campplus", "emotion2vec"):
        before["localModels"][name] = {"instance_metrics": [{"instance_index": 0, "calls": 10}],
            "inference_outcomes": {"success": 10}, "duration_skips": 0}
        after["localModels"][name] = {"configured_instances": 1, "instances": 1, "loaded": True,
            "instance_metrics": [{"instance_index": 0, "calls": 14}],
            "inference_outcomes": {"success": 14}, "duration_skips": 0}
    for index in range(4):
        filename = f"unique-round-{index}.wav"
        rows.append({"sample_id": f"a{index + 1}", "upload_filename": filename,
                     "segments": 1, "role_nonnull": 1, "emotion_nonnull": 1})
        stages.append({"filename": filename, "request_id": f"request-{index}", "phases": {
            "qwen_calls": 1, "spk_nn_calls": 1, "spk_nn_success": 1, "emo_nn_calls": 1, "emo_nn_success": 1}})
    after["resourceDiagnostics"] = {"requestTelemetry": {"recent_stages": stages}}
    return before, after, rows


def test_b4_requires_current_round_every_instance_and_each_request_nn_evidence():
    before, after, rows = enhancement_fixture()
    assert controller.enhancement_validation(before, after, rows)["passed"]
    after["localModels"]["emotion2vec"]["instance_metrics"][0]["calls"] = 10
    assert not controller.enhancement_validation(before, after, rows)["passed"]


def test_b4_duration_only_is_not_nn_success_and_old_filename_cannot_prove_coverage():
    before, after, rows = enhancement_fixture()
    first = after["resourceDiagnostics"]["requestTelemetry"]["recent_stages"][0]
    first["phases"].update(emo_nn_calls=0, emo_nn_success=0, emo_duration_skips=1)
    after["localModels"]["emotion2vec"]["instance_metrics"][0]["calls"] = 13
    after["localModels"]["emotion2vec"]["inference_outcomes"]["success"] = 13
    assert controller.enhancement_validation(before, after, rows)["passed"]
    assert rows[0]["emotion_duration_only"] and not rows[0]["real_dual_nn_success"]
    before, after, rows = enhancement_fixture()
    after["resourceDiagnostics"]["requestTelemetry"]["recent_stages"][0]["filename"] = "old-round-0.wav"
    assert not controller.enhancement_validation(before, after, rows)["passed"]


def test_a_request_cannot_silently_run_enhancement_and_stage_matching_is_exact():
    before, after, rows = enhancement_fixture()
    assert not controller.stage_evidence(before, after, rows, "A")["passed"]
    for row in after["resourceDiagnostics"]["requestTelemetry"]["recent_stages"]:
        row["phases"] = {"itn_ms": 2}
    assert controller.stage_evidence(before, after, rows, "A")["passed"]
    rows[0]["upload_filename"] = "new-unmatched-round.wav"
    assert not controller.stage_evidence(before, after, rows, "A")["passed"]


def test_integrity_gates_drift_and_shared_original_port_preservation():
    identity = {"complete": True, "files": {"core/a.py": {"sha256": "one"}}}
    shared = {"available": True, "alive": True, "birth": 1, "listening_ports": [("127.0.0.1", 8083)]}
    assert controller.integrity_result(identity, identity, b"cfg", b"cfg", shared, shared)["passed"]
    changed = deepcopy(identity)
    changed["files"]["core/a.py"]["sha256"] = "two"
    assert not controller.integrity_result(identity, changed, b"cfg", b"cfg", shared, shared)["passed"]
    assert not controller.integrity_result(identity, identity, b"cfg", b"other", shared, shared)["passed"]
    assert not controller.integrity_result(identity, identity, b"cfg", b"cfg", shared,
                                         dict(shared, listening_ports=[]))["passed"]


def test_main_integrity_failure_affects_exit_code(monkeypatch):
    async def fake_execute(args):
        return {"A6": {"complete": True, "passed": True}, "shutdown": {"passed": True}, "integrity": {"passed": False}}
    monkeypatch.setattr(controller, "execute", fake_execute)
    monkeypatch.setattr(controller.sys, "argv", ["benchmark", "--output", "unused"])
    assert controller.main() == 1


@async_test
async def test_resident_phase_without_same_service_b4_is_rejected_before_start(tmp_path):
    with pytest.raises(ValueError):
        await controller.execute(SimpleNamespace(phases=["A6-resident"], output=str(tmp_path / "never-started")))
    assert not (tmp_path / "never-started").exists()


@async_test
async def test_controller_repeated_cancel_keeps_owned_close_and_final_report(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_bytes(config_bytes())
    output = tmp_path / "run"
    request_entered, close_entered, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    class Owner:
        process = SimpleNamespace(poll=lambda: None)
        def register(self): pass
        async def close(self, **kwargs):
            close_entered.set()
            await finish.wait()
            return {"passed": True, "exit_code": 0, "forced": False}
    async def never_status(client):
        request_entered.set()
        await asyncio.Event().wait()
    async def background():
        return hbm_snapshot(7, used=100)
    monkeypatch.setattr(controller, "OwnedService", lambda *args, **kwargs: Owner())
    monkeypatch.setattr(controller, "source_identity", lambda: {"complete": True, "files": {}})
    monkeypatch.setattr(controller, "load_inputs", lambda path: [])
    monkeypatch.setattr(controller, "shared_identity", lambda: {
        "available": True, "alive": True, "birth": 1, "listening_ports": [("127.0.0.1", 8083)]})
    monkeypatch.setattr(controller, "background_snapshot", background)
    monkeypatch.setattr(controller, "get_status", never_status)
    args = SimpleNamespace(phases=["A6"], output=str(output), config=str(config), manifest="unused", warmup_clip="unused")
    task = asyncio.create_task(controller.execute(args))
    try:
        await asyncio.wait_for(request_entered.wait(), 5)
        task.cancel()
        await asyncio.wait_for(close_entered.wait(), 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        result = json.loads((output / "run.json").read_text())
        assert result["phases"]["shutdown"]["passed"]
        assert result["phases"]["shutdown"]["exit_code"] == 0
        assert result["integrity"]["passed"]
    finally:
        finish.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


class ProbeResponse:
    status_code = 200
    def __init__(self, code=None): self.code = code
    def raise_for_status(self): pass
    def json(self):
        if self.code is not None:
            return {"code": self.code}
        return {"segments": [{"segment_text": "fixture"}], "processing_status": "complete", "segment_warnings": []}


@async_test
async def test_admission_probe_requires_observed_real_capacity_windows(tmp_path):
    # 仅测试controller时序的CPU替身；生产函数始终发送真实HTTP，不注入假hold服务。
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"fixture")
    active, enhanced, rejected = 0, 0, 0
    finish = asyncio.Event()
    class Client:
        async def get(self, *args, **kwargs):
            value = {"audioConcurrency": {"active_total": active}, "spkEmoConcurrency": {"active": enhanced}}
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: value)
        async def post(self, *args, **kwargs):
            nonlocal active, enhanced, rejected
            is_b = kwargs["data"]["showSpk"] == "true"
            if active == 6:
                rejected += 1
                if rejected == 2:
                    finish.set()
                return ProbeResponse(4013)
            if is_b and enhanced == 4:
                return ProbeResponse(4017)
            active += 1
            enhanced += int(is_b)
            try:
                await finish.wait()
                return ProbeResponse()
            finally:
                active -= 1
                enhanced -= int(is_b)
    owner = SimpleNamespace(process=SimpleNamespace(poll=lambda: None), register=lambda: None)
    result = await controller.admission_probes(Client(), clip, tmp_path / "probe", owner)
    assert result["passed"] and result["records"]["fifthB"]["code"] == 4017
    assert result["records"]["fullA"]["code"] == result["records"]["fullB"]["code"] == 4013
    assert any(row["audio"]["active_total"] == 6 and row["B"]["active"] == 4 for row in result["observations"])
    assert active == enhanced == 0


@async_test
async def test_admission_probe_without_activity_window_is_inconclusive(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"fixture")
    class Client:
        async def post(self, *args, **kwargs): return ProbeResponse()
        async def get(self, *args, **kwargs):
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
                "audioConcurrency": {"active_total": 0}, "spkEmoConcurrency": {"active": 0}})
    owner = SimpleNamespace(process=SimpleNamespace(poll=lambda: None), register=lambda: None)
    result = await controller.admission_probes(Client(), clip, tmp_path / "probe", owner)
    assert not result["passed"] and result["status"] == "inconclusive"


def hbm_snapshot(index=7, *, used=100, total=64000):
    return {"npu": {"available": True, "npus": [{"index": index, "chip": 0, "bus_id": "0000:01:00.0",
        "available": True, "hbm_used_mb": used, "hbm_total_mb": total, "hbm_free_mb": total - used}]}}


def test_background_hbm_uses_only_physical_npu7_actual_values():
    before, after = hbm_snapshot(7, used=100), hbm_snapshot(7, used=160)
    after["npu"]["npus"].append(hbm_snapshot(0, used=1)["npu"]["npus"][0])
    value = controller.hbm_recovery_evidence(before, after)
    assert value["observation_complete"] and value["passed"] is None
    assert value["before"]["index"] == value["after"]["index"] == 7
    assert value["used_delta_mb"] == 60 and value["free_delta_mb"] == -60
    assert value["near_background"] is None
    missing = controller.hbm_recovery_evidence(before, hbm_snapshot(0))
    assert not missing["passed"] and missing["status"] == "unknown"
    assert missing["used_delta_mb"] is None


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), "100"])
def test_physical_npu7_hbm_missing_or_non_numeric_is_unknown(value):
    before, after = hbm_snapshot(), hbm_snapshot()
    after["npu"]["npus"][0]["hbm_used_mb"] = value
    result = controller.hbm_recovery_evidence(before, after)
    assert not result["passed"] and not result["hbm_observed"]
    assert result["after"]["status"] == "unknown"


def test_observing_zero_free_hbm_does_not_reintroduce_a_threshold():
    sample = hbm_snapshot(7, used=64000)
    assert controller.hbm_recovery_evidence(sample, sample, owned_normal_exit=True)["passed"]
    assert Safety().check({"resourceDiagnostics": sample}) is None


def test_real_decode_steps_prevent_false_no_progress_with_no_eos():
    progress = controller.ExecutionProgress(0)
    status = {"qwenModel": {"instances": [{"index": 0, "continuous": {
        "completed": 0, "steps": {"prefill": 1, "decode": 10}}}]}}
    progress.observe(status, 0, 0)
    for now, count in ((110, 11), (220, 12), (330, 13)):
        status["qwenModel"]["instances"][0]["continuous"]["steps"]["decode"] = count
        progress.observe(status, now, 0)
        assert not progress.expired(now)
    assert progress.expired(450)
    status["qwenModel"]["instances"][0]["continuous"].pop("steps")
    status["resourceDiagnostics"] = {"sampling": {"age_s": .1}}
    progress.observe(status, 451, 0)
    assert progress.expired(451) and progress.step_observation["status"] == "unknown"


def test_live_window_and_final_peak_limits_are_separate_and_unknown_is_not_zero():
    status = good_status()
    status["qwenModel"]["instances"][0]["continuous"]["active_sequences"] = 17
    assert Safety().check(status) == "qwen_owner_active_limit"
    status = good_status()
    status["qwenModel"]["batch_scheduler"]["running_members"] = 97
    assert Safety().check(status) == "qwen_global_active_limit"
    status = good_status()
    covered = covered_owners(status)
    expected = controller.deployment_contract(config_bytes())
    status["qwenModel"]["instances"][0]["continuous"]["peak_active"] = 17
    assert not controller.service_validation(status, expected, baseline=covered)["passed"]
    status = good_status()
    status["qwenModel"]["batch_scheduler"].pop("peak_inflight")
    result = controller.service_validation(status, expected, baseline=covered)
    assert not result["passed"] and "global.peak_inflight" in result["unknown"]


def test_enhancement_conservation_uses_qwen_logical_calls_and_outcome_totals():
    before, after, rows = enhancement_fixture()
    phases = after["resourceDiagnostics"]["requestTelemetry"]["recent_stages"][0]["phases"]
    phases["qwen_calls"] = 2
    assert not controller.enhancement_validation(before, after, rows)["passed"]
    before, after, rows = enhancement_fixture()
    after["resourceDiagnostics"]["requestTelemetry"]["recent_stages"][0]["phases"]["spk_nn_success"] = 0
    assert not controller.enhancement_validation(before, after, rows)["passed"]


def test_hbm_observation_and_recovery_conclusion_do_not_conflate_external_changes():
    before, after = hbm_snapshot(used=100), hbm_snapshot(used=160)
    before["npu"]["compute_apps"] = [{"index": 7, "chip": 0, "pid": 1601397, "memory_mb": 50}]
    after["npu"]["compute_apps"] = [{"index": 7, "chip": 0, "pid": 1601397, "memory_mb": 110}]
    value = controller.hbm_recovery_evidence(before, after, owned_normal_exit=True)
    assert value["observation_complete"] and value["hbm_observed"]
    assert value["used_delta_mb"] == value["external_process_changes"][0]["delta_mb"] == 60
    assert value["passed"] is None and value["status"] == "unknown"


def test_namespace_socket_probe_matches_only_target_owned_inodes(tmp_path):
    import os
    import scripts.benchmark_qwen_multi_instance as controller
    process = tmp_path / "123"
    (process / "fd").mkdir(parents=True)
    (process / "net").mkdir()
    os.symlink("socket:[777]", process / "fd/5")
    os.symlink("socket:[888]", process / "fd/6")
    header = "sl local_address rem_address st tx_queue tr retrnsmt uid timeout inode\n"
    (process / "net/tcp").write_text(header +
        "0: 0100007F:1F47 00000000:0000 0A 0:0 0:0 0 0 0 777\n" +
        "1: 0100007F:FFFF 00000000:0000 0A 0:0 0:0 0 0 0 999\n" +
        "2: 0100007F:BBBB 00000000:0000 01 0:0 0:0 0 0 0 777\n")
    (process / "net/tcp6").write_text(header +
        "0: 00000000000000000000000001000000:1F48 00000000000000000000000000000000:0000 0A 0:0 0:0 0 0 0 888\n")
    assert controller.namespace_listening_ports(123, proc_root=tmp_path) == [("127.0.0.1", 8007), ("::1", 8008)]
