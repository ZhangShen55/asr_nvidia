import asyncio
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("throughput_harness", Path(__file__).resolve().parents[1]/"scripts/benchmark_asr_throughput.py")
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


def samples():
    return [dict(sample_id=str(i), sha256=str(i), audio_seconds=10) for i in range(8)]


def args(**updates):
    values = dict(mode="A", concurrency=2, url="http://127.0.0.1:1", diagnostic_wall_timeout_s=420,
                  no_progress_s=120, monitor_interval_s=.002, diagnostic_window_s=.05)
    values.update(updates)
    return SimpleNamespace(**values)


def test_same_workload_continuous_refill_and_timestamps(tmp_path):
    async def exercise():
        active = 0
        peak = 0
        events = []
        async def request(sample, fields):
            nonlocal active, peak
            active += 1; peak = max(peak, active)
            events.append((sample["sample_id"], "start"))
            await asyncio.sleep(.03 if sample["sample_id"] == "0" else .001)
            events.append((sample["sample_id"], "end"))
            active -= 1
            return {"segments": []}
        async def status():
            return {"completed": len([e for e in events if e[1] == "end"])}
        result = await harness.run_round(args(), samples(), tmp_path, request_fn=request, status_fn=status)
        assert result["complete"] and result["successful"] == 8 and peak == 2
        assert events.index(("2", "start")) < events.index(("0", "end"))
        assert all(row["elapsed_s"] == row["ended_monotonic"]-row["started_monotonic"] for row in result["records"])
        assert result["batch_elapsed_s"] >= max(row["elapsed_s"] for row in result["records"])
    asyncio.run(exercise())


def test_watchdog_aborts_and_keeps_unsubmitted_and_cancellations(tmp_path):
    async def exercise():
        async def request(sample, fields):
            await asyncio.sleep(10)
        async def stuck_status():
            await asyncio.sleep(10)
        result = await harness.run_round(args(diagnostic_wall_timeout_s=.01, no_progress_s=1), samples(), tmp_path,
                                         request_fn=request, status_fn=stuck_status)
        assert result["status"] == "diagnostic_aborted"
        assert not result["complete"] and result["successful"] == 0
        assert len(result["records"]) == 8
        assert sum(row["status"] == "cancelled" for row in result["records"]) == 2
        assert sum(row["status"] == "not_submitted" for row in result["records"]) == 6
        assert result["source_audio_seconds_per_wall_second"] is None
        assert (tmp_path/"diagnostic-trigger.json").is_file()
    asyncio.run(exercise())


def test_failed_response_not_dropped(tmp_path):
    async def exercise():
        async def request(sample, fields):
            return {"code": 4017} if sample["sample_id"] == "0" else {"segments": []}
        async def status():
            return {}
        result = await harness.run_round(args(), samples(), tmp_path, request_fn=request, status_fn=status)
        assert result["status"] == "incomplete" and result["successful"] == 7
        assert result["success_subset_latency"]["n"] == 7 and len(result["records"]) == 8
    asyncio.run(exercise())


def test_same_source_manifest_hash_validation(tmp_path):
    rows = []
    for i in range(8):
        path = tmp_path/str(i)
        path.write_bytes(bytes([i]))
        rows.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), audio_seconds=1))
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps(rows))
    assert len(harness.load_samples(manifest)) == 8
    rows[-1] = rows[0]
    manifest.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        harness.load_samples(manifest)


def test_progress_excludes_time_and_memory():
    assert harness.progress_token({"age": 1,"rss": 10,"completed": 2}) == harness.progress_token({"age": 9,"rss": 99,"completed": 2})
    assert harness.fields_for("mixed", 0)["showSpk"] == "true"
    assert harness.fields_for("mixed-swap", 0)["showSpk"] == "false"


def test_owned_cleanup_preserves_unrelated_process(tmp_path):
    import psutil
    import subprocess
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        port = connection.getsockname()[1]
    owner = harness.OwnedService([sys.executable,"-c","import time;time.sleep(30)"],dict(os.environ),"127.0.0.1",port,tmp_path,
                                 expected_exit_codes=(0, -15))
    try:
        result = asyncio.run(owner.close(deadline_s=1, observe_s=.01))
        assert result["passed"] and not result["remaining_pids"]
        assert unrelated.poll() is None
        assert owner.process.poll() is not None
    finally:
        unrelated.terminate(); unrelated.wait()


def test_pid_reuse_is_not_owned(tmp_path):
    import psutil
    owner = object.__new__(harness.OwnedService)
    owner.identities = {os.getpid(): psutil.Process().create_time()-100}
    assert owner.remaining() == []


def test_owned_service_uses_formal_main_entry():
    assert harness.service_command(18083) == [sys.executable, "main.py", "--host", "127.0.0.1",
        "--port", "18083", "--workers", "1", "--timeout-graceful-shutdown", "180"]


def test_exit_one_cannot_pass_cleanup_even_without_listener_or_children(tmp_path):
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0)); port=connection.getsockname()[1]
    owner=harness.OwnedService([sys.executable,"-c","import time,sys;time.sleep(.1);sys.exit(1)"],
                               dict(os.environ),"127.0.0.1",port,tmp_path)
    owner.process.wait(timeout=5)
    result=asyncio.run(owner.close(deadline_s=1,observe_s=.01))
    assert result["exit_code"]==1 and not result["exit_code_ok"]
    assert not result["forced"] and not result["remaining_pids"] and not any(result["port_observations"])
    assert not result["passed"]


def test_owned_shutdown_deadline_reserves_main_graceful_window():
    with pytest.raises(SystemExit):
        harness.parse_args(["--manifest","unused", "--output","unused", "--concurrency","1",
                            "--mode","A", "--launch-service", "--config","unused",
                            "--shutdown-timeout-s","180"])


def test_active_progress_only_tracks_done_counters():
    def status(age, done, noise):
        return {"active": {"count": noise, "requests": [{"request_id": "r", "stage": "qwen",
                "age_s": age, "no_progress_s": age, "progress": {"segments_total": noise,
                "segments_done": done, "qwen_done": done, "spk_done": 0, "emo_done": 0,
                "cut_done": 3, "time": noise, "queue": noise}}]}}
    assert harness.progress_token(status(1, 0, 1)) == harness.progress_token(status(99, 0, 99))
    assert harness.progress_token(status(1, 0, 1)) != harness.progress_token(status(2, 1, 1))
    clock = [0.0]
    watchdog = harness.PerformanceWatchdog(no_progress_s=10, clock=lambda:clock[0])
    watchdog.sample([], {"r": 0}, harness.progress_token(status(1, 0, 1)))
    clock[0] = 11
    assert watchdog.sample([], {"r": 0}, harness.progress_token(status(99, 0, 99))).reason == "no_progress"


def test_b64_legacy_path_and_ascii_multipart(tmp_path, monkeypatch):
    import httpx
    rows = []
    for i in range(8):
        rawpath = os.fsencode(tmp_path) + b"/legacy-\xff-" + str(i).encode() + b".wav"
        with open(rawpath, "wb") as stream:
            stream.write(bytes([i]))
        rows.append({"id": str(i), "path_bytes_b64": base64.b64encode(rawpath).decode(),
                     "sha256": hashlib.sha256(bytes([i])).hexdigest(), "duration_seconds": 1})
    manifest = tmp_path/"manifest.json"
    harness.write_json(manifest, {"samples": rows})
    loaded = harness.load_samples(manifest)
    assert os.fsencode(loaded[0]["path"]) == base64.b64decode(rows[0]["path_bytes_b64"])
    assert loaded[0]["path_bytes_b64"] == rows[0]["path_bytes_b64"]
    output = tmp_path/"run"; output.mkdir()
    sent = []
    async def post(self, url, files, data):
        filename, stream = files["audioFile"]
        assert filename.isascii() and "legacy" not in filename
        sent.append(stream.read())
        return httpx.Response(200, json={"segments": []}, request=httpx.Request("POST", "http://localhost"))
    async def status():
        return {}
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    result = asyncio.run(harness.run_round(args(), loaded, output, status_fn=status))
    assert result["complete"] and len(sent) == 8
    harness.write_json(output/"samples.json", loaded)
    assert os.fsencode(json.loads((output/"samples.json").read_text())[0]["path"]) == os.fsencode(loaded[0]["path"])


def test_parent_exits_before_eight_workers_are_registered(tmp_path):
    import psutil
    worker_file = tmp_path/"workers.json"
    code = ("import subprocess,sys,time,json;time.sleep(.1);"
            "children=[subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']) for _ in range(8)];"
            f"open({str(worker_file)!r},'w').write(json.dumps([p.pid for p in children]))")
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0)); port = connection.getsockname()[1]
    owner = harness.OwnedService([sys.executable,"-c",code], dict(os.environ),"127.0.0.1",port,tmp_path)
    try:
        owner.process.wait(timeout=5)
        workers = json.loads(worker_file.read_text())
        owner.register()
        assert all(pid in owner.identities for pid in workers)
        result = asyncio.run(owner.close(deadline_s=1, observe_s=.01))
        assert result["orphan_cleanup"] and not result["passed"]
        assert not result["remaining_pids"]
        assert all(not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE for pid in workers)
    finally:
        for process in owner.remaining():
            process.kill()


@pytest.mark.parametrize("mode,enhanced", [("A","false"),("B","true"),("mixed","true"),("mixed-swap","true")])
def test_warmup_preserves_flags_and_separate_measurement(tmp_path,mode,enhanced):
    audio=tmp_path/"clip.wav"; audio.write_bytes(b"real clip supplied by caller")
    async def request(source,fields):
        assert source == audio
        assert fields=={"showSpk":enhanced,"showEmotion":enhanced,"showRoleIdentify":"false"}
        return {"segments":[], "gpu_time_ms":"12", "warmup_marker":True}
    async def status():
        return {"active":{"count":1,"requests":[{"age_s":99,"progress":{"qwen_done":1}}]}}
    options=args(mode=mode,warmup_audio=str(audio))
    result=asyncio.run(harness.warmup(options,tmp_path,request_fn=request,status_fn=status))
    assert result["complete"] and result["response"]["warmup_marker"]
    assert result["ended_monotonic"] >= result["started_monotonic"]
    assert not (tmp_path/"summary.json").exists()
    assert not list(tmp_path.glob("request-*.json"))


def test_warmup_timeout_does_not_wait_for_slow_status(tmp_path):
    async def request(source,fields):
        await asyncio.sleep(5)
    async def status():
        await asyncio.sleep(5)
    result=asyncio.run(harness.warmup(args(warmup_audio="unused",diagnostic_wall_timeout_s=.01),tmp_path,
                                    request_fn=request,status_fn=status))
    assert result["status"]=="diagnostic_aborted" and not result["complete"]
    assert result["diagnostic_trigger"]["reason"]=="inflight_wall_timeout"


def loading_status(ready_count, *, age=0, births=8):
    workers = []
    for index in range(births):
        identity = {"pid": 1000 + index, "create_time": 100. + index,
                    "start_ticks": str(10000 + index), "sid": 999}
        workers.append({"worker_identity": identity, "worker_alive": True,
                        "birth_in_progress": False, "loaded": index < ready_count,
                        "ready": {"kind": "READY", "launch": "launch-" + str(index),
                                  "identity": identity} if index < ready_count else None,
                        "allocator": {"memory_allocated_bytes": age * 1024}})
    return {"localModels": {"speech_campplus": {"execution_mode": "process",
            "configured_instances": 8, "instances": ready_count, "model_workers": workers}},
            "active": {"requests": [{"age_s": age, "progress": {"qwen_done": 0, "spk_done": 0}}]},
            "resourceDiagnostics": {"sampling": {"age_s": age}, "cpu": age}}


def test_warmup_ready_progress_per_worker_allows_128_second_loading_without_widening_gates():
    clock = [0.]
    watchdog = harness.PerformanceWatchdog(420, 120, clock=lambda: clock[0])
    progress = harness.WarmupProgress()
    for second in range(129):
        clock[0] = second
        state = loading_status(second // 16, age=second)
        assert watchdog.sample([], {"warmup": 0}, progress.sample(state)) is None
    clock[0] = 248.
    assert watchdog.sample([], {"warmup": 0}, progress.sample(loading_status(8, age=248))) is None
    clock[0] = 248.001
    trigger = watchdog.sample([], {"warmup": 0}, progress.sample(loading_status(8, age=249)))
    assert trigger.reason == "no_progress" and trigger.threshold_s == 120
    assert trigger.observed_s == pytest.approx(120.001)


def test_warmup_birth_resource_configured_count_and_loaded_flags_are_not_ready_progress():
    clock = [0.]
    watchdog = harness.PerformanceWatchdog(420, 120, clock=lambda: clock[0])
    progress = harness.WarmupProgress()
    for second in range(121):
        clock[0] = second
        state = loading_status(0, age=second, births=second % 9)
        model = state["localModels"]["speech_campplus"]
        model.update(instances=8, loaded=bool(second % 2))  # Process counts cannot replace accepted READY.
        assert watchdog.sample([], {"warmup": 0}, progress.sample(state)) is None
    clock[0] = 120.001
    trigger = watchdog.sample([], {"warmup": 0}, progress.sample(loading_status(0, age=999)))
    assert trigger.reason == "no_progress" and trigger.observed_s == pytest.approx(120.001)


def test_warmup_ready_identity_must_match_and_native_actual_count_is_monotonic():
    progress = harness.WarmupProgress()
    empty = progress.sample({})
    cfg_only = {"localModels": {"speech_campplus": {"configured_instances": 8, "loaded": True}},
                "vadPool": {"configured_instances": 8}, "qwenModel": {"configured_instances": 1, "loaded": True}}
    assert progress.sample(cfg_only) == empty
    state = loading_status(1)
    state["localModels"]["speech_campplus"]["model_workers"][0]["ready"]["identity"] = {"pid": 999}
    # Compare the readiness portion: existing segment completion fields are independent.
    assert progress.sample(state)[1:] == empty[1:]
    actual = {"localModels": {"emotion2vec": {"execution_mode": "thread", "instances": 2}},
              "vadPool": {"loaded_instances": 8}, "qwenModel": {"execution_mode": "thread", "instances": [{"loaded": True}]}}
    ready = progress.sample(actual)
    assert ready != empty
    actual["localModels"]["emotion2vec"].update(instances=0, loaded=False)
    actual["vadPool"]["loaded_instances"] = 0
    actual["qwenModel"]["instances"][0]["loaded"] = False
    assert progress.sample(actual) == ready
    actual["localModels"]["emotion2vec"].update(instances=2, loaded=True)
    actual["vadPool"]["loaded_instances"] = 8
    actual["qwenModel"]["instances"][0]["loaded"] = True
    assert progress.sample(actual) == ready


def test_warmup_qwen_constructor_ack_is_not_loaded_and_repeated_ready_is_not_progress():
    progress = harness.WarmupProgress()
    worker = {"worker_pid": 2000, "worker_create_time": 200., "worker_start_ticks": "20000",
              "ready_verified": True, "launch_id": "qwen-launch"}
    state = {"qwenModel": {"execution_mode": "process", "instances": [{"loaded": False, "model_worker": worker}]}}
    baseline = progress.sample(state)
    state["qwenModel"]["instances"][0]["loaded"] = True
    worker["ready_verified"] = False
    assert progress.sample(state) == baseline
    worker["ready_verified"] = True
    ready = progress.sample(state)
    assert ready != baseline
    state["qwenModel"]["instances"][0]["loaded"] = False
    worker.update(worker_alive=False, parent_rss_mb=999)
    assert progress.sample(state) == ready
    state["qwenModel"]["instances"][0]["loaded"] = True
    assert progress.sample(state) == ready


def test_warmup_continuous_ready_cannot_bypass_420_wall_and_formal_token_ignores_loading():
    clock = [0.]
    watchdog = harness.PerformanceWatchdog(420, 120, clock=lambda: clock[0])
    progress = harness.WarmupProgress()
    for second in range(421):
        clock[0] = second
        state = loading_status(second // 60, age=second)
        assert watchdog.sample([], {"warmup": 0}, progress.sample(state)) is None
    clock[0] = 420.001
    trigger = watchdog.sample([], {"warmup": 0}, progress.sample(loading_status(8)))
    assert trigger.reason == "inflight_wall_timeout" and trigger.threshold_s == 420
    assert harness.progress_token(loading_status(0)) == harness.progress_token(loading_status(8, age=999))


def test_warmup_uses_loading_progress_helper_without_changing_formal_progress(tmp_path, monkeypatch):
    calls = []
    original = harness.WarmupProgress
    class ObservedProgress(original):
        def sample(self, status):
            calls.append(status)
            return super().sample(status)
    monkeypatch.setattr(harness, "WarmupProgress", ObservedProgress)
    async def request(source, fields):
        await asyncio.sleep(.01)
        return {"segments": []}
    async def status():
        return loading_status(1)
    result = asyncio.run(harness.warmup(args(warmup_audio="injected-clip"), tmp_path,
                                       request_fn=request, status_fn=status))
    assert result["complete"] and calls
    assert any(value.get("localModels") for value in calls)


@pytest.mark.parametrize("warm_success", [False, True])
def test_execute_warmup_failure_still_closes_owned_and_success_keeps_n8(tmp_path,monkeypatch,warm_success):
    import httpx
    audio=tmp_path/"clip.wav"; audio.write_bytes(b"real clip supplied by caller")
    config=tmp_path/"config.toml"; config.write_text("instance_count=1\n")
    closed=[]; formal=[]
    class Owner:
        def __init__(self,*arguments):
            self.process=SimpleNamespace(poll=lambda:None)
        def register(self):
            pass
        async def close(self,*arguments):
            closed.append(True)
            return {"passed":True}
    async def get(self,*arguments,**kwargs):
        return httpx.Response(200,json={},request=httpx.Request("GET","http://localhost"))
    async def post(self,*arguments,**kwargs):
        payload={"segments":[]} if warm_success else {"code":4012,"msg":"原始推理错误"}
        return httpx.Response(200,json=payload,request=httpx.Request("POST","http://localhost"))
    async def round_fn(options,source_samples,output,owner):
        assert (output/"warmup.json").is_file()
        formal.append(True)
        return {"complete":True,"successful":8,"expected":8,"records":[{"sample_id":str(i)} for i in range(8)]}
    monkeypatch.setattr(harness,"OwnedService",Owner)
    monkeypatch.setattr(httpx.AsyncClient,"get",get)
    monkeypatch.setattr(httpx.AsyncClient,"post",post)
    monkeypatch.setattr(harness,"run_round",round_fn)
    options=args(launch_service=True,config=str(config),warmup_audio=str(audio),startup_timeout_s=1,shutdown_timeout_s=210,observe_s=5)
    result=asyncio.run(harness.execute(options,samples(),tmp_path))
    assert closed==[True]
    if warm_success:
        assert formal==[True] and len(result["records"])==8 and result["temperature"]=="warmup_completed"
    else:
        assert not formal and not result["formal_round_started"] and not result["complete"]
        assert result["warmup"]["code"]==4012 and result["warmup"]["response"]["msg"]=="原始推理错误"


def test_warmup_expected_capacity_is_optional_and_a_does_not_require_enhancement(tmp_path):
    async def request(source,fields):
        return {"segments":[]}
    async def status():
        return {"vadPool":{"loaded_instances":8,"failed_instances":0},
                "localModels":{"speech_campplus":{"configured_instances":8,"instances":0},
                               "emotion2vec":{"configured_instances":8,"instances":0}}}
    options=args(mode="A",warmup_audio="provided-by-injected-request",expected_model_instances=8)
    result=asyncio.run(harness.warmup(options,tmp_path,request_fn=request,status_fn=status))
    assert result["complete"] and result["final_status"]["vadPool"]["loaded_instances"]==8
    options.mode="B"
    result=asyncio.run(harness.warmup(options,tmp_path,request_fn=request,status_fn=status))
    assert not result["complete"] and result["status"]=="failed"
    assert "speech_campplus" in result["error"]


def fixture_source_tree(root):
    (root/"core").mkdir(parents=True)
    (root/"api").mkdir()
    (root/"main.py").write_text("entry bytes\n")
    (root/"core/compute.py").write_text("python source\n")
    (root/"core/compute.so").write_bytes(b"protected artifact bytes")
    (root/"api/route.py").write_text("route source\n")
    for directory in ("model","test","cache",".git","core/__pycache__"):
        (root/directory).mkdir(parents=True,exist_ok=True)
        (root/directory/"excluded.py").write_text("do not include\n")
    (root/"core/generated.c").write_text("generated C must be excluded\n")


def test_source_manifest_hashes_actual_files_and_prefers_existing_extension(tmp_path):
    fixture_source_tree(tmp_path)
    identity=harness.source_identity(tmp_path)
    assert identity["complete"]
    assert set(identity["files"])=={"main.py","core/compute.py","core/compute.so","api/route.py"}
    assert identity["preferred_module_files"]["core/compute"]=="core/compute.so"
    assert identity["preferred_module_files"]["main"]=="main.py"
    assert identity["files"]["core/compute.so"]["sha256"]==hashlib.sha256(b"protected artifact bytes").hexdigest()
    assert identity["git_head"] is None and identity["git_head_status"]=="unknown"
    assert identity["python_executable"]==sys.executable


def test_protected_only_source_manifest_never_fabricates_missing_python(tmp_path):
    fixture_source_tree(tmp_path)
    (tmp_path/"core/compute.py").unlink()
    identity=harness.source_identity(tmp_path)
    assert "core/compute.py" not in identity["files"]
    assert identity["preferred_module_files"]["core/compute"]=="core/compute.so"


def test_source_manifest_limits_are_bounded_and_incomplete(tmp_path):
    fixture_source_tree(tmp_path)
    identity=harness.source_identity(tmp_path,max_files=1)
    assert not identity["complete"] and identity["file_count"]==1
    assert any(error["reason"]=="bounded_manifest_limit" for error in identity["errors"])


def test_source_added_removed_and_changed_are_explicit(tmp_path):
    fixture_source_tree(tmp_path)
    before=harness.source_identity(tmp_path)
    (tmp_path/"core/compute.so").write_bytes(b"new protected bytes")
    (tmp_path/"api/route.py").unlink()
    (tmp_path/"core/added.py").write_text("new module\n")
    after=harness.source_identity(tmp_path)
    differences=harness.source_differences(before,after)
    assert {row["path"]:row["change"] for row in differences}=={
        "core/compute.so":"modified","api/route.py":"removed","core/added.py":"added"}


def test_source_hot_change_invalidates_pass_without_erasing_original_evidence(tmp_path):
    source=tmp_path/"source";source.mkdir();fixture_source_tree(source)
    output=tmp_path/"output";output.mkdir()
    before=harness.source_identity(source)
    harness.write_json(output/"run.json",{"source_identity":before,"arguments":{"mode":"A"}})
    harness.write_json(output/"summary.json",{"complete":True,"status":"completed","successful":8})
    (source/"core/compute.py").write_text("changed during the round\n")
    assert harness.finish_source_identity(output,before)
    run=json.loads((output/"run.json").read_text());summary=json.loads((output/"summary.json").read_text())
    assert run["source_changed_during_round"] and len(run["source_changes"])==1
    assert run["source_identity"]==before
    assert summary["status"]=="source_changed" and not summary["complete"] and summary["successful"]==8


def test_owned_request_drain_cli_is_independent_and_strict():
    argv = ["--manifest", "unused", "--output", "unused", "--concurrency", "1", "--mode", "A"]
    options = harness.parse_args(argv)
    assert options.owned_request_drain_grace_s == 180
    assert options.diagnostic_wall_timeout_s == 420 and options.observe_s == 5
    explicit = harness.parse_args(argv + ["--owned-request-drain-grace-s", "10", "--shutdown-timeout-s", "240"])
    assert explicit.owned_request_drain_grace_s == 10 and explicit.shutdown_timeout_s == 240
    assert harness.service_command(18083, 10)[-2:] == ["--timeout-graceful-shutdown", "10"]
    assert harness.service_command(18083, 10)[1] == "main.py"
    for value in ("0", "-1", "1.5", "true", "nan", "inf", "181"):
        with pytest.raises(SystemExit):
            harness.parse_args(argv + ["--owned-request-drain-grace-s", value])
    for value in (True, 0, 1.5, 181):
        with pytest.raises(ValueError):
            harness.service_command(18083, value)
    with pytest.raises(SystemExit):
        harness.parse_args(argv + ["--owned-request-drain-grace-s", "10", "--diagnostic-wall-timeout-s", "421"])
    with pytest.raises(SystemExit):
        harness.parse_args(argv + ["--owned-request-drain-grace-s", "10", "--observe-s", "4"])
    with pytest.raises(SystemExit):
        harness.parse_args(argv + ["--owned-request-drain-grace-s", "10", "--launch-service", "--config", "unused",
                                  "--shutdown-timeout-s", "180"])


@pytest.mark.parametrize("request_grace", [None, 10])
def test_owned_request_drain_passes_command_and_records_actual_launch(tmp_path, monkeypatch, request_grace):
    import httpx
    commands = []
    closes = []
    class Owner:
        def __init__(self, command, *arguments):
            commands.append(command)
            self.process = SimpleNamespace(poll=lambda: None)
        def register(self):
            pass
        async def close(self, *arguments):
            closes.append(arguments)
            return {"passed": True}
    async def get(self, *arguments, **kwargs):
        return httpx.Response(200, json={}, request=httpx.Request("GET", "http://localhost"))
    async def run_round(*arguments):
        return {"complete": True, "successful": 8, "records": []}
    monkeypatch.setattr(harness, "OwnedService", Owner)
    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    monkeypatch.setattr(harness, "run_round", run_round)
    harness.write_json(tmp_path / "run.json", {"source_identity": {"fixture": True}, "arguments": {"kept": True}})
    options = args(launch_service=True, config="unused", startup_timeout_s=1, shutdown_timeout_s=240, observe_s=5)
    if request_grace is not None:
        options.owned_request_drain_grace_s = request_grace
    result = asyncio.run(harness.execute(options, samples(), tmp_path))
    expected = 180 if request_grace is None else request_grace
    record = json.loads((tmp_path / "run.json").read_text())
    assert record["source_identity"] == {"fixture": True} and record["arguments"] == {"kept": True}
    assert record["owned_service"]["command"] == commands[0]
    assert commands[0][-2:] == ["--timeout-graceful-shutdown", str(expected)]
    assert record["owned_service"]["request_drain_grace_s"] == expected
    assert record["owned_service"]["cleanup_deadline_s"] == 240
    assert record["owned_service"]["port_observe_s"] == 5
    assert closes == [(240, 5)] and result["complete"]


@pytest.mark.parametrize("already_exited", [False, True])
def test_owned_term_exit_times_are_observed_once_without_signals_or_processes(tmp_path, monkeypatch, already_exited):
    from io import StringIO
    clock = [100.0]
    process = SimpleNamespace(pid=123456, exited=already_exited, terms=0)
    def terminate():
        process.terms += 1
    process.terminate = terminate
    process.poll = lambda: 0 if process.exited else None
    def kill():
        raise AssertionError("unit fixture must drain normally, never send KILL")
    process.kill = kill
    owner = object.__new__(harness.OwnedService)
    owner.process = process
    owner.identities = {process.pid: 1.0}
    owner.temp_dir = str(tmp_path)
    owner.cache_dir = str(tmp_path / "seacraft_asr")
    owner.expected_exit_codes = (0,)
    owner.output = tmp_path
    owner.host, owner.port = "127.0.0.1", 1
    owner.log = StringIO()
    owner.register = lambda: None
    owner.remaining = lambda: [] if process.exited else [process]
    async def sleep(seconds):
        clock[0] += seconds
        process.exited = True
    monkeypatch.setattr(harness.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(harness.asyncio, "sleep", sleep)
    monkeypatch.setattr(harness, "listening", lambda *arguments: False)
    result = asyncio.run(owner.close(deadline_s=240, observe_s=5))
    assert result["passed"] and not result["forced"] and not result["remaining_pids"]
    assert process.terms == (0 if already_exited else 1)
    assert result["observe_s"] == 5 and not any(result["port_observations"])
    if already_exited:
        assert result["term_sent_monotonic"] is None
        assert result["term_to_parent_exit_s"] is None
        assert result["parent_exit_observed_monotonic"] == 100.0
    else:
        assert result["term_sent_monotonic"] == 100.0
        assert result["parent_exit_observed_monotonic"] == pytest.approx(100.1)
        assert result["term_to_parent_exit_s"] == pytest.approx(.1)


def test_owned_tempdir_creation_failure_never_spawns(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "listening", lambda *args: False)
    def fail(**kwargs):
        assert kwargs == {"prefix": "asr-svc-", "dir": "/tmp"}
        raise OSError("private TMPDIR unavailable")
    monkeypatch.setattr(harness.tempfile, "mkdtemp", fail)
    monkeypatch.setattr(harness.subprocess, "Popen", lambda *a, **k: pytest.fail("must not spawn after TMPDIR failure"))
    with pytest.raises(OSError, match="private TMPDIR"):
        harness.OwnedService([sys.executable, "unused"], dict(os.environ), "127.0.0.1", 1, tmp_path)


@pytest.mark.parametrize("protected", [False, True])
def test_owned_tmpdir_isolates_actual_source_and_protected_cache_helpers(tmp_path, monkeypatch, protected):
    import time
    root = os.environ.get("ASR_OWNED_CACHE_PROTECTED_ROOT") if protected else os.environ.get("ASR_OWNED_CACHE_SOURCE_ROOT", str(harness.ROOT))
    if not root:
        pytest.skip("independent protected stage not supplied")
    private_external = tmp_path / "PRIVATE-production-simulation"
    private_external.mkdir()
    sentinel = private_external / "sentinel"
    sentinel.write_bytes(b"preserve external private cache")
    monkeypatch.setenv("TMPDIR", str(private_external))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [os.environ.get("ASR_CPU_GUARD_ROOT"), root]))
    probe = tmp_path / "cache_probe.py"
    ready, release = tmp_path / "cache-proof.json", tmp_path / "release"
    probe.write_text('''import json,os,sys,time,tempfile
from pathlib import Path
import utils.audio_utils as audio
import core.resource_diagnostics as diagnostic
upload=Path(audio.write_audio_bytes_to_temp_file(b"CPU proof only", "proof", suffix=".wav"))
record={"TMPDIR":os.environ["TMPDIR"],"gettempdir":tempfile.gettempdir(),"cache":audio.SERVICE_CACHE_DIR,"diagnostic_cache":str(diagnostic.DEFAULT_CACHE_DIR),"upload":str(upload),"audio_origin":audio.__file__,"diagnostic_origin":diagnostic.__file__}
p=Path(sys.argv[1]);q=p.with_suffix(".tmp");q.write_text(json.dumps(record));q.replace(p)
while not Path(sys.argv[2]).exists():time.sleep(.01)
upload.unlink()
''')
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    owner = harness.OwnedService([sys.executable, str(probe), str(ready), str(release)], env, "127.0.0.1", port, tmp_path)
    try:
        end = time.monotonic() + 15
        while not ready.exists() and time.monotonic() < end:
            assert owner.process.poll() is None, (tmp_path / "service.log").read_text()
            time.sleep(.01)
        proof = json.loads(ready.read_text())
        assert proof["TMPDIR"] == proof["gettempdir"] == owner.temp_dir
        assert proof["cache"] == proof["diagnostic_cache"] == owner.cache_dir
        assert Path(proof["upload"]).parent == Path(owner.cache_dir)
        assert Path(owner.temp_dir).parent == Path("/tmp") and Path(owner.temp_dir).name.startswith("asr-svc-")
        assert owner.temp_dir != str(private_external)
        assert os.environ["TMPDIR"] == env["TMPDIR"] == str(private_external)
        for field in ("audio_origin", "diagnostic_origin"):
            assert Path(proof[field]).resolve().is_relative_to(Path(root).resolve())
            assert Path(proof[field]).suffix == (".so" if protected else ".py")
        ownership = json.loads((tmp_path / "ownership.json").read_text())
        assert ownership["temp_dir"] == owner.temp_dir and ownership["cache_dir"] == owner.cache_dir
        release.touch()
        owner.process.wait(timeout=10)
        result = asyncio.run(owner.close(deadline_s=240, observe_s=5))
        assert result["passed"] and result["temp_dir_retained"] and result["temp_dir"] == owner.temp_dir
        assert not Path(proof["upload"]).exists() and Path(owner.cache_dir).is_dir()
        assert sentinel.read_bytes() == b"preserve external private cache"
    finally:
        release.touch()
        if owner.process.poll() is None:
            owner.process.wait(timeout=10)


@pytest.mark.parametrize("state", ["missing", "empty", "residual", "unreadable"])
def test_owned_audio_cache_gate_is_readonly_after_normal_shutdown(tmp_path, monkeypatch, state):
    from io import StringIO
    clock = [100.]
    cache = tmp_path / "seacraft_asr"
    unrelated = tmp_path / "other-runtime-file"
    unrelated.write_bytes(b"TMPDIR files outside audio cache do not count")
    if state != "missing":
        cache.mkdir()
    residue = cache / "seacraft-asr-owned-leftover.wav"
    if state == "residual":
        residue.write_bytes(b"retain evidence")
    owner = object.__new__(harness.OwnedService)
    owner.temp_dir, owner.cache_dir = str(tmp_path), str(cache)
    owner.process = SimpleNamespace(pid=123456, poll=lambda: 0)
    owner.identities = {123456: 1.}
    owner.expected_exit_codes = (0,)
    owner.output = tmp_path
    owner.host, owner.port = "127.0.0.1", 1
    owner.log = StringIO()
    owner.register = lambda: None
    owner.remaining = lambda: []
    async def sleep(seconds):
        clock[0] += seconds
    monkeypatch.setattr(harness.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(harness.asyncio, "sleep", sleep)
    monkeypatch.setattr(harness, "listening", lambda *args: False)
    if state == "unreadable":
        def denied(path):
            assert Path(path) == cache
            raise PermissionError("CPU fixture denied cache scan")
        monkeypatch.setattr(harness.os, "scandir", denied)
    result = asyncio.run(owner.close(deadline_s=240, observe_s=5))
    assert result["exit_code"] == 0 and not result["forced"] and result["temp_dir_retained"]
    assert result["observe_s"] == 5 and not any(result["port_observations"])
    assert result["passed"] == (state in ("missing", "empty"))
    assert result["remaining_cache_files"] == ([residue.name] if state == "residual" else [])
    assert ("PermissionError" in result["cache_scan_error"]) if state == "unreadable" else result["cache_scan_error"] is None
    if state == "residual":
        assert residue.read_bytes() == b"retain evidence"
    assert unrelated.read_bytes() == b"TMPDIR files outside audio cache do not count"
    if state != "missing":
        assert cache.is_dir()


def test_owned_service_explicit_cwd_keeps_formal_log_untouched(tmp_path, monkeypatch):
    source, work, output = (tmp_path / name for name in ("source", "work", "output"))
    for path in (source, work, output):
        path.mkdir()
    sentinel = source / "asr_service.log"
    sentinel.write_text("historical log must remain")
    monkeypatch.setattr(harness, "ROOT", source)
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    async def exercise():
        owner = harness.OwnedService(
            [sys.executable, "-c", "from pathlib import Path; import time; Path('asr_service.log').write_text('owned runtime log'); time.sleep(30)"],
            dict(os.environ), "127.0.0.1", port, output, expected_exit_codes=(-15,), cwd=work)
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while not (work / "asr_service.log").exists():
                assert owner.process.poll() is None
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError("owned log fixture startup")
                await asyncio.sleep(.02)
        finally:
            result = await owner.close(deadline_s=5, observe_s=.05)
        assert result["passed"]
        assert json.loads((output / "ownership.json").read_text())["cwd"] == str(work)
    asyncio.run(exercise())
    assert sentinel.read_text() == "historical log must remain"
    assert (work / "asr_service.log").read_text() == "owned runtime log"
