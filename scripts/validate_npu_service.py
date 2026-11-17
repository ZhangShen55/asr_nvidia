"""独立服务验收：只关闭本次启动的进程组，finally核对端口/worker回收。"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import wave

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[1]
TERMINATION_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
# 同长度/内容重复请求的预登记门槛；允许分配器块及运行时缓存常驻，禁止持续持有请求资源。
STEADY_LIMITS = {"allocated_growth_bytes": 16 * 2**20, "reserved_growth_bytes": 256 * 2**20,
                 "rss_growth_bytes": 256 * 2**20, "thread_growth": 8, "fd_growth": 4}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def listening(port):
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def shutdown_log_evidence(path):
    """进程退出0不代表lifespan成功；核对应用和worker的原始关闭记录。"""
    text = Path(path).read_text(errors="replace")
    workers = []
    for line in text.splitlines():
        if "VAD_WORKER_SHUTDOWN " in line:
            try:
                workers.append(json.loads(line.split("VAD_WORKER_SHUTDOWN ", 1)[1]))
            except ValueError:
                workers.append({"parse_error": line, "cleanup_complete": False})
    application_failed = any(marker in text for marker in (
        "Application shutdown failed", "服务收尾失败", "VAD 池关闭失败"))
    worker_failed = any(row.get("forced") or not row.get("cleanup_complete")
                        or row.get("exitcode") != 0 for row in workers)
    return {"application_shutdown_failed": application_failed,
            "worker_shutdown_failed": worker_failed, "worker_shutdown": workers}


def quantiles(values):
    import numpy as np
    return {"count": len(values), "p50_s": float(np.percentile(values, 50)),
            "p95_s": float(np.percentile(values, 95))} if values else {"count": 0}


def select_samples(manifest, count, *, exclude_ids=()):
    unique = {}
    for item in sorted(manifest["samples"], key=lambda row: row["media"].get("duration_seconds", 1e9)):
        if item["extension"] == ".wav" and not item["errors"] and item["sample_id"] not in exclude_ids:
            unique.setdefault(item["sha256"], item)
    result = list(unique.values())[:count]
    if len(result) != count:
        raise ValueError("不足指定数量的独立SHA样本")
    return result


def clip_bytes(item, offset=60, seconds=12):
    with wave.open(os.fsdecode(bytes.fromhex(item["path_hex"])), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (16000, 1, 2):
            raise ValueError("片段源要求16k/mono/PCM16")
        source.setpos(int(offset * 16000))
        frames = source.readframes(int(seconds * 16000))
    out = io.BytesIO()
    with wave.open(out, "wb") as target:
        target.setparams((1, 2, 16000, 0, "NONE", ""))
        target.writeframes(frames)
    return out.getvalue()


def validate_response(payload):
    if payload.get("code") not in (None, 0):
        raise AssertionError(f"业务失败: {payload}")
    assert isinstance(payload["segments"], list)
    assert isinstance(payload["load_audio_time_ms"], str)
    assert isinstance(payload["gpu_time_ms"], str)
    assert isinstance(payload["speed_info"], list)
    assert isinstance(payload["language_ratio"], dict)
    import re
    previous = 0
    for segment in payload["segments"]:
        for key in ("bg", "ed"):
            assert isinstance(segment[key], str) and re.fullmatch(r"\d+\.\d{2}", segment[key])
        assert float(segment["bg"]) >= previous and float(segment["ed"]) >= float(segment["bg"])
        previous = float(segment["bg"])


async def request(client, data, filename, fields, output, request_id):
    started = time.perf_counter()
    mime = "audio/mpeg" if filename.lower().endswith(".mp3") else "audio/wav"
    response = await client.post("/v3/seacraft_asr", files={"audioFile": (filename, data, mime)}, data=fields)
    response.raise_for_status()
    payload = response.json()
    record = {"request_id": request_id, "elapsed_s": time.perf_counter() - started,
              "fields": fields, "http_status": response.status_code, "payload": payload}
    write_json(output / f"request-{request_id}.json", record)
    return record


async def exercise(args, out, process, samples, mode):
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{args.port}", timeout=7200, trust_env=False) as client:
        for _ in range(300):
            if process.poll() is not None:
                raise RuntimeError(f"服务启动失败exit={process.returncode}")
            try:
                ready = await client.get("/get_status")
                if ready.status_code == 200:
                    break
            except httpx.RequestError:
                pass
            await asyncio.sleep(1)
        else:
            raise TimeoutError("服务300s未就绪")
        initial = ready.json()
        write_json(out / "status-initial.json", initial)
        # Operational pools allow partial capacity; migration acceptance requires
        # all four configured workers and records every failure in raw status.
        pool = initial["vadPool"]
        assert pool["configured_instances"] == 4
        assert pool["loaded_instances"] == 4 and pool["failed_instances"] == 0
        assert not initial["qwenModel"]["loaded"]
        records, snapshots = [], []
        stop = asyncio.Event()
        async def sample_status():
            while not stop.is_set():
                try:
                    status = compact_status((await client.get("/get_status")).json(), process.pid)
                    snapshots.append(status)
                    with (out / "status.jsonl").open("a") as stream:
                        stream.write(json.dumps(status, ensure_ascii=False) + "\n")
                except Exception as exc:
                    with (out / "sampling-errors.log").open("a") as stream:
                        stream.write(repr(exc) + "\n")
                await asyncio.sleep(5)
        sampler = asyncio.create_task(sample_status())
        try:
            if mode == "short":
                data = clip_bytes(samples[0])
                write_json(out / "clip.json", {"source": samples[0], "offset_s": 60, "duration_s": 12,
                                             "sha256": hashlib.sha256(data).hexdigest(), "reference": "unannotated"})
                for index, flags in enumerate(itertools.product((False, True), repeat=3)):
                    fields = {key: str(flag).lower() for key, flag in zip(("showSpk", "showEmotion", "showRoleIdentify"), flags)}
                    record = await request(client, data, "regression.wav", fields, out, f"combo-{index}")
                    validate_response(record["payload"])
                    records.append(record)
                for index, language in enumerate(("auto", "", "English", "en", "invalid-language")):
                    record = await request(client, data, "language.wav", {"language": language}, out, f"language-{index}")
                    expected = 4009 if language == "invalid-language" else None
                    if expected:
                        assert record["payload"].get("code") == expected
                    else:
                        validate_response(record["payload"])
                    records.append(record)
                silence = io.BytesIO()
                with wave.open(silence, "wb") as wav:
                    wav.setparams((1, 2, 16000, 0, "NONE", "")); wav.writeframes(bytes(3 * 32000))
                for label, audio in (("silence", silence.getvalue()), ("empty", b""), ("broken", b"not-audio")):
                    record = await request(client, audio, f"{label}.wav", {}, out, label)
                    assert record["payload"].get("code") not in (None, 0)
                    records.append(record)
                # 预登记16轮连续短请求、冷却10秒，比较预热后资源与排空计数。
                for index in range(16):
                    record = await request(client, data, "steady.wav", {}, out, f"steady-{index}")
                    validate_response(record["payload"]); records.append(record)
            elif mode.startswith("failure-"):
                record = await request(client, clip_bytes(samples[0]), "failure.wav", {}, out, mode)
                assert record["payload"].get("code") == int(mode.split("-")[1])
                records.append(record)
            elif mode == "cold-mixed":
                # 首次A/B并发，覆盖HF BF16初始化与FP32增强模型构造的交错窗口。
                data = clip_bytes(samples[0])
                fields = {"showSpk": "true", "showEmotion": "true", "showRoleIdentify": "true"}
                records = await asyncio.gather(
                    request(client, data, "cold-A.wav", {}, out, "cold-A"),
                    request(client, data, "cold-B.wav", fields, out, "cold-B"))
                for record in records:
                    validate_response(record["payload"])
                assert all(record["payload"]["segments"] and record["payload"]["text"].strip()
                           for record in records), "冷并发必须实际完成转写，空segments不能作为通过"
                assert all(row["role"] is not None and row["emotion"] is not None
                           for row in records[1]["payload"]["segments"])
            elif mode == "calibrate":
                # 同一冻结12秒片段只作容量/时延标定，不作为8份独立质量样本。
                data = clip_bytes(samples[0])
                write_json(out / "calibration-source.json", {"source": samples[0], "offset_s": 60,
                           "seconds": 12, "clip_sha256": hashlib.sha256(data).hexdigest(),
                           "scope": "repeated capacity workload; not independent quality samples"})
                for index in range(args.instances):
                    warm = await request(client, clip_bytes(samples[0], seconds=4.1), "warmup-instance.wav", {}, out, f"warmup-instance-{index}")
                    validate_response(warm["payload"])
                warm = await request(client, data, "warmup.wav", {"showSpk": "true", "showEmotion": "true"}, out, "warmup")
                validate_response(warm["payload"])
                for round_id in range(2):
                    async def timed(index):
                        fields = {"showSpk": "true", "showEmotion": "true"} if index < 4 else {}
                        record = await request(client, data, "calibration.wav", fields, out, f"cal-{round_id}-{index}")
                        validate_response(record["payload"])
                        record["audio_seconds"] = 12
                        record["rtf"] = record["elapsed_s"] / 12
                        return record
                    records.extend(await asyncio.gather(*(timed(index) for index in range(8))))
            elif mode == "steady":
                write_json(out / "steady-plan.json", {"rounds": 16, "cooldown_s": 10,
                    "limits": STEADY_LIMITS, "basis": "fixed 12s input/shape, models warmed; allow allocator blocks and bounded runtime caches"})
                data = clip_bytes(samples[0])
                fields = {"showSpk": "true", "showEmotion": "true", "showRoleIdentify": "true"}
                for index in range(3):
                    warm = await request(client, data, "steady-warmup.wav", fields, out, f"warmup-{index}")
                    validate_response(warm["payload"])
                baseline = compact_status((await client.get("/get_status")).json(), process.pid)
                write_json(out / "steady-baseline.json", baseline)
                for index in range(16):
                    record = await request(client, data, "steady.wav", fields if index % 2 else {}, out, f"steady-{index}")
                    validate_response(record["payload"]); records.append(record)
            else:
                # 同一批4B+4A，使B峰值实际可达4，完整源文件不裁切、不用重复SHA充数。
                manifest = json.loads(Path(args.manifest).read_text())
                if mode == "long":
                    # 原始MP3完整上传，验证PyAV标准化后的v3全链路；该课已用于短回归。
                    full_samples = sorted((row for row in manifest["samples"] if row["extension"] == ".mp3"),
                                          key=lambda row: row["media"]["duration_seconds"])[:1]
                else:
                    # 正式8路使用未参与短片段调试/标定的源文件，按SHA去重。
                    full_samples = select_samples(manifest, 8, exclude_ids={
                        "audio-6156a4b38b576557c2be", "audio-96862e257ed1629907b8",
                        "audio-4b897698e079522a81b8", "audio-cee89c0a4814d7e72f15"})
                write_json(out / "full-samples.json", full_samples)
                async def full(index, item):
                    fields = {"showSpk": "true", "showEmotion": "true"} if index < 4 else {}
                    path = os.fsdecode(bytes.fromhex(item["path_hex"]))
                    with open(path, "rb") as source:
                        digest = hashlib.sha256()
                        for chunk in iter(lambda: source.read(4 * 2**20), b""):
                            digest.update(chunk)
                        assert digest.hexdigest() == item["sha256"], "测试源文件SHA已变化"
                        source.seek(0)
                        before_stat = os.fstat(source.fileno())
                        result = await request(client, source, item["sample_id"] + item["extension"], fields, out, f"full-{index}")
                        after_stat = os.fstat(source.fileno())
                        assert (before_stat.st_size, before_stat.st_mtime_ns) == (after_stat.st_size, after_stat.st_mtime_ns)
                    validate_response(result["payload"])
                    assert result["payload"]["segments"] and result["payload"]["text"].strip(), "完整课堂录音应有有效输出"
                    assert all(float(row["ed"]) <= item["media"]["duration_seconds"] + 0.05
                               for row in result["payload"]["segments"]), "时间轴超出完整音频"
                    result["sample_id"], result["sha256"] = item["sample_id"], item["sha256"]
                    result["audio_seconds"] = item["media"]["duration_seconds"]
                    result["rtf"] = result["elapsed_s"] / result["audio_seconds"]
                    write_json(out / f"request-full-{index}.json", result)
                    return result
                records = await asyncio.gather(*(full(index, item) for index, item in enumerate(full_samples)))
            await asyncio.sleep(10)
            final = (await client.get("/get_status")).json()
            write_json(out / "status-drained.json", final)
            assert final["audioConcurrency"]["active_total"] == 0
            assert final["audioConcurrency"]["waiting_total"] == 0
            assert final["spkEmoConcurrency"]["active"] == 0
            assert final["qwenModel"]["busy"] == 0
            assert final["qwenModel"]["waiting"] == 0
            if mode == "cold-mixed":
                assert final["resourceDiagnostics"]["backend"]["npu_allocator"]["torch_default_dtype"] == "torch.float32"
                assert final["localModels"]["speech_campplus"]["failed_instances"] == 0
                assert final["localModels"]["emotion2vec"]["failed_instances"] == 0
            if mode == "steady":
                after = compact_status(final, process.pid)
                before_memory = baseline["resourceDiagnostics"]["backend"]["npu_allocator"]
                after_memory = after["resourceDiagnostics"]["backend"]["npu_allocator"]
                checks = {key: after_memory[key] - before_memory[key] <= STEADY_LIMITS[limit]
                          for key, limit in (("allocated_bytes", "allocated_growth_bytes"), ("reserved_bytes", "reserved_growth_bytes"))}
                for key, limit in (("rss_bytes", "rss_growth_bytes"), ("threads", "thread_growth"), ("num_fds", "fd_growth")):
                    checks[key] = after["owned_process"][key] - baseline["owned_process"][key] <= STEADY_LIMITS[limit]
                checks["vad_workers"] = final["vadPool"]["loaded_instances"] == len(initial["vadPool"]["devices"])
                checks["cache_empty"] = after["resourceDiagnostics"]["cache"].get("files") == 0
                checks["model_leases_empty"] = all(not row.get("busy") and not row.get("waiting")
                                                     for row in final["localModels"].values())
                write_json(out / "steady-check.json", {"baseline": baseline, "after": after, "checks": checks, "limits": STEADY_LIMITS})
                assert all(checks.values()), checks
            write_json(out / "summary.json", {"mode": mode, "measurements": [{key: row.get(key) for key in (
                "request_id", "elapsed_s", "fields", "sample_id", "sha256", "audio_seconds", "rtf")} for row in records],
                "latency": quantiles([r["elapsed_s"] for r in records]),
                "b_peak": max((r.get("spkEmoConcurrency", {}).get("active", 0) for r in snapshots), default=0),
                "total_peak": max((r.get("audioConcurrency", {}).get("active_total", 0) for r in snapshots), default=0),
                "shared_card": True, "manual_reference": None})
        finally:
            stop.set()
            await sampler


def compact_status(status, process_pid):
    """保留稳态所需数值，周期快照不重复写入全文遥测和其他卡原始输出。"""
    diagnostics = status.get("resourceDiagnostics") or {}
    npu = diagnostics.get("npu") or {}
    result = {key: status.get(key) for key in (
        "audioConcurrency", "spkEmoConcurrency", "localModels", "qwenModel", "vadPool")}
    result["sampled_at"] = time.time()
    result["resourceDiagnostics"] = {
        "host": {key: diagnostics.get("host", {}).get(key) for key in (
            "available", "total_bytes", "available_bytes", "available_ratio", "swap_used_bytes")},
        "backend": diagnostics.get("backend"),
        "cache": {key: diagnostics.get("cache", {}).get(key) for key in (
            "available", "files", "bytes", "disk_free_bytes", "disk_total_bytes", "disk_free_ratio")},
        "processes": diagnostics.get("processes"),
        "npu": {
            "available": npu.get("available", False), "reason": npu.get("reason"),
            "memory_scope": npu.get("memory_scope"),
            "npus": [row for row in npu.get("npus", []) if row.get("index") == 7],
            "service_compute_apps": [row for row in npu.get("service_compute_apps", []) if row.get("index") == 7],
            "other_compute_apps": [row for row in npu.get("other_compute_apps", []) if row.get("index") == 7],
        },
    }
    try:
        process = psutil.Process(process_pid)
        cpu = process.cpu_times()
        result["owned_process"] = {"available": True, "pid": process_pid,
                                   "rss_bytes": process.memory_info().rss,
                                   "threads": process.num_threads(), "num_fds": process.num_fds(),
                                   "cpu_user_s": float(cpu.user), "cpu_system_s": float(cpu.system),
                                   "cpu_scope": "main_service_process_excludes_workers_and_shared_processes"}
    except (psutil.Error, OSError) as exc:
        result["owned_process"] = {"available": False, "pid": process_pid, "reason": str(exc)}
    return result


def _owned_group_members(process):
    """Popen(start_new_session=True) 的进程组，包括父进程提前退出后的后代。"""
    members = {}
    try:
        for child in psutil.Process(process.pid).children(recursive=True):
            members[child.pid] = child.create_time()
    except psutil.NoSuchProcess:
        pass
    for item in psutil.process_iter(attrs=["pid", "create_time"]):
        try:
            if item.pid != process.pid and os.getpgid(item.pid) == process.pid:
                members[item.pid] = item.info["create_time"]
        except (ProcessLookupError, PermissionError, psutil.NoSuchProcess):
            continue
    return list(members.items())


def _remaining_owned(identities):
    remaining = []
    for pid, created in identities:
        try:
            child = psutil.Process(pid)
            if child.create_time() == created and child.status() != psutil.STATUS_ZOMBIE:
                remaining.append(child)
        except psutil.NoSuchProcess:
            pass
    return remaining


def close_owned_service(process, port, out, *, original_failure=None, graceful_timeout=210,
                        terminate_timeout=20, kill_timeout=10, observe_s=5):
    """有界回收自有进程；重复中断不能打断清理，超时本身就是验收失败。"""
    handlers = {signum: signal.getsignal(signum) for signum in TERMINATION_SIGNALS}
    for signum in handlers:
        signal.signal(signum, signal.SIG_IGN)
    descendants, errors, timed_out, forced_children = [], [], False, False
    try:
        if process is not None:
            try:
                descendants = _owned_group_members(process)
            except Exception as exc:
                errors.append(f"collect_owned_children: {type(exc).__name__}: {exc}")
            write_json(out / "shutdown-before.json", {"parent": process.pid, "children": descendants, "listening": listening(port)})

            def group_signal(signum):
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    errors.append(f"signal_owned_group: {type(exc).__name__}: {exc}")

            if process.poll() is None:
                try:
                    process.send_signal(signal.SIGTERM)
                    process.wait(timeout=graceful_timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    write_json(out / "shutdown-timeout.json", {"owned_pgid": process.pid, "error": f"graceful shutdown >{graceful_timeout}s"})
                    group_signal(signal.SIGTERM)
                    try:
                        process.wait(timeout=terminate_timeout)
                    except subprocess.TimeoutExpired:
                        group_signal(signal.SIGKILL)
                        try:
                            process.wait(timeout=kill_timeout)
                        except subprocess.TimeoutExpired:
                            errors.append("owned parent still alive after SIGKILL")
                except ProcessLookupError:
                    pass
                except Exception as exc:
                    errors.append(f"wait_owned_parent: {type(exc).__name__}: {exc}")
            # TIME_WAIT 不等于 LISTEN；不要求共享卡设备 HBM 归零。
            time.sleep(observe_s)
            remaining = _remaining_owned(descendants)
            if remaining:
                forced_children = True
                for child in remaining:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(remaining, timeout=terminate_timeout)
                for child in alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                psutil.wait_procs(alive, timeout=kill_timeout)
            remaining_pids = [child.pid for child in _remaining_owned(descendants)]
        else:
            remaining_pids = []
        evidence = {"exit_code": process.returncode if process is not None else None,
                    "listening": listening(port), "remaining_children": remaining_pids,
                    "observed_s": observe_s, "original_failure": original_failure,
                    "shutdown_timeout": timed_out, "forced_child_cleanup": forced_children,
                    "cleanup_errors": errors}
        evidence["parent_exit_failed"] = process is not None and process.returncode != 0
        evidence["failed"] = bool(timed_out or forced_children or errors or evidence["listening"]
                                  or remaining_pids or evidence["parent_exit_failed"])
        write_json(out / "shutdown-after.json", evidence)
        return evidence
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def run(args):
    if listening(args.port):
        raise RuntimeError("测试端口已被占用，不接管/关闭既有监听者")
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(args.manifest).read_text())
    samples = select_samples(manifest, 8)
    text = (ROOT / "config.toml").read_text()
    import re
    text = re.sub(r'^log_path = .*', 'log_path = ' + json.dumps(str(out / "application.log")), text, flags=re.M)
    if args.models_root:
        for key, folder in (("vad_model_dir", "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
                            ("spk_model_dir", "speech_campplus_sv_zh_en_16k-common_advanced"),
                            ("emotion_model_dir", "emotion2vec_plus_large")):
            text = re.sub(rf'^{key} = .*', key + ' = ' + json.dumps(str(Path(args.models_root).resolve() / folder)), text, flags=re.M)
    for key in ("segment_concurrency", "request_concurrency"):
        text = re.sub(rf"^{key} = .*", f"{key} = {args.concurrency}", text, flags=re.M)
    text = re.sub(r'^local_instance_count = .*', f'local_instance_count = {args.instances}', text, flags=re.M)
    mode = args.mode
    if mode == "failure-4011":
        text = re.sub(r'^local_model_dir = .*', 'local_model_dir = "/tmp/asr-intentionally-missing-model-20261117"', text, flags=re.M)
    if mode == "failure-4012":
        text = re.sub(r'^request_timeout_s = .*', 'request_timeout_s = 0.000001', text, flags=re.M)
        text = re.sub(r'^retry_count = .*', 'retry_count = 0', text, flags=re.M)
    config = out / "config.toml"; config.write_text(text)
    temp = out / "tmp"; temp.mkdir()
    env = dict(os.environ, CONFIG_PATH=str(config), TMPDIR=str(temp),
               ASR_STATS_PATH=str(out / "asr_stats.json"), ASR_RUNTIME_DIR=str(temp),
               PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    # 无守护/自动重启，信号只发给本次Popen确定持有的进程组。
    process = None; failure = None
    shutdown_handlers = {signum: signal.getsignal(signum) for signum in TERMINATION_SIGNALS}
    with (out / "service.log").open("w") as log:
        try:
            process = subprocess.Popen([sys.executable, "main.py", "--host", "127.0.0.1", "--port", str(args.port), "--workers", "1", "--timeout-graceful-shutdown", "180"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            identity = psutil.Process(process.pid)
            snapshot_file = ROOT / "source-snapshot.json"
            write_json(out / "ownership.json", {"run_id": out.name, "app_root": str(ROOT), "cwd": str(ROOT),
                "source_snapshot_sha256": hashlib.sha256(snapshot_file.read_bytes()).hexdigest() if snapshot_file.exists() else None,
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(), "python": sys.executable,
                "harness_pid": os.getpid(), "harness_create_time": psutil.Process().create_time(), "pid": process.pid,
                "create_time": identity.create_time(), "pgid": os.getpgid(process.pid), "port": args.port,
                "command": identity.cmdline(), "started_at": time.time(), "concurrency": args.concurrency, "instances": args.instances})
            asyncio.run(exercise(args, out, process, samples, mode))
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
            write_json(out / "failure.json", {"error": failure})
            raise
        finally:
            # The original handlers were captured before starting the service.
            # Ignore both signals before any cleanup work, including a first
            # interrupt that arrives after exercise has already returned.
            for signum in TERMINATION_SIGNALS:
                signal.signal(signum, signal.SIG_IGN)
            try:
                try:
                    evidence = close_owned_service(process, args.port, out, original_failure=failure)
                except BaseException as cleanup_error:
                    write_json(out / "cleanup-failure.json", {"error": f"{type(cleanup_error).__name__}: {cleanup_error}", "original_failure": failure})
                    if failure is None:
                        raise
                else:
                    evidence.update(shutdown_log_evidence(out / "service.log"))
                    evidence["failed"] = bool(evidence["failed"] or evidence["application_shutdown_failed"]
                                              or evidence["worker_shutdown_failed"])
                    write_json(out / "shutdown-after.json", evidence)
                    if evidence["failed"] and failure is None:
                        raise RuntimeError(f"本次服务关闭验收失败: {evidence}")
            finally:
                for signum, handler in shutdown_handlers.items():
                    signal.signal(signum, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT / "docs/npu-910b-migration-20261117/audio-manifest.json"))
    parser.add_argument("--app-root", default=str(ROOT), help="explicit source or protected staging app directory")
    parser.add_argument("--models-root", help="read-only original model directory for protected staging validation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--mode", choices=("short", "steady", "cold-mixed", "calibrate", "long", "full", "failure-4011", "failure-4012"), default="short")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--instances", type=int, choices=(1, 2), default=1,
                        help="shared-card calibration limits Qwen copies to at most two")
    options = parser.parse_args()
    ROOT = Path(options.app_root).resolve()
    if options.concurrency < 1: parser.error("concurrency必须为正数")
    def interrupted(*_):
        # 第二次信号也不能打断finally的有界关闭。
        for signum in TERMINATION_SIGNALS:
            signal.signal(signum, signal.SIG_IGN)
        raise KeyboardInterrupt("测试被中止，正在关闭自有服务")
    for signum in TERMINATION_SIGNALS:
        signal.signal(signum, interrupted)
    run(options)
