"""同八完整文件的v3并发测量，七分钟诊断与owned服务统一收尾。"""
from __future__ import annotations

import argparse
import asyncio
import base64
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.performance_watchdog import PerformanceWatchdog, percentile


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # 历史源文件名可能包含surrogateescape，JSON转义保留原字节往返能力。
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n")
    temporary.replace(path)


def source_identity(root=ROOT, *, max_files=2048, max_bytes=512*1024*1024):
    """仅散列运行时代码文件，不读正文入报告、不遍历权重/音频/构建C产物。"""
    import importlib.machinery
    root=Path(root).resolve()
    files={};preferred={};errors=[];total=0
    candidates=[]
    for folder in ("core","api","utils","entity"):
        base=root/folder
        if base.is_dir():
            candidates.extend(path for path in base.rglob("*") if path.suffix in (".py",".so")
                              and not set(path.relative_to(base).parts[:-1]) & {"__pycache__","cache","caches",".git","model","test","tests"})
    for module in ("main","model_status_runtime","telemetry_runtime"):
        candidates.extend(path for path in root.glob(module+".*") if path.suffix in (".py",".so") and path.name.split(".")[0]==module)
    for path in sorted(set(candidates)):
        relative=str(path.relative_to(root))
        if path.is_symlink():
            errors.append({"path":relative,"reason":"symlink_source_not_hashed"});continue
        if not path.is_file():
            continue
        try:
            before=path.stat()
            if len(files)>=max_files or total+before.st_size>max_bytes:
                errors.append({"path":relative,"reason":"bounded_manifest_limit"});break
            digest=hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda:source.read(1024*1024),b""):
                    digest.update(chunk)
            after=path.stat()
            if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
                errors.append({"path":relative,"reason":"changed_while_hashing"})
            files[relative]={"sha256":digest.hexdigest(),"bytes":after.st_size,"kind":path.suffix[1:]}
            total+=after.st_size
            module=str(Path(relative).parent/path.name.split(".")[0])
            loadable_extension=any(path.name.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)
            previous=preferred.get(module)
            # 保护stage通常只剩.so；两者并存按扩展模块优先记录候选，绝不伪造缺失.py。
            if previous is None or (loadable_extension and Path(previous).suffix!=".so"):
                preferred[module]=relative
        except OSError as exc:
            errors.append({"path":relative,"reason":type(exc).__name__})
    # 实际入口命令为python main.py，即便同目录存在main.so也不改写入口事实。
    if "main.py" in files:
        preferred["main"]="main.py"
    head=None;head_reason="not_a_git_root"
    try:
        git=subprocess.run(["git","-C",str(root),"rev-parse","--show-toplevel","HEAD"],capture_output=True,text=True,timeout=5)
        lines=git.stdout.strip().splitlines()
        if git.returncode==0 and len(lines)==2 and Path(lines[0]).resolve()==root:
            head=lines[1];head_reason=None
    except (OSError,subprocess.TimeoutExpired) as exc:
        head_reason=type(exc).__name__
    return {"root":str(root),"git_head":head,"git_head_status":"known" if head else "unknown",
            "git_head_reason":head_reason,"python_executable":sys.executable,"python_version":sys.version,
            "files":files,"preferred_module_files":preferred,
            "preferred_module_scope":"existing extension/source import candidates, not an in-memory import trace; main.py is the explicit entry",
            "file_count":len(files),"hashed_bytes":total,"complete":bool(files) and not errors,"errors":errors}


def source_differences(before,after):
    changes=[]
    original=before.get("files",{});current=after.get("files",{})
    for path in sorted(set(original)|set(current)):
        old=original.get(path,{}).get("sha256");new=current.get(path,{}).get("sha256")
        if old!=new:
            changes.append({"path":path,"before_sha256":old,"after_sha256":new,
                            "change":"added" if old is None else "removed" if new is None else "modified"})
    for key in ("root","python_executable","python_version"):
        if before.get(key)!=after.get(key):
            changes.append({"identity_field":key,"before":before.get(key),"after":after.get(key)})
    return changes


def finish_source_identity(output, before, *, collector=source_identity):
    output=Path(output)
    after=collector(Path(before["root"]))
    changes=source_differences(before,after)
    invalid=bool(changes) or not before.get("complete") or not after.get("complete")
    path=output/"run.json"
    record=json.loads(path.read_text())
    fields={"source_identity_after":after,"source_changed_during_round":bool(changes),
            "source_changes":changes,"source_identity_complete":before.get("complete") and after.get("complete"),
            "git_head_changed_during_round":before.get("git_head")!=after.get("git_head")}
    record.update(fields);write_json(path,record)
    summary_path=output/"summary.json"
    if summary_path.exists():
        summary=json.loads(summary_path.read_text());summary.update(fields)
        if invalid:
            if summary.get("complete"):
                summary["status"]="source_changed" if changes else "source_identity_incomplete"
            summary["complete"]=False
        write_json(summary_path,summary)
    return invalid


def load_samples(path):
    """兼容资产审计清单和显式选定的八条samples；校验完整文件SHA。"""
    manifest = json.loads(Path(path).read_text())
    source = manifest["samples"] if isinstance(manifest, dict) else manifest
    if len(source) != 8:
        raise ValueError("必须显式提供八条完整音频，不自动挑选或截断清单")
    result, hashes, identifiers = [], set(), set()
    for index, row in enumerate(source):
        if row.get("path_bytes_b64"):
            filename = os.fsdecode(base64.b64decode(row["path_bytes_b64"], validate=True))
        else:
            filename = os.fsdecode(bytes.fromhex(row["path_hex"])) if row.get("path_hex") else row["path"]
        filename = str(Path(filename).resolve())
        with open(filename, "rb") as stream:
            sha = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(chunk)
        actual = sha.hexdigest()
        if row.get("sha256") != actual or actual in hashes:
            raise ValueError("源SHA不匹配或不是八条独立音频")
        seconds = row.get("audio_seconds", row.get("duration_seconds", row.get("media", {}).get("duration_seconds")))
        if not isinstance(seconds, (int, float)) or not 0 < seconds < float("inf"):
            raise ValueError("每条需有效audio_seconds或media.duration_seconds")
        sample_id = row.get("sample_id", row.get("id", f"audio-{index}"))
        if sample_id in identifiers:
            raise ValueError("sample_id必须唯一")
        hashes.add(actual); identifiers.add(sample_id)
        result.append(dict(row, path=filename, sha256=actual, audio_seconds=seconds, sample_id=sample_id))
    return result


def fields_for(mode, index):
    enhanced = mode == "B" or (mode == "mixed" and index < 4) or (mode == "mixed-swap" and index >= 4)
    return {"showSpk": str(enhanced).lower(), "showEmotion": str(enhanced).lower(), "showRoleIdentify": "false"}


def listening(host, port):
    with socket.socket() as connection:
        connection.settimeout(.2)
        return connection.connect_ex((host, port)) == 0


def service_command(port, request_drain_grace_s=180):
    # 只改变本轮owned测试HTTP请求drain窗口；main/core的缺省与模型收尾不变。
    if type(request_drain_grace_s) is not int or not 1 <= request_drain_grace_s <= 180:
        raise ValueError("owned request drain grace必须是1～180秒整数")
    # main.py负责TERM重放及设备收尾，不能旁路为uvicorn模块入口。
    return [sys.executable, "main.py", "--host", "127.0.0.1", "--port", str(port),
            "--workers", "1", "--timeout-graceful-shutdown", str(request_drain_grace_s)]


def owned_cache_files(cache_dir, *, max_entries=4096):
    """只读独占音频cache；不扫描TMPDIR其它文件，不跟随符号链接。"""
    root = Path(cache_dir)
    remaining = []
    try:
        root.lstat()
    except FileNotFoundError:
        return remaining, None
    except OSError as exc:
        return remaining, f"{type(exc).__name__}: {exc}"
    try:
        if root.is_symlink() or not root.is_dir():
            return remaining, "owned cache root is not a regular directory"
    except OSError as exc:
        return remaining, f"{type(exc).__name__}: {exc}"
    pending, visited = [root], 0
    try:
        while pending:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    visited += 1
                    if visited > max_entries:
                        return sorted(remaining), "owned cache scan exceeded bounded entry limit"
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    else:
                        remaining.append(str(Path(entry.path).relative_to(root)))
    except OSError as exc:
        return sorted(remaining), f"{type(exc).__name__}: {exc}"
    return sorted(remaining), None


class OwnedService:
    """仅操作Popen确定归属、且PID创建时间仍匹配的资源。"""
    def __init__(self, command, env, host, port, output, *, expected_exit_codes=(0,), cwd=None):
        import psutil
        if host not in ("localhost", "127.0.0.1") or listening(host, port):
            raise RuntimeError("owned启动仅支持空闲本机端口，不接管既有服务")
        self.host, self.port, self.output = host, port, Path(output)
        # 正式源码入口可配独立工作目录，防止相对日志路径轮转历史项目日志。
        self.cwd = Path(cwd).resolve() if cwd is not None else ROOT
        # 生产main.py正常TERM必须exit0；裸Python子进程测试可显式采用不同策略。
        self.expected_exit_codes = tuple(expected_exit_codes)
        # 每服务独占短临时根；不继承外部TMPDIR，不删除旧缓存，也不污染父环境。
        self.temp_dir = tempfile.mkdtemp(prefix="asr-svc-", dir="/tmp")
        self.cache_dir = str(Path(self.temp_dir) / "seacraft_asr")
        child_env = dict(env, TMPDIR=self.temp_dir)
        self.log = (self.output / "service.log").open("w")
        try:
            self.process = subprocess.Popen(command, cwd=self.cwd, env=child_env, stdout=self.log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            self.log.close()
            raise
        self.identities = {self.process.pid: psutil.Process(self.process.pid).create_time()}
        self.register()

    def register(self):
        import psutil
        parent = self.identities.get(self.process.pid)
        try:
            proc = psutil.Process(self.process.pid)
            if proc.create_time() == parent:
                for child in proc.children(recursive=True):
                    self.identities[child.pid] = child.create_time()
        except psutil.NoSuchProcess:
            pass
        # spawn worker可能在父进程已退出后才被采样。独立session是Popen创建，
        # 同session/进程组且创建不早于父进程的成员必属该生命周期，再记录身份。
        for member in psutil.process_iter(attrs=["pid", "create_time"]):
            try:
                if (member.info["create_time"] >= parent
                        and os.getsid(member.pid) == self.process.pid
                        and os.getpgid(member.pid) == self.process.pid):
                    self.identities[member.pid] = member.info["create_time"]
            except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError, PermissionError):
                continue
        write_json(self.output / "ownership.json", {"pid": self.process.pid,
                   "identities": self.identities, "host": self.host, "port": self.port,
                   "temp_dir": self.temp_dir, "cache_dir": self.cache_dir,
                   "temp_dir_scope": "owned service TMPDIR; retained after shutdown, no recursive deletion",
                   "cwd": str(self.cwd)})

    def remaining(self):
        import psutil
        result = []
        for pid, created in self.identities.items():
            try:
                proc = psutil.Process(pid)
                if proc.create_time() == created and proc.status() != psutil.STATUS_ZOMBIE:
                    result.append(proc)
            except psutil.NoSuchProcess:
                pass
        return result

    async def close(self, deadline_s=210, observe_s=5):
        import psutil
        term_sent = None
        parent_exit_observed = None
        def poll_exit():
            nonlocal parent_exit_observed
            code = self.process.poll()
            if code is not None and parent_exit_observed is None:
                parent_exit_observed = time.monotonic()
            return code
        self.register()
        initial = list(self.identities.items())
        forced = False
        parent = next((p for p in self.remaining() if p.pid == self.process.pid), None)
        orphan_cleanup = parent is None and bool(self.remaining())
        if parent:
            try:
                parent.terminate()
                term_sent = time.monotonic()
            except psutil.NoSuchProcess:
                pass
        if orphan_cleanup:
            for proc in self.remaining():
                try:
                    proc.terminate()
                except psutil.NoSuchProcess:
                    pass
        deadline = time.monotonic() + deadline_s
        while self.remaining() and time.monotonic() < deadline:
            self.register()
            poll_exit()
            await asyncio.sleep(.1)
        if self.remaining():
            forced = True
            for proc in self.remaining():
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
            await asyncio.sleep(.2)
        poll_exit()
        # 连续观察而非只看一次无监听，避免接口短暂释放后又自动拉起。
        observations = []
        end = time.monotonic() + observe_s
        while True:
            observations.append(listening(self.host, self.port))
            if time.monotonic() >= end:
                break
            await asyncio.sleep(min(.2, max(0, end-time.monotonic())))
        remaining = [p.pid for p in self.remaining()]
        exit_code = poll_exit()
        exit_code_ok = exit_code in self.expected_exit_codes
        cache_files, cache_error = owned_cache_files(self.cache_dir)
        result = {"owned": True, "identities": initial, "forced": forced,
                  "temp_dir": self.temp_dir, "cache_dir": self.cache_dir,
                  "temp_dir_retained": Path(self.temp_dir).is_dir(),
                  "remaining_cache_files": cache_files, "cache_scan_error": cache_error,
                  "cache_scan_scope": "owned service audio cache only, after process/port cleanup; read-only, no deletion",
                  "exit_code": exit_code, "expected_exit_codes": self.expected_exit_codes,
                  "exit_code_ok": exit_code_ok, "remaining_pids": remaining,
                  "port_observations": observations, "observe_s": observe_s,
                  "orphan_cleanup": orphan_cleanup,
                  "term_sent_monotonic": term_sent,
                  "parent_exit_observed_monotonic": parent_exit_observed,
                  "term_to_parent_exit_s": (parent_exit_observed-term_sent
                                            if term_sent is not None and parent_exit_observed is not None else None),
                  "exit_timing_scope": "parent process poll observation, not device/kernel time; None when no TERM was sent",
                  "passed": exit_code_ok and not forced and not orphan_cleanup and not remaining and not any(observations)
                            and not cache_files and cache_error is None}
        self.log.close()
        write_json(self.output / "shutdown.json", result)
        return result


def progress_token(status):
    """仅阶段完成计数，不使用动态时延/CPU/HBM，避免伪进展。"""
    tokens = []
    def visit(value, prefix=""):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("completed", "completed_segments", "completed_batches", "finished", "processed",
                           "segments_done", "qwen_done", "spk_done", "emo_done", "cut_done") and type(item) in (int, float):
                    tokens.append((prefix+key, item))
                elif isinstance(item, (dict, list)):
                    visit(item, prefix+key+".")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, prefix+str(index)+".")
    visit(status)
    return tuple(tokens)


class WarmupProgress:
    """仅预热累计已验证加载完成；资源/出生/loaded抖动不重置无进展计时。"""
    def __init__(self):
        self._ready = set()
        self._native_completed = {}

    @staticmethod
    def _birth(identity):
        if not isinstance(identity, dict) or type(identity.get("pid")) is not int or identity["pid"] <= 0:
            return None
        created, ticks = identity.get("create_time"), identity.get("start_ticks")
        if type(created) not in (int, float) or not math.isfinite(created) or created <= 0:
            return None
        if isinstance(ticks, bool) or not isinstance(ticks, (str, int)) or not str(ticks).isdigit() or int(ticks) <= 0:
            return None
        return identity["pid"], float(created), str(int(ticks))

    def _native(self, name, count):
        if type(count) is int and count > 0:
            self._native_completed[name] = max(self._native_completed.get(name, 0), count)

    def sample(self, status):
        for name, model in status.get("localModels", {}).items():
            if not isinstance(model, dict):
                continue
            if model.get("execution_mode", "thread") != "process":
                self._native("local." + name, model.get("instances"))
                continue
            for worker in model.get("model_workers", []):
                if not isinstance(worker, dict):
                    continue
                ready = worker.get("ready")
                identity = worker.get("worker_identity")
                birth = self._birth(identity)
                if (birth is not None and isinstance(ready, dict) and ready.get("kind") == "READY"
                        and ready.get("identity") == identity and isinstance(ready.get("launch"), str)
                        and ready["launch"]):
                    self._ready.add(("local." + name, ready["launch"], *birth))
        self._native("vad", status.get("vadPool", {}).get("loaded_instances"))
        qwen = status.get("qwenModel", {})
        instances = qwen.get("instances", [])
        if isinstance(instances, list):
            if qwen.get("execution_mode", "thread") == "process":
                for instance in instances:
                    if not isinstance(instance, dict) or instance.get("loaded") is not True:
                        continue
                    worker = instance.get("model_worker", {})
                    if not isinstance(worker, dict):
                        continue
                    birth = self._birth({"pid": worker.get("worker_pid"),
                                         "create_time": worker.get("worker_create_time"),
                                         "start_ticks": worker.get("worker_start_ticks")})
                    if (birth is not None and worker.get("ready_verified") is True
                            and isinstance(worker.get("launch_id"), str) and worker["launch_id"]):
                        self._ready.add(("qwen", worker["launch_id"], *birth))
            else:
                self._native("qwen", sum(isinstance(row, dict) and row.get("loaded") is True
                                         and not row.get("failure") for row in instances))
        return (progress_token(status), tuple(sorted(self._ready)), tuple(sorted(self._native_completed.items())))


def summarize(records, batch_start, batch_end, aborted):
    succeeded = [r for r in records if r["status"] == "completed"]
    values = [r["elapsed_s"] for r in succeeded]
    complete = len(succeeded) == len(records) and not aborted
    elapsed = batch_end - batch_start
    strict = [r for r in succeeded if r.get("processing_status", "complete") == "complete"
              and not r.get("warning_count", 0)]
    strict_values = [r["elapsed_s"] for r in strict]
    warned = [r for r in succeeded if r.get("processing_status") == "completed_with_warnings"]
    return {"status": "diagnostic_aborted" if aborted else
                     "completed_with_warnings" if complete and warned else "completed" if complete else "incomplete",
            "expected": len(records), "successful": len(strict), "complete": complete,
            "processed_completed": len(succeeded), "completed_with_warnings": len(warned),
            "strict_success_rate": len(strict)/len(records) if records else None,
            "processing_completion_rate": len(succeeded)/len(records) if records else None,
            "requests_with_usable_text": sum(bool(r.get("has_usable_text")) for r in succeeded),
            "warning_segments": sum(r.get("warning_count", 0) for r in succeeded),
            "unrecoverable_segments": sum(r.get("unrecoverable_segment_count", 0) for r in succeeded),
            "completion_subset_latency": {"n": len(values), "p50_s": percentile(values, .5),
                "p95_s": percentile(values, .95), "scope": "all processed completions including explicit warnings"},
            "batch_started_monotonic": batch_start, "batch_ended_monotonic": batch_end,
            "batch_elapsed_s": elapsed, "success_subset_latency": {"n": len(strict_values),
            "p50_s": percentile(strict_values, .5), "p95_s": percentile(strict_values, .95),
            "method": "linear interpolation; incomplete rounds are censored and not acceptance"},
            "source_audio_seconds": sum(r["audio_seconds"] for r in records),
            "source_audio_seconds_per_wall_second": sum(r["audio_seconds"] for r in records)/elapsed if complete and elapsed else None,
            "records": records}


async def run_round(args, samples, output, owner=None, *, request_fn=None, status_fn=None):
    """注入request/status便于无硬件测试；同时在途窗口，而非按波等待。"""
    import httpx
    output = Path(output)
    watchdog = PerformanceWatchdog(args.diagnostic_wall_timeout_s, args.no_progress_s)
    records = [dict(sample_id=s["sample_id"], sha256=s["sha256"], audio_seconds=s["audio_seconds"],
                    status="not_submitted", fields=fields_for(args.mode, i)) for i, s in enumerate(samples)]
    active, active_start = {}, {}
    started, started_utc = time.monotonic(), utc_now()
    trigger, failure = None, None
    async with httpx.AsyncClient(base_url=args.url, timeout=None, trust_env=False) as client:
        async def status_call():
            if status_fn:
                return await status_fn()
            response = await client.get("/get_status", timeout=5)
            response.raise_for_status()
            return response.json()

        async def one(index):
            row = records[index]
            row.update(status="running", started_monotonic=time.monotonic(), started_utc=utc_now())
            active_start[row["sample_id"]] = row["started_monotonic"]
            try:
                if request_fn:
                    payload = await request_fn(samples[index], row["fields"])
                    row["http_status"] = 200
                else:
                    with open(samples[index]["path"], "rb") as audio:
                        response = await client.post("/v3/seacraft_asr", files={"audioFile": ("audio-"+row["sha256"][:20]+".wav", audio)}, data=row["fields"])
                    row["http_status"] = response.status_code
                    response.raise_for_status()
                    payload = response.json()
                row["code"] = payload.get("code")
                row["status"] = "completed" if row["code"] in (None, 0) and isinstance(payload.get("segments"), list) else "failed"
                if row["status"] == "completed":
                    warnings = payload.get("segment_warnings", [])
                    state = payload.get("processing_status", "complete")
                    if state not in {"complete", "completed_with_warnings"} or not isinstance(warnings, list):
                        raise ValueError("invalid processing completion metadata")
                    if (state == "complete" and warnings) or (state == "completed_with_warnings" and not warnings):
                        raise ValueError("inconsistent processing warning metadata")
                    row.update(processing_status=state, warning_count=len(warnings),
                        unrecoverable_segment_count=sum(w.get("action") in {"discard_unrecoverable", "unrecoverable"}
                                                        for w in warnings if isinstance(w, dict)),
                        has_usable_text=any(isinstance(x.get("segment_text"), str) and bool(x["segment_text"].strip())
                                            for x in payload["segments"] if isinstance(x, dict)))
                write_json(output / f"response-{index}.json", payload)
            except asyncio.CancelledError:
                row["status"] = "cancelled"
                raise
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            finally:
                row.update(ended_monotonic=time.monotonic(), ended_utc=utc_now())
                row["elapsed_s"] = row["ended_monotonic"] - row["started_monotonic"]
                row["rtf"] = row["elapsed_s"] / row["audio_seconds"]
                active_start.pop(row["sample_id"], None)
                write_json(output / f"request-{index}.json", row)

        pending = iter(range(len(samples)))
        exhausted = False
        latest_status = {}
        status_task = None
        try:
            while True:
                if owner:
                    owner.register()
                values = [r["elapsed_s"] for r in records if r["status"] == "completed"]
                trigger = watchdog.sample(values, active_start, progress_token(latest_status))
                if trigger:
                    write_json(output / "diagnostic-trigger.json", {**asdict(trigger), "utc": utc_now(), "records": records})
                    # 先停止追加，再有界取证；不等待完整请求自然结束才记录卡点。
                    try:
                        latest_status = await asyncio.wait_for(status_call(), args.diagnostic_window_s)
                    except Exception as exc:
                        latest_status = {"snapshot_error": repr(exc)}
                    write_json(output / "diagnostic-status.json", latest_status)
                    break
                while len(active) < args.concurrency and not exhausted:
                    index = next(pending, None)
                    if index is None:
                        exhausted = True
                        break
                    active[index] = asyncio.create_task(one(index))
                if not active:
                    break
                done, _ = await asyncio.wait(active.values(), timeout=args.monitor_interval_s,
                                              return_when=asyncio.FIRST_COMPLETED)
                for index, task in list(active.items()):
                    if task in done:
                        task.result()
                        del active[index]
                # 状态接口慢/失效不能阻塞一秒watchdog采样。
                if status_task is not None and status_task.done():
                    try:
                        latest_status = status_task.result()
                    except Exception as exc:
                        latest_status = {"snapshot_error": repr(exc)}
                    with (output / "status.jsonl").open("a") as stream:
                        stream.write(json.dumps({"monotonic": time.monotonic(), "utc": utc_now(), "status": latest_status}, ensure_ascii=True)+"\n")
                    status_task = None
                if status_task is None:
                    status_task = asyncio.create_task(status_call())
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if status_task is not None:
                status_task.cancel()
                await asyncio.gather(status_task, return_exceptions=True)
            for task in active.values():
                task.cancel()
            await asyncio.gather(*active.values(), return_exceptions=True)
            ended = time.monotonic()
            summary = summarize(records, started, ended, trigger is not None)
            summary.update(batch_started_utc=started_utc, batch_ended_utc=utc_now(), mode=args.mode,
                           concurrency=args.concurrency, service_ownership="owned" if owner else "external_nonowned",
                           failure=failure, diagnostic_trigger=asdict(trigger) if trigger else None)
            write_json(output / "summary.json", summary)
    return summary


async def warmup(args, output, *, request_fn=None, status_fn=None):
    """独立一次真实片段预热，严禁混入八完整文件的计时与分位数。"""
    import httpx
    source = Path(args.warmup_audio)
    fields = fields_for("A" if args.mode == "A" else "B", 0)
    record = {"kind": "warmup_clip_not_full_audio", "source_path_bytes_b64": base64.b64encode(os.fsencode(source)).decode(),
              "mode": args.mode, "fields": fields, "started_monotonic": time.monotonic(),
              "started_utc": utc_now(), "status": "running", "complete": False}
    watchdog = PerformanceWatchdog(args.diagnostic_wall_timeout_s, args.no_progress_s)
    warmup_progress = WarmupProgress()
    request_task = status_task = None
    latest = {}
    async with httpx.AsyncClient(base_url=args.url, timeout=None, trust_env=False) as client:
        async def request():
            if request_fn:
                return await request_fn(source, fields)
            with source.open("rb") as audio:
                response = await client.post("/v3/seacraft_asr", files={"audioFile": ("warmup.wav", audio)}, data=fields)
            record["http_status"] = response.status_code
            try:
                record["response"] = response.json()
                record["code"] = record["response"].get("code")
            except (ValueError, AttributeError):
                record["response_text"] = response.text
            response.raise_for_status()
            return response.json()
        async def status():
            if status_fn:
                return await status_fn()
            response = await client.get("/get_status", timeout=5)
            response.raise_for_status()
            return response.json()
        try:
            request_task = asyncio.create_task(request())
            while not request_task.done():
                trigger = watchdog.sample([], {"warmup": record["started_monotonic"]}, warmup_progress.sample(latest))
                if trigger:
                    record.update(status="diagnostic_aborted", diagnostic_trigger=asdict(trigger))
                    try:
                        latest = await asyncio.wait_for(status(), args.diagnostic_window_s)
                    except Exception as exc:
                        latest = {"snapshot_error": repr(exc)}
                    record["diagnostic_status"] = latest
                    break
                if status_task is None:
                    status_task = asyncio.create_task(status())
                if status_task.done():
                    try:
                        latest = status_task.result()
                    except Exception as exc:
                        latest = {"snapshot_error": repr(exc)}
                    status_task = None
                await asyncio.wait({request_task}, timeout=args.monitor_interval_s)
            if request_task.done() and record["status"] != "diagnostic_aborted":
                payload = request_task.result()
                record.update(response=payload, code=payload.get("code"))
                elapsed = time.monotonic()-record["started_monotonic"]
                trigger = watchdog.sample([elapsed], {})
                if trigger:
                    record.update(status="diagnostic_aborted", diagnostic_trigger=asdict(trigger), diagnostic_status=latest)
                elif record["code"] in (None, 0) and isinstance(payload.get("segments"), list):
                    try:
                        latest = await asyncio.wait_for(status(), args.diagnostic_window_s)
                    except Exception as exc:
                        latest = {"snapshot_error": repr(exc)}
                    record["final_status"] = latest
                    expected = getattr(args, "expected_model_instances", None)
                    if expected is not None:
                        vad = latest.get("vadPool", {})
                        if vad.get("loaded_instances") != expected or vad.get("failed_instances", 0):
                            raise ValueError("预热后VAD健康容量不符合expected-model-instances")
                        if args.mode != "A":
                            models = latest.get("localModels", {})
                            for name in ("speech_campplus", "emotion2vec"):
                                model = models.get(name, {})
                                if (model.get("configured_instances") != expected or model.get("instances") != expected
                                        or model.get("failed_instances", 0)):
                                    raise ValueError("预热后"+name+"健康容量不符合expected-model-instances")
                    record.update(status="completed", complete=True)
                else:
                    record["status"] = "failed"
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            raise
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            pending = [task for task in (request_task, status_task) if task is not None]
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            record.update(ended_monotonic=time.monotonic(), ended_utc=utc_now())
            record["elapsed_s"] = record["ended_monotonic"]-record["started_monotonic"]
            save_status = latest if record["complete"] else record.get("diagnostic_status", latest)
            record["last_status"] = save_status
            write_json(Path(output)/"warmup.json", record)
    return record


async def execute(args, samples, output):
    owner = None
    summary = None
    try:
        if args.launch_service:
            parsed = urlparse(args.url)
            env = dict(os.environ, CONFIG_PATH=str(Path(args.config).resolve()),
                       PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get("PYTHONPATH", ""))
            request_grace = getattr(args, "owned_request_drain_grace_s", 180)
            command = service_command(parsed.port or 80, request_grace)
            run_path = Path(output) / "run.json"
            if run_path.is_file():
                run_record = json.loads(run_path.read_text())
                run_record["owned_service"] = {
                    "command": command, "request_drain_grace_s": request_grace,
                    "cleanup_deadline_s": args.shutdown_timeout_s, "port_observe_s": args.observe_s,
                    "scope": "test-owned HTTP request drain only; model DONE/lifespan cleanup remains separate",
                }
                write_json(run_path, run_record)
            owner = OwnedService(command, env, parsed.hostname, parsed.port or 80, output)
            import httpx
            deadline = time.monotonic()+args.startup_timeout_s
            async with httpx.AsyncClient(base_url=args.url, trust_env=False, timeout=2) as client:
                while True:
                    owner.register()
                    if owner.process.poll() is not None:
                        raise RuntimeError("owned服务启动失败")
                    try:
                        response = await client.get("/get_status")
                        if response.status_code == 200:
                            write_json(output / "initial-status.json", response.json())
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("服务启动超过有界期限")
                    await asyncio.sleep(1)
        warmup_record = None
        if getattr(args, "warmup_audio", None):
            warmup_record = await warmup(args, output)
            if not warmup_record["complete"]:
                summary = {"status": "warmup_"+warmup_record["status"], "complete": False,
                           "expected": len(samples), "successful": 0, "records": [],
                           "formal_round_started": False, "temperature": "warmup_failed",
                           "warmup": warmup_record}
                write_json(Path(output)/"summary.json", summary)
                return summary
        summary = await run_round(args, samples, output, owner)
        summary.update(temperature="warmup_completed" if warmup_record else "unknown/cold_possible",
                       formal_round_started=True, warmup=warmup_record)
        write_json(Path(output)/"summary.json", summary)
        return summary
    finally:
        if owner:
            # shield清理任务，重复中断也必须等owned资源收尾。
            cleanup = asyncio.create_task(owner.close(args.shutdown_timeout_s, args.observe_s))
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
            shutdown = cleanup.result()
            if summary is not None:
                summary["shutdown"] = shutdown
                if not shutdown["passed"]:
                    summary["complete"] = False
                    summary["status"] = "cleanup_failed"
                write_json(output / "summary.json", summary)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18083")
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--mode", choices=("A", "B", "mixed", "mixed-swap"), required=True)
    parser.add_argument("--launch-service", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--owned-request-drain-grace-s", type=int, default=180,
                        help="owned测试HTTP请求drain秒数(1～180)；不改模型/业务超时或cleanup期限")
    parser.add_argument("--warmup-audio", help="独立真实短片路径；预热不计入八文件正式轮次")
    parser.add_argument("--expected-model-instances", type=int, help="可选预热后健康容量校验；A只校验VAD，不要求增强加载")
    for name, default in (("diagnostic-wall-timeout-s",420), ("monitor-interval-s",1),
                          ("no-progress-s",120), ("diagnostic-window-s",5),
                          ("shutdown-timeout-s",210), ("startup-timeout-s",300), ("observe-s",5)):
        parser.add_argument("--"+name, type=float, default=default)
    args = parser.parse_args(argv)
    import math
    for name in ("diagnostic_wall_timeout_s", "monitor_interval_s", "no_progress_s", "diagnostic_window_s", "shutdown_timeout_s", "startup_timeout_s", "observe_s"):
        if not math.isfinite(getattr(args,name)) or getattr(args,name) <= 0:
            parser.error(f"{name}必须为有限正值")
    if not 1 <= args.owned_request_drain_grace_s <= 180:
        parser.error("owned request drain grace必须是1～180秒整数")
    if args.observe_s < 5 or args.diagnostic_wall_timeout_s > 420:
        parser.error("不得放宽420秒门槛或缩短5秒端口观察")
    if args.launch_service and not args.config:
        parser.error("owned服务启动必须指定--config")
    if args.launch_service and args.shutdown_timeout_s < 210:
        parser.error("owned关闭期限至少210秒以保留真实资源收尾余量，独立于HTTP request drain grace")
    if args.expected_model_instances is not None and (args.expected_model_instances < 1 or not args.warmup_audio):
        parser.error("expected-model-instances需正整数并同时指定warmup-audio")
    return args


def main():
    args = parse_args()
    samples = load_samples(args.manifest)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity=source_identity()
    write_json(output / "run.json", {"arguments": vars(args), "samples": samples, "utc": utc_now(),"source_identity":identity})
    if args.config:
        import shutil
        shutil.copy2(args.config, output / "config.toml")
    async def run():
        task = asyncio.current_task()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, task.cancel)
        return await execute(args, samples, output)
    exit_code=2
    try:
        if not identity["complete"]:
            raise RuntimeError("实际运行时代码身份未完整记录，拒绝启动测试服务")
        summary = asyncio.run(run())
        exit_code=0 if summary["complete"] else 2
    except BaseException as exc:
        write_json(output / "runner-error.json", {"error": repr(exc), "utc": utc_now()})
    finally:
        if finish_source_identity(output,identity):
            exit_code=2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
