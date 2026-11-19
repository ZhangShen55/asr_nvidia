"""真实短音频HTTP准入功能验证；默认plan，拒绝探针不进入性能矩阵。"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlparse
import wave

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.benchmark_asr_throughput import (
    OwnedService, service_command, write_json, warmup, source_identity, source_differences,
)

PRESERVED_PID = 1601397


def capacity(status):
    # 实际顶层准入字段；segmentPipeline在resourceDiagnostics内，不参与准入判定。
    total = status["audioConcurrency"]["active_total"]
    enhanced = status["spkEmoConcurrency"]["active"]
    if type(total) is not int or type(enhanced) is not int or not 0 <= enhanced <= total <= 8:
        raise ValueError("准入status计数非法")
    return total, enhanced


def classify_probe(row, expected):
    if row.get("http_status") != 200:
        return "failed"
    code = row.get("payload", {}).get("code")
    if code == expected:
        return "passed"
    if code in (None, 0, 4013, 4017):
        # status与随后的POST之间没有原子屏障；容量释放后的成功不能假报为通过。
        return "not_observed"
    return "failed"


async def conduct_checks(request, status, record, *, observation_s=10.0, poll_s=.05,
                         clock=time.monotonic, checkpoint=lambda: None):
    all_tasks = []
    record.setdefault("requests", [])
    record.setdefault("status_samples", [])
    record.setdefault("stages", [])

    async def one(label, kind):
        row = {"id": label, "kind": kind, "started_monotonic": clock(), "status": "running"}
        record["requests"].append(row)
        try:
            response = await request(label, kind)
            row.update(response, status="completed")
        except asyncio.CancelledError:
            row["status"] = "cancelled"
            raise
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            row["ended_monotonic"] = clock()
            checkpoint()
        return row

    def launch(specs):
        gate = asyncio.Event()
        async def gated(label, kind):
            await gate.wait()
            return await one(label, kind)
        tasks = [asyncio.create_task(gated(label, kind)) for label, kind in specs]
        all_tasks.extend(tasks)
        gate.set()
        return tasks

    async def observe(target, tasks=()):
        end = clock() + observation_s
        last = None
        while clock() < end:
            last = await status()
            measured = capacity(last)
            if len(record["status_samples"]) < 4096:
                record["status_samples"].append({"monotonic": clock(), "counts": list(measured)})
            record["latest_status"] = last
            if measured == target:
                return {"observed": True, "monotonic": clock(), "raw": last}
            if tasks and all(task.done() for task in tasks):
                break
            await asyncio.sleep(poll_s)
        return {"observed": False, "monotonic": clock(), "raw": last}

    try:
        initial = await observe((0, 0))
        if not initial["observed"]:
            raise RuntimeError("预热后owned服务未回到零准入，拒绝追加测试")
        for label, base, target, probes in [
            ("B4", ["B"] * 4, (4, 4), [("fifth_B", "B", 4017)]),
            ("total8", ["B"] * 4 + ["A"] * 4, (8, 4),
             [("ninth_A", "A", 4013), ("ninth_B", "B", 4013)]),
        ]:
            stage = {"id": label, "expected_active": list(target), "status": "running"}
            record["stages"].append(stage)
            bases = launch([(f"{label}-base-{index}", kind) for index, kind in enumerate(base)])
            stage["capacity_observation"] = await observe(target, bases)
            checks = []
            if stage["capacity_observation"]["observed"]:
                probe_tasks = launch([(f"{label}-{name}", kind) for name, kind, _ in probes])
                rows = await asyncio.gather(*probe_tasks)
                checks = [{"id": row["id"], "expected_code": expected,
                           "status": classify_probe(row, expected), "raw_request": row}
                          for row, (_, _, expected) in zip(rows, probes)]
                # 额外快照只供诊断；失败不能取消已经获准的base请求。
                # target/drain的关键observe仍保留原失败语义，不以此快照代替。
                try:
                    stage["after_probes_status"] = await status()
                    capacity(stage["after_probes_status"])
                    stage["after_probes_status_observation"] = {"status": "observed", "gate": False}
                except Exception as exc:
                    stage["after_probes_status"] = None
                    stage["after_probes_status_observation"] = {
                        "status": "unknown", "gate": False,
                        "error_type": type(exc).__name__, "error": str(exc),
                    }
            stage["probes"] = checks
            base_rows = await asyncio.gather(*bases)
            base_ok = all(row.get("http_status") == 200 and row.get("payload", {}).get("code") in (None, 0)
                          and isinstance(row.get("payload", {}).get("segments"), list) for row in base_rows)
            drained = await observe((0, 0))
            stage["drained"] = drained
            stage["base_requests_successful"] = base_ok
            if not base_ok or not drained["observed"] or any(item["status"] == "failed" for item in checks):
                stage["status"] = "failed"
            elif not checks or any(item["status"] == "not_observed" for item in checks):
                stage["status"] = "not_observed"
            else:
                stage["status"] = "passed"
            checkpoint()
            if stage["status"] == "failed":
                break
        states = [item["status"] for item in record["stages"]]
        return "passed" if len(states) == 2 and all(s == "passed" for s in states) else "failed" if "failed" in states else "not_observed"
    finally:
        for task in all_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)
        checkpoint()


async def close_owned(owner, timeout_s=240.0):
    if PRESERVED_PID in getattr(owner, "identities", {}):
        raise RuntimeError("保留PID不能属于owned关闭集合")
    cleanup = asyncio.create_task(owner.close(deadline_s=timeout_s, observe_s=5))
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            continue
    return cleanup.result()


def preserved_identity():
    import psutil
    try:
        process = psutil.Process(PRESERVED_PID)
        return {"pid": process.pid, "create_time": process.create_time(), "present": True}
    except psutil.NoSuchProcess:
        return {"pid": PRESERVED_PID, "present": False}


def plan(args):
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    config = tomllib.loads(args.config.read_text())
    settings = config.get("qwen_asr", {})
    if settings.get("audio_concurrency", 8) != 8 or settings.get("audio_spk_emo_concurrency", 4) != 4:
        raise ValueError("准入功能测试必须总8/B4 profile")
    if settings.get("queue_wait_timeout_s", 0) != 0:
        raise ValueError("必须queue_wait_timeout_s=0，验证即时4013/4017拒绝")
    actual_mode = settings.get("execution_mode", "thread")
    if actual_mode != args.expected_execution_mode:
        raise ValueError("profile execution_mode与显式期望不符，不能误测另一候选")
    with wave.open(str(args.wav), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
            raise ValueError("真实warmup WAV必须16k单声道PCM16")
        seconds = audio.getnframes() / audio.getframerate()
    parsed = urlparse(args.url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost") or not parsed.port:
        raise ValueError("仅允许显式空闲本机HTTP端口")
    return {"execution": args.execute, "config": str(args.config), "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "wav": str(args.wav), "wav_sha256": hashlib.sha256(args.wav.read_bytes()).hexdigest(), "wav_seconds": seconds,
            "url": args.url, "qwen_execution_mode": actual_mode, "expected_capacity": {"total": 8, "B": 4}, "global_test_deadline_s": args.deadline_s,
            "shutdown_timeout_s": args.shutdown_timeout_s, "status_timeout_s": getattr(args, "status_timeout_s", 20.0),
            "preserved_pid": PRESERVED_PID,
            "cases": ["4B active -> fifth B4017", "4B+4A active8 -> ninth A/B both4013"],
            "scope": "functional admission only; excluded from throughput matrix", "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


async def execute(args, info):
    import httpx
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    before = source_identity()
    if not before.get("complete"):
        raise RuntimeError("运行时代码身份不完整，未启动服务")
    record = dict(info, status="running", source_identity_before=before, preserved_before=preserved_identity())
    owner = None
    write_json(output / "admission.json", record)
    def checkpoint(): write_json(output / "admission.json", record)
    async def experiment():
        nonlocal owner
        parsed = urlparse(args.url)
        env = dict(os.environ, CONFIG_PATH=str(args.config), PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get("PYTHONPATH", ""))
        owner = OwnedService(service_command(parsed.port), env, parsed.hostname, parsed.port, output)
        if PRESERVED_PID in owner.identities:
            raise RuntimeError("保留PID意外进入owned集合")
        async with httpx.AsyncClient(base_url=args.url, timeout=None, trust_env=False) as client:
            async def status():
                owner.register()
                started = time.monotonic()
                try:
                    response = await client.get("/get_status", timeout=getattr(args, "status_timeout_s", 20.0))
                    response.raise_for_status()
                    return response.json()
                finally:
                    ended = time.monotonic()
                    timings = record.setdefault("status_http_timings", [])
                    if len(timings) < 4096:
                        timings.append({"started_monotonic_s": started, "ended_monotonic_s": ended,
                            "elapsed_s": ended-started,
                            "scope": "HTTP GET and response JSON; excludes owned PID scan"})
            startup_end = time.monotonic() + 300
            while True:
                if owner.process.poll() is not None:
                    raise RuntimeError("owned服务启动失败")
                try:
                    record["initial_status"] = await status()
                    break
                except httpx.HTTPError:
                    if time.monotonic() >= startup_end:
                        raise TimeoutError("owned服务启动超过300秒")
                    await asyncio.sleep(.2)
            warm_args = SimpleNamespace(url=args.url, warmup_audio=str(args.wav), mode="B", expected_model_instances=8,
                diagnostic_wall_timeout_s=min(420, args.deadline_s), no_progress_s=120, diagnostic_window_s=5, monitor_interval_s=.2)
            record["warmup"] = await warmup(warm_args, output)
            checkpoint()
            if not record["warmup"]["complete"]:
                raise RuntimeError("预热未通过，不追加容量探针")
            wav = args.wav.read_bytes()
            async def request(label, kind):
                enhanced = str(kind == "B").lower()
                response = await client.post("/v3/seacraft_asr", files={"audioFile": (label+".wav", wav, "audio/wav")},
                    data={"showSpk": enhanced, "showEmotion": enhanced, "showRoleIdentify": "false"})
                payload = response.json()
                return {"http_status": response.status_code, "payload": payload}
            return await conduct_checks(request, status, record, observation_s=args.observation_s,
                                        poll_s=args.poll_s, checkpoint=checkpoint)
    try:
        record["status"] = await asyncio.wait_for(experiment(), timeout=args.deadline_s)
    except asyncio.TimeoutError:
        record.update(status="diagnostic_aborted", error="global test deadline reached; cleanup has separate bounded deadline")
    except asyncio.CancelledError:
        record["status"] = "cancelled"
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if owner is not None:
            try:
                record["shutdown"] = await close_owned(owner, args.shutdown_timeout_s)
                if not record["shutdown"].get("passed"):
                    record["status"] = "cleanup_failed"
            except Exception as exc:
                record.update(status="cleanup_failed", shutdown={"passed": False, "error": f"{type(exc).__name__}: {exc}"})
        after = source_identity()
        record["source_identity_after"] = after
        record["source_changes"] = source_differences(before, after)
        record["preserved_after"] = preserved_identity()
        record["preserved_pid_unchanged"] = record["preserved_before"] == record["preserved_after"]
        record["config_unchanged"] = info["config_sha256"] == hashlib.sha256(args.config.read_bytes()).hexdigest()
        if record["source_changes"] or not after.get("complete") or not record["config_unchanged"]:
            record["status"] = "source_changed"
        if not record["preserved_pid_unchanged"]:
            record["status"] = "preserved_pid_changed"
        record["complete"] = record["status"] == "passed"
        checkpoint()
    return record


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18084")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-execution-mode", choices=("thread", "process"), default="process")
    for name, default in (("deadline-s", 420), ("shutdown-timeout-s", 240), ("observation-s", 10), ("poll-s", .05), ("status-timeout-s", 20)):
        parser.add_argument("--"+name, type=float, default=default)
    args = parser.parse_args(argv)
    for name in ("config", "wav", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if any(not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0 for name in
           ("deadline_s", "shutdown_timeout_s", "observation_s", "poll_s", "status_timeout_s")):
        parser.error("时间参数必须有限正数")
    if args.status_timeout_s > 60:
        parser.error("状态读取超时不可超过60秒")
    if args.deadline_s > 420 or args.shutdown_timeout_s < 240:
        parser.error("测试截止不可超过420秒；关闭期限至少240秒，独立于测试截止")
    return args


def main(argv=None):
    args = arguments(argv)
    info = plan(args)
    if not args.execute:
        print(json.dumps(info, ensure_ascii=True, indent=2))
        return 0
    async def run():
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        for sig in (signal.SIGINT, signal.SIGTERM): loop.add_signal_handler(sig, task.cancel)
        return await execute(args, info)
    record = asyncio.run(run())
    print(json.dumps({"output": str(args.output), "status": record["status"], "complete": record["complete"]}))
    return 0 if record["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
