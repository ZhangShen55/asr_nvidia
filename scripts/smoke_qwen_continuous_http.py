#!/usr/bin/env python3
"""Real v3 short-audio smoke with owned service and diagnostic-only shutdown hook.

Source CANN before running. This script does not bypass main.py preflight or
change the environment installation. --help/import never starts a service.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.benchmark_asr_throughput import OwnedService, fields_for, utc_now, write_json


def same_owned_parent(owner):
    import psutil
    try:
        proc = psutil.Process(owner.process.pid)
        return proc if proc.create_time() == owner.identities.get(proc.pid) and proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE else None
    except psutil.NoSuchProcess:
        return None


async def diagnose_slow_close(owner, output, close_task):
    # OwnedService sends TERM at close entry; do not shorten its 210-second grace.
    await asyncio.sleep(20)
    if close_task.done():
        return
    proc = same_owned_parent(owner)
    if proc is None:
        return
    record = {"utc": utc_now(), "monotonic": time.monotonic(), "pid": proc.pid,
              "create_time": proc.create_time(), "action": "SIGUSR2 traceback only; no kill", "tasks": []}
    try:
        if same_owned_parent(owner) is not None:
            os.kill(proc.pid, signal.SIGUSR2)
            record["signal_sent"] = True
    except (ProcessLookupError, PermissionError) as exc:
        record["signal_error"] = repr(exc)
    for task in sorted(Path(f"/proc/{proc.pid}/task").glob("*"))[:256]:
        row = {"tid": task.name}
        for name in ("comm", "wchan", "stack"):
            try:
                row[name] = (task/name).read_text()[:8192]
            except OSError as exc:
                row[name+"_error"] = str(exc)
        record["tasks"].append(row)
    write_json(output/"shutdown-diagnostics.json", record)


def diagnostic_hook(output):
    hook = Path(tempfile.mkdtemp(prefix="qc-http-hook-", dir="/tmp"))
    # Child stderr is the OwnedService service.log; no additional shared logger.
    (hook/"sitecustomize.py").write_text(
        "import faulthandler,signal\n"
        "faulthandler.enable(all_threads=True)\n"
        "faulthandler.register(signal.SIGUSR2, all_threads=True, chain=False)\n")
    write_json(output/"diagnostic-hook.json", {"directory": str(hook), "retained": True,
               "scope": "child-only PYTHONPATH startup hook, SIGUSR2 traceback; no main or installed dependency edits"})
    return hook


async def run(args):
    import httpx
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile, audio = args.config.resolve(), args.audio.resolve()
    if not profile.is_file() or not audio.is_file():
        raise FileNotFoundError("config and short WAV must exist")
    hook = diagnostic_hook(output)
    env = dict(os.environ, CONFIG_PATH=str(profile), PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = str(hook) + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, str(ROOT/"main.py"), "--host", args.host, "--port", str(args.port), "--workers", "1"]
    summary = {"kind": "real_main_v3_single_short_audio_smoke", "started_utc": utc_now(),
        "command": command, "config": str(profile), "config_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
        "audio": str(audio), "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
        "success": False, "request": {}, "failure": None, "cleanup": None}
    owner = None
    try:
        owner = OwnedService(command, env, args.host, args.port, output)
        started = time.monotonic()
        deadline = started + args.startup_timeout_s
        async with httpx.AsyncClient(base_url=f"http://{args.host}:{args.port}", trust_env=False, timeout=5) as client:
            last_error = None
            while True:
                if owner.process.poll() is not None:
                    raise RuntimeError(f"main exited before healthy: {owner.process.returncode}")
                owner.register()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"GET healthy deadline exceeded: {last_error}")
                try:
                    response = await client.get("/get_status", timeout=max(.01, min(5, deadline-time.monotonic())))
                    response.raise_for_status()
                    initial = response.json()
                    if initial.get("status") == "living":
                        write_json(output/"initial-status.json", initial)
                        break
                    last_error = "status is not living"
                except (httpx.HTTPError, ValueError) as exc:
                    last_error = str(exc)
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"GET healthy deadline exceeded: {last_error}")
                await asyncio.sleep(min(.5, max(0, deadline-time.monotonic())))
            summary["startup_to_healthy_s"] = time.monotonic()-started
            fields = fields_for("A", 0)
            request = summary["request"]
            request.update(fields=fields, started_utc=utc_now(), started_monotonic=time.monotonic())
            try:
                with audio.open("rb") as stream:
                    response = await asyncio.wait_for(client.post("/v3/seacraft_asr", files={"audioFile": ("smoke.wav", stream, "audio/wav")},
                                                 data=fields, timeout=args.request_timeout_s), args.request_timeout_s)
                request["http_status"] = response.status_code
                request["response_text"] = response.text
                body = None
                try:
                    body = response.json()
                    request["response"] = body
                    request["code"] = body.get("code") if isinstance(body, dict) else None
                except ValueError:
                    request["code"] = None
                response.raise_for_status()
                # Benchmark's business-success contract, not HTTP200 alone.
                summary["success"] = (request.get("code") in (None, 0) and isinstance(body, dict)
                                      and isinstance(body.get("segments"), list))
                if not summary["success"]:
                    summary["failure"] = "non-success business code"
            finally:
                request["ended_utc"] = utc_now()
                request["elapsed_s"] = time.monotonic()-request["started_monotonic"]
            try:
                response = await client.get("/get_status")
                response.raise_for_status()
                write_json(output/"last-status.json", response.json())
            except Exception as exc:
                summary["last_status_error"] = repr(exc)
    except BaseException as exc:
        summary["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        if owner is not None:
            close_task = asyncio.create_task(owner.close(deadline_s=210, observe_s=5))
            diagnosis_task = asyncio.create_task(diagnose_slow_close(owner, output, close_task))
            # Even cancellation of the runner must not abandon owned shutdown.
            try:
                while not close_task.done():
                    try:
                        await asyncio.shield(close_task)
                    except asyncio.CancelledError:
                        summary["failure"] = summary["failure"] or "runner cancelled during cleanup"
                summary["cleanup"] = close_task.result()
            except BaseException as exc:
                summary["cleanup_error"] = repr(exc)
            finally:
                if not diagnosis_task.done():
                    diagnosis_task.cancel()
                await asyncio.gather(diagnosis_task, return_exceptions=True)
        summary["ended_utc"] = utc_now()
        summary["passed"] = bool(summary["success"] and summary.get("cleanup", {}).get("passed")) if summary.get("cleanup") else False
        write_json(output/"summary.json", summary)
    print(json.dumps({k: summary.get(k) for k in ("passed", "success", "failure", "startup_to_healthy_s")}, ensure_ascii=False))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT/"test/config.qwen-continuous128.toml")
    p.add_argument("--audio", type=Path, default=Path("/tmp/qc-warmup.wav"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--host", choices=["127.0.0.1", "localhost"], default="127.0.0.1")
    p.add_argument("--port", type=int, default=18083)
    p.add_argument("--startup-timeout-s", type=float, default=300)
    p.add_argument("--request-timeout-s", type=float, default=180)
    args = p.parse_args()
    if not 0 < args.startup_timeout_s <= 300 or not 0 < args.request_timeout_s <= 420 or not 1 <= args.port <= 65535:
        p.error("timeouts must be positive (startup<=300s, request<=420s), port1..65535")
    result = asyncio.run(run(args))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
