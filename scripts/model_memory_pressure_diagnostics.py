#!/usr/bin/env python3
"""受安全闸门保护的模型内存/Swap 诊断与分阶段压测工具。

默认只采集和压测 SeaCraftASR 8085，不会停止或重启 Qwen 8806。出现红色条件
时停止提交后续请求并保存现场；只有显式传入 ``--stop-backend-on-red`` 才会向
诊断期间由用户指定的后端 PID 发送 SIGTERM。

示例（先确认 8085 已由运维命令启动）：
    conda run -n asr python scripts/model_memory_pressure_diagnostics.py \
      --base-url http://127.0.0.1:8085 --audio-dir test/audio_file/音频文件 \
      --out-dir /tmp/seacraft-memory-diagnostics --cooldown-s 60

只生成可复算的阶段计划而不连接后端：
    ... --plan-only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.resource_diagnostics import (
    BYTES_PER_GIB,
    SafetyBaseline,
    SafetyPolicy,
    collect_snapshot,
    evaluate_safety,
    write_json_line,
)


DEFAULT_AUDIO_DIR = ROOT / "test" / "audio_file" / "音频文件"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 3)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def load_config(path: Path) -> dict[str, Any]:
    try:
        import tomli

        with path.open("rb") as handle:
            return tomli.load(handle)
    except (OSError, ValueError, ImportError) as exc:
        return {"available": False, "reason": str(exc), "path": str(path)}


def get_json(url: str, timeout: float = 3.0) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.get(url, timeout=timeout)
        payload = response.json()
        if not isinstance(payload, dict):
            return None, f"响应不是对象: HTTP {response.status_code}"
        return payload, None
    except Exception as exc:  # noqa: BLE001 - 诊断必须保留失败原因
        return None, f"{type(exc).__name__}: {exc}"


def post_audio(
    base_url: str,
    path: Path,
    *,
    show_spk: bool,
    show_emotion: bool,
    timeout_s: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {}
    error = None
    http_status = None
    try:
        with path.open("rb") as handle:
            response = requests.post(
                f"{base_url.rstrip('/')}/v3/seacraft_asr",
                files={"audioFile": (path.name, handle, "application/octet-stream")},
                data={
                    "showSpk": "true" if show_spk else "false",
                    "showEmotion": "true" if show_emotion else "false",
                },
                timeout=timeout_s,
            )
        http_status = response.status_code
        try:
            payload = response.json()
        except ValueError:
            payload = {}
            error = f"响应不是 JSON: HTTP {response.status_code}"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "file": path.name,
        "elapsed_ms": round(elapsed_ms, 3),
        "http_status": http_status,
        "code": payload.get("code"),
        "msg": payload.get("msg"),
        "segments": len(payload.get("segments") or []) if isinstance(payload, dict) else None,
        "error": error,
        "ok": error is None and payload.get("code") in (None, 0),
    }


def _service_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    services = snapshot.get("services", {})
    return {
        name: {"count": len(rows), "pids": [row.get("pid") for row in rows]}
        for name, rows in services.items()
    }


class Sampler:
    """后台 1 秒采样器；触发红色条件后只设置 stop_event，不杀进程。"""

    def __init__(self, *, backend_pid: int | None, base_url: str, qwen_url: str, cache_dir: Path, out_file: Path, policy: SafetyPolicy):
        self.backend_pid = backend_pid
        self.base_url = base_url.rstrip("/")
        self.qwen_url = qwen_url.rstrip("/")
        self.cache_dir = cache_dir
        self.out_file = out_file
        self.policy = policy
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.baseline: SafetyBaseline | None = None
        self.expected_vad_workers: int | None = None
        self.samples: list[dict[str, Any]] = []
        self.red_events: list[dict[str, Any]] = []
        self.swap_samples = 0
        self.health_failures = 0
        self._last_vm: dict[str, Any] | None = None

    def start(self) -> None:
        status, _ = get_json(f"{self.base_url}/get_status")
        if status:
            pool = status.get("vadPool") or {}
            configured = pool.get("configured_instances")
            if isinstance(configured, int):
                self.expected_vad_workers = configured
        initial = collect_snapshot(
            backend_pid=self.backend_pid,
            # /get_status 会同步采集 GPU 和进程树，模型加载时可能超过 2 秒；
            # 用轻量 openapi 端点作为健康检查，状态快照仍单独读取 /get_status。
            backend_url=f"{self.base_url}/openapi.json",
            qwen_url=f"{self.qwen_url}/v1/models",
            cache_dir=self.cache_dir,
            service_status=get_json(f"{self.base_url}/get_status")[0],
        )
        self.baseline = SafetyBaseline.from_snapshot(initial)
        self._record(initial, "baseline")
        self.thread = threading.Thread(target=self._run, name="memory-pressure-sampler", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        self.out_file.parent.mkdir(parents=True, exist_ok=True)
        with self.out_file.open("a", encoding="utf-8") as handle:
            while not self.stop_event.is_set():
                snapshot = collect_snapshot(
                    backend_pid=self.backend_pid,
                    backend_url=f"{self.base_url}/openapi.json",
                    qwen_url=f"{self.qwen_url}/v1/models",
                    cache_dir=self.cache_dir,
                    service_status=get_json(f"{self.base_url}/get_status")[0],
                )
                decision = self._record(snapshot, "sample")
                if decision.should_stop:
                    event = {"timestamp": snapshot.get("timestamp"), "decision": decision.__dict__}
                    self.red_events.append(event)
                    write_json_line(handle, {"type": "snapshot", "snapshot": snapshot, "decision": decision.__dict__})
                    write_json_line(handle, {"type": "safety_red", **event})
                    self.stop_event.set()
                    break
                write_json_line(handle, {"type": "snapshot", "snapshot": snapshot, "decision": decision.__dict__})
                self.stop_event.wait(1.0)

    def _record(self, snapshot: dict[str, Any], kind: str):
        self.samples.append(snapshot)
        if self.baseline is None:
            return evaluate_safety(snapshot, SafetyBaseline.from_snapshot(snapshot), self.policy)
        vm = snapshot.get("vmstat", {})
        if vm.get("available") and self._last_vm and self._last_vm.get("available"):
            if (vm.get("swap_in_pages", 0) > self._last_vm.get("swap_in_pages", 0) or
                    vm.get("swap_out_pages", 0) > self._last_vm.get("swap_out_pages", 0)):
                self.swap_samples += 1
            else:
                self.swap_samples = 0
        health = snapshot.get("health", {})
        if not (health.get("backend", {}).get("ok") and health.get("qwen", {}).get("ok")):
            self.health_failures += 1
        else:
            self.health_failures = 0
        self._last_vm = vm
        return evaluate_safety(
            snapshot,
            self.baseline,
            self.policy,
            swap_samples=self.swap_samples,
            health_failures=self.health_failures,
            expected_vad_workers=self.expected_vad_workers,
        )

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)


def _stage_plan(configured_segment_concurrency: int | None = None) -> list[dict[str, Any]]:
    # A 项 VAD-only 已确认正常，严格跳过；模型单独加载通过开关保证只触发一类懒加载。
    return [
        {"id": "B-cam-only", "show_spk": True, "show_emotion": False, "audio_concurrency": 1, "segment_concurrency": configured_segment_concurrency},
        {"id": "B-emo-only", "show_spk": False, "show_emotion": True, "audio_concurrency": 1, "segment_concurrency": configured_segment_concurrency},
        {"id": "C-both", "show_spk": True, "show_emotion": True, "audio_concurrency": 1, "segment_concurrency": configured_segment_concurrency},
        *[
            {"id": f"D-audio-{level}", "show_spk": True, "show_emotion": True, "audio_concurrency": level, "segment_concurrency": configured_segment_concurrency}
            for level in (2, 4, 8)
        ],
    ]


def resource_analysis(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """从时间序列计算增长斜率、缓存差值和 GPU/模型池联合证据。"""
    if not snapshots:
        return {"available": False, "reason": "没有采样"}

    def values(path: tuple[str, ...]) -> list[tuple[float, float]]:
        result = []
        for row in snapshots:
            value: Any = row
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if isinstance(value, (int, float)):
                try:
                    timestamp = datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00")).timestamp()
                    result.append((timestamp, float(value)))
                except (KeyError, TypeError, ValueError):
                    continue
        return result

    def slope(path: tuple[str, ...]) -> float | None:
        points = values(path)
        if len(points) < 2 or points[-1][0] <= points[0][0]:
            return None
        return round((points[-1][1] - points[0][1]) / (points[-1][0] - points[0][0]), 3)

    first_cache = snapshots[0].get("cache", {})
    last_cache = snapshots[-1].get("cache", {})
    gpu_rows = [row for sample in snapshots for row in (sample.get("gpu", {}).get("gpus", []) or [])]
    gpu_util = [row.get("utilization_gpu") for row in gpu_rows if isinstance(row.get("utilization_gpu"), int)]
    model_status = snapshots[-1].get("status") or {}
    models = model_status.get("localModels") or {}
    return {
        "available": True,
        "backend_pss_slope_bytes_per_s": slope(("backend", "memory", "pss_bytes")),
        "backend_rss_slope_bytes_per_s": slope(("backend", "memory", "rss_bytes")),
        "cache_delta": {
            "files": (last_cache.get("files") - first_cache.get("files")) if isinstance(last_cache.get("files"), int) and isinstance(first_cache.get("files"), int) else None,
            "bytes": (last_cache.get("bytes") - first_cache.get("bytes")) if isinstance(last_cache.get("bytes"), int) and isinstance(first_cache.get("bytes"), int) else None,
        },
        "gpu": {
            "utilization_max_percent": max(gpu_util, default=None),
            "utilization_mean_percent": round(sum(gpu_util) / len(gpu_util), 3) if gpu_util else None,
        },
        "model_pool_waiting": {
            name: {"waiting": entry.get("waiting"), "wait_total_ms": entry.get("wait_total_ms"), "inference_total_ms": entry.get("inference_total_ms")}
            for name, entry in models.items()
        },
        "interpretation": (
            "PSS 持续上升且缓存回到基线附近，优先检查请求对象引用；"
            "GPU 利用率高且模型池等待增加时，优先检查同卡算力争用。"
        ),
    }


def stage_summary(stage: dict[str, Any], results: list[dict[str, Any]], snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [float(item["elapsed_ms"]) for item in results]
    successful = [item for item in results if item.get("ok")]
    host_available = [item.get("host", {}).get("available_bytes") for item in snapshots]
    rss = [item.get("backend", {}).get("memory", {}).get("rss_bytes") for item in snapshots]
    pss = [item.get("backend", {}).get("memory", {}).get("pss_bytes") for item in snapshots]
    swap = [item.get("host", {}).get("swap_used_bytes") for item in snapshots]
    return {
        **stage,
        "requests": len(results),
        "success": len(successful),
        "failed": len(results) - len(successful),
        "p50_ms": percentile(durations, 0.50),
        "p95_ms": percentile(durations, 0.95),
        "resource_peak": {
            "backend_rss_bytes": max((x for x in rss if isinstance(x, int)), default=None),
            "backend_pss_bytes": max((x for x in pss if isinstance(x, int)), default=None),
            "mem_available_min_bytes": min((x for x in host_available if isinstance(x, int)), default=None),
            "swap_used_max_bytes": max((x for x in swap if isinstance(x, int)), default=None),
        },
        "errors": [
            {key: item.get(key) for key in ("file", "http_status", "code", "msg", "error")}
            for item in results if not item.get("ok")
        ],
    }


def graceful_stop(pid: int) -> dict[str, Any]:
    try:
        os.kill(pid, signal.SIGTERM)
        return {"pid": pid, "signal": "SIGTERM", "sent": True}
    except ProcessLookupError:
        return {"pid": pid, "signal": "SIGTERM", "sent": False, "reason": "进程已退出"}
    except OSError as exc:
        return {"pid": pid, "signal": "SIGTERM", "sent": False, "reason": str(exc)}


def run(args: argparse.Namespace) -> int:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"memory-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    config = load_config(args.config.resolve())
    config_qwen = config.get("qwen_asr", {}) if isinstance(config, dict) else {}
    configured_segment_concurrency = config_qwen.get("segment_concurrency") if isinstance(config_qwen, dict) else None
    plan = _stage_plan(int(configured_segment_concurrency) if isinstance(configured_segment_concurrency, int) else None)
    metadata = {
        "run_id": run_id,
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "base_url": args.base_url,
        "qwen_url": args.qwen_url,
        "backend_pid": args.backend_pid,
        "qwen_must_remain_running": True,
        "skip_a_vad_only": True,
        "config_path": str(args.config.resolve()),
        "config": config,
        "audio_dir": str(args.audio_dir.resolve()),
        "stages": plan,
        "requested_segment_concurrency_matrix": [4, 8, 16],
        "segment_concurrency_note": "工具不在线修改生产配置；stage-*.json 记录启动时实际配置，切换 4/8/16 需重启隔离后端后重复运行。",
        "safety_policy": SafetyPolicy().__dict__,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.plan_only:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0

    backend, backend_error = get_json(f"{args.base_url.rstrip('/')}/get_status")
    qwen, qwen_error = get_json(f"{args.qwen_url.rstrip('/')}/v1/models")
    if backend is None or qwen is None:
        report = {
            "run_id": run_id,
            "status": "blocked_before_requests",
            "backend": {"ok": backend is not None, "error": backend_error},
            "qwen": {"ok": qwen is not None, "error": qwen_error},
            "reason": "8085 或 8806 健康检查失败，未提交任何音频请求",
        }
        (out_dir / "run-summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    audio_files = sorted(args.audio_dir.rglob("*.wav"))
    if not audio_files:
        raise SystemExit(f"音频目录没有 wav: {args.audio_dir}")
    rng = random.Random(args.seed)
    if args.audio_count:
        audio_files = rng.sample(audio_files, min(args.audio_count, len(audio_files)))
    sampler = Sampler(
        backend_pid=args.backend_pid,
        base_url=args.base_url,
        qwen_url=args.qwen_url,
        cache_dir=args.cache_dir,
        out_file=out_dir / "snapshots.jsonl",
        policy=SafetyPolicy(),
    )
    sampler.start()
    stage_reports: list[dict[str, Any]] = []
    run_status = "completed"
    try:
        for stage in plan:
            if sampler.stop_event.is_set():
                run_status = "stopped_by_safety_gate"
                break
            level = int(stage["audio_concurrency"])
            files = audio_files[: min(level, len(audio_files))]
            print(f"[{stage['id']}] 提交 {len(files)} 个完整音频", flush=True)
            before = len(sampler.samples)
            results = []
            with ThreadPoolExecutor(max_workers=level, thread_name_prefix="diag-request") as executor:
                futures = [
                    executor.submit(
                        post_audio,
                        args.base_url,
                        path,
                        show_spk=stage["show_spk"],
                        show_emotion=stage["show_emotion"],
                        timeout_s=args.request_timeout_s,
                    )
                    for path in files
                    if not sampler.stop_event.is_set()
                ]
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    print(f"  {result['file']}: {result['elapsed_ms'] / 1000:.2f}s code={result['code']}", flush=True)
            after = len(sampler.samples)
            report = stage_summary(stage, results, sampler.samples[before:after] or sampler.samples)
            status, _ = get_json(f"{args.base_url.rstrip('/')}/get_status")
            report["status_after"] = status
            stage_reports.append(report)
            (out_dir / f"stage-{stage['id']}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            if any(not item.get("ok") for item in results):
                run_status = "stopped_after_stage_failure"
                break
            if args.cooldown_s > 0:
                print(f"  冷却 {args.cooldown_s:.0f}s", flush=True)
                sampler.stop_event.wait(args.cooldown_s)
            if sampler.stop_event.is_set():
                run_status = "stopped_by_safety_gate"
                break
    finally:
        sampler.stop()
    final_snapshot = collect_snapshot(
        backend_pid=args.backend_pid,
        backend_url=f"{args.base_url.rstrip('/')}/openapi.json",
        qwen_url=f"{args.qwen_url.rstrip('/')}/v1/models",
        cache_dir=args.cache_dir,
        service_status=get_json(f"{args.base_url.rstrip('/')}/get_status")[0],
    )
    stop_action = None
    if sampler.red_events and args.stop_backend_on_red and args.backend_pid:
        stop_action = graceful_stop(args.backend_pid)
    summary = {
        "run_id": run_id,
        "status": run_status,
        "stages": stage_reports,
        "safety_red_events": sampler.red_events,
        "sample_count": len(sampler.samples),
        "final_snapshot": final_snapshot,
        "resource_analysis": resource_analysis(sampler.samples),
        "qwen_health_after": final_snapshot.get("health", {}).get("qwen"),
        "backend_stop_action": stop_action,
        "qwen_stop_action": None,
        "qwen_was_not_stopped": True,
    }
    (out_dir / "run-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "status": run_status, "stages": len(stage_reports), "red": len(sampler.red_events)}, ensure_ascii=False))
    return 0 if run_status == "completed" else 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8085")
    parser.add_argument("--qwen-url", default="http://127.0.0.1:8806")
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--audio-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--config", type=Path, default=ROOT / "config.toml")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/seacraft_asr"))
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/seacraft-model-memory-diagnostics"))
    parser.add_argument("--backend-pid", type=int, default=None)
    parser.add_argument("--request-timeout-s", type=float, default=7200)
    parser.add_argument("--cooldown-s", type=float, default=60)
    parser.add_argument("--stop-backend-on-red", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
