"""CAM++/emotion2vec 模型池完整音频并发基准。

示例：
    conda run -n asr python scripts/bench_cam_emo_model_pool.py --dry-run
    conda run -n asr python scripts/bench_cam_emo_model_pool.py \
        --base-url http://127.0.0.1:8084 --concurrency 2,4,8,16,32 \
        --audio 'test/audio_file/教师1.aac' --out-dir /tmp/cam_emo_pool

脚本只操作本次指定的 HTTP 服务，不会停止或重启 qwen-asr 服务。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import threading
import time
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIO = REPO_ROOT / "test/audio_file/教师1.aac"
V2_PHASE_RE = re.compile(
    r"Qwen 等待 ([\d.]+)ms/调用 ([\d.]+)ms/(\d+)次, "
    r"说话人等待 ([\d.]+)ms/推理 ([\d.]+)ms/(\d+)次, "
    r"情绪等待 ([\d.]+)ms/推理 ([\d.]+)ms/(\d+)次"
)


def gpu_sample() -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        rows = []
        for line in result.stdout.splitlines():
            fields = [value.strip() for value in line.split(",")]
            if len(fields) >= 4:
                rows.append({"index": int(fields[0]), "utilization_gpu": float(fields[1]),
                             "memory_used_mb": float(fields[2]), "memory_total_mb": float(fields[3])})
        return {"gpus": rows}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "gpus": []}


def status(base_url: str) -> dict:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/get_status", timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def one_request(base_url: str, audio: Path, timeout: float) -> dict:
    started = time.perf_counter()
    try:
        with audio.open("rb") as handle:
            response = requests.post(
                f"{base_url.rstrip('/')}/v2/seacraft_asr",
                files={"audioFile": (audio.name, handle, "audio/aac")},
                data={"showSpk": "true", "showEmotion": "true"},
                timeout=timeout,
            )
        payload = response.json()
        segments = payload.get("segments") or []
        role_count = sum(1 for item in segments if item.get("role") is not None)
        emotion_count = sum(1 for item in segments if item.get("emotion") is not None)
        return {"ok": response.ok and payload.get("code", 0) == 0,
                "http_status": response.status_code,
                "elapsed_s": round(time.perf_counter() - started, 3),
                "segments": len(segments),
                "role_count": role_count,
                "emotion_count": emotion_count,
                "text_chars": len(str(payload.get("text") or "")),
                "error": payload.get("msg") if not response.ok else None}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "elapsed_s": round(time.perf_counter() - started, 3), "error": str(exc)}


def run_level(
    base_url: str,
    audio: Path,
    concurrency: int,
    timeout: float,
    log_path: Path = REPO_ROOT / "asr_service.log",
) -> dict:
    before = {"status": status(base_url), "gpu": gpu_sample()}
    try:
        log_offset = log_path.stat().st_size
    except OSError:
        log_offset = 0
    status_samples = []
    stop_sampling = threading.Event()

    def sample_status():
        while not stop_sampling.is_set():
            snapshot = status(base_url)
            if snapshot.get("localModels"):
                status_samples.append({
                    "time": time.time(),
                    "localModels": snapshot["localModels"],
                    "gpu": gpu_sample(),
                })
            stop_sampling.wait(1.0)

    sampler = threading.Thread(target=sample_status, daemon=True)
    sampler.start()
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(one_request, base_url, audio, timeout) for _ in range(concurrency)]
            results = [future.result() for future in futures]
    finally:
        stop_sampling.set()
        sampler.join(timeout=5)
    elapsed = time.perf_counter() - started
    after = {"status": status(base_url), "gpu": gpu_sample()}
    times = [item["elapsed_s"] for item in results]
    successes = [item for item in results if item.get("ok")]
    audio_duration_s = None
    try:
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "default=noprint_wrappers=1:nokey=1", str(audio)],
                               capture_output=True, text=True, check=False, timeout=10)
        audio_duration_s = float(probe.stdout.strip())
    except Exception:
        pass
    pool_metrics = {}
    for name in ("speech_campplus", "emotion2vec"):
        entries = [item["localModels"].get(name, {}) for item in status_samples]
        if entries:
            pool_metrics[name] = {
                "max_busy": max(int(item.get("busy", 0)) for item in entries),
                "max_waiting": max(int(item.get("waiting", 0)) for item in entries),
                "wait_total_ms_before": entries[0].get("wait_total_ms", 0.0),
                "wait_total_ms_after": entries[-1].get("wait_total_ms", 0.0),
                "calls_before": entries[0].get("calls", 0),
                "calls_after": entries[-1].get("calls", 0),
                "configured_instances": entries[-1].get("configured_instances", 0),
                "loaded_instances": entries[-1].get("instances", 0),
            }
        else:
            pool_metrics[name] = {}
    phase_logs = []
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(log_offset)
            for line in handle:
                match = V2_PHASE_RE.search(line)
                if match:
                    values = [float(match.group(i)) for i in (1, 2, 4, 5, 7, 8)]
                    phase_logs.append({
                        "qwen_wait_ms": values[0],
                        "qwen_ms": values[1],
                        "spk_wait_ms": values[2],
                        "spk_ms": values[3],
                        "emo_wait_ms": values[4],
                        "emo_ms": values[5],
                        "qwen_calls": int(match.group(3)),
                        "spk_calls": int(match.group(6)),
                        "emo_calls": int(match.group(9)),
                    })
    except OSError:
        pass
    return {
        "concurrency": concurrency,
        "audio": str(audio),
        "audio_duration_s": audio_duration_s,
        "wall_s": round(elapsed, 3),
        "success": len(successes),
        "failed": len(results) - len(successes),
        "p50_s": round(statistics.median(times), 3) if times else None,
        "p95_s": round(sorted(times)[max(0, math.ceil(len(times) * 0.95) - 1)], 3) if times else None,
        "max_s": round(max(times), 3) if times else None,
        "audio_per_min": round(len(successes) * 60 / elapsed, 3) if elapsed else 0,
        "segments_per_s": round(sum(item.get("segments", 0) for item in successes) / elapsed, 3) if elapsed else 0,
        "before": before,
        "after": after,
        "status_samples": status_samples,
        "model_pool_metrics": pool_metrics,
        "qwen_phase_logs": phase_logs,
        "requests": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8084")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--concurrency", default="2,4,8,16,32")
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/cam_emo_pool"))
    parser.add_argument("--log-path", type=Path, default=REPO_ROOT / "asr_service.log")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    levels = [int(value) for value in args.concurrency.split(",")]
    config = {"base_url": args.base_url, "audio": str(args.audio), "concurrency": levels,
              "showSpk": True, "showEmotion": True, "port": args.base_url.rsplit(":", 1)[-1]}
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    args.audio = args.audio if args.audio.is_absolute() else REPO_ROOT / args.audio
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for level in levels:
        result = run_level(args.base_url, args.audio, level, args.timeout, args.log_path)
        results.append(result)
        output = args.out_dir / f"concurrency_{level}.json"
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: result[key] for key in
                          ("concurrency", "wall_s", "success", "failed", "p50_s", "p95_s",
                           "max_s", "audio_per_min", "segments_per_s")}, ensure_ascii=False), flush=True)
    (args.out_dir / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
