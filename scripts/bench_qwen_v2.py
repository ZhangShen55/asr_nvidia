"""v2 接口的并发压测脚本：观测端到端时延、吞吐与 GPU 算力占用。

用法：
    python scripts/bench_qwen_v2.py --base-url http://127.0.0.1:8083 \
        --audio test/audio_file/教师1.wav --concurrency 1,2,4,8 --repeat 1
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def sample_gpu():
    """采样所有 GPU 的利用率与显存占用。"""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return []
    rows = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        rows.append(
            {"index": int(parts[0]), "util": int(parts[1]), "mem_mb": int(parts[2])}
        )
    return rows


def post_audio(base_url, path, show_spk=False, show_emotion=False, timeout=1800):
    url = f"{base_url.rstrip('/')}/v2/seacraft_asr"
    data = {}
    if show_spk:
        data["showSpk"] = "true"
    if show_emotion:
        data["showEmotion"] = "true"
    started = time.perf_counter()
    with open(path, "rb") as audio:
        response = requests.post(
            url,
            files={"audioFile": (os.path.basename(path), audio, "audio/wav")},
            data=data,
            timeout=timeout,
        )
    elapsed = time.perf_counter() - started
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    return {
        "elapsed_s": elapsed,
        "http": response.status_code,
        "segments": len(payload.get("segments", []) or []),
        "segment_texts": [
            s.get("segment_text") for s in (payload.get("segments") or [])
        ],
        "roles": [s.get("role") for s in (payload.get("segments") or [])],
        "emotions": [s.get("emotion") for s in (payload.get("segments") or [])],
        "chars": len(payload.get("text", "") or ""),
        "code": payload.get("code"),
        "language": payload.get("language"),
        "language_ratio": payload.get("language_ratio"),
    }


def percentile(values, q):
    """线性插值分位数。"""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (pos - low), 3)


def run_round(base_url, audio, concurrency, show_spk, show_emotion, keep_texts=False):
    samples = []
    gpu_samples = []
    stop = False

    def _sampler():
        while not stop:
            gpu_samples.append(sample_gpu())
            time.sleep(0.5)

    import threading

    thread = threading.Thread(target=_sampler, daemon=True)
    thread.start()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(post_audio, base_url, audio, show_spk, show_emotion)
            for _ in range(concurrency)
        ]
        for future in futures:
            samples.append(future.result())

    stop = True
    thread.join(timeout=2)

    latencies = [s["elapsed_s"] for s in samples]
    errors = [s for s in samples if s["code"] not in (None, 0)]
    gpu_util = {}
    gpu_mem = {}
    for snapshot in gpu_samples:
        for row in snapshot:
            gpu_util.setdefault(row["index"], []).append(row["util"])
            gpu_mem.setdefault(row["index"], []).append(row["mem_mb"])

    total_segments = sum(s["segments"] for s in samples)
    wall_s = max(latencies) if latencies else 0.0
    result = {
        "concurrency": concurrency,
        "ok": len(samples) - len(errors),
        "failed": len(errors),
        "p50_s": round(statistics.median(latencies), 3),
        "p95_s": percentile(latencies, 0.95),
        "min_s": round(min(latencies), 3),
        "max_s": round(max(latencies), 3),
        "mean_s": round(statistics.mean(latencies), 3),
        "segments_total": total_segments,
        "segments_per_s": round(total_segments / wall_s, 2) if wall_s else 0.0,
        "audio_per_min": round(
            concurrency / (statistics.mean(latencies) / 60), 2
        ) if latencies and statistics.mean(latencies) else 0.0,
        "ms_per_segment": round(
            statistics.mean(latencies) * 1000 / (total_segments / max(1, len(samples))),
            1,
        ) if total_segments else 0.0,
        "gpu_util_avg": {k: round(statistics.mean(v), 1) for k, v in gpu_util.items()},
        "gpu_util_max": {k: max(v) for k, v in gpu_util.items()},
        "gpu_mem_max_mb": {k: max(v) for k, v in gpu_mem.items()},
        "errors": [e["code"] for e in errors][:5],
    }
    if keep_texts:
        result["texts"] = [s["segment_texts"] for s in samples]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8083")
    parser.add_argument("--audio", default="test/audio_file/教师1.wav")
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--show-spk", action="store_true")
    parser.add_argument("--show-emotion", action="store_true")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--keep-texts", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    results = []
    for level in levels:
        for round_index in range(args.rounds):
            print(f"=== 并发 {level} 第 {round_index + 1} 轮 ===", flush=True)
            result = run_round(
                args.base_url,
                args.audio,
                level,
                args.show_spk,
                args.show_emotion,
                keep_texts=args.keep_texts,
            )
            result["round"] = round_index + 1
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        print(f"结果已写入 {args.output}")


if __name__ == "__main__":
    main()
