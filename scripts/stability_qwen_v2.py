"""稳定性测试：8 音频并发 + role/emotion 全开连续跑 N 分钟。

观测：端到端时延、失败率、GPU 显存与利用率趋势、/tmp 临时文件数、槽位是否泄漏。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def sample_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], text=True)
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            rows.append({"index": int(parts[0]), "util": int(parts[1]), "mem": int(parts[2])})
    return rows


def count_tmp():
    try:
        return len([n for n in os.listdir("/tmp") if n.startswith("tmp") or n.startswith("v2_")])
    except Exception:
        return -1


def post(base_url, path, timeout=3600):
    started = time.perf_counter()
    try:
        with open(path, "rb") as handle:
            r = requests.post(
                f"{base_url.rstrip('/')}/v2/seacraft_asr",
                files={"audioFile": (os.path.basename(path), handle, "audio/wav")},
                data={"showSpk": "true", "showEmotion": "true"},
                timeout=timeout,
            )
        payload = r.json()
        code = payload.get("code")
    except Exception as e:  # noqa: BLE001
        return {"elapsed": time.perf_counter() - started, "code": f"EXC:{e}", "segments": 0}
    return {
        "elapsed": time.perf_counter() - started,
        "code": code,
        "segments": len(payload.get("segments") or []),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8085")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    deadline = time.time() + args.minutes * 60
    samples, gpus, tmps = [], [], []
    stop = False

    def _sampler():
        while not stop:
            gpus.append(sample_gpu())
            tmps.append(count_tmp())
            time.sleep(10)

    t = threading.Thread(target=_sampler, daemon=True)
    t.start()

    rounds = 0
    while time.time() < deadline:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(post, args.base_url, args.audio)
                       for _ in range(args.concurrency)]
            for f in futures:
                samples.append(f.result())
        rounds += 1
        done = sum(1 for s in samples if s["code"] in (None, 0))
        print(f"第 {rounds} 轮: 累计 {len(samples)} 次, 成功 {done}, "
              f"P50 {statistics.median([s['elapsed'] for s in samples]):.1f}s", flush=True)

    stop = True
    t.join(timeout=3)

    ok = [s for s in samples if s["code"] in (None, 0)]
    failed = [s for s in samples if s["code"] not in (None, 0)]
    lat = [s["elapsed"] for s in ok]

    mem_by_gpu = {}
    util_by_gpu = {}
    for snap in gpus:
        for row in snap:
            mem_by_gpu.setdefault(row["index"], []).append(row["mem"])
            util_by_gpu.setdefault(row["index"], []).append(row["util"])

    def trend(values):
        if not values:
            return None
        half = len(values) // 2
        return {
            "start": values[0],
            "mid": values[half],
            "end": values[-1],
            "max": max(values),
            "growth_mb": values[-1] - values[0],
            "half_growth_mb": values[-1] - values[half],
        }

    result = {
        "minutes": args.minutes,
        "concurrency": args.concurrency,
        "rounds": rounds,
        "requests": len(samples),
        "ok": len(ok),
        "failed": len(failed),
        "failure_codes": sorted({str(s["code"]) for s in failed}),
        "p50_s": round(statistics.median(lat), 2) if lat else None,
        "p95_s": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 2) if lat else None,
        "min_s": round(min(lat), 2) if lat else None,
        "max_s": round(max(lat), 2) if lat else None,
        "gpu_mem_trend": {k: trend(v) for k, v in mem_by_gpu.items()},
        "gpu_util_avg": {k: round(statistics.mean(v), 1) for k, v in util_by_gpu.items()},
        "tmp_file_count": {"start": tmps[0] if tmps else None,
                           "end": tmps[-1] if tmps else None,
                           "max": max(tmps) if tmps else None},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as h:
            json.dump(result, h, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

