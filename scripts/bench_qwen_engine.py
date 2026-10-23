"""直接压 Qwen 引擎（绕过本服务），测前端 CPU 成本与吞吐天花板。

回答两个问题：
1. Qwen 侧处理一段 VAD 音频要多少 CPU（核）；
2. 在 --api-server-count 4 下，引擎的吞吐上限与排队表现。

用法：
    python scripts/bench_qwen_engine.py --clips 400 --concurrency 32
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent


def container_cpu_seconds(container: str) -> float:
    """容器内所有进程累计 CPU 秒数。"""
    out = subprocess.run(
        ["sudo", "-n", "docker", "inspect", "-f", "{{.State.Pid}}", container],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if out.returncode != 0 or not out.stdout.strip():
        return 0.0
    root = int(out.stdout.strip())
    pids = [root]
    try:
        listing = subprocess.check_output(["ps", "-eo", "pid,ppid"], text=True, timeout=15)
        children: Dict[int, List[int]] = {}
        for line in listing.strip().splitlines()[1:]:
            parts = line.split()
            if len(parts) == 2:
                children.setdefault(int(parts[1]), []).append(int(parts[0]))
        stack, seen = list(pids), set(pids)
        while stack:
            for child in children.get(stack.pop(), []):
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        pids = sorted(seen)
    except Exception:
        pass
    ticks = os.sysconf("SC_CLK_TCK")
    total = 0.0
    for p in pids:
        try:
            with open(f"/proc/{p}/stat", "rb") as handle:
                data = handle.read().decode("utf-8", "ignore")
            fields = data[data.rindex(")") + 2:].split()
            total += (int(fields[11]) + int(fields[12])) / ticks
        except Exception:
            continue
    return total


def engine_metrics(base_url: str) -> Dict[str, float]:
    wanted = {
        "vllm:num_requests_running": "running",
        "vllm:num_requests_waiting": "waiting",
        "vllm:kv_cache_usage_perc": "kv_usage",
        "vllm:request_queue_time_seconds_sum": "q_sum",
        "vllm:request_queue_time_seconds_count": "q_count",
    }
    out: Dict[str, float] = {}
    try:
        text = requests.get(f"{base_url}/metrics", timeout=5).text
    except Exception:
        return out
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        for prefix, key in wanted.items():
            if line.startswith(prefix):
                try:
                    out[key] = float(line.rsplit(" ", 1)[-1])
                except ValueError:
                    pass
    return out


def make_clips(src: str, count: int, clip_s: int) -> List[str]:
    """从长音频里切出 count 段 clip_s 秒的片段。"""
    outdir = Path(tempfile.mkdtemp(prefix="vadclips_"))
    paths = []
    for i in range(count):
        dst = outdir / f"clip_{i:04d}.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-ss", str(i * clip_s),
                "-t", str(clip_s), "-i", src,
                "-ar", "16000", "-ac", "1", str(dst),
            ],
            check=True,
        )
        paths.append(str(dst))
    return paths


def post_clip(base_url: str, path: str) -> Dict[str, Any]:
    started = time.perf_counter()
    with open(path, "rb") as handle:
        resp = requests.post(
            f"{base_url}/v1/audio/transcriptions",
            files={"file": (os.path.basename(path), handle, "audio/wav")},
            timeout=600,
        )
    return {
        "elapsed_s": time.perf_counter() - started,
        "status": resp.status_code,
        "chars": len(resp.text),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8806")
    parser.add_argument("--container", default="qwen3-asr-0.6b-8806")
    parser.add_argument(
        "--audio",
        default=str(REPO_ROOT / "test" / "audio_file" / "教师1.wav"),
    )
    parser.add_argument("--clips", type=int, default=400)
    parser.add_argument("--clip-s", type=int, default=5)
    parser.add_argument("--concurrency", default="8,16,32,64")
    args = parser.parse_args()

    print(f"切分 {args.clips} 段 {args.clip_s}s 片段 ...")
    clips = make_clips(args.audio, args.clips, args.clip_s)
    print(f"片段目录：{os.path.dirname(clips[0])}")

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    for conc in levels:
        samples: List[Dict[str, Any]] = []
        metrics: List[Dict[str, float]] = []
        stop = threading.Event()

        def sampler():
            while not stop.is_set():
                m = engine_metrics(args.base_url)
                if m:
                    metrics.append(m)
                time.sleep(0.5)

        thread = threading.Thread(target=sampler, daemon=True)
        thread.start()
        cpu0 = container_cpu_seconds(args.container)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=conc) as pool:
            futures = [pool.submit(post_clip, args.base_url, c) for c in clips]
            for f in futures:
                samples.append(f.result())
        wall = time.perf_counter() - t0
        cpu1 = container_cpu_seconds(args.container)
        stop.set()
        thread.join(timeout=2)

        ok = [s for s in samples if s["status"] == 200]
        cpu_s = cpu1 - cpu0
        lat = [s["elapsed_s"] for s in ok]
        run_max = max((m.get("running", 0) for m in metrics), default=0)
        wait_max = max((m.get("waiting", 0) for m in metrics), default=0)
        kv_max = max((m.get("kv_usage", 0) for m in metrics), default=0)
        q_avg = "-"
        if metrics and metrics[-1].get("q_count"):
            q_avg = round(metrics[-1]["q_sum"] / metrics[-1]["q_count"], 4)
        per_req_ms = cpu_s / max(1, len(ok)) * 1000
        cores = cpu_s / wall
        print(
            "并发{:<4} 成功 {}/{} 墙钟 {:.2f}s 吞吐 {:.1f} 段/s | 容器CPU {:.1f}s "
            "({:.2f} 核, {:.1f} ms/请求) | p50 {:.2f}s | running峰 {:.0f} waiting峰 {:.0f} "
            "KV峰 {:.2f}% 排队均 {}s".format(
                conc, len(ok), len(samples), wall, len(ok) / wall, cpu_s, cores,
                per_req_ms, statistics.median(lat), run_max, wait_max, kv_max * 100, q_avg,
            )
        )


if __name__ == "__main__":
    main()
