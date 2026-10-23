"""验证 VAD 的扩展性瓶颈是 GIL 还是算力：同进程多线程 vs 多进程。

用法：
    python scripts/bench_vad_process.py --slice-s 300 --workers 4 --mode process
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "model" / "speech_fsmn_vad_zh-cn-16k-common-pytorch"


def make_slice(src: str, seconds: int) -> str:
    dst = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", src, "-t", str(seconds),
         "-ar", "16000", "-ac", "1", dst],
        check=True,
    )
    return dst


def build_model(device: str):
    from funasr import AutoModel

    return AutoModel(
        model=str(MODEL_DIR),
        device=device,
        vad_kwargs={"max_single_segment_time": 30000, "max_end_silence_time": 800},
        disable_update=True,
        disable_pbar=True,
    )


def run_vad(model, path: str) -> float:
    started = time.perf_counter()
    model.generate(input=path, cache={}, vad_kwargs={
        "max_single_segment_time": 30000, "max_end_silence_time": 800,
    })
    return time.perf_counter() - started


def _worker(args):
    device, path, barrier = args
    model = build_model(device)
    run_vad(model, path)  # 预热
    barrier.wait()  # 所有 worker 就绪后同时开跑，排除启动开销
    started = time.perf_counter()
    run_vad(model, path)
    return started, time.perf_counter()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio", default=str(REPO_ROOT / "test" / "audio_file" / "教师1.wav")
    )
    parser.add_argument("--slice-s", type=int, default=300)
    parser.add_argument("--workers", default="1,2,4,8")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=["process", "thread"], default="process")
    args = parser.parse_args()

    slice_path = make_slice(args.audio, args.slice_s)
    print(f"基准：{args.slice_s}s 片段，设备 {args.device}，模式 {args.mode}")

    for workers in [int(x) for x in args.workers.split(",")]:
        if args.mode == "process":
            ctx = mp.get_context("spawn")
            # Manager().Barrier 是代理对象，可以跨 spawn 进程传递
            barrier = ctx.Manager().Barrier(workers)
            with ctx.Pool(workers) as pool:
                t0 = time.perf_counter()
                results = pool.map(
                    _worker, [(args.device, slice_path, barrier)] * workers
                )
                wall = time.perf_counter() - t0
        else:
            import threading

            models = [build_model(args.device) for _ in range(workers)]
            for m in models:
                run_vad(m, slice_path)
            results: List[float] = []
            lock = threading.Lock()

            def worker(m):
                started = time.perf_counter()
                run_vad(m, slice_path)
                r = (started, time.perf_counter())
                with lock:
                    results.append(r)

            threads = [threading.Thread(target=worker, args=(m,)) for m in models]
            t0 = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            wall = time.perf_counter() - t0

        # 只统计「真正开跑」的区间：进程启动、模型加载、预热都不计入吞吐
        batch_start = min(r[0] for r in results)
        batch_end = max(r[1] for r in results)
        wall = batch_end - batch_start
        per = sum(r[1] - r[0] for r in results) / len(results)
        print(
            "{:>6} 并发：实测区间墙钟 {:.2f}s，单实例均 {:.2f}s，总吞吐 {:.1f}x 实时".format(
                workers, wall, per, workers * args.slice_s / wall
            )
        )

    os.unlink(slice_path)


if __name__ == "__main__":
    main()
