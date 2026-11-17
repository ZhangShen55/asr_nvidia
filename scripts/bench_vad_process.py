"""FSMN-VAD 同进程独立实例多线程与 spawn 多进程扩展性。

CPU 默认；显式 --device npu:7 才使用指定卡，先 source CANN。
每个实例独占调用，模型关闭和 spawn manager 回收均在 finally 中完成。
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bench_vad_cpu_vs_gpu import build_model, run_vad, make_slice


def _worker(args):
    device, path, barrier = args
    model = None
    try:
        model = build_model(device)
        run_vad(model, path)
        barrier.wait(timeout=600)
        started = time.perf_counter()
        run_vad(model, path)
        return started, time.perf_counter()
    except BaseException:
        # 任一实例启动/推理失败须唤醒其他等待者，避免遗留600秒Barrier等待。
        try:
            barrier.abort()
        except Exception:
            pass
        raise
    finally:
        if model is not None:
            model.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--slice-s", type=int, default=300)
    parser.add_argument("--workers", default="1,2,4,8")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=["process", "thread"], default="process")
    args = parser.parse_args()
    from core.local_inference import backend_for_device
    backend_for_device(args.device)
    counts = [int(value) for value in args.workers.split(",")]
    if not counts or min(counts) < 1 or args.slice_s <= 0:
        parser.error("positive workers and --slice-s required")
    path = make_slice(args.audio, args.slice_s)
    try:
        for count in counts:
            if args.mode == "process":
                ctx = mp.get_context("spawn")
                with ctx.Manager() as manager:
                    barrier = manager.Barrier(count)
                    with ctx.Pool(count) as pool:
                        results = pool.map(_worker, [(args.device, path, barrier)] * count)
            else:
                from threading import Barrier
                barrier = Barrier(count)
                with ThreadPoolExecutor(max_workers=count) as executor:
                    results = list(executor.map(_worker, [(args.device, path, barrier)] * count))
            wall = max(item[1] for item in results) - min(item[0] for item in results)
            print(json.dumps({"workers": count, "mode": args.mode, "device": args.device,
                              "precision": "fp32", "wall_s": wall,
                              "mean_instance_s": sum(end - begin for begin, end in results) / count,
                              "throughput_realtime": count * args.slice_s / max(wall, 1e-9)}), flush=True)
    finally:
        os.unlink(path)


if __name__ == "__main__":
    main()
