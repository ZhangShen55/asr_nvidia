"""FSMN-VAD CPU/NPU 原始 FP32 单实例和多实例扩展性。

先 source CANN，默认宿主 npu:7；容器映射该卡时显式 --device npu:0。
独立实例各自预热；报告不包括加载时间，不启动服务。
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bench_vad_cpu_vs_gpu import build_model, run_vad, make_slice, allocated_mb


def measure(model, audio):
    started = time.perf_counter()
    spans = run_vad(model, audio)
    return {"elapsed_s": time.perf_counter() - started, "segments": len(spans)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--slice-s", type=int, default=300)
    parser.add_argument("--device", default="npu:7")
    parser.add_argument("--concurrency", "--gpu-concurrency", dest="concurrency", default="1,2,4,8")
    args = parser.parse_args()
    from core.local_inference import backend_for_device
    backend_for_device(args.device)
    counts = [int(value) for value in args.concurrency.split(",")]
    if not counts or min(counts) < 1 or args.slice_s <= 0:
        parser.error("positive concurrency and --slice-s required")
    audio = make_slice(args.audio, args.slice_s)
    try:
        for device in dict.fromkeys(("cpu", args.device)):
            for count in counts:
                models = []
                try:
                    before = allocated_mb(device)
                    started = time.perf_counter()
                    for _ in range(count):
                        models.append(build_model(device))
                    load_s = time.perf_counter() - started
                    for model in models:
                        run_vad(model, audio)
                    after = allocated_mb(device)
                    started = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=count) as executor:
                        results = list(executor.map(lambda model: measure(model, audio), models))
                    wall = time.perf_counter() - started
                    print(json.dumps({"device": device, "precision": "fp32", "instances": count,
                                      "load_s": load_s, "wall_s": wall,
                                      "mean_instance_s": sum(item["elapsed_s"] for item in results) / count,
                                      "throughput_realtime": count * args.slice_s / wall,
                                      "allocated_mb": after,
                                      "allocation_delta_mb": after - before if before is not None else None,
                                      "results": results}, ensure_ascii=False), flush=True)
                finally:
                    close_errors = []
                    for model in models:
                        try:
                            model.close()
                        except Exception as exc:
                            close_errors.append(exc)
                    if close_errors:
                        raise RuntimeError(f"{len(close_errors)} VAD instances failed to close") from close_errors[0]
    finally:
        os.unlink(audio)


if __name__ == "__main__":
    main()
