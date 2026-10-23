"""VAD 实例的基准测试：GPU/CPU 速度、每实例显存/内存、多实例并发扩展性。

用法：
    python scripts/bench_vad_scaling.py --audio test/audio_file/教师1.wav --slice-s 300
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


def gpu_mem_mb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return 0.0


def make_slice(src: str, seconds: int) -> str:
    """切出前 N 秒音频到临时 wav（16k 单声道）。"""
    dst = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", src,
            "-t", str(seconds), "-ar", "16000", "-ac", "1", dst,
        ],
        check=True,
    )
    return dst


def build_model(device: str):
    from funasr import AutoModel

    return AutoModel(
        model=str(REPO_ROOT / "model" / "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
        device=device,
        vad_kwargs={
            "max_single_segment_time": 30000,
            "max_end_silence_time": 800,
        },
        disable_update=True,
        disable_pbar=True,
    )


def run_vad(model, path: str) -> Dict[str, Any]:
    started = time.perf_counter()
    result = model.generate(input=path, cache={}, vad_kwargs={
        "max_single_segment_time": 30000,
        "max_end_silence_time": 800,
    })
    elapsed = time.perf_counter() - started
    spans = 0
    try:
        spans = len(result[0]["value"])
    except Exception:
        pass
    return {"elapsed_s": round(elapsed, 3), "segments": spans}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default=str(REPO_ROOT / "test" / "audio_file" / "教师1.wav"))
    parser.add_argument("--slice-s", type=int, default=300)
    parser.add_argument("--gpu-concurrency", default="1,2,4,8")
    args = parser.parse_args()

    slice_path = make_slice(args.audio, args.slice_s)
    print(f"基准音频：{args.audio} 前 {args.slice_s}s -> {slice_path}")

    # ---- 单实例：GPU vs CPU
    print("\n=== 单实例速度 ===")
    before = gpu_mem_mb()
    t0 = time.perf_counter()
    gpu_model = build_model("cuda:0")
    load_s = time.perf_counter() - t0
    after = gpu_mem_mb()
    print(f"GPU 实例加载：{load_s:.2f}s，显存 {after - before:.1f} MB")
    gpu_single = run_vad(gpu_model, slice_path)
    print(f"GPU 单实例：{gpu_single['elapsed_s']}s（{gpu_single['segments']} 段）")

    t0 = time.perf_counter()
    cpu_model = build_model("cpu")
    cpu_load_s = time.perf_counter() - t0
    cpu_single = run_vad(cpu_model, slice_path)
    print(f"CPU 实例加载：{cpu_load_s:.2f}s")
    print(f"CPU 单实例：{cpu_single['elapsed_s']}s（{cpu_single['segments']} 段）")
    ratio = cpu_single["elapsed_s"] / max(gpu_single["elapsed_s"], 1e-6)
    print(f"CPU/GPU 速度比：{ratio:.1f}x")

    # ---- 并发扩展性（GPU / CPU 各档，先预热排除首次调用的 CUDA 惰性加载干扰）
    for device, base_model, label in (("cuda:0", gpu_model, "GPU"), ("cpu", cpu_model, "CPU")):
        print(f"\n=== {label} 多实例并发（每个实例跑同一段，先预热）===")
        for conc in [int(x) for x in args.gpu_concurrency.split(",")]:
            models = [base_model] + [build_model(device) for _ in range(conc - 1)]
            # 预热：每个实例先跑一次，排除首次 CUDA 调用/内核加载
            for m in models:
                run_vad(m, slice_path)
            mem = gpu_mem_mb()
            results: List[Dict[str, Any]] = []
            lock = threading.Lock()

            def worker(model):
                r = run_vad(model, slice_path)
                with lock:
                    results.append(r)

            threads = [threading.Thread(target=worker, args=(m,)) for m in models]
            t0 = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            wall = time.perf_counter() - t0
            per = sum(r["elapsed_s"] for r in results) / len(results)
            print(
                f"  {label} 并发{conc:>2}：墙钟 {wall:.2f}s，单实例均 {per:.2f}s，"
                f"总吞吐 {conc * args.slice_s / wall:.1f}x 实时，显存 {mem:.0f} MB"
            )
            del models

    os.unlink(slice_path)


if __name__ == "__main__":
    main()
