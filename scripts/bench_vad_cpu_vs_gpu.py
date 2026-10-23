"""VAD 的 CPU/GPU 等价性 + 每实例 CPU 核数消耗。

用法：
    python scripts/bench_vad_cpu_vs_gpu.py --audio test/audio_file/储能学术英语实践.wav
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "model" / "speech_fsmn_vad_zh-cn-16k-common-pytorch"


def make_slice(src: str, seconds: int) -> str:
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
        model=str(MODEL_DIR),
        device=device,
        vad_kwargs={
            "max_single_segment_time": 30000,
            "max_end_silence_time": 800,
        },
        disable_update=True,
        disable_pbar=True,
    )


def run_vad(model, path: str) -> List[Tuple[float, float]]:
    result = model.generate(
        input=path,
        cache={},
        vad_kwargs={
            "max_single_segment_time": 30000,
            "max_end_silence_time": 800,
        },
    )
    payload = result[0]
    if isinstance(payload, dict):
        spans = payload.get("value") or []
    elif isinstance(payload, list) and payload and isinstance(payload[0], list):
        spans = payload[0]
    else:
        spans = []
    return [(float(s[0]), float(s[1])) for s in spans if float(s[1]) > float(s[0])]


def cpu_time() -> float:
    """本进程（含已回收子进程）的 CPU 时间，单位秒。"""
    me = resource.getrusage(resource.RUSAGE_SELF)
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (
        me.ru_utime + me.ru_stime + kids.ru_utime + kids.ru_stime
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio", default=str(REPO_ROOT / "test" / "audio_file" / "储能学术英语实践.wav")
    )
    parser.add_argument("--slice-s", type=int, default=0, help="0 表示用整个文件")
    parser.add_argument("--threads", type=int, default=0, help="torch 线程数，0 表示不干预")
    args = parser.parse_args()

    if args.threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)

    audio = args.audio
    tmp = None
    if args.slice_s > 0:
        tmp = make_slice(audio, args.slice_s)
        audio = tmp

    print(f"音频：{audio}")
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '未设置')}")

    results: Dict[str, Any] = {}
    for device in ("cpu", "cuda:0"):
        model = build_model(device)
        run_vad(model, audio)  # 预热，不计入统计
        c0 = cpu_time()
        w0 = time.perf_counter()
        spans = run_vad(model, audio)
        wall = time.perf_counter() - w0
        cpu_used = cpu_time() - c0
        results[device] = {"spans": spans, "wall": wall, "cpu": cpu_used}
        print(
            f"[{device}] 段数 {len(spans)}  墙钟 {wall:.2f}s  "
            f"CPU 时间 {cpu_used:.2f}s  折合 {cpu_used / max(wall, 1e-9):.2f} 核"
        )

    cpu_spans = results["cpu"]["spans"]
    gpu_spans = results["cuda:0"]["spans"]
    same = cpu_spans == gpu_spans
    print(f"\nCPU/GPU 分段边界完全一致：{same}")
    if not same:
        print(f"  CPU 段数 {len(cpu_spans)}，GPU 段数 {len(gpu_spans)}")
        diff = [
            (i, a, b)
            for i, (a, b) in enumerate(zip(cpu_spans, gpu_spans))
            if a != b
        ]
        print(f"  前 10 处差异：{diff[:10]}")

    if tmp:
        os.unlink(tmp)


if __name__ == "__main__":
    main()
