"""Measure CPU consumption of the VAD process pool at several pool sizes.

Usage:
    conda run -n asr env PYTHONPATH=. python scripts/bench_vad_pool_scaling.py \
        --workers 4,8,16 --audio test/audio_file/储能学术英语实践.wav
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "model" / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
VAD_KWARGS = {
    "max_single_segment_time": 30000,
    "max_end_silence_time": 800,
}


def _cpu_seconds(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            data = handle.read()
        fields = data[data.rindex(")") + 2 :].split()
        ticks = os.sysconf("SC_CLK_TCK")
        return (int(fields[11]) + int(fields[12])) / ticks
    except (FileNotFoundError, IndexError, ValueError, OSError):
        return 0.0


async def measure(audio: Path, workers: int) -> Dict[str, object]:
    from core.vad_pool import VadPool
    from scripts.bench_vad_cpu_vs_gpu import vad_kwargs
    audio = audio.resolve()
    pool = VadPool(
        devices=["cpu"] * workers,
        model_dir=str(MODEL_DIR),
        vad_kwargs=vad_kwargs(),
    )
    try:
        await pool.start()
        pids = [worker.process.pid for worker in pool._workers.values()]
        before = {pid: _cpu_seconds(pid) for pid in pids}
        started = time.perf_counter()
        await asyncio.gather(*[pool.detect(str(audio)) for _ in range(workers)])
        elapsed = time.perf_counter() - started
        after = {pid: _cpu_seconds(pid) for pid in pids}
        cpu_seconds = sum(max(0.0, after[pid] - before[pid]) for pid in pids)
        return {
            "workers": workers,
            "audio": str(audio),
            "wall_s": round(elapsed, 3),
            "aggregate_cpu_s": round(cpu_seconds, 3),
            "aggregate_cpu_cores": round(cpu_seconds / max(elapsed, 1e-9), 3),
            "cpu_cores_per_worker": round(cpu_seconds / max(elapsed, 1e-9) / workers, 3),
            "pids": pids,
        }
    finally:
        await pool.stop()


async def run(audio: Path, workers: List[int]) -> None:
    for count in workers:
        result = await measure(audio, count)
        print(json.dumps(result, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
    )
    parser.add_argument("--workers", default="4,8,16")
    args = parser.parse_args()
    counts = [int(value) for value in args.workers.split(",")]
    if not counts or min(counts) < 1:
        parser.error("--workers must be positive")
    asyncio.run(run(args.audio.resolve(), counts))


if __name__ == "__main__":
    main()
