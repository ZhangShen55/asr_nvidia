"""Compare direct FSMN-VAD output with the process-pool client.

Usage:
    conda run -n asr python scripts/verify_vad_pool_regression.py --device cpu
    conda run -n asr python scripts/verify_vad_pool_regression.py --device npu:7

The direct model is the pre-change execution path. The pool is constrained to
one worker here so the comparison measures result equivalence rather than
throughput or scheduling behavior.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "model" / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
VAD_KWARGS = {
    "max_single_segment_time": 30000,
    "max_end_silence_time": 800,
}



def _spans(payload: Any) -> List[Tuple[float, float]]:
    from core.vad import _parse_vad_result
    return [(segment.start_s, segment.end_s) for segment in _parse_vad_result(payload)]


def _direct_model(device: str):
    from core.vad_pool import build_vad_model
    return build_vad_model(str(MODEL_DIR), device, VAD_KWARGS)



def _direct_spans(model: Any, audio: Path) -> List[Tuple[float, float]]:
    payload = model.generate(
        input=str(audio),
        cache={},
        vad_kwargs=VAD_KWARGS,
    )
    return _spans(payload)


async def _pool_spans(pool: Any, audio: Path) -> List[Tuple[float, float]]:
    return _spans(await pool.detect(str(audio)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="cpu or npu:7 (container mapping: npu:0)")
    parser.add_argument("--audio", action="append", type=Path, required=True)
    args = parser.parse_args()
    from core.vad_pool import VadPool
    from core.local_inference import backend_for_device
    from scripts.bench_vad_cpu_vs_gpu import vad_kwargs
    backend_for_device(args.device)
    VAD_KWARGS.update(vad_kwargs())
    audio_files = tuple(args.audio)

    import asyncio

    direct = _direct_model(args.device)
    results = []
    pool = VadPool(
        devices=[args.device],
        model_dir=str(MODEL_DIR),
        vad_kwargs=VAD_KWARGS,
    )

    async def run() -> None:
        try:
            await pool.start()
            for audio in audio_files:
                started = time.perf_counter()
                baseline = _direct_spans(direct, audio)
                baseline_s = time.perf_counter() - started
                started = time.perf_counter()
                pooled = await _pool_spans(pool, audio)
                pooled_s = time.perf_counter() - started
                same = baseline == pooled
                result = {
                    "audio": str(audio),
                    "baseline_segments": len(baseline),
                    "pool_segments": len(pooled),
                    "same_boundaries": same,
                    "baseline_s": round(baseline_s, 3),
                    "pool_s": round(pooled_s, 3),
                }
                if not same:
                    result["first_difference"] = next(
                        (
                            {"index": i, "baseline": a, "pool": b}
                            for i, (a, b) in enumerate(zip(baseline, pooled))
                            if a != b
                        ),
                        {"index": min(len(baseline), len(pooled))},
                    )
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
        finally:
            await pool.stop()

    try:
        asyncio.run(run())
    finally:
        close = getattr(direct, "close", None)
        if close is not None:
            close()
    if not all(result["same_boundaries"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
