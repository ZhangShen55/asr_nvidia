"""Compare direct FSMN-VAD output with the process-pool client.

Usage:
    conda run -n asr python scripts/verify_vad_pool_regression.py --device cpu
    conda run -n asr python scripts/verify_vad_pool_regression.py --device cuda:0

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

from core.vad import _parse_vad_result
from core.vad_pool import VadPool


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "model" / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
VAD_KWARGS = {
    "max_single_segment_time": 30000,
    "max_end_silence_time": 800,
}
AUDIO_FILES = (
    REPO_ROOT / "test/audio_file/储能学术英语实践.wav",
    REPO_ROOT / "test/audio_file/传播学_1124501_马道全_2025年4月24号8时30分.aac",
    REPO_ROOT
    / "test/audio_file/形式语言与自动机理论_1623301_1623302_1623303;1623304_康达周_2025年4月24号11时20分.aac",
)


def _spans(payload: Any) -> List[Tuple[float, float]]:
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return [(segment.start_s, segment.end_s) for segment in _parse_vad_result(payload)]
    return [(segment.start_s, segment.end_s) for segment in _parse_vad_result(payload)]


def _direct_model(device: str):
    from funasr import AutoModel

    return AutoModel(
        model=str(MODEL_DIR),
        device=device,
        vad_kwargs=VAD_KWARGS,
        disable_update=True,
        disable_pbar=True,
    )


def _direct_spans(model: Any, audio: Path) -> List[Tuple[float, float]]:
    payload = model.generate(
        input=str(audio),
        cache={},
        vad_kwargs=VAD_KWARGS,
    )
    return _spans(payload)


async def _pool_spans(pool: VadPool, audio: Path) -> List[Tuple[float, float]]:
    return _spans(await pool.detect(str(audio)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--audio", action="append", type=Path)
    args = parser.parse_args()
    audio_files = tuple(args.audio) if args.audio else AUDIO_FILES

    import asyncio

    direct = _direct_model(args.device)
    results = []
    pool = VadPool(
        pool_size=1,
        pool_max=1,
        watermark=0,
        queue_timeout_s=120,
        device=args.device,
        model_dir=str(MODEL_DIR),
        vad_kwargs=VAD_KWARGS,
    )

    async def run() -> None:
        await pool.start()
        try:
            for audio in audio_files:
                started = time.perf_counter()
                baseline = _direct_spans(direct, audio)
                baseline_s = time.perf_counter() - started
                started = time.perf_counter()
                pooled = await _pool_spans(pool, audio)
                pooled_s = time.perf_counter() - started
                same = baseline == pooled
                result = {
                    "audio": str(audio.relative_to(REPO_ROOT)),
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

    asyncio.run(run())
    if not all(result["same_boundaries"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
