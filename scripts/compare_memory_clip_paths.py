"""Compare the legacy temporary-WAV clip with the in-memory WAV clip.

The comparison is sample-level: both paths receive the same VAD-like start and
duration windows from a real source WAV, then their decoded mono PCM, sample
rate and duration are compared.  This is the deterministic part of the old
and new Qwen upload contract and avoids sending duplicate long recordings to
Qwen during a regression check.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from utils.audio_utils import encode_audio_clip_wav, read_audio_array, save_audio_clip_array


def compare_file(path: Path, windows: list[tuple[float, float]]) -> dict:
    source, sample_rate = read_audio_array(str(path))
    source = source.reshape(-1).astype("float32")
    rows = []
    for start_s, duration_s in windows:
        start = max(0, int(start_s * sample_rate))
        end = min(len(source), int((start_s + duration_s) * sample_rate))
        array = source[start:end]
        legacy_path = save_audio_clip_array(array, sample_rate)
        try:
            old, old_rate = read_audio_array(legacy_path)
            new, new_rate = sf.read(io.BytesIO(encode_audio_clip_wav(array, sample_rate)), dtype="float32")
        finally:
            try:
                os.remove(legacy_path)
            except FileNotFoundError:
                pass
        new = np.asarray(new).reshape(-1)
        old = np.asarray(old).reshape(-1)
        rows.append({
            "start_s": start_s,
            "duration_s": duration_s,
            "sample_rate_equal": int(old_rate) == int(new_rate) == int(sample_rate),
            "sample_count_equal": len(old) == len(new) == len(array),
            "pcm_equal": bool(np.array_equal(old, new)),
            "max_abs_delta": float(np.max(np.abs(old - new))) if len(old) and len(old) == len(new) else None,
        })
    return {"audio": str(path), "sample_rate": sample_rate, "windows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    windows = [(0.0, 2.0), (17.25, 3.5), (120.0, 8.0)]
    report = {
        "windows": windows,
        "files": [compare_file(path, windows) for path in args.audio],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
