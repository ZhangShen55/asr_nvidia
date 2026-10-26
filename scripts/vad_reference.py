"""Reference-audio VAD regression helpers and optional threshold scanner.

The supplied annotation file is a human-edited JSON-like document, so this module
deliberately parses its segment objects without rewriting the source fixture.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import wave
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

import numpy as np

# Allow the documented ``python scripts/vad_reference.py`` invocation to import
# repository packages when the script directory is placed first on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import settings
from core.vad import VadSegment, _parse_vad_result
from core.vad_pool import VadPool

DEFAULT_AUDIO = Path("test/audio_file/理论计算机科学导论_时间am9.55_191803.wav")
DEFAULT_ANNOTATION = Path("test/test_result/理论计算机科学导论_时间am9.55_191803.json")


def parse_reference_annotations(path: str | Path = DEFAULT_ANNOTATION) -> list[dict]:
    """Parse marked segments from the human-edited reference document."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("细小声音"):
            # The marker is inside the object, after its opening brace. Insert a
            # private JSON field immediately after that brace.
            for index in range(len(cleaned) - 1, -1, -1):
                if cleaned[index].strip() == "{":
                    indent = cleaned[index][: len(cleaned[index]) - len(cleaned[index].lstrip())] + "    "
                    cleaned.insert(index + 1, f'{indent}"__small_noise__": true,')
                    break
            continue
        # Human notes such as "转写错误应该是 ..." are intentionally skipped.
        if stripped and not stripped.startswith(('"', "{", "}", "[", "]", ",")):
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)
    # A few hand-edited objects omit the comma before the bg field.
    text = re.sub(
        r'("segment_text"\s*:\s*"(?:[^"\\]|\\.)*")\s*(?="bg")',
        r"\1,\n",
        text,
    )
    payload = json.loads(text)
    segments = []
    for segment in payload.get("segments", []):
        try:
            item = {
                "bg": float(segment["bg"]),
                "ed": float(segment["ed"]),
                "segment_text": str(segment.get("segment_text", "")),
                "small_noise": bool(segment.get("__small_noise__", False)),
            }
        except (KeyError, TypeError, ValueError):
            continue
        segments.append(item)
    return segments


def overlap_ratio(candidate: VadSegment | Sequence[float], marked: Sequence[float]) -> float:
    """Return candidate overlap divided by marked interval duration."""
    start, end = (candidate.start_s, candidate.end_s) if isinstance(candidate, VadSegment) else candidate
    marked_start, marked_end = float(marked[0]), float(marked[1])
    duration = max(0.0, marked_end - marked_start)
    if duration <= 0:
        return 0.0
    return max(0.0, min(float(end), marked_end) - max(float(start), marked_start)) / duration


def interval_hit(candidates: Iterable[VadSegment | Sequence[float]], marked: Sequence[float], threshold: float = 0.5) -> bool:
    """A marked interval is detected when any VAD span covers at least 50%."""
    return any(overlap_ratio(candidate, marked) >= threshold for candidate in candidates)


def _audio_diagnostics(path: Path, annotations: list[dict]) -> dict:
    marked_rms: list[float] = []
    other_rms: list[float] = []
    marked_peak: list[float] = []
    other_peak: list[float] = []
    try:
        with wave.open(str(path), "rb") as stream:
            rate = stream.getframerate()
            frames = stream.readframes(stream.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        for item in annotations:
            start = max(0, int(item["bg"] * rate))
            end = min(len(audio), int(item["ed"] * rate))
            chunk = audio[start:end]
            if not len(chunk):
                continue
            rms = 20 * np.log10(max(float(np.sqrt(np.mean(chunk * chunk))), 1e-12))
            peak = 20 * np.log10(max(float(np.max(np.abs(chunk))), 1e-12))
            item["rms_dbfs"] = rms
            item["peak_dbfs"] = peak
            (marked_rms if item["small_noise"] else other_rms).append(rms)
            (marked_peak if item["small_noise"] else other_peak).append(peak)
    except (OSError, wave.Error):
        pass
    return {
        "marked_rms_median_dbfs": round(median(marked_rms), 2) if marked_rms else None,
        "unmarked_rms_median_dbfs": round(median(other_rms), 2) if other_rms else None,
        "marked_peak_median_dbfs": round(median(marked_peak), 2) if marked_peak else None,
        "unmarked_peak_median_dbfs": round(median(other_peak), 2) if other_peak else None,
    }


async def scan_thresholds(
    audio_path: str | Path = DEFAULT_AUDIO,
    annotation_path: str | Path = DEFAULT_ANNOTATION,
    thresholds: Iterable[float] = (0.3, 0.6, 0.8),
    pool_factory=VadPool,
) -> dict:
    """Run the optional long test and return a JSON-serializable report."""
    annotations = parse_reference_annotations(annotation_path)
    marked = [item for item in annotations if item["small_noise"]]
    unmarked = [item for item in annotations if not item["small_noise"]]
    report = {"audio": str(audio_path), "annotation": str(annotation_path), "thresholds": []}
    report["annotation_count"] = len(marked)
    report["diagnostics"] = _audio_diagnostics(Path(audio_path), annotations)
    for threshold in thresholds:
        started = time.perf_counter()
        pool = pool_factory(vad_kwargs={
            "max_single_segment_time": settings.qwen_vad_max_single_segment_time_ms,
            "max_end_silence_time": settings.qwen_vad_max_end_silence_time_ms,
            "speech_noise_thres": float(threshold),
        })
        try:
            raw = await pool.detect(str(audio_path))
            candidates = _parse_vad_result(raw)
        finally:
            await pool.stop()
        marked_hits = sum(interval_hit(candidates, (item["bg"], item["ed"])) for item in marked)
        unmarked_hits = sum(interval_hit(candidates, (item["bg"], item["ed"])) for item in unmarked)
        report["thresholds"].append({
            "threshold": float(threshold),
            "marked_hit_rate": round(marked_hits / len(marked), 4) if marked else 0.0,
            "marked_suppression_rate": round(1 - marked_hits / len(marked), 4) if marked else 0.0,
            "unmarked_retention_rate": round(unmarked_hits / len(unmarked), 4) if unmarked else 0.0,
            "unmarked_false_positive_rate": round(unmarked_hits / len(unmarked), 4) if unmarked else 0.0,
            "filtered_segment_count": len(candidates),
            "qwen_request_count": len(candidates),
            "elapsed_s": round(time.perf_counter() - started, 3),
        })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="扫描细小声音参考音频的 VAD 阈值")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.3, 0.6, 0.8])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = asyncio.run(scan_thresholds(args.audio, args.annotation, args.thresholds))
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
