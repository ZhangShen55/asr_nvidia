#!/usr/bin/env python3
"""Compare FireRedVAD intervals with existing seaco Paraformer intervals."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = ROOT / "test/audio_file/音频文件"
FIRERED_RESULTS = ROOT / "test/artifacts/fireredvad/firered-gpu-results"
OUT_DIR = ROOT / "test/artifacts/fireredvad/paraformer-comparison"


def pct(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(values[lo])
    return float(values[lo] + (values[hi] - values[lo]) * (pos - lo))


def intervals_from_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if "segments" in data and isinstance(data["segments"], list):
        out = []
        for seg in data["segments"]:
            try:
                a, b = float(seg["bg"]), float(seg["ed"])
            except (KeyError, TypeError, ValueError):
                continue
            if b > a:
                out.append((a, b))
        return out
    return []


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def iou(a, b):
    ov = overlap(a, b)
    return ov / (max(a[1], b[1]) - min(a[0], b[0])) if ov else 0.0


def stats(values):
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else 0.0,
        "p50": pct(values, 0.50),
        "p95": pct(values, 0.95),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def result_path(wav: Path) -> Path:
    rel = str(wav.relative_to(AUDIO_ROOT))
    key = hashlib.sha1(("fireredvad\0" + rel).encode()).hexdigest()[:16]
    return FIRERED_RESULTS / f"{key}.json"


def compare_one(json_path: Path):
    wav_name = json_path.name[len("seaco_paraformer-") : -5]
    wav = json_path.with_name(wav_name + ".wav")
    para = intervals_from_json(json_path)
    fire_record = json.loads(result_path(wav).read_text(encoding="utf-8"))
    fire = [(float(x["start_s"]), float(x["end_s"])) for x in fire_record["segments"]]
    duration = float(fire_record["duration_s"])

    para_lengths = [b - a for a, b in para]
    fire_lengths = [b - a for a, b in fire]
    para_speech = sum(para_lengths)
    fire_speech = sum(fire_lengths)
    para_to_fire_counts = []
    fire_to_para_counts = []
    boundary_start, boundary_end, best_ious = [], [], []
    for p in para:
        ovs = [f for f in fire if overlap(p, f) > 0]
        para_to_fire_counts.append(len(ovs))
        if ovs:
            best = max(ovs, key=lambda f: iou(p, f))
            boundary_start.append(abs(p[0] - best[0]))
            boundary_end.append(abs(p[1] - best[1]))
            best_ious.append(iou(p, best))
    for f in fire:
        fire_to_para_counts.append(sum(1 for p in para if overlap(f, p) > 0))

    para_gaps = [max(0.0, para[i + 1][0] - para[i][1]) for i in range(len(para) - 1)]
    fire_gaps = [max(0.0, fire[i + 1][0] - fire[i][1]) for i in range(len(fire) - 1)]
    return {
        "json": str(json_path.relative_to(AUDIO_ROOT)),
        "wav": str(wav.relative_to(AUDIO_ROOT)),
        "duration_s": duration,
        "paraformer": {
            "count": len(para),
            "speech_s": para_speech,
            "coverage": para_speech / duration if duration else 0.0,
            "length_s": stats(para_lengths),
            "short_lt_1s": sum(x < 1 for x in para_lengths),
            "long_ge_10s": sum(x >= 10 for x in para_lengths),
            "gap_s": stats(para_gaps),
        },
        "fireredvad": {
            "count": len(fire),
            "speech_s": fire_speech,
            "coverage": fire_speech / duration if duration else 0.0,
            "length_s": stats(fire_lengths),
            "short_lt_1s": sum(x < 1 for x in fire_lengths),
            "long_ge_10s": sum(x >= 10 for x in fire_lengths),
            "gap_s": stats(fire_gaps),
        },
        "difference": {
            "count_delta": len(fire) - len(para),
            "count_ratio": len(fire) / len(para) if para else None,
            "coverage_delta": fire_speech / duration - para_speech / duration if duration else 0.0,
            "para_segments_split_by_fire": sum(x >= 2 for x in para_to_fire_counts),
            "fire_segments_merge_para": sum(x >= 2 for x in fire_to_para_counts),
            "para_segments_without_fire_overlap": sum(x == 0 for x in para_to_fire_counts),
            "fire_segments_without_para_overlap": sum(x == 0 for x in fire_to_para_counts),
            "boundary_start_s": stats(boundary_start),
            "boundary_end_s": stats(boundary_end),
            "best_iou": stats(best_ious),
        },
        "firered_elapsed_s": fire_record.get("elapsed_s", 0.0),
        "lengths_s": {"paraformer": para_lengths, "fireredvad": fire_lengths},
    }


def aggregate(rows):
    def collect(path):
        return [float(path(r)) for r in rows]

    def nested(model, key):
        return [float(r[model][key]) for r in rows]

    def diff(key):
        return [float(r["difference"][key]) for r in rows]

    return {
        "files": len(rows),
        "duration_s": sum(r["duration_s"] for r in rows),
        "paraformer": {
            "segments": sum(r["paraformer"]["count"] for r in rows),
            "speech_s": sum(r["paraformer"]["speech_s"] for r in rows),
            "coverage": sum(r["paraformer"]["speech_s"] for r in rows) / sum(r["duration_s"] for r in rows),
            "length_s": stats([x for r in rows for x in r["lengths_s"]["paraformer"]]),
            "short_lt_1s": sum(r["paraformer"]["short_lt_1s"] for r in rows),
            "long_ge_10s": sum(r["paraformer"]["long_ge_10s"] for r in rows),
        },
        "fireredvad": {
            "segments": sum(r["fireredvad"]["count"] for r in rows),
            "speech_s": sum(r["fireredvad"]["speech_s"] for r in rows),
            "coverage": sum(r["fireredvad"]["speech_s"] for r in rows) / sum(r["duration_s"] for r in rows),
            "length_s": stats([x for r in rows for x in r["lengths_s"]["fireredvad"]]),
            "short_lt_1s": sum(r["fireredvad"]["short_lt_1s"] for r in rows),
            "long_ge_10s": sum(r["fireredvad"]["long_ge_10s"] for r in rows),
        },
        "difference": {
            "count_delta": sum(diff("count_delta")),
            "count_ratio": sum(r["fireredvad"]["count"] for r in rows) / sum(r["paraformer"]["count"] for r in rows),
            "coverage_delta": sum(r["fireredvad"]["speech_s"] for r in rows) / sum(r["duration_s"] for r in rows) - sum(r["paraformer"]["speech_s"] for r in rows) / sum(r["duration_s"] for r in rows),
            "para_segments_split_by_fire": sum(diff("para_segments_split_by_fire")),
            "fire_segments_merge_para": sum(diff("fire_segments_merge_para")),
            "para_segments_without_fire_overlap": sum(diff("para_segments_without_fire_overlap")),
            "fire_segments_without_para_overlap": sum(diff("fire_segments_without_para_overlap")),
            "boundary_start_s": stats(collect(lambda r: r["difference"]["boundary_start_s"]["mean"])),
            "boundary_end_s": stats(collect(lambda r: r["difference"]["boundary_end_s"]["mean"])),
            "best_iou": stats(collect(lambda r: r["difference"]["best_iou"]["mean"])),
        },
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for p in sorted(AUDIO_ROOT.rglob("seaco_paraformer-*.json")):
        wav = p.with_name(p.name[len("seaco_paraformer-") : -5] + ".wav")
        if not wav.exists() or not result_path(wav).exists():
            missing.append(str(p.relative_to(AUDIO_ROOT)))
            continue
        rows.append(compare_one(p))
    payload = {"rows": rows, "missing": missing, "aggregate": aggregate(rows)}
    (OUT_DIR / "comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(rows), "missing": len(missing), "aggregate": payload["aggregate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
