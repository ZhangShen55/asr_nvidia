#!/usr/bin/env python3
"""Compare seaco JSON, direct FunASR, and FireRedVAD intervals."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = ROOT / "test/audio_file/音频文件"
FUNASR_DIR = ROOT / "test/artifacts/fireredvad/funasr-final-results"
FIRERED_DIR = ROOT / "test/artifacts/fireredvad/firered-gpu-results"
OUT_DIR = ROOT / "test/artifacts/fireredvad/three-baseline-comparison"


def quantile(values, q):
    values = sorted(values)
    if not values:
        return 0.0
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (pos - lo)


def dist(values):
    return {"count": len(values), "mean": statistics.mean(values) if values else 0.0,
            "p50": quantile(values, .5), "p95": quantile(values, .95),
            "min": min(values) if values else 0.0, "max": max(values) if values else 0.0}


def parse_intervals(data, source):
    out = []
    for seg in data.get("segments", []):
        try:
            if source == "json":
                a, b = float(seg["bg"]), float(seg["ed"])
            else:
                a, b = float(seg["start_s"]), float(seg["end_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if b > a:
            out.append((a, b))
    return out


def path_key(kind, rel):
    return hashlib.sha1((kind + "\0" + rel).encode()).hexdigest()[:16]


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def iou(a, b):
    ov = overlap(a, b)
    return ov / (max(a[1], b[1]) - min(a[0], b[0])) if ov else 0.0


def summarize(xs, duration):
    lengths = [b - a for a, b in xs]
    gaps = [max(0.0, xs[i + 1][0] - xs[i][1]) for i in range(len(xs) - 1)]
    speech = sum(lengths)
    return {"segments": len(xs), "speech_s": speech, "coverage": speech / duration if duration else 0.0,
            "segments_per_min": len(xs) / duration * 60 if duration else 0.0,
            "length_s": dist(lengths), "gap_s": dist(gaps),
            "short_lt_1s": sum(v < 1 for v in lengths), "long_ge_10s": sum(v >= 10 for v in lengths),
            "lengths_s": lengths}


def pair(a, b):
    a_to_b = [sum(overlap(x, y) > 0 for y in b) for x in a]
    b_to_a = [sum(overlap(x, y) > 0 for x in a) for y in b]
    starts, ends, ious = [], [], []
    for x in a:
        candidates = [y for y in b if overlap(x, y) > 0]
        if candidates:
            y = max(candidates, key=lambda z: iou(x, z))
            starts.append(abs(x[0] - y[0])); ends.append(abs(x[1] - y[1])); ious.append(iou(x, y))
    return {"count_delta": len(b) - len(a), "count_ratio": len(b) / len(a) if a else None,
            "coverage_delta_s": sum(y[1] - y[0] for y in b) - sum(x[1] - x[0] for x in a),
            "segments_a_split_by_b": sum(v >= 2 for v in a_to_b),
            "segments_b_merge_a": sum(v >= 2 for v in b_to_a),
            "segments_a_without_b_overlap": sum(v == 0 for v in a_to_b),
            "segments_b_without_a_overlap": sum(v == 0 for v in b_to_a),
            "boundary_start_s": dist(starts), "boundary_end_s": dist(ends), "best_iou": dist(ious)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for jp in sorted(AUDIO_ROOT.rglob("seaco_paraformer-*.json")):
        stem = jp.name[len("seaco_paraformer-"):-5]
        wav = jp.with_name(stem + ".wav")
        rel = str(wav.relative_to(AUDIO_ROOT))
        fk = FUNASR_DIR / (path_key("funasr", rel) + ".json")
        rk = FIRERED_DIR / (path_key("fireredvad", rel) + ".json")
        if not wav.exists() or not fk.exists() or not rk.exists():
            missing.append(rel); continue
        j = json.loads(jp.read_text(encoding="utf-8")); f = json.loads(fk.read_text()); r = json.loads(rk.read_text())
        duration = float(r["duration_s"])
        xs = {"json": parse_intervals(j, "json"), "funasr": parse_intervals(f, "result"), "fireredvad": parse_intervals(r, "result")}
        summary = {k: summarize(v, duration) for k, v in xs.items()}
        pairs = {"json_vs_funasr": pair(xs["json"], xs["funasr"]),
                 "json_vs_fireredvad": pair(xs["json"], xs["fireredvad"]),
                 "funasr_vs_fireredvad": pair(xs["funasr"], xs["fireredvad"])}
        rows.append({"wav": rel, "duration_s": duration, "summary": summary, "pairs": pairs,
                     "funasr_elapsed_s": f.get("elapsed_s"), "firered_elapsed_s": r.get("elapsed_s")})

    total_duration = sum(x["duration_s"] for x in rows)
    aggregate = {"files": len(rows), "duration_s": total_duration, "baselines": {}, "pairs": {}}
    for name in ("json", "funasr", "fireredvad"):
        lengths = [z for row in rows for z in row["summary"][name]["lengths_s"]]
        gaps = [z for row in rows for z in row["summary"][name]["gap_s"]["lengths_s"]] if False else []
        aggregate["baselines"][name] = {
            "segments": sum(row["summary"][name]["segments"] for row in rows),
            "speech_s": sum(row["summary"][name]["speech_s"] for row in rows),
            "coverage": sum(row["summary"][name]["speech_s"] for row in rows) / total_duration,
            "segments_per_min": sum(row["summary"][name]["segments"] for row in rows) / total_duration * 60,
            "length_s": dist(lengths),
            "short_lt_1s": sum(row["summary"][name]["short_lt_1s"] for row in rows),
            "long_ge_10s": sum(row["summary"][name]["long_ge_10s"] for row in rows),
        }
    for key in ("json_vs_funasr", "json_vs_fireredvad", "funasr_vs_fireredvad"):
        aggregate["pairs"][key] = {
            "count_delta": sum(row["pairs"][key]["count_delta"] for row in rows),
            "count_ratio": aggregate["baselines"][key.split("_vs_")[1]]["segments"] / aggregate["baselines"][key.split("_vs_")[0]]["segments"],
            "segments_a_split_by_b": sum(row["pairs"][key]["segments_a_split_by_b"] for row in rows),
            "segments_b_merge_a": sum(row["pairs"][key]["segments_b_merge_a"] for row in rows),
            "segments_a_without_b_overlap": sum(row["pairs"][key]["segments_a_without_b_overlap"] for row in rows),
            "segments_b_without_a_overlap": sum(row["pairs"][key]["segments_b_without_a_overlap"] for row in rows),
            "boundary_start_s_mean_per_file": statistics.mean(row["pairs"][key]["boundary_start_s"]["mean"] for row in rows),
            "boundary_end_s_mean_per_file": statistics.mean(row["pairs"][key]["boundary_end_s"]["mean"] for row in rows),
            "best_iou_mean_per_file": statistics.mean(row["pairs"][key]["best_iou"]["mean"] for row in rows),
        }
    payload = {"rows": rows, "missing": missing, "aggregate": aggregate}
    out = OUT_DIR / "comparison.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(rows), "missing": len(missing), "aggregate": aggregate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
