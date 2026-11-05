#!/usr/bin/env python3
"""Evaluate FireRedVAD thresholds against inline `细小声音` annotations."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_FILE = ROOT / "test/test_result/理论计算机科学导论_时间am9.55_191803.json"
RESULT_DIRS = {
    "0.5": ROOT / "test/artifacts/fireredvad/theory-cs-threshold-05",
    "0.6": ROOT / "test/artifacts/fireredvad/theory-cs-threshold-06",
    "0.7": ROOT / "test/artifacts/fireredvad/theory-cs-threshold-07",
}
OUT = ROOT / "test/artifacts/fireredvad/theory-cs-threshold-comparison.json"


def annotations():
    lines = ANNOTATION_FILE.read_text(encoding="utf-8").splitlines()
    output = []
    for index, line in enumerate(lines):
        if "细小声音" not in line:
            continue
        block = "\n".join(lines[index : index + 15])
        match = re.search(r'"bg"\s*:\s*"([0-9.]+)"\s*,\s*"ed"\s*:\s*"([0-9.]+)"', block)
        if not match:
            raise RuntimeError(f"无法解析第 {index + 1} 行标注后的 bg/ed")
        output.append({"index": len(output) + 1, "line": index + 1,
                       "start_s": float(match.group(1)), "end_s": float(match.group(2))})
    return output


def result_segments(directory: Path):
    files = sorted(directory.glob("*.json"))
    if len(files) != 1:
        raise RuntimeError(f"{directory} 结果文件数量为 {len(files)}")
    data = json.loads(files[0].read_text(encoding="utf-8"))
    return data, [(float(x["start_s"]), float(x["end_s"])) for x in data["segments"]]


def evaluate(annotation, segments):
    start, end = annotation["start_s"], annotation["end_s"]
    length = end - start
    matches = []
    for seg_index, (a, b) in enumerate(segments, 1):
        overlap = max(0.0, min(end, b) - max(start, a))
        if overlap > 0:
            matches.append({"segment_index": seg_index, "start_s": a, "end_s": b,
                            "overlap_s": overlap, "annotation_coverage": overlap / length})
    best = max(matches, key=lambda x: x["overlap_s"]) if matches else None
    return {"annotation_index": annotation["index"], "line": annotation["line"],
            "start_s": start, "end_s": end, "duration_s": length,
            "detected_any_overlap": bool(matches),
            "detected_coverage_ge_50pct": bool(best and best["annotation_coverage"] >= 0.5),
            "detected_coverage_ge_80pct": bool(best and best["annotation_coverage"] >= 0.8),
            "best_match": best, "matches": matches}


def main():
    anns = annotations()
    output = {"annotation_file": str(ANNOTATION_FILE), "annotation_count": len(anns), "thresholds": {}}
    for threshold, directory in RESULT_DIRS.items():
        data, segments = result_segments(directory)
        rows = [evaluate(a, segments) for a in anns]
        output["thresholds"][threshold] = {
            "result_file": str(next(directory.glob("*.json"))),
            "elapsed_s": data.get("elapsed_s"), "segment_count": len(segments),
            "annotations": rows,
            "summary": {
                "any_overlap": sum(x["detected_any_overlap"] for x in rows),
                "coverage_ge_50pct": sum(x["detected_coverage_ge_50pct"] for x in rows),
                "coverage_ge_80pct": sum(x["detected_coverage_ge_80pct"] for x in rows),
                "missed_any_overlap": [x["annotation_index"] for x in rows if not x["detected_any_overlap"]],
                "missed_coverage_ge_50pct": [x["annotation_index"] for x in rows if not x["detected_coverage_ge_50pct"]],
            },
        }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v["summary"] for k, v in output["thresholds"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
