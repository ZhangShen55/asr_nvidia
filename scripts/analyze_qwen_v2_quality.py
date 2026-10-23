"""对压测落盘的原始响应做效果分析：幻觉重复、标点卫生、文本连贯性。

用法：
    python scripts/analyze_qwen_v2_quality.py --bench-dir /tmp/bench_qwen/phaseA
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

PUNCTUATION = set("，。？！；：、,.?!;:…—“”‘’\"'()（）《》【】[]{}-—～~·")
SENTENCE_END = set("。！？!?…")
CJK = re.compile(r"[\u4e00-\u9fff]")
SHINGLE = 12

# 段内长文本重复：同一子串连续重复 >= 3 次
LONG_LOOP_RE = re.compile(r"(.{3,120}?)\1{2,}")
# 字符级复读：同一字符连续出现 >= 6 次
CHAR_LOOP_RE = re.compile(r"(.)\1{5,}")
# 跨段幻觉候选：同一 12 字 shingle 出现在 >= 3 个不同分段


def analyze_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    segments = payload.get("segments") or []
    texts = [str(s.get("segment_text", "") or "").strip() for s in segments]
    non_empty = [t for t in texts if t]

    pure_punct = sum(1 for t in non_empty if all(ch in PUNCTUATION for ch in t))
    leading_punct = sum(1 for t in non_empty if t[0] in PUNCTUATION)
    empty = len(texts) - len(non_empty)

    long_loop_segments = 0
    longest_loop = 0
    longest_loop_text = ""
    char_loop_segments = 0
    longest_char_run = 0
    for t in non_empty:
        m = LONG_LOOP_RE.search(t)
        if m:
            long_loop_segments += 1
            if len(m.group(1)) > longest_loop:
                longest_loop = len(m.group(1))
                longest_loop_text = m.group(0)[:80]
        cm = CHAR_LOOP_RE.search(t)
        if cm:
            char_loop_segments += 1
            longest_char_run = max(longest_char_run, len(cm.group(0)))

    # 跨段重复 shingle
    shingle_seg: Dict[str, set] = defaultdict(set)
    for idx, t in enumerate(non_empty):
        for i in range(0, max(0, len(t) - SHINGLE + 1)):
            shingle_seg[t[i:i + SHINGLE]].add(idx)
    cross_repeat = {k: v for k, v in shingle_seg.items() if len(v) >= 3}
    cross_repeat_max = max((len(v) for v in cross_repeat.values()), default=0)

    consecutive_dup = 0
    prev = None
    for t in non_empty:
        if prev is not None and t == prev:
            consecutive_dup += 1
        prev = t

    # 连贯性代理指标
    end_punct = sum(1 for t in non_empty if t[-1] in SENTENCE_END)
    cjk_ratio = (
        sum(len(CJK.findall(t)) for t in non_empty) / max(1, sum(len(t) for t in non_empty))
    )
    prefix_residue = sum(1 for t in non_empty if "asr_text" in t or "<|" in t)

    return {
        "segments": len(segments),
        "non_empty_segments": len(non_empty),
        "chars": sum(len(t) for t in non_empty),
        "avg_segment_chars": round(statistics.mean([len(t) for t in non_empty]), 2)
        if non_empty
        else 0.0,
        "avg_sentence_chars": round(
            statistics.mean([len(t) for t in non_empty]) if non_empty else 0.0, 2
        ),
        "pure_punctuation_segments": pure_punct,
        "leading_punctuation_segments": leading_punct,
        "empty_segments": empty,
        "long_loop_segments": long_loop_segments,
        "longest_loop": longest_loop,
        "longest_loop_text": longest_loop_text,
        "char_loop_segments": char_loop_segments,
        "longest_char_run": longest_char_run,
        "cross_segment_repeat_shingles": len(cross_repeat),
        "cross_segment_repeat_max": cross_repeat_max,
        "consecutive_duplicate_segments": consecutive_dup,
        "sentence_end_punct_rate": round(end_punct / len(non_empty), 4) if non_empty else 0.0,
        "cjk_char_ratio": round(cjk_ratio, 4),
        "prefix_residue_segments": prefix_residue,
        "language": payload.get("language"),
        "language_ratio": payload.get("language_ratio"),
    }


def collect(bench_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """按 responses_<mode>_c<N> 目录分组收集。"""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for d in sorted(bench_dir.glob("responses_*")):
        per_file = []
        for f in sorted(d.glob("*.json")):
            try:
                payload = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not payload:
                continue
            item = analyze_response(payload)
            item["file"] = f.name
            per_file.append(item)
        if per_file:
            groups[d.name] = per_file
    return groups


def aggregate(per_file: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "segments",
        "chars",
        "pure_punctuation_segments",
        "leading_punctuation_segments",
        "empty_segments",
        "long_loop_segments",
        "char_loop_segments",
        "cross_segment_repeat_shingles",
        "consecutive_duplicate_segments",
        "prefix_residue_segments",
    ]
    agg = {k: sum(x[k] for x in per_file) for k in keys}
    agg["files"] = len(per_file)
    agg["longest_loop"] = max((x["longest_loop"] for x in per_file), default=0)
    agg["longest_char_run"] = max((x["longest_char_run"] for x in per_file), default=0)
    agg["cross_segment_repeat_max"] = max(
        (x["cross_segment_repeat_max"] for x in per_file), default=0
    )
    agg["avg_segment_chars"] = round(
        statistics.mean([x["avg_segment_chars"] for x in per_file]), 2
    ) if per_file else 0.0
    agg["sentence_end_punct_rate"] = round(
        statistics.mean([x["sentence_end_punct_rate"] for x in per_file]), 4
    ) if per_file else 0.0
    agg["cjk_char_ratio"] = round(
        statistics.mean([x["cjk_char_ratio"] for x in per_file]), 4
    ) if per_file else 0.0
    worst = sorted(per_file, key=lambda x: -x["long_loop_segments"])[:3]
    agg["worst_files"] = [
        {
            "file": w["file"],
            "long_loop_segments": w["long_loop_segments"],
            "longest_loop": w["longest_loop"],
            "longest_loop_text": w["longest_loop_text"],
            "cross_segment_repeat_max": w["cross_segment_repeat_max"],
        }
        for w in worst
    ]
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-dir", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    bench_dir = Path(args.bench_dir)
    groups = collect(bench_dir)
    result = {name: aggregate(files) for name, files in groups.items()}

    print(f"{'分组':<34}{'文件':>5}{'段数':>8}{'纯标点':>8}{'段首标点':>10}"
          f"{'长复读':>8}{'字符复读':>10}{'跨段重复':>10}{'连续重复':>10}{'句末标点率':>12}")
    for name, agg in result.items():
        print(
            f"{name:<34}{agg['files']:>5}{agg['segments']:>8}"
            f"{agg['pure_punctuation_segments']:>8}{agg['leading_punctuation_segments']:>10}"
            f"{agg['long_loop_segments']:>8}{agg['char_loop_segments']:>10}"
            f"{agg['cross_segment_repeat_shingles']:>10}"
            f"{agg['consecutive_duplicate_segments']:>10}"
            f"{agg['sentence_end_punct_rate']:>12}"
        )
    print()
    for name, agg in result.items():
        print(f"[{name}] 最差样例：")
        for w in agg["worst_files"]:
            print(f"   {w['file'][:60]} 复读段={w['long_loop_segments']} "
                  f"最长={w['longest_loop']} 跨段={w['cross_segment_repeat_max']} "
                  f"样例={w['longest_loop_text'][:50]!r}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n结果已写入 {args.out}")


if __name__ == "__main__":
    main()
