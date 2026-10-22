"""对比 /v1.1.8 与 /v2 的转写效果：文本一致性、漏字/重复、时间戳对齐。"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter

import requests


def _chars(text):
    return Counter(re.sub(r"[\s，。！？、,.!?；;：:\"'“”‘’（）()\-—…]+", "", text or ""))


def window_coverage(segs_a, segs_b, window_s=30.0):
    """按固定时间窗比较两条链路的字符覆盖情况。

    字符级 SequenceMatcher 在两条链路切分差异大时会整体塌陷（对齐崩塌），
    这里改用「同一时间窗内的字符多重集覆盖率」，衡量的是同一段时间里
    双方说了同样的内容，指标与切分方式无关。
    """
    if not segs_a or not segs_b:
        return {}
    end = max(float(s.get("ed") or 0) for s in segs_a + segs_b)
    total_a = Counter()
    total_b = Counter()
    matched = 0
    windows = 0
    for start in range(0, int(end) + 1, int(window_s)):
        stop = start + window_s
        ca = Counter()
        cb = Counter()
        for seg in segs_a:
            if float(seg.get("ed") or 0) > start and float(seg.get("bg") or 0) < stop:
                ca.update(_chars(seg.get("segment_text")))
        for seg in segs_b:
            if float(seg.get("ed") or 0) > start and float(seg.get("bg") or 0) < stop:
                cb.update(_chars(seg.get("segment_text")))
        if not ca and not cb:
            continue
        windows += 1
        total_a.update(ca)
        total_b.update(cb)
        if ca and cb:
            overlap = sum((ca & cb).values())
            if overlap >= 0.5 * min(sum(ca.values()), sum(cb.values())):
                matched += 1
    covered = sum((total_a & total_b).values())
    return {
        "window_s": window_s,
        "windows": windows,
        "windows_matched": matched,
        "window_match_rate": round(matched / windows, 4) if windows else 0.0,
        "v118_chars_covered_by_v2": round(covered / sum(total_a.values()), 4) if total_a else 0.0,
        "v2_chars_covered_by_v118": round(covered / sum(total_b.values()), 4) if total_b else 0.0,
    }


def timestamp_offset(segs_a, segs_b, min_ratio=0.5):
    """比较两链路对同一句内容给出的时间戳偏差（秒）。"""
    offsets = []
    for seg_b in segs_b:
        text_b = normalize(seg_b.get("segment_text"))
        if not text_b:
            continue
        best = None
        for seg_a in segs_a:
            text_a = normalize(seg_a.get("segment_text"))
            if not text_a:
                continue
            ratio = difflib.SequenceMatcher(None, text_a, text_b).ratio()
            if ratio >= min_ratio and (best is None or ratio > best[0]):
                best = (ratio, seg_a)
        if best is None:
            continue
        offsets.append(
            {
                "ratio": round(best[0], 3),
                "start_diff_s": round(float(seg_b.get("bg") or 0) - float(best[1].get("bg") or 0), 3),
                "end_diff_s": round(float(seg_b.get("ed") or 0) - float(best[1].get("ed") or 0), 3),
            }
        )
    if not offsets:
        return {"matched_segments": 0}
    start_diffs = sorted(abs(o["start_diff_s"]) for o in offsets)
    end_diffs = sorted(abs(o["end_diff_s"]) for o in offsets)

    def median(arr):
        return arr[len(arr) // 2]

    return {
        "matched_segments": len(offsets),
        "start_offset_median_s": median(start_diffs),
        "start_offset_p90_s": start_diffs[min(len(start_diffs) - 1, int(len(start_diffs) * 0.9))],
        "end_offset_median_s": median(end_diffs),
        "end_offset_p90_s": end_diffs[min(len(end_diffs) - 1, int(len(end_diffs) * 0.9))],
    }


def post(base_url, path, audio, **fields):
    data = {k: v for k, v in fields.items()}
    import os

    with open(audio, "rb") as handle:
        response = requests.post(
            f"{base_url.rstrip('/')}{path}",
            files={"audioFile": (os.path.basename(audio), handle, "audio/wav")},
            data=data,
            timeout=3600,
        )
    try:
        return response.json()
    except ValueError:
        print(f"[错误] {path} 返回非 JSON，状态码 {response.status_code}: {response.text[:300]}")
        raise


def normalize(text):
    return re.sub(r"[\s，。！？、,.!?；;：:\"'“”‘’（）()\-—…]+", "", text or "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8083")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    v118 = post(args.base_url, "/v1.1.8/seacraft_asr", args.audio, showSpk="true")
    v2 = post(args.base_url, "/v2/seacraft_asr", args.audio, showSpk="true", showEmotion="true")

    text_118 = normalize(v118.get("text", ""))
    text_v2 = normalize(v2.get("text", ""))

    ratio = difflib.SequenceMatcher(None, text_118, text_v2).ratio()
    matcher = difflib.SequenceMatcher(None, text_118, text_v2)
    diffs = [
        {
            "op": tag,
            "v118": text_118[i1:i2][:60],
            "v2": text_v2[j1:j2][:60],
        }
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    ]

    # 重复检测：v2 文本中连续重复 4 次以上的片段
    repeated = [m.group(0) for m in re.finditer(r"(.+?)\1{3,}", text_v2)]
    # 幻觉检测：v2 中出现而 v1.1.8 没有的较长片段
    hallucination = [d for d in diffs if d["op"] == "insert" and len(d["v2"]) >= 8]

    result = {
        "audio": args.audio,
        "v118_segments": len(v118.get("segments", []) or []),
        "v2_segments": len(v2.get("segments", []) or []),
        "v118_chars": len(text_118),
        "v2_chars": len(text_v2),
        "char_ratio": round(ratio, 4),
        "diff_count": len(diffs),
        "diff_sample": diffs[:20],
        "repeated_patterns": repeated[:5],
        "hallucination_like": hallucination[:10],
        "v118_language": v118.get("language"),
        "v2_language": v2.get("language"),
        "v2_language_ratio": v2.get("language_ratio"),
        "v2_gpu_time_ms": v2.get("gpu_time_ms"),
    }

    roles_118 = [s.get("role") for s in v118.get("segments", []) or []]
    roles_v2 = [s.get("role") for s in v2.get("segments", []) or []]
    emotions_v2 = [s.get("emotion") for s in v2.get("segments", []) or []]
    result["v118_role_dist"] = {r: roles_118.count(r) for r in set(roles_118)}
    result["v2_role_dist"] = {r: roles_v2.count(r) for r in set(roles_v2)}
    result["v2_emotion_dist"] = {e: emotions_v2.count(e) for e in set(emotions_v2)}
    segs_118 = v118.get("segments") or []
    segs_v2 = v2.get("segments") or []
    result["window_coverage"] = window_coverage(segs_118, segs_v2)
    result["timestamp_offset"] = timestamp_offset(segs_118, segs_v2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
