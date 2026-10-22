"""并发边界测试：逐档位提高并发，检测幻觉 / 重复 / 空结果。

判据（以同一次构建的串行基线为参照）：
1. 空结果：某段转写为空；
2. 幻觉：与串行基线相似度 < 0.5 的段数占比；
3. 重复：段内出现长度 >= 6 且连续重复 >= 3 次的子串；
4. 失控：单段字符数显著超过基线最大值（> 3 倍）。
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import statistics
from concurrent.futures import ThreadPoolExecutor

import requests


def post(base_url, path, timeout=3600):
    with open(path, "rb") as handle:
        response = requests.post(
            f"{base_url.rstrip('/')}/v2/seacraft_asr",
            files={"audioFile": (os.path.basename(path), handle, "audio/wav")},
            timeout=timeout,
        )
    payload = response.json()
    return {
        "code": payload.get("code"),
        "segments": payload.get("segments") or [],
        "text": payload.get("text") or "",
    }


def norm(text):
    return re.sub(r"[\s，。！？、,.!?；;：:\"'“”‘’（）()\-—…]+", "", text or "")


def repeated_spans(text, min_len=6, min_repeat=3):
    """找出长度 >= min_len 且连续重复 >= min_repeat 次的子串。"""
    found = []
    pattern = rf"(.{{{min_len},}}?)\1{{{min_repeat - 1},}}"
    for m in re.finditer(pattern, text):
        found.append(m.group(0)[:60])
    return found


def compare_to_baseline(baseline, result):
    base = [norm(s.get("segment_text")) for s in baseline["segments"]]
    cur = [norm(s.get("segment_text")) for s in result["segments"]]
    if not base or not cur:
        return {"segments": 0, "similarity": 1.0, "low_similarity": 0, "empty": 0}
    matcher = difflib.SequenceMatcher(None, "".join(base), "".join(cur))
    ratio = matcher.ratio()
    low = 0
    for i in range(min(len(base), len(cur))):
        r = difflib.SequenceMatcher(None, base[i], cur[i]).ratio()
        if r < 0.5:
            low += 1
    return {
        "segments": len(cur),
        "similarity": round(ratio, 4),
        "low_similarity": low,
        "empty": sum(1 for t in cur if not t),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8085")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--levels", default="1,4,8,16,32")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    baseline = post(args.base_url, args.audio)
    base_chars = [len(norm(s.get("segment_text"))) for s in baseline["segments"]]
    base_max = max(base_chars) if base_chars else 0
    print(f"串行基线: {len(baseline['segments'])} 段, 总字符 {len(norm(baseline['text']))}")

    results = []
    for level in [int(x) for x in args.levels.split(",") if x.strip()]:
        with ThreadPoolExecutor(max_workers=level) as pool:
            samples = list(pool.map(lambda _: post(args.base_url, args.audio), range(level)))
        codes = [s["code"] for s in samples]
        ok = [s for s in samples if s["code"] in (None, 0)]
        stat = compare_to_baseline(baseline, ok[0]) if ok else {}
        repeats = []
        runaway = 0
        empty_total = 0
        for s in ok:
            empty_total += sum(1 for seg in s["segments"] if not norm(seg.get("segment_text")))
            for seg in s["segments"]:
                t = norm(seg.get("segment_text"))
                repeats.extend(repeated_spans(t))
                if base_max and len(t) > base_max * 3:
                    runaway += 1
        row = {
            "concurrency": level,
            "ok": len(ok),
            "failed": len(samples) - len(ok),
            "codes": sorted({c for c in codes if c is not None}),
            "segments": stat.get("segments"),
            "text_similarity_vs_serial": stat.get("similarity"),
            "low_similarity_segments": stat.get("low_similarity"),
            "empty_segments": empty_total,
            "repeat_patterns": repeats[:5],
            "runaway_segments": runaway,
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

