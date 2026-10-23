"""效果比对：调用 /v2/seacraft_asr 生成新结果，并与变更前的基线结果对比。

用法：
    python scripts/run_effect_compare.py --base-url http://127.0.0.1:8083 \
        --out-dir /tmp/asr_run/after_change

基线结果（变更前）放在 --baseline-dir，文件名为 after_skip_<音频名>.json。
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

AUDIO_DIR = "test/audio_file"

# 基线文件名 -> 音频文件名
PAIRS = {
    "after_skip_教师1.json": "教师1.wav",
    "after_skip_储能.json": "储能学术英语实践.wav",
    "after_skip_课堂实录.json": "课堂实录.wav",
}

PUNCTUATION = set("，。！？；：、,.?!;:~…—–“”‘’\"'()（）《》〈〉【】〔〕「」『』")


def post(base_url: str, path: str, timeout: int = 3600) -> dict:
    started = time.perf_counter()
    with open(path, "rb") as handle:
        response = requests.post(
            f"{base_url.rstrip('/')}/v2/seacraft_asr",
            files={"audioFile": (os.path.basename(path), handle, "audio/wav")},
            data={"showSpk": "true", "showEmotion": "true"},
            timeout=timeout,
        )
    payload = response.json()
    payload["_elapsed_s"] = round(time.perf_counter() - started, 2)
    return payload


def punctuation_issues(payload: dict) -> dict:
    """统计标点卫生规则的违例数。"""
    segments = payload.get("segments") or []
    only_punct = 0
    leading = 0
    for segment in segments:
        text = str(segment.get("segment_text", "") or "").strip()
        if not text:
            continue
        if all(char in PUNCTUATION for char in text):
            only_punct += 1
        if text[0] in PUNCTUATION:
            leading += 1
    return {
        "segments": len(segments),
        "pure_punctuation_segments": only_punct,
        "leading_punctuation_segments": leading,
    }


def char_similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)
    return round(matcher.ratio(), 4)


def compare(baseline: dict, current: dict) -> dict:
    base_segments = baseline.get("segments") or []
    new_segments = current.get("segments") or []
    base_text = str(baseline.get("text") or "")
    new_text = str(current.get("text") or "")
    base_ratio = baseline.get("language_ratio") or {}
    new_ratio = current.get("language_ratio") or {}
    return {
        "baseline_segments": len(base_segments),
        "new_segments": len(new_segments),
        "text_similarity": char_similarity(base_text, new_text),
        "baseline_language_ratio": base_ratio,
        "new_language_ratio": new_ratio,
        "ratio_key_changed": set(base_ratio) != set(new_ratio),
        "baseline_punctuation": punctuation_issues(baseline),
        "new_punctuation": punctuation_issues(current),
        "elapsed_s": current.get("_elapsed_s"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8083")
    parser.add_argument("--audio-dir", default=AUDIO_DIR)
    parser.add_argument("--baseline-dir", default="/tmp/asr_run")
    parser.add_argument("--out-dir", default="/tmp/asr_run/after_change")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    def _one(item):
        baseline_name, audio_name = item
        audio_path = os.path.join(args.audio_dir, audio_name)
        out_path = os.path.join(args.out_dir, baseline_name)
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as handle:
                return baseline_name, audio_name, json.load(handle)
        payload = post(args.base_url, audio_path)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
        return baseline_name, audio_name, payload

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(_one, PAIRS.items()))

    report = {}
    for baseline_name, audio_name, payload in results:
        baseline_path = os.path.join(args.baseline_dir, baseline_name)
        with open(baseline_path, "r", encoding="utf-8") as handle:
            baseline = json.load(handle)
        report[audio_name] = compare(baseline, payload)

    report_path = os.path.join(args.out_dir, "compare_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
