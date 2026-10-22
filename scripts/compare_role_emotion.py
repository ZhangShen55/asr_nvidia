"""对比 /v1.1.8 与 /v2 的 role / emotion 效果（按时间轴对齐后逐段比对）。

两条链路的 VAD 参数一致，但 v1.1.8（Paraformer）与 v2（Qwen）切出的段边界
并不完全相同，因此先按时间重叠把 v2 的段映射到 v1.1.8 的段上再比对。

用法：
    python scripts/compare_role_emotion.py --base-url http://127.0.0.1:8085 \
        --audio test/audio_file/课堂实录.wav --output /tmp/asr_run/role_cmp.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import requests


def post(base_url, path, audio, **fields):
    data = {k: v for k, v in fields.items()}
    with open(audio, "rb") as handle:
        response = requests.post(
            f"{base_url.rstrip('/')}{path}",
            files={"audioFile": (os.path.basename(audio), handle, "audio/wav")},
            data=data,
            timeout=7200,
        )
    try:
        return response.json()
    except ValueError:
        print(f"[错误] {path} 返回非 JSON，状态码 {response.status_code}")
        raise


def spans(segments):
    """返回 (start, end, segment) 列表，容忍 bg/ed 为字符串。"""
    result = []
    for seg in segments or []:
        try:
            start = float(seg.get("bg") or 0.0)
            end = float(seg.get("ed") or 0.0)
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        result.append((start, end, seg))
    return result


def align(v2_segments, v118_segments):
    """为每个 v2 段找出时间上重叠最多的 v1.1.8 段。"""
    refs = spans(v118_segments)
    mapping = []
    for start, end, seg in spans(v2_segments):
        best = None
        best_overlap = 0.0
        for ref_start, ref_end, ref in refs:
            overlap = min(end, ref_end) - max(start, ref_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best = ref
        mapping.append((seg, best, best_overlap))
    return mapping


def cluster_agreement(mapping, key_a, key_b):
    """按重叠做贪心最佳匹配，得到「聚类一致率」与映射表。"""
    pairs = defaultdict(Counter)
    for seg_a, seg_b, overlap in mapping:
        if seg_b is None or overlap <= 0:
            continue
        label_a = seg_a.get(key_a)
        label_b = seg_b.get(key_b)
        if label_a is None or label_b is None:
            continue
        pairs[label_a][label_b] += 1

    mapping_table = {}
    used = set()
    for label_a in sorted(pairs, key=lambda k: -sum(pairs[k].values())):
        candidates = [c for c in pairs[label_a] if c not in used]
        if not candidates:
            continue
        best = max(candidates, key=lambda c: pairs[label_a][c])
        mapping_table[label_a] = best
        used.add(best)

    matched = sum(pairs[a][b] for a, b in mapping_table.items())
    total = sum(sum(c.values()) for c in pairs.values())
    rate = matched / total if total else 0.0
    return round(rate, 4), mapping_table, matched, total


def semantic_class(label):
    """把 role 归到 teacher / student / spk 三大类。"""
    if not label:
        return None
    text = str(label)
    if text.startswith("teacher"):
        return "teacher"
    if text.startswith("student"):
        return "student"
    if text.startswith("spk"):
        return "spk"
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8085")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--role-identify", action="store_true")
    args = parser.parse_args()

    common = {"showSpk": "true", "showEmotion": "true"}
    if args.role_identify:
        common["showRoleIdentify"] = "true"

    v118 = post(args.base_url, "/v1.1.8/seacraft_asr", args.audio, **common)
    v2 = post(args.base_url, "/v2/seacraft_asr", args.audio, **common)

    mapping = align(v2.get("segments"), v118.get("segments"))

    role_rate, role_map, role_matched, role_total = cluster_agreement(
        mapping, "role", "role"
    )
    emotion_rate, emotion_map, emo_matched, emo_total = cluster_agreement(
        mapping, "emotion", "emotion"
    )

    # 语义级一致率（teacher / student 分类，不看 student 编号）
    semantic_hits = semantic_total = 0
    semantic_mismatch = []
    for seg_v2, seg_118, overlap in mapping:
        if seg_118 is None or overlap <= 0:
            continue
        cls_a = semantic_class(seg_v2.get("role"))
        cls_b = semantic_class(seg_118.get("role"))
        if cls_a is None or cls_b is None:
            continue
        semantic_total += 1
        if cls_a == cls_b:
            semantic_hits += 1
        elif len(semantic_mismatch) < 10:
            semantic_mismatch.append(
                {
                    "bg": seg_v2.get("bg"),
                    "ed": seg_v2.get("ed"),
                    "v2_role": seg_v2.get("role"),
                    "v118_role": seg_118.get("role"),
                }
            )

    result = {
        "audio": args.audio,
        "v118_segments": len(v118.get("segments") or []),
        "v2_segments": len(v2.get("segments") or []),
        "role_cluster_agreement": role_rate,
        "role_matched": role_matched,
        "role_total": role_total,
        "role_mapping": role_map,
        "emotion_agreement": emotion_rate,
        "emotion_matched": emo_matched,
        "emotion_total": emo_total,
        "emotion_mapping": emotion_map,
        "role_semantic_agreement": round(
            semantic_hits / semantic_total, 4
        ) if semantic_total else 0.0,
        "role_semantic_total": semantic_total,
        "role_semantic_mismatch": semantic_mismatch,
        "v118_role_dist": dict(
            Counter(s.get("role") for s in v118.get("segments") or [])
        ),
        "v2_role_dist": dict(
            Counter(s.get("role") for s in v2.get("segments") or [])
        ),
        "v118_emotion_dist": dict(
            Counter(s.get("emotion") for s in v118.get("segments") or [])
        ),
        "v2_emotion_dist": dict(
            Counter(s.get("emotion") for s in v2.get("segments") or [])
        ),
        "v2_language": v2.get("language"),
        "v2_language_ratio": v2.get("language_ratio"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
