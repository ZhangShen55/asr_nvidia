"""CAM++ 说话人聚类阈值 sweep：抽一次嵌入，离线扫多个阈值。

步骤：
1. 本地 VAD 切分 + CAM++ 抽取每段嵌入，缓存到 .npz（重复运行不再重算）；
2. 可选拉取 /v1.1.8 的 role 作为参照，按时间重叠算「聚类一致率」；
3. 对每个阈值输出簇数与一致率，用于选定生产取值。

用法：
    python scripts/speaker_threshold_sweep.py --audio test/audio_file/课堂实录.wav \
        --base-url http://127.0.0.1:8085 --thresholds 0.35,0.4,0.45,0.5,0.6,0.7,0.75,0.8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes.asr_qwen_common import SegmentOutcome, cluster_speakers
from core.config import settings
from core.models import get_spk_model, load_v2_models_if_needed
from core.vad import VadSegment, detect_segments
from utils.audio_utils import (
    extract_audio_clip_array,
    extract_audio_clip_from_tensor,
    load_audio_tensor,
)


def collect_embeddings(audio_path):
    """抽取每个 VAD 段的说话人嵌入，返回 (时间区间列表, 嵌入矩阵)。"""
    segments = asyncio.run(detect_segments(audio_path))
    print(f"[VAD] 切出 {len(segments)} 段", flush=True)

    async def _run():
        await load_v2_models_if_needed(need_role=True, need_emotion=False)
        model = get_spk_model()
        if model is None:
            raise RuntimeError("CAM++ 模型未加载（检查 open_v2_role 开关）")

        audio_tensor, sample_rate = load_audio_tensor(audio_path)
        embeddings = []
        for index, segment in enumerate(segments):
            clip_path = extract_audio_clip_from_tensor(
                audio_tensor, sample_rate, segment.start_s, segment.duration_s
            )
            array, _ = extract_audio_clip_array(
                audio_tensor, sample_rate, segment.start_s, segment.duration_s
            )
            try:
                with torch.cuda.device(settings.model_device_speech_campplus):
                    result = model.generate(input=array, fs=sample_rate)
                payload = result[0]
                embedding = payload.get("spk_embedding")
                if embedding is None:
                    continue
                if not isinstance(embedding, torch.Tensor):
                    embedding = torch.tensor(embedding, dtype=torch.float32)
                embeddings.append(embedding.detach().float().reshape(-1).cpu())
            finally:
                os.unlink(clip_path)
            if (index + 1) % 20 == 0:
                print(f"  已抽取 {index + 1}/{len(segments)}", flush=True)
        return segments, embeddings

    segments, embeddings = asyncio.run(_run())
    return segments, torch.stack(embeddings) if embeddings else torch.zeros(0)


def fetch_reference(base_url, audio_path):
    """拉取 /v1.1.8 的 role 作为聚类参照。"""
    import requests

    with open(audio_path, "rb") as handle:
        response = requests.post(
            f"{base_url.rstrip('/')}/v1.1.8/seacraft_asr",
            files={"audioFile": (os.path.basename(audio_path), handle, "audio/wav")},
            data={"showSpk": "true"},
            timeout=7200,
        )
    payload = response.json()
    reference = []
    for seg in payload.get("segments") or []:
        try:
            reference.append((float(seg["bg"]), float(seg["ed"]), seg.get("role")))
        except (KeyError, TypeError, ValueError):
            continue
    return reference


def reference_for_span(reference, start, end):
    """取时间重叠最多的参照 role。"""
    best_role = None
    best_overlap = 0.0
    for ref_start, ref_end, role in reference:
        overlap = min(end, ref_end) - max(start, ref_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_role = role
    return best_role if best_overlap > 0 else None


def evaluate(segments, matrix, threshold, reference):
    """给定阈值做聚类，返回簇数与参照一致率。"""
    outcomes = []
    for index, segment in enumerate(segments):
        if index >= matrix.shape[0]:
            break
        outcomes.append(
            SegmentOutcome(index=index, segment=segment, embedding=matrix[index])
        )
    labels = cluster_speakers(outcomes, threshold=threshold)

    clusters = defaultdict(Counter)
    ref_labels = {}
    for outcome in outcomes:
        label = labels.get(outcome.index)
        if label is None:
            continue
        ref = reference_for_span(
            reference, outcome.segment.start_s, outcome.segment.end_s
        )
        if ref is not None:
            ref_labels[outcome.index] = ref
            clusters[label][ref] += 1

    matched = sum(
        counter.most_common(1)[0][1] for counter in clusters.values() if counter
    )
    total = sum(sum(counter.values()) for counter in clusters.values())
    return {
        "threshold": threshold,
        "clusters": len(set(labels.values())),
        "labeled_segments": len(labels),
        "reference_agreement": round(matched / total, 4) if total else None,
        "reference_total": total,
        "bcubed_precision": _bcubed(labels, ref_labels, part="precision"),
        "bcubed_recall": _bcubed(labels, ref_labels, part="recall"),
        "bcubed_f1": _bcubed(labels, ref_labels, part="f1"),
        "reference_speakers": len(set(ref_labels.values())),
    }


def _bcubed(labels, ref_labels, part="f1"):
    """BCubed：同时惩罚「把不同说话人合并」与「把同一说话人拆散」。

    - precision：被判为同簇的段对中，参照说话人确实相同的比例；
    - recall：参照为同一说话人的段对中，被判为同簇的比例。
    """
    indices = [i for i in ref_labels if i in labels]
    if not indices:
        return None

    cluster_of = {i: labels[i] for i in indices}
    ref_of = {i: ref_labels[i] for i in indices}

    # 同簇对数、同参照对数与两者交集的对数
    cluster_pairs = 0
    ref_pairs = 0
    joint_pairs = 0
    for position, i in enumerate(indices):
        for j in indices[position + 1:]:
            same_cluster = cluster_of[i] == cluster_of[j]
            same_ref = ref_of[i] == ref_of[j]
            if same_cluster:
                cluster_pairs += 1
            if same_ref:
                ref_pairs += 1
            if same_cluster and same_ref:
                joint_pairs += 1

    if cluster_pairs == 0 or ref_pairs == 0:
        return None

    precision = joint_pairs / cluster_pairs if cluster_pairs else 0.0
    recall = joint_pairs / ref_pairs if ref_pairs else 0.0
    if part == "precision":
        return round(precision, 4)
    if part == "recall":
        return round(recall, 4)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return round(f1, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8085")
    parser.add_argument(
        "--thresholds",
        default="0.35,0.4,0.45,0.5,0.6,0.7,0.75,0.8",
    )
    parser.add_argument("--cache", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cache = args.cache or f"/tmp/asr_run/sweep_{os.path.basename(args.audio)}.npz"
    segments = None
    matrix = None
    if os.path.exists(cache):
        print(f"[缓存] 复用 {cache}")
        data = np.load(cache, allow_pickle=True)
        segments = [
            VadSegment(start_s=float(row[0]), end_s=float(row[1]))
            for row in data["segments"]
        ]
        matrix = torch.tensor(data["embeddings"])
    else:
        segments, matrix = collect_embeddings(args.audio)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(
            cache,
            segments=np.array(
                [(s.start_s, s.end_s) for s in segments], dtype=np.float64
            ),
            embeddings=matrix.numpy(),
        )

    reference = []
    if args.base_url:
        try:
            reference = fetch_reference(args.base_url, args.audio)
            print(f"[参照] /v1.1.8 role 段数 {len(reference)}")
        except Exception as e:  # noqa: BLE001
            print(f"[参照] 拉取失败（忽略）：{e}")

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    results = [
        evaluate(segments, matrix, threshold, reference) for threshold in thresholds
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

