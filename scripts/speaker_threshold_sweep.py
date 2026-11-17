"""CAM++ 说话人聚类阈值 sweep：抽一次嵌入，离线扫多个阈值。

步骤：
1. 本地 VAD 切分 + CAM++ 抽取每段嵌入，缓存到 .npz（重复运行不再重算）；
2. 可选拉取 /v3 的 role 作为参照，按时间重叠算「聚类一致率」；
3. 对每个阈值输出簇数与一致率，用于选定生产取值。

用法：
    python scripts/speaker_threshold_sweep.py --audio test/音频文件/课堂实录.wav \
        --base-url http://127.0.0.1:8084 --thresholds 0.35,0.4,0.45,0.5,0.6,0.7,0.75,0.8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter, defaultdict

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def collect_embeddings(audio_path, device="cpu"):
    """用本工具独占的 VAD 池和 FP32 运行器，保持时间段与嵌入严格对齐。"""
    import torch
    from core.vad import _parse_vad_result
    from core.vad_pool import VadPool
    from scripts.bench_vad_cpu_vs_gpu import MODEL_DIR, vad_kwargs
    from scripts.bench_local_model_latency import REPO_ROOT, build_runner, read_audio

    async def run():
        pool = VadPool(devices=["cpu"], model_dir=str(MODEL_DIR), vad_kwargs=vad_kwargs())
        runner = None
        try:
            await pool.start()
            segments = _parse_vad_result(await pool.detect(str(audio_path)))
            runner = build_runner("speech_campplus", REPO_ROOT / "model/speech_campplus_sv_zh_en_16k-common_advanced", device)
            audio, rate = read_audio(audio_path)
            embeddings = []
            for index, segment in enumerate(segments):
                array = audio[int(segment.start_s * rate):int(segment.end_s * rate)].copy()
                payload = runner.generate(input=array, fs=rate)[0]
                embedding = payload.get("spk_embedding")
                if embedding is None:
                    raise RuntimeError(f"CAM++ missing embedding for segment {index}; refusing shifted cache")
                embeddings.append(torch.as_tensor(embedding).detach().float().reshape(-1).cpu())
                if (index + 1) % 20 == 0:
                    print(f"  已抽取 {index + 1}/{len(segments)}", flush=True)
            return segments, torch.stack(embeddings) if embeddings else torch.empty((0, 0))
        finally:
            try:
                if runner is not None:
                    runner.close()
            finally:
                await pool.stop()
    return asyncio.run(run())


def fetch_reference(base_url, audio_path):
    """拉取 /v3 的 role 作为聚类参照。"""
    import requests

    with open(audio_path, "rb") as handle:
        response = requests.post(
            f"{base_url.rstrip('/')}/v3/seacraft_asr",
            files={"audioFile": ("threshold-sample.wav", handle, "application/octet-stream")},
            data={"showSpk": "true"},
            timeout=7200,
        )
    response.raise_for_status()
    return parse_reference(response.json())


def parse_reference(payload):
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
        "pairwise_precision": _bcubed(labels, ref_labels, part="precision"),
        "pairwise_recall": _bcubed(labels, ref_labels, part="recall"),
        "pairwise_f1": _bcubed(labels, ref_labels, part="f1"),
        "reference_speakers": len(set(ref_labels.values())),
    }


def _bcubed(labels, ref_labels, part="f1"):
    """历史函数名保留；实际指标是 pairwise precision/recall/F1，并非 B-cubed。

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
    parser.add_argument("--base-url", default="", help="optional existing v3 service; this tool never starts/stops it")
    parser.add_argument("--reference-json", type=Path, help="preferred saved v3 response")
    parser.add_argument("--device", default="cpu", help="cpu or npu:7; container mapping npu:0")
    parser.add_argument(
        "--thresholds",
        default="0.35,0.4,0.45,0.5,0.6,0.7,0.75,0.8",
    )
    parser.add_argument("--cache", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    global np, torch, VadSegment, SegmentOutcome, cluster_speakers
    import numpy as np
    import torch
    from core.vad import VadSegment
    from api.routes.asr_qwen_common import SegmentOutcome, cluster_speakers
    from core.local_inference import backend_for_device
    from scripts.bench_local_model_latency import REPO_ROOT, sha256_file
    from scripts.bench_vad_cpu_vs_gpu import vad_kwargs
    backend_for_device(args.device)
    cam_dir = REPO_ROOT / "model/speech_campplus_sv_zh_en_16k-common_advanced"
    identity = {"audio_sha256": sha256_file(args.audio), "device": args.device, "precision": "fp32", "schema": 2,
                "model_sha256": sha256_file(cam_dir / "campplus_cn_en_common.pt"),
                "model_config_sha256": sha256_file(cam_dir / "config.yaml"), "vad_kwargs": vad_kwargs()}

    cache = args.cache or f"/tmp/asr_run/sweep_{os.path.basename(args.audio)}.npz"
    segments = None
    matrix = None
    if os.path.exists(cache):
        print(f"[缓存] 复用 {cache}")
        data = np.load(cache, allow_pickle=False)
        if "identity" not in data or json.loads(str(data["identity"])) != identity:
            raise ValueError("cache identity differs or old cache lacks provenance; use a fresh --cache path")
        segments = [
            VadSegment(start_s=float(row[0]), end_s=float(row[1]))
            for row in data["segments"]
        ]
        matrix = torch.tensor(data["embeddings"])
    else:
        segments, matrix = collect_embeddings(args.audio, args.device)
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.savez(
            cache,
            segments=np.array(
                [(s.start_s, s.end_s) for s in segments], dtype=np.float64
            ),
            embeddings=matrix.numpy(),
            identity=json.dumps(identity),
        )

    reference = parse_reference(json.loads(args.reference_json.read_text())) if args.reference_json else []
    if args.base_url and not args.reference_json:
        try:
            reference = fetch_reference(args.base_url, args.audio)
            print(f"[参照] /v3 role 段数 {len(reference)}")
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

