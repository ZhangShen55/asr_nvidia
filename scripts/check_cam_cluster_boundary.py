"""固定响应段集合对比 CPU 原始 FP32 与 NPU 的 CAM++ 聚类边界，不调用 Qwen。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--course", type=int, default=5)
    parser.add_argument("--device", default="npu:7", help="cpu or npu:7; container mapping npu:0")
    parser.add_argument("--response", type=Path, help="saved v3 response, overrides legacy evidence layout")
    parser.add_argument("--audio", type=Path, help="source audio, overrides baseline.json")
    args = parser.parse_args()
    import numpy as np
    import torch
    from sklearn.metrics import adjusted_rand_score
    from api.routes.asr_qwen_common import SegmentOutcome, cluster_speakers
    from core.local_inference import backend_for_device
    from core.vad import VadSegment
    from scripts.bench_local_model_latency import REPO_ROOT, build_runner, read_audio
    backend_for_device(args.device)
    torch.set_num_threads(4)
    root = args.evidence.resolve()
    source = str(args.audio) if args.audio else json.loads((root / "baseline.json").read_text())["samples"][args.course - 1]["path"]
    response = args.response or root / "full-audio/pytorch" / f"audio-{args.course}-response.json"
    segments = json.loads(response.read_text())["segments"]
    audio, rate = read_audio(source)
    waves = [audio[int(float(segment["bg"]) * rate):int(float(segment["ed"]) * rate)].copy() for segment in segments]
    matrices, labels, outputs = {}, {}, {}
    for device in dict.fromkeys(("cpu", args.device)):
        runner = build_runner("speech_campplus", REPO_ROOT / "model/speech_campplus_sv_zh_en_16k-common_advanced", device)
        try:
            outcomes, embeddings = [], []
            for index, (segment, wave) in enumerate(zip(segments, waves)):
                outcome = SegmentOutcome(index=index, segment=VadSegment(float(segment["bg"]), float(segment["ed"])),
                                         text=segment.get("segment_text", ""))
                if outcome.segment.duration_s >= 1:
                    outcome.embedding = torch.as_tensor(runner.generate(input=wave, fs=rate)[0]["spk_embedding"]).reshape(-1)
                    embeddings.append(outcome.embedding)
                outcomes.append(outcome)
            labels[device] = cluster_speakers(outcomes)
            outputs[device] = outcomes
            matrix = torch.stack(embeddings) if embeddings else torch.empty((0, 0))
            matrix = matrix / torch.linalg.norm(matrix, dim=1, keepdim=True).clamp_min(1e-12)
            matrices[device] = (matrix @ matrix.T).numpy()
        finally:
            runner.close()
    result = {"course": args.course, "source": source, "response": str(response), "segments": len(segments),
              "scope": "fixed saved response; original FP32 checkpoint; threshold 0.3; no Qwen calls", "backends": {}}
    # ARI compares only common labeled indices; unlabeled segments must not form an artificial None class.
    for device in matrices:
        common = sorted(set(labels["cpu"]) & set(labels[device]))
        diff = np.abs(matrices[device] - matrices["cpu"])
        result["backends"][device] = {
            "clusters": len(set(labels[device].values())), "compared_segments": len(common),
            "ARI_vs_cpu_fp32": float(adjusted_rand_score([labels["cpu"][i] for i in common], [labels[device][i] for i in common])) if common else None,
            "pairwise_cosine_max_abs_vs_cpu": float(diff.max()) if diff.size else None,
            "threshold_sweep": {str(t): len(set(cluster_speakers(outputs[device], t).values()))
                                for t in (0.299, 0.2995, 0.2999, 0.3, 0.3001, 0.3005, 0.301)},
            "labels": labels[device]}
    np.savez(root / f"cam-course-{args.course}-similarity.npz", **matrices)
    (root / f"cam-course-{args.course}-fixed-clusters.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
