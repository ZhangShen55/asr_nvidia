"""固定整课输入复核 CAM++ 聚类边界，排除 Qwen 文本筛选造成的分段集合变化。"""
import argparse
import json
from pathlib import Path
import sys
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    import torch
    from funasr import AutoModel
    from sklearn.metrics import adjusted_rand_score
    from api.routes.asr_qwen_common import SegmentOutcome, cluster_speakers
    from core.local_inference import TorchRunner
    from core.tensorrt_runner import TensorRTRunner
    from core.vad import VadSegment

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--course", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.evidence.resolve()
    source = json.loads((root / "baseline.json").read_text())["samples"][args.course - 1]["path"]
    segments = json.loads((root / "full-audio/pytorch" / f"audio-{args.course}-response.json").read_text())["segments"]
    waves = []
    for segment in segments:
        start, end = float(segment["bg"]), float(segment["ed"])
        with wave.open(source, "rb") as audio:
            a, b = int(start * 16000), int((start + (end - start)) * 16000)
            audio.setpos(a)
            values = np.frombuffer(audio.readframes(b - a), dtype="<i2").astype(np.float32) / 32768
        waves.append(values)
    model_dir = Path("model/speech_campplus_sv_zh_en_16k-common_advanced")
    matrices, labels, outputs = {}, {}, {}
    for backend in ("pytorch", "pytorch_fp32", "fp32", "fp16"):
        half = backend == "pytorch"
        runner = (TorchRunner(AutoModel(model=str(model_dir), device="cuda:0", fp16=half,
                    disable_update=True, disable_pbar=True), "cuda:0", fp16=half) if backend.startswith("pytorch")
                  else TensorRTRunner("speech_campplus", model_dir, "cuda:0", backend))
        outcomes = []
        embeddings = []
        for index, (segment, audio) in enumerate(zip(segments, waves)):
            outcome = SegmentOutcome(index=index, segment=VadSegment(float(segment["bg"]), float(segment["ed"])),
                                     text=segment["segment_text"])
            if outcome.segment.duration_s >= 1:
                outcome.embedding = torch.as_tensor(runner.generate(input=audio, fs=16000)[0]["spk_embedding"]).reshape(-1)
                embeddings.append(outcome.embedding)
            outcomes.append(outcome)
        labels[backend] = cluster_speakers(outcomes)
        outputs[backend] = outcomes
        matrix = torch.stack(embeddings)
        matrix = matrix / torch.linalg.norm(matrix, dim=1, keepdim=True)
        matrices[backend] = (matrix @ matrix.T).numpy()
        runner.close()
        print(backend, "clusters", len(set(labels[backend].values())), flush=True)
    result = {"course": args.course, "source": source, "segments": len(segments),
              "scope": "fixed PyTorch response segment list; unchanged threshold 0.3; no Qwen calls", "backends": {}}
    baseline_labels = [labels["pytorch"].get(i) for i in range(len(segments))]
    for backend in ("pytorch", "pytorch_fp32", "fp32", "fp16"):
        candidate = [labels[backend].get(i) for i in range(len(segments))]
        result["backends"][backend] = {
            "clusters": len(set(candidate)), "ARI_vs_fixed_pytorch": float(adjusted_rand_score(baseline_labels, candidate)),
            "ARI_vs_pytorch_fp32": float(adjusted_rand_score(
                [labels["pytorch_fp32"].get(i) for i in range(len(segments))], candidate)),
            "pairwise_cosine_max_abs_vs_pytorch": float(np.abs(matrices[backend] - matrices["pytorch"]).max()),
            "threshold_sweep": {str(t): len(set(cluster_speakers(outputs[backend], t).values()))
                                for t in (0.299, 0.2995, 0.2999, 0.3, 0.3001, 0.3005, 0.301)},
            "labels": candidate,
        }
    np.savez(root / f"cam-course-{args.course}-similarity.npz", **matrices)
    (root / f"cam-course-{args.course}-fixed-clusters.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: {a: b for a, b in v.items() if a != "labels"} for k, v in result["backends"].items()}, indent=2))


if __name__ == "__main__":
    main()
