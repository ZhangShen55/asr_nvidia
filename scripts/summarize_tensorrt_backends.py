"""复算八路分位数、整课角色/情绪一致性，并生成扩展课堂裁剪清单。"""
import argparse
import json
from pathlib import Path
import sys
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.model_assets import sha256_file


def main():
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    args = parser.parse_args()
    directory = args.evidence.resolve()
    full = directory / "full-audio"
    baseline = json.loads((directory / "baseline.json").read_text())
    performance = []
    for backend in ("pytorch", "fp32", "fp16"):
        summary = json.loads((full / backend / "summary.json").read_text())
        values = [r["elapsed_s"] for r in summary["samples"] if r["success"]]
        assert len(values) == 8 and summary["peak_active_B_sampled"] == 8
        assert np.isclose(np.percentile(values, 50), summary["p50_s"])
        assert np.isclose(np.percentile(values, 95), summary["p95_s"])
        observations = json.loads((full / backend / "observations.json").read_text())
        gpu = [row["gpu"].split(",") for row in observations if "gpu" in row]
        telemetry = json.loads((full / backend / "telemetry.json").read_text())
        rows = []
        for row in summary["samples"]:
            event = next(e for e in telemetry["recent"] if e["filename"].startswith(f"course-{row['index']}-"))
            rows.append({**row, "server_request_id": event["request_id"], "server_task_id": event["task_id"],
                         "phases": event["phases"], "admission": event["admission"]})
        log = (full / backend / "service.log").read_text()
        performance.append({k: summary[k] for k in ["backend", "p50_s", "p95_s", "batch_wall_s", "N",
            "successful", "peak_active_B_sampled", "cold_load_and_warmup_s", "throughput_audio_s_per_wall_s"]} |
            {"samples": rows, "models_after": summary["models_after"],
             "speaker_inference_failures_logged": log.count("[说话人] 嵌入提取失败"),
             "emotion_inference_failures_logged": log.count("[情绪] 推理失败"),
             "gpu_peak_memory_mb": max(float(row[1]) for row in gpu),
             "gpu_peak_util_percent": max(float(row[2]) for row in gpu)})
    quality = []
    for backend in ("fp32", "fp16"):
        for index in range(1, 9):
            left = json.loads((full / "pytorch" / f"audio-{index}-response.json").read_text())["segments"]
            right = json.loads((full / backend / f"audio-{index}-response.json").read_text())["segments"]
            a = {(s["bg"], s["ed"]): s for s in left}
            b = {(s["bg"], s["ed"]): s for s in right}
            common = sorted(a.keys() & b.keys())
            labels_a = [a[k]["role"] for k in common]
            labels_b = [b[k]["role"] for k in common]
            ca, cb = sorted(set(labels_a)), sorted(set(labels_b))
            counts = np.array([[sum(x == l and y == r for x, y in zip(labels_a, labels_b)) for r in cb] for l in ca])
            aa, bb = linear_sum_assignment(-counts)
            eligible = [k for k in common if 1 <= float(k[1]) - float(k[0]) <= 10]
            differences = [{"bg": k[0], "ed": k[1], "pytorch": a[k]["emotion"], "tensorrt": b[k]["emotion"]}
                           for k in common if a[k]["emotion"] != b[k]["emotion"]]
            quality.append({"backend": backend, "course": index, "pytorch_segments": len(left),
                "tensorrt_segments": len(right), "aligned_segments": len(common),
                "output_boundaries_equal": a.keys() == b.keys(),
                "aligned_text_equal": sum(a[k]["segment_text"] == b[k]["segment_text"] for k in common),
                "pytorch_clusters": len(ca), "tensorrt_clusters": len(cb),
                "role_ARI": float(adjusted_rand_score(labels_a, labels_b)),
                "role_best_mapping_agreement": float(counts[aa, bb].sum() / len(common)),
                "emotion_agreement": 1 - len(differences) / len(common),
                "emotion_eligible_segments": len(eligible),
                "emotion_eligible_agreement": sum(a[k]["emotion"] == b[k]["emotion"] for k in eligible) / max(1, len(eligible)),
                "emotion_differences": differences})
    crop_dir = directory / "classroom-crops"
    crop_dir.mkdir(exist_ok=True)
    crops = []
    for index, source in enumerate(baseline["samples"], 1):
        output = json.loads((full / "pytorch" / f"audio-{index}-response.json").read_text())
        eligible = [s for s in output["segments"] if 1 <= float(s["ed"]) - float(s["bg"]) <= 10]
        for number, offset in enumerate(np.linspace(0, len(eligible) - 1, 6).astype(int)):
            segment = eligible[offset]
            start, end = float(segment["bg"]), float(segment["ed"])
            with wave.open(source["path"], "rb") as audio:
                audio.setpos(round(start * 16000))
                values = np.frombuffer(audio.readframes(round((end - start) * 16000)), dtype="<i2").astype(np.float32) / 32768
            path = crop_dir / f"course-{index}-{number}.npy"
            np.save(path, values)
            crops.append({"id": path.stem, "file": str(path), "source": source["path"],
                          "source_start_s": start, "duration_s": len(values) / 16000,
                          "sha256": sha256_file(path)})
    (directory / "classroom-samples.json").write_text(json.dumps(crops, ensure_ascii=False, indent=2) + "\n")
    boundaries = {}
    for row in quality:
        for change in row["emotion_differences"]:
            key = (row["course"], change["bg"], change["ed"])
            boundaries[key] = change
    boundary_samples = []
    for (course, start_text, end_text), change in boundaries.items():
        source = baseline["samples"][course - 1]["path"]
        start, end = float(start_text), float(end_text)
        if not 1 <= end - start <= 10:
            continue
        a, b = int(start * 16000), int((start + (end - start)) * 16000)
        with wave.open(source, "rb") as audio:
            audio.setpos(a)
            values = np.frombuffer(audio.readframes(b - a), dtype="<i2").astype(np.float32) / 32768
        path = crop_dir / f"boundary-course-{course}-{a}.npy"
        np.save(path, values)
        boundary_samples.append({"id": path.stem, "file": str(path), "source": source,
            "source_start_s": start, "duration_s": len(values) / 16000, "sha256": sha256_file(path),
            "observed_business_difference": change})
    (directory / "emotion-boundary-samples.json").write_text(json.dumps(boundary_samples, ensure_ascii=False, indent=2) + "\n")
    result = {"performance": performance, "quality": quality, "classroom_crops": len(crops),
              "notes": ["Output alignment is after ASR text filtering, not raw VAD boundary equivalence.",
                        "No human speaker/emotion ground truth; consistency is not absolute accuracy.",
                        "N=8 per backend, linear interpolation; P95 is descriptive for this batch."]}
    (directory / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for row in quality:
        print(row["backend"], row["course"], "ARI", round(row["role_ARI"], 6),
              "emotion", round(row["emotion_agreement"], 6), "aligned", row["aligned_segments"])


if __name__ == "__main__":
    main()
