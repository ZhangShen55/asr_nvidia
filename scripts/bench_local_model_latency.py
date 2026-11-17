"""串行原始 FP32 CPU/NPU 延迟；包含预处理、复制和同步，不是完整 API 延迟。

先 source CANN，宿主仅使用 npu:7；容器映射该卡时显式传 --device npu:0。
本工具不启动或关闭 HTTP 服务。
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


class RawCpuRunner:
    """与 NPU 相同原始 checkpoint/前后处理的 CPU FP32 参考。"""
    effective_backend = "pytorch_cpu_native"
    actual_precision = "fp32"

    def __init__(self, name, model_dir):
        from core.local_inference import emotion_labels, load_native_network
        self.name = name
        self.model = load_native_network(name, model_dir)
        self.labels = emotion_labels(model_dir) if name == "emotion2vec" else None

    def generate(self, *, input, fs=16000, **kwargs):
        import torch
        import torch.nn.functional as F
        from core.local_inference import waveform_tensor, campplus_features
        wave = waveform_tensor(input, fs)
        with torch.inference_mode():
            if self.name == "speech_campplus":
                return [{"spk_embedding": self.model(campplus_features(wave)).detach().float().cpu()}]
            source = F.layer_norm(wave, wave.shape) if self.model.cfg.normalize else wave
            features = self.model.extract_features(source.view(1, -1), padding_mask=None)["x"]
            logits = self.model.proj(features.mean(dim=1))
            scores = torch.softmax(logits[:, [0, 2, 3, 4, 5, 7, 8]], dim=-1)
            return [{"labels": self.labels, "scores": scores[0].detach().float().cpu().tolist()}]

    def close(self):
        self.model = None


def build_runner(name, model_dir, device):
    from core.local_inference import backend_for_device, TorchNpuRunner
    return (RawCpuRunner(name, model_dir) if backend_for_device(device) == "pytorch_cpu"
            else TorchNpuRunner(name, model_dir, device))


def read_audio(path, max_seconds=None):
    """PyAV 流式解码为单声道16k float32；支持原始文件路径编码且不依赖 ffmpeg CLI。"""
    import av
    import numpy as np
    chunks, samples = [], 0
    limit = int(max_seconds * 16000) if max_seconds is not None else None
    with open(path, "rb") as handle, av.open(handle) as container:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        complete = True
        for frame in container.decode(audio=0):
            for converted in resampler.resample(frame):
                array = converted.to_ndarray().reshape(-1)
                chunks.append(array)
                samples += array.size
            if limit is not None and samples >= limit:
                complete = False
                break
        if complete:
            chunks.extend(frame.to_ndarray().reshape(-1) for frame in resampler.resample(None))
    if not chunks:
        raise ValueError("audio has no decoded samples")
    audio = np.concatenate(chunks)
    return np.ascontiguousarray(audio[:limit], dtype=np.float32), 16000


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True, help="JSON list: file, sha256, duration_s, id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:7", help="cpu or npu:N; host default npu:7")
    parser.add_argument("--devices", default="", help="optional comma list; overrides --device")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    import numpy as np
    import torch
    from core.local_inference import backend_for_device
    devices = args.devices.split(",") if args.devices else ["cpu", args.device]
    devices = list(dict.fromkeys(device.strip() for device in devices))
    for device in devices:
        backend_for_device(device)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    torch.set_num_threads(4)
    samples = json.loads(args.samples.read_text())
    for row in samples:
        if sha256_file(row["file"]) != row["sha256"]:
            raise ValueError(f"sample hash mismatch: {row['id']}")
    result = {"devices": devices, "samples_manifest": str(args.samples.resolve()),
              "scope": "raw FP32 checkpoint; serial preprocessing/copies/synchronize included; no Qwen or API calls; no lease queue",
              "percentile_method": "numpy percentile linear interpolation", "rows": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, directory in (("speech_campplus", "speech_campplus_sv_zh_en_16k-common_advanced"),
                            ("emotion2vec", "emotion2vec_plus_large")):
        for device in devices:
            runner = build_runner(name, REPO_ROOT / "model" / directory, device)
            try:
                rows = []
                for row in samples:
                    if name == "emotion2vec" and not 1 <= row["duration_s"] <= 10:
                        continue
                    audio = np.load(row["file"], allow_pickle=False)
                    kwargs = dict(input=audio, fs=16000, granularity="utterance", extract_embedding=False)
                    runner.generate(**kwargs)
                    times = []
                    for _ in range(args.repeats):
                        started = time.perf_counter()
                        runner.generate(**kwargs)
                        times.append((time.perf_counter() - started) * 1000)
                    rows.append({"id": row["id"], "duration_s": row["duration_s"], "latencies_ms": times})
                times = [t for row in rows for t in row["latencies_ms"]]
                entry = {"model": name, "device": device, "backend": runner.effective_backend,
                         "precision": "fp32", "samples": len(rows), "N": len(times),
                         "p50_ms": float(np.percentile(times, 50)) if times else None,
                         "p95_ms": float(np.percentile(times, 95)) if times else None, "rows": rows}
                result["rows"].append(entry)
                args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                print(name, device, entry["p50_ms"], entry["p95_ms"], flush=True)
            finally:
                runner.close()


if __name__ == "__main__":
    main()
