"""相同课堂裁剪的串行单段运行器延迟；与完整 API 并发指标分开报告。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    import torch
    from funasr import AutoModel
    from core.local_inference import TorchRunner
    from core.model_assets import sha256_file
    from core.tensorrt_runner import TensorRTRunner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(4)
    samples = json.loads(args.samples.read_text())
    for row in samples:
        assert sha256_file(row["file"]) == row["sha256"]
    inventory = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=uuid,memory.used,utilization.gpu",
                                         "--format=csv,noheader,nounits"], text=True).strip()
    result = {"device": args.device, "gpu_before": inventory, "samples_manifest": str(args.samples.resolve()),
              "scope": "serial raw waveform -> runner CPU output including preprocessing/copies/synchronize; "
                       "Qwen remains resident on same GPU, no extra Qwen requests in this microbenchmark; "
                       "no model-lease queue; not full-request latency",
              "percentile_method": "numpy percentile linear interpolation", "rows": []}
    for name, directory in (("speech_campplus", "speech_campplus_sv_zh_en_16k-common_advanced"),
                            ("emotion2vec", "emotion2vec_plus_large")):
        model_dir = root / "model" / directory
        for backend in ("pytorch", "fp32", "fp16"):
            runner = (TorchRunner(AutoModel(model=str(model_dir), device=args.device, fp16=True,
                        disable_update=True, disable_pbar=True), args.device, fp16=True)
                      if backend == "pytorch" else TensorRTRunner(name, model_dir, args.device, backend))
            rows = []
            for row in samples:
                if name == "emotion2vec" and not 1 <= row["duration_s"] <= 10:
                    continue
                audio = np.load(row["file"])
                kwargs = dict(input=audio, fs=16000, granularity="utterance", extract_embedding=False)
                runner.generate(**kwargs)  # 每个 shape 先预热一次，不进入实测。
                times = []
                for _ in range(3):
                    started = time.perf_counter()
                    runner.generate(**kwargs)
                    times.append((time.perf_counter() - started) * 1000)
                rows.append({"id": row["id"], "duration_s": row["duration_s"], "latencies_ms": times})
            times = [t for row in rows for t in row["latencies_ms"]]
            entry = {"model": name, "backend": backend, "samples": len(rows), "N": len(times),
                     "p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
                     "rows": rows}
            result["rows"].append(entry)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            print(name, backend, entry["p50_ms"], entry["p95_ms"], flush=True)
            runner.close()
            del runner
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
