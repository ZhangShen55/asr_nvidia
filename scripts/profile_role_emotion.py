"""剖析 v3 本地 CAM++/emotion2vec FP32 运行器与独占模型池扩展性。

先 source CANN；默认 CPU，宿主显式 --device npu:7，容器映射该卡时 npu:0。
不启动 HTTP 服务。原始网络串行统计与生产 ModelPool 统计分开报告。
"""
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def summarize(name, latencies):
    values = sorted(latencies)
    return {"name": name, "calls": len(values), "mean_ms": statistics.mean(values),
            "p50_ms": statistics.median(values), "p90_ms": values[min(len(values) - 1, int(len(values) * .9))],
            "total_s": sum(values) / 1000}


def time_calls(fn, items):
    latencies = []
    for item in items:
        started = time.perf_counter()
        fn(item)
        latencies.append((time.perf_counter() - started) * 1000)
    return latencies


async def profile_pool(name, model_dir, device, arrays, count):
    from core.models import ModelPool
    from core.local_inference import generate_local
    pool = ModelPool(name, str(model_dir), devices_getter=lambda: [device] * count)
    try:
        await pool.ensure_loaded()
        async def call(item):
            async with pool.lease() as (model, actual_device):
                return await asyncio.to_thread(generate_local, model, actual_device, input=item[0], fs=item[1],
                                               granularity="utterance", extract_embedding=False)
        # 预热所有实例，不计入墙钟；生产 CPU 池保持 FunASR 兼容路径。
        await asyncio.gather(*(call(arrays[0]) for _ in range(count)))
        started = time.perf_counter()
        await asyncio.gather(*(call(item) for item in arrays))
        wall = time.perf_counter() - started
        return {"wall_s": wall, "instances": count, "calls": len(arrays),
                "per_call_effective_ms": wall * 1000 / len(arrays),
                "backend": "pytorch_cpu_funasr" if device == "cpu" else "torch_npu", "precision": "fp32"}
    finally:
        await pool.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments-dir", default="/tmp/asr_run/segs")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--spk-model", default="model/speech_campplus_sv_zh_en_16k-common_advanced")
    parser.add_argument("--emotion-model", default="model/emotion2vec_plus_large")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8], help="exclusive model-pool replicas/tasks")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    from core.local_inference import backend_for_device
    from scripts.bench_local_model_latency import REPO_ROOT, build_runner, read_audio
    backend_for_device(args.device)
    if min(args.threads) < 1 or args.limit < 0:
        parser.error("positive --threads and nonnegative --limit required")
    paths = sorted(Path(args.segments_dir).glob("*.wav"))
    paths = paths[:args.limit] if args.limit else paths
    if not paths:
        parser.error("segments directory has no wav files")
    arrays = [read_audio(path) for path in paths]
    report = {"segments": len(paths), "device": args.device, "precision": "fp32", "variants": {}, "scaling": {},
              "batch_policy": "Only single-utterance inputs; padded batch semantics are not equivalent and are excluded."}
    for name, directory, prefix in (("speech_campplus", args.spk_model, "spk"), ("emotion2vec", args.emotion_model, "emotion")):
        model_dir = Path(directory)
        model_dir = model_dir if model_dir.is_absolute() else REPO_ROOT / model_dir
        runner = build_runner(name, model_dir, args.device)
        try:
            def numpy_call(item):
                return runner.generate(input=item[0], fs=item[1], granularity="utterance", extract_embedding=False)
            def file_call(path):
                return numpy_call(read_audio(path))
            numpy_call(arrays[0])
            report["variants"][f"{prefix}_numpy"] = summarize("raw checkpoint / numpy", time_calls(numpy_call, arrays))
            report["variants"][f"{prefix}_file"] = summarize("decode + raw checkpoint", time_calls(file_call, paths))
            report["variants"][f"{prefix}_audio_load"] = summarize("decode only", time_calls(read_audio, paths))
            fixed = []
            for seconds in (.2, 1., 3., 10.):
                piece = arrays[0][0][:int(seconds * 16000)]
                if not piece.size:
                    continue
                lat = time_calls(numpy_call, [(piece, 16000)])[0]
                fixed.append({"requested_s": seconds, "actual_s": len(piece) / 16000, "elapsed_ms": lat})
            report["variants"][f"{prefix}_fixed_overhead"] = fixed
        finally:
            runner.close()
        for count in args.threads:
            entry = asyncio.run(profile_pool(name, model_dir, args.device, arrays, count))
            report["scaling"][f"{prefix}_{count}t"] = entry
        one = report["scaling"].get(f"{prefix}_1t")
        for count in args.threads:
            entry = report["scaling"][f"{prefix}_{count}t"]
            entry["speedup_vs_pool_1"] = one["wall_s"] / entry["wall_s"] if one else None
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
