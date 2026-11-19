"""FSMN-VAD CPU/NPU 等价性与每实例 CPU 核数；文件名保留以兼容旧命令。

先 source CANN；宿主默认 npu:7，容器映射该卡时显式 --device npu:0。
只比较原始 FP32 网络，不启动 HTTP 服务。音频请显式指定，支持 manifest path_hex 还原路径。
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
MODEL_DIR = REPO_ROOT / "model/speech_fsmn_vad_zh-cn-16k-common-pytorch"


def make_slice(src, seconds):
    from scripts.bench_local_model_latency import read_audio
    import numpy as np
    import wave
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as handle:
        dst = handle.name
    try:
        audio, rate = read_audio(src, max_seconds=seconds)
        pcm = np.clip(np.rint(audio * 32768), -32768, 32767).astype("<i2")
        with wave.open(dst, "wb") as output:
            output.setparams((1, 2, rate, 0, "NONE", ""))
            output.writeframes(pcm.tobytes())
        return dst
    except BaseException:
        os.unlink(dst)
        raise


def vad_kwargs():
    # 使用模型原始FSMN选项，与正式池保持一致，不转发无效nested阈值。
    return {}

def build_model(device):
    from core.local_inference import NativeVadRunner
    return NativeVadRunner(str(MODEL_DIR), device, vad_kwargs())


def run_vad(model, path):
    from core.vad import _parse_vad_result
    result = model.generate(input=str(path), cache={}, vad_kwargs=vad_kwargs())
    return [(segment.start_s, segment.end_s) for segment in _parse_vad_result(result)]


def allocated_mb(device):
    if device == "cpu":
        return None
    from core.local_inference import npu_api
    api = npu_api()
    api.set_device(device)
    return float(api.memory_allocated(device)) / 1024 ** 2


def cpu_time():
    me = resource.getrusage(resource.RUSAGE_SELF)
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    return me.ru_utime + me.ru_stime + kids.ru_utime + kids.ru_stime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", default="npu:7", help="NPU candidate; cpu enables CPU-only smoke")
    parser.add_argument("--slice-s", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    from core.local_inference import backend_for_device
    backend_for_device(args.device)
    if args.threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
        import torch
        torch.set_num_threads(args.threads)
    tmp = None
    results = {}
    try:
        audio = args.audio
        if args.slice_s > 0:
            tmp = make_slice(audio, args.slice_s)
            audio = tmp
        for device in dict.fromkeys(("cpu", args.device)):
            model = build_model(device)
            try:
                run_vad(model, audio)
                c0, w0 = cpu_time(), time.perf_counter()
                spans = run_vad(model, audio)
                wall, cpu_used = time.perf_counter() - w0, cpu_time() - c0
                results[device] = {"spans": spans, "wall_s": wall, "cpu_s": cpu_used,
                                   "cpu_cores": cpu_used / max(wall, 1e-9), "precision": "fp32"}
            finally:
                model.close()
        print(json.dumps({"results": results, "same_boundaries": results["cpu"]["spans"] == results[args.device]["spans"]},
                         ensure_ascii=False, indent=2))
    finally:
        if tmp:
            os.unlink(tmp)


if __name__ == "__main__":
    main()
