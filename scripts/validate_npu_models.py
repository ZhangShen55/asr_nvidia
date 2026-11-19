#!/usr/bin/env python3
"""只读原音频片段，比较 CPU FP32 与物理 npu:7 上的原始小模型候选。

本工具不启动服务、不安装依赖、不改模型/原音频。容差先写入 JSONL，再加载模型；
每项结果立即 flush。低精度/TorchAir 仅是显式实验候选，不改变生产默认 FP32。
未提供 CUDA golden，因此输出仅能证明 CPU FP32 参考比较，不能宣称 CUDA 一致。
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import sys
import tempfile
import time
import traceback
import wave
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# These limits are fixed before the run. They are engineering numerical gates,
# not a substitute for labeled speaker/emotion quality or a CUDA golden.
TOLERANCES = {
    "fp32": {"cam_cosine_min": 0.9999, "cam_max_abs": 0.001, "emo_max_abs": 0.001},
    "bf16": {"cam_cosine_min": 0.999, "cam_max_abs": 0.03, "emo_max_abs": 0.02},
    "fp16": {"cam_cosine_min": 0.9999, "cam_max_abs": 0.01, "emo_max_abs": 0.01},
}
VAD_BOUNDARY_TOLERANCE_MS = 20.0
DEVICE = "npu:7"


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def write_record(handle, payload: dict[str, Any]) -> None:
    # Preserve invalid numerical evidence as explicit strings rather than lose
    # the result when a low precision candidate produces NaN/Inf.
    handle.write(json.dumps(_json_safe(payload), ensure_ascii=True, allow_nan=False) + "\n")
    handle.flush()


def manifest_samples(path: Path, sample_ids: list[str] | None = None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("samples", payload.get("files", []))
    if not isinstance(rows, list):
        raise ValueError("manifest 必须包含 samples 列表")
    selected = [row for row in rows if isinstance(row, dict) and row.get("path_hex")]
    if sample_ids:
        by_id = {row.get("sample_id"): row for row in selected}
        missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"manifest 中不存在样本: {missing}")
        return [by_id[sample_id] for sample_id in sample_ids]
    return selected[:1]


def read_clip(sample: dict[str, Any], offset_s: float, duration_s: float, *, verify_source_hash=False) -> tuple[np.ndarray, dict[str, Any]]:
    raw_path = bytes.fromhex(sample["path_hex"])
    if not os.path.isabs(raw_path) or offset_s < 0 or duration_s <= 0:
        raise ValueError("需要绝对 path_hex、非负 offset 和正 duration")
    verified_hash = None
    if verify_source_hash:
        digest = hashlib.sha256()
        with open(raw_path, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        verified_hash = digest.hexdigest()
        if sample.get("sha256") and verified_hash != sample["sha256"]:
            raise ValueError(f"source SHA256 mismatch: {sample.get('sample_id')}")
    with wave.open(os.fsdecode(raw_path), "rb") as source:
        rate = source.getframerate()
        if (rate, source.getnchannels(), source.getsampwidth(), source.getcomptype()) != (16000, 1, 2, "NONE"):
            raise ValueError("验证片段要求原始 16kHz/mono/PCM16 WAV；不隐式重采样或转声道")
        start, frames = round(offset_s * rate), round(duration_s * rate)
        if start + frames > source.getnframes():
            raise ValueError("片段超出原音频，拒绝静默截短")
        source.setpos(start)
        pcm = source.readframes(frames)
    if len(pcm) != frames * 2:
        raise ValueError("原音频片段解码长度不完整")
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    return audio, {
        "sample_id": sample.get("sample_id"), "source_path_hex": sample["path_hex"],
        "source_sha256_declared": sample.get("sha256"), "source_sha256_verified": verified_hash,
        "offset_s": offset_s, "duration_s": duration_s, "sample_rate": rate,
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(), "frames": frames,
        "reference": "independent read-only slice; no human labels inferred",
    }


def vector_metrics(reference, candidate) -> dict[str, Any]:
    a, b = np.asarray(reference, dtype=np.float64).reshape(-1), np.asarray(candidate, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"valid": False, "reason": "shape mismatch, empty or nonfinite output"}
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {"valid": denominator > 0, "cosine": float(np.dot(a, b) / denominator) if denominator else None,
            "max_abs": float(np.max(np.abs(a - b))), "mean_abs": float(np.mean(np.abs(a - b)))}


def compare_output(name: str, reference: dict, candidate: dict, precision: str) -> dict[str, Any]:
    tolerance = TOLERANCES[precision]
    metrics = vector_metrics(reference["values"], candidate["values"])
    if name == "speech_campplus":
        metrics["passed"] = bool(metrics.get("valid") and metrics["cosine"] >= tolerance["cam_cosine_min"] and metrics["max_abs"] <= tolerance["cam_max_abs"])
    else:
        labels_match = reference.get("labels") == candidate.get("labels") and len(candidate.get("labels", [])) == 7
        a, b = reference["values"], candidate["values"]
        top1_match = bool(len(a) and len(b) and np.argmax(a) == np.argmax(b))
        metrics.update(labels_match=labels_match, top1_match=top1_match,
                       probability_sum=float(sum(b)) if metrics.get("valid") else None)
        metrics["passed"] = bool(metrics.get("valid") and labels_match and top1_match and metrics["max_abs"] <= tolerance["emo_max_abs"])
    return metrics


def compare_vad(reference: list, candidate: list, tolerance_ms=VAD_BOUNDARY_TOLERANCE_MS) -> dict[str, Any]:
    same_count = len(reference) == len(candidate)
    errors = [abs(float(a) - float(b)) for left, right in zip(reference, candidate) for a, b in zip(left, right)] if same_count else []
    max_error = max(errors, default=0.0) if same_count else None
    return {"same_count": same_count, "max_boundary_error_ms": max_error,
            "tolerance_ms": tolerance_ms, "passed": bool(same_count and max_error <= tolerance_ms),
            "speech_evidence": bool(reference)}


def _tensor_forward(network, name: str):
    if name == "speech_campplus":
        return lambda features: network(features)
    import torch
    import torch.nn.functional as F

    def forward(source):
        if network.cfg.normalize:
            source = F.layer_norm(source, source.shape)
        features = network.extract_features(source.view(1, -1), padding_mask=None)["x"]
        logits = network.proj(features.mean(dim=1))
        return torch.softmax(logits[:, [0, 2, 3, 4, 5, 7, 8]], dim=-1)
    return forward


def create_torchair_backend():
    try:
        torchair = importlib.import_module("torchair")
        compiler = importlib.import_module("torchair.configs.compiler_config")
        return torchair.get_npu_backend(compiler_config=compiler.CompilerConfig())
    except Exception as exc:
        raise RuntimeError(f"TorchAir unavailable: {type(exc).__name__}: {exc}") from exc


def graph_counter() -> int | None:
    try:
        counters = importlib.import_module("torch._dynamo.utils").counters
        return int(counters["stats"]["unique_graphs"])
    except Exception:
        return None


class Candidate:
    def __init__(self, name, model_dir, device, precision="fp32", compiler="eager"):
        import torch
        from core.local_inference import load_native_network, npu_api, npu_scope, emotion_labels, synchronize_npu
        self.name, self.device, self.compiler = name, device, compiler
        self.api = npu_api() if device == DEVICE else None
        self.dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
        self.network = None
        self.forward = None
        self.labels = emotion_labels(model_dir) if name == "emotion2vec" else None
        if device not in ("cpu", DEVICE) or (device == "cpu" and precision != "fp32"):
            raise ValueError("本工具只允许 CPU FP32 参考和物理 npu:7 候选")
        self._scope = lambda: npu_scope(self.api, device) if self.api is not None else nullcontext()
        self._sync = lambda: synchronize_npu(self.api, device) if self.api is not None else None
        try:
            with self._scope():
                self.network = load_native_network(name, model_dir).to(device=device, dtype=self.dtype)
                devices = {str(parameter.device) for parameter in self.network.parameters()}
                if devices != {device}:
                    raise RuntimeError(f"模型实际落卡不符: expected={device}, actual={sorted(devices)}")
                self.operator_adaptations = {"emotion2vec_exact_gelu_modules": getattr(self.network, "_npu_exact_gelu_count", 0),
                                             "campplus_segment_pooling": name == "speech_campplus"}
                self.parameter_dtypes = sorted({str(parameter.dtype) for parameter in self.network.parameters()})
                self.parameter_devices = sorted(devices)
                self.forward = _tensor_forward(self.network, name)
                if compiler == "torchair":
                    self.forward = torch.compile(self.forward, backend=create_torchair_backend(), dynamic=True, fullgraph=True)
                self._sync()
        except BaseException:
            self.close()
            raise

    def infer(self, audio: np.ndarray) -> tuple[dict, dict]:
        import torch
        from core.local_inference import campplus_features, waveform_tensor
        count_before = graph_counter()
        started = time.perf_counter()
        with self._scope(), torch.inference_mode():
            wave_tensor = waveform_tensor(audio, 16000)
            source = campplus_features(wave_tensor) if self.name == "speech_campplus" else wave_tensor
            result = self.forward(source.to(device=self.device, dtype=self.dtype))
            self._sync()
            if str(result.device) != self.device:
                raise RuntimeError(f"输出实际落卡不符: expected={self.device}, actual={result.device}")
            values = result.detach().float().cpu().reshape(-1).tolist()
        count_after = graph_counter()
        return {"values": values, "labels": self.labels}, {
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "unique_graphs_before": count_before, "unique_graphs_after": count_after,
            "new_graphs": count_after - count_before if count_after is not None and count_before is not None else None,
            "timing_scope": "CPU frontend + transfer + forward + explicit NPU synchronization + CPU result",
        }

    def close(self):
        try:
            with self._scope():
                self._sync()
        finally:
            self.forward = None
            self.network = None
            if self.compiler == "torchair":
                # This is a standalone experiment process; release captured
                # compiled graph references before measuring cleanup HBM.
                import torch._dynamo
                torch._dynamo.reset()
            gc.collect()
            if self.api is not None:
                with self._scope():
                    self.api.empty_cache()


def interference_snapshot() -> dict[str, Any]:
    from core.resource_diagnostics import npu_snapshot
    snapshot = npu_snapshot()
    return {"available": snapshot.get("available", False),
            "npus": [row for row in snapshot.get("npus", []) if row.get("index") == 7],
            "other_processes": [row for row in snapshot.get("compute_apps", []) if row.get("index") == 7 and row.get("pid") != os.getpid()],
            "scope": "observed shared-card background; no causal or exclusive-card comparison claimed"}


def cluster_comparison(references: dict[str, dict], candidates: dict[str, dict], clips: list[dict]) -> dict:
    import torch
    from api.routes.asr_qwen_common import SegmentOutcome, cluster_speakers
    from core.vad import VadSegment

    def cluster(values):
        outcomes = []
        for index, clip in enumerate(clips):
            key = clip["clip_id"]
            if key not in values:
                continue
            outcome = SegmentOutcome(index=index, segment=VadSegment(0.0, clip["duration_s"]), text="validation slice")
            outcome.embedding = torch.as_tensor(values[key]["values"], dtype=torch.float32)
            outcomes.append(outcome)
        return cluster_speakers(outcomes, threshold=0.3)

    a, b = cluster(references), cluster(candidates)
    def partition(labels):
        return sorted(sorted(index for index, value in labels.items() if value == label) for label in set(labels.values()))
    return {"threshold": 0.3, "reference_labels": a, "candidate_labels": b,
            "reference_clusters": len(set(a.values())), "candidate_clusters": len(set(b.values())),
            "same_partition": partition(a) == partition(b),
            "scope": "production average-linkage algorithm on fixed independent slices; no human speaker labels"}


def run_vad(clips: list[dict], model_dir: str, vad_kwargs: dict, handle, temp_dir: Path) -> bool:
    from core.local_inference import NativeVadRunner
    from core.vad_pool import build_vad_model
    from core.vad import _parse_vad_result
    all_passed, baseline = True, {}
    for backend in ("automodel_cpu", "native_cpu", "native_npu"):
        runner = None
        try:
            started = time.perf_counter()
            runner = build_vad_model(model_dir, "cpu", vad_kwargs) if backend == "automodel_cpu" else NativeVadRunner(model_dir, DEVICE if backend == "native_npu" else "cpu", vad_kwargs)
            write_record(handle, {"type": "vad_load", "backend": backend, "elapsed_ms": round((time.perf_counter() - started) * 1000, 3), "vad_kwargs_nested": vad_kwargs})
            for clip in clips:
                path = temp_dir / f"{clip['clip_id']}.wav"
                with wave.open(str(path), "wb") as stream:
                    stream.setnchannels(1)
                    stream.setsampwidth(2)
                    stream.setframerate(16000)
                    # float32 came from PCM16 /32768, so roundtrip is exact.
                    stream.writeframes(np.round(clip["audio"] * 32768).astype("<i2").tobytes())
                started = time.perf_counter()
                raw = runner.generate(input=str(path), cache={}, vad_kwargs=vad_kwargs)
                spans = [[span.start_s * 1000, span.end_s * 1000] for span in _parse_vad_result(raw)]
                if backend == "automodel_cpu":
                    baseline[clip["clip_id"]] = spans
                    comparison = None
                else:
                    comparison = compare_vad(baseline[clip["clip_id"]], spans, 0.0 if backend == "native_cpu" else VAD_BOUNDARY_TOLERANCE_MS)
                    all_passed &= comparison["passed"]
                write_record(handle, {"type": "vad_result", "backend": backend, "clip_id": clip["clip_id"], "raw": raw,
                                      "spans_ms": spans, "elapsed_ms": round((time.perf_counter() - started) * 1000, 3), "comparison": comparison})
        except Exception as exc:
            all_passed = False
            write_record(handle, {"type": "vad_error", "backend": backend, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        finally:
            if runner is not None:
                close = getattr(runner, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception as exc:
                        all_passed = False
                        write_record(handle, {"type": "cleanup_error", "backend": backend, "error": str(exc)})
                runner = None
            gc.collect()
    return all_passed


def run(args: argparse.Namespace) -> int:
    if args.device != DEVICE:
        raise SystemExit("硬件验证只允许 --device npu:7")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        write_record(handle, {"type": "plan", "device": DEVICE, "pid": os.getpid(), "tolerances": TOLERANCES,
                              "vad_boundary_tolerance_ms": VAD_BOUNDARY_TOLERANCE_MS,
                              "precisions": args.precisions, "compilers": args.compilers,
                              "models_requested": getattr(args, "models", None), "vad_requested": not getattr(args, "skip_vad", False),
                              "reference": "CPU FP32 original checkpoint; CUDA golden not provided",
                              "candidate_precision": "explicit parameter and input dtype; no production setting changed",
                              "limits": "short-slice numerical/latency comparison; no labeled quality acceptance or full-course result"})
        if args.plan_only:
            return 0
        all_passed, clips = True, []
        for sample in manifest_samples(args.manifest, args.sample_id):
            offset = args.offset
            for index, duration in enumerate(args.durations):
                audio, metadata = read_clip(sample, offset, duration, verify_source_hash=args.verify_source_hash and index == 0)
                metadata["clip_id"] = f"{sample['sample_id']}-{index}"
                write_record(handle, {"type": "clip", **metadata})
                clips.append(dict(metadata, audio=audio))
                offset += duration + args.gap
        if not clips:
            raise ValueError("没有可验证的片段")
        import torch
        from core.config import settings
        from core.local_inference import npu_api, configure_npu_precision
        # Select device 7 before any context, option, allocator query or compile.
        api = npu_api()
        math_mode = configure_npu_precision(api, DEVICE)
        torch.set_num_threads(args.cpu_threads)
        write_record(handle, {"type": "environment", "torch": torch.__version__, "torch_npu": importlib.import_module("torch_npu").__version__,
                              "device": DEVICE, "math_mode": math_mode, "background": interference_snapshot()})
        model_dirs = {"speech_campplus": settings.spk_model_dir, "emotion2vec": settings.emotion_model_dir}
        for name in args.models:
            references, runner = {}, None
            try:
                started = time.perf_counter()
                runner = Candidate(name, model_dirs[name], "cpu")
                write_record(handle, {"type": "model_load", "model": name, "backend": "cpu_fp32", "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                                              "operator_adaptations": runner.operator_adaptations})
                for clip in clips:
                    output, timing = runner.infer(clip["audio"])
                    references[clip["clip_id"]] = output
                    write_record(handle, {"type": "reference", "model": name, "clip_id": clip["clip_id"], "output": output, **timing})
            except Exception as exc:
                all_passed = False
                write_record(handle, {"type": "reference_error", "model": name, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
            finally:
                if runner is not None:
                    runner.close()
                    runner = None
                gc.collect()
            if len(references) != len(clips):
                continue
            for compiler in args.compilers:
                for precision in args.precisions:
                    runner, outputs, candidate_ok = None, {}, True
                    try:
                        started = time.perf_counter()
                        runner = Candidate(name, model_dirs[name], DEVICE, precision, compiler)
                        write_record(handle, {"type": "model_load", "model": name, "compiler": compiler, "precision": precision,
                                              "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                                              "parameter_dtypes": runner.parameter_dtypes, "parameter_devices": runner.parameter_devices,
                                              "operator_adaptations": runner.operator_adaptations,
                                              "background": interference_snapshot()})
                        # Return to first shape after visiting all lengths; unique_graphs
                        # deltas distinguish observable recompilation from raw latency.
                        for position, clip in enumerate(clips + clips[:1]):
                            latencies = []
                            for repeat in range(args.repeat + 1):
                                output, timing = runner.infer(clip["audio"])
                                comparison = compare_output(name, references[clip["clip_id"]], output, precision)
                                candidate_ok &= comparison["passed"]
                                outputs[clip["clip_id"]] = output
                                latencies.append(timing["elapsed_ms"])
                                write_record(handle, {"type": "candidate", "model": name, "compiler": compiler, "precision": precision,
                                                      "clip_id": clip["clip_id"], "repeat": repeat, "revisited_first_shape": position == len(clips),
                                                      "phase": "first_for_shape" if repeat == 0 else "steady", "output": output,
                                                      "comparison": comparison, **timing})
                            write_record(handle, {"type": "latency_summary", "model": name, "compiler": compiler, "precision": precision,
                                                  "clip_id": clip["clip_id"], "first_ms": latencies[0],
                                                  "steady_median_ms": float(np.median(latencies[1:])) if len(latencies) > 1 else None})
                        if name == "speech_campplus":
                            cluster = cluster_comparison(references, outputs, clips)
                            candidate_ok &= cluster["same_partition"]
                            write_record(handle, {"type": "cluster", "compiler": compiler, "precision": precision, **cluster})
                    except Exception as exc:
                        candidate_ok = False
                        write_record(handle, {"type": "candidate_error", "model": name, "compiler": compiler, "precision": precision,
                                              "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
                    finally:
                        if runner is not None:
                            try:
                                runner.close()
                            except Exception as exc:
                                candidate_ok = False
                                write_record(handle, {"type": "cleanup_error", "model": name, "compiler": compiler, "precision": precision, "error": str(exc)})
                            runner = None
                        gc.collect()
                        write_record(handle, {"type": "candidate_summary", "model": name, "compiler": compiler, "precision": precision,
                                              "passed": candidate_ok, "background_after_cleanup": interference_snapshot()})
                    # Optional optimization failures remain explicit evidence; only
                    # FP32 eager correctness is a required baseline gate here.
                    if compiler == "eager" and precision == "fp32":
                        all_passed &= candidate_ok
        if not args.skip_vad:
            # 模型配置提供真实FSMN选项，保持与正式VAD池一致。
            vad_kwargs = {}
            with tempfile.TemporaryDirectory(prefix="asr-model-validation-") as directory:
                all_passed &= run_vad(clips, settings.vad_model_dir, vad_kwargs, handle, Path(directory))
        write_record(handle, {"type": "complete", "baseline_passed": bool(all_passed), "background": interference_snapshot()})
        return 0 if all_passed else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs/npu-910b-migration-20261117/audio-manifest.json")
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--offset", type=float, default=60.0)
    parser.add_argument("--durations", type=float, nargs="+", default=[1.0, 3.0, 5.0])
    parser.add_argument("--gap", type=float, default=1.0)
    parser.add_argument("--device", default=DEVICE, choices=[DEVICE])
    parser.add_argument("--models", nargs="+", choices=["speech_campplus", "emotion2vec"], default=["speech_campplus", "emotion2vec"])
    parser.add_argument("--precisions", nargs="+", choices=list(TOLERANCES), default=["fp32", "bf16", "fp16"])
    parser.add_argument("--compilers", nargs="+", choices=["eager", "torchair"], default=["eager", "torchair"])
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-vad", action="store_true")
    parser.add_argument("--verify-source-hash", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.offset < 0 or args.gap < 0 or any(value <= 0 for value in args.durations) or args.repeat < 1 or args.cpu_threads < 1:
        parser.error("offset/gap 必须非负，durations/cpu-threads 必须为正，repeat >= 1")
    if "fp32" not in args.precisions or "eager" not in args.compilers:
        parser.error("必须保留 eager FP32 基线，额外优化候选可选")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
