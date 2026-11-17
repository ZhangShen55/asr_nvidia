#!/usr/bin/env python3
"""Offline Qwen eager/native-batch candidates; never starts a server.

Manifest: {"samples": [{"id": "zh", "path": "/source.wav", "start_s": 0,
"end_s": 4, "language": "auto"}]}. Use path_bytes_b64 for non-UTF8 names.
Run one candidate per process for comparable cold-load measurements:
  python scripts/benchmark_qwen_candidates.py --manifest clips.json \
    --candidate eager --warmup 2 --repeats 5 --output eager.json
  python scripts/benchmark_qwen_candidates.py --manifest clips.json \
    --candidate batch --batch-size 2 --bucket-seconds 2 --output batch.json
Optional --compile torchair compiles forward, not the Python generate loop.
Optional --attention eager switches only the candidate model after loading and
before compilation; default preserves the installed model's attention choice.
First invocation/compilation remains visible in warmup records. This tool
does not implement production microbatch scheduling or quality acceptance.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import statistics
import sys
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass
class Sample:
    id: str
    path: str
    wav_bytes: bytes = field(repr=False)
    frames: int
    language: str | None
    source_sha256: str
    start_frame: int
    end_frame: int

    @property
    def seconds(self):
        return self.frames / 16000

    def metadata(self):
        return {"id": self.id, "path": self.path,
                "path_bytes_b64": base64.b64encode(os.fsencode(self.path)).decode(),
                "source_sha256": self.source_sha256,
                "clip_wav_sha256": hashlib.sha256(self.wav_bytes).hexdigest(),
                "start_frame": self.start_frame, "end_frame": self.end_frame,
                "frames": self.frames, "sample_rate": 16000,
                "duration_s": self.seconds, "language": self.language}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_samples(manifest_path):
    manifest = Path(manifest_path)
    data = json.loads(manifest.read_text())
    rows = data["samples"] if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest.samples must be a nonempty list")
    samples, hashes, ids = [], {}, set()
    for row in rows:
        identifier = str(row["id"])
        if not identifier or identifier in ids:
            raise ValueError(f"empty/duplicate sample id: {identifier!r}")
        ids.add(identifier)
        path = Path(os.fsdecode(base64.b64decode(row["path_bytes_b64"], validate=True))
                    if "path_bytes_b64" in row else row["path"])
        if not path.is_absolute():
            path = manifest.parent / path
        path = str(path.resolve())
        with wave.open(path, "rb") as reader:
            if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth(), reader.getcomptype()) != (16000, 1, 2, "NONE"):
                raise ValueError(f"{identifier}: only 16kHz mono PCM16 WAV is supported")
            total = reader.getnframes()
            start_s = float(row.get("start_s", 0))
            end_s = float(row.get("end_s", total / 16000))
            if not math.isfinite(start_s) or not math.isfinite(end_s):
                raise ValueError(f"{identifier}: nonfinite clip boundary")
            start, end = round(start_s * 16000), round(end_s * 16000)
            if not 0 <= start < end <= total:
                raise ValueError(f"{identifier}: clip outside source bounds")
            reader.setpos(start)
            pcm = reader.readframes(end - start)
            if len(pcm) != (end - start) * 2:
                raise ValueError(f"{identifier}: incomplete PCM data")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as writer:
            writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            writer.writeframes(pcm)
        if path not in hashes:
            hashes[path] = sha256_file(path)
        language = (row.get("language") or "").strip()
        samples.append(Sample(identifier, path, buffer.getvalue(), end - start,
                              None if language.lower() in {"", "auto"} else language,
                              hashes[path], start, end))
    return samples


def length_batches(samples, batch_size, bucket_seconds):
    if batch_size < 1 or not math.isfinite(bucket_seconds) or bucket_seconds <= 0:
        raise ValueError("batch size and bucket width must be positive")
    buckets = {}
    for index, sample in enumerate(samples):
        bucket = math.floor(sample.seconds / bucket_seconds)
        buckets.setdefault(bucket, []).append((index, sample))
    batches = []
    for bucket in sorted(buckets):
        ordered = sorted(buckets[bucket], key=lambda pair: (pair[1].frames, pair[0]))
        batches.extend([ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size)])
    return batches


def validate_masks(inputs, batch_size):
    """Fail closed if token/acoustic padding loses its effective lengths."""
    for key in ("input_ids", "attention_mask", "input_features", "input_features_mask"):
        if key not in inputs:
            raise ValueError(f"processor output missing {key}; masked native batch is required")
    if tuple(inputs["input_ids"].shape) != tuple(inputs["attention_mask"].shape):
        raise ValueError("token mask shape does not match input_ids")
    if (inputs["input_features"].shape[0] != batch_size
            or inputs["input_features"].shape[-1] != inputs["input_features_mask"].shape[-1]):
        raise ValueError("acoustic mask shape does not match input_features")
    result = {}
    for key in ("attention_mask", "input_features_mask"):
        mask = inputs[key]
        if mask.ndim != 2 or mask.shape[0] != batch_size:
            raise ValueError(f"invalid {key} batch shape")
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError(f"{key} must retain a binary padding mask")
        lengths = mask.sum(-1).tolist()
        if any(value <= 0 for value in lengths):
            raise ValueError(f"empty effective lengths in {key}")
        result[key] = {"shape": list(mask.shape), "valid_lengths": lengths}
    return result


def attention_configurations(model):
    """Observe actual shared/nested config objects and their module aliases."""
    rows, known = [], {}

    def visit(config, path):
        if config is None:
            return None
        identity = id(config)
        if identity in known:
            row = known[identity]
            if path not in row["config_paths"]:
                row["config_paths"].append(path)
            return row
        row = {"config_paths": [path], "module_paths": [],
               "config_class": type(config).__name__,
               "implementation": getattr(config, "_attn_implementation", None)}
        rows.append(row)
        known[identity] = row
        for key in getattr(config, "sub_configs", {}) or {}:
            visit(getattr(config, key, None), f"{path}.{key}")
        return row

    visit(getattr(model, "config", None), "config")
    named_modules = getattr(model, "named_modules", None)
    for name, module in named_modules() if callable(named_modules) else [("", model)]:
        config = getattr(module, "config", None)
        if config is not None:
            row = known.get(id(config)) or visit(config, f"module:{name or '<root>'}.config")
            row["module_paths"].append(name or "<root>")
    return rows


def set_candidate_attention(model, mode):
    if mode not in {"default", "eager"}:
        raise ValueError(f"unsupported candidate attention mode: {mode}")
    before = attention_configurations(model)
    if mode == "eager":
        setter = getattr(model, "set_attn_implementation", None)
        if not callable(setter):
            raise RuntimeError("loaded model does not support set_attn_implementation")
        setter("eager")
    after = attention_configurations(model)
    if mode == "eager" and (not after or any(row["implementation"] != "eager" for row in after)):
        raise RuntimeError(f"candidate eager attention did not propagate to every model config: {after}")
    return {"requested": mode, "before": before, "after": after}


class CandidateRunner:
    def __init__(self, model_dir, device, dtype, max_new_tokens, compile_mode="none", attention="default"):
        from core.qwen_asr_runtime import LocalQwenRuntime
        self.runtime = LocalQwenRuntime(model_dir, device, dtype, max_new_tokens)
        self.compile_mode = compile_mode
        self.attention = attention
        self.attention_state = None
        self.compile_setup_ms = 0.0

    def load(self):
        self.runtime.ensure_loaded()
        self.attention_state = set_candidate_attention(self.runtime.model, self.attention)
        if self.compile_mode == "torchair":
            import torch
            import torchair  # lazily exposed by the matching torch-npu wheel
            started = time.perf_counter()
            config = torchair.CompilerConfig()
            backend = torchair.get_npu_backend(compiler_config=config)
            self.runtime.model.forward = torch.compile(self.runtime.model.forward, backend=backend, dynamic=True)
            self.compile_setup_ms = (time.perf_counter() - started) * 1000

    def infer(self, samples, native_batch):
        if not native_batch:
            raw = self.runtime.transcribe(samples[0].wav_bytes, samples[0].language)
            return [raw], {}
        import torch
        from core.qwen_asr_runtime import decode_wav_bytes, move_inputs
        runtime = self.runtime
        if runtime.failure:
            raise RuntimeError(f"candidate runtime is unavailable: {runtime.failure}")
        runtime.api.set_device(runtime.device)
        with runtime.api.device(runtime.device), torch.inference_mode():
            try:
                # Each array retains its own exact PCM length; never concatenate.
                inputs = runtime.processor.apply_transcription_request(
                    audio=[decode_wav_bytes(sample.wav_bytes) for sample in samples],
                    language=[sample.language for sample in samples],
                )
                masks = validate_masks(inputs, len(samples))
                inputs = move_inputs(inputs, device=runtime.device, dtype=runtime.model.dtype)
                output = runtime.model.generate(**inputs, max_new_tokens=runtime.max_new_tokens, do_sample=False)
                sequences = getattr(output, "sequences", output)
                generated = sequences[:, inputs["input_ids"].shape[1]:]
                raw = runtime.processor.batch_decode(generated, skip_special_tokens=False,
                                                     clean_up_tokenization_spaces=False)
                if len(raw) != len(samples):
                    raise ValueError("processor batch result count mismatch")
                # Raw evidence keeps controls; parsing strips the same endings as
                # core/qwen_asr_runtime.py and preserves each result's identity.
                return list(raw), masks
            finally:
                try:
                    runtime.api.synchronize(runtime.device)
                except Exception as exc:
                    runtime.failure = f"candidate NPU synchronize failed: {exc}"
                    raise

    def metadata(self):
        import inspect
        runtime = self.runtime
        config = runtime.model.config
        processor_file = inspect.getfile(type(runtime.processor))
        return {"model_class": type(runtime.model).__name__,
                "model_config_commit_hash": getattr(config, "_commit_hash", None),
                "model_config_name_or_path": getattr(config, "_name_or_path", None),
                "model_type": getattr(config, "model_type", None),
                "actual_model_dtype": str(runtime.model.dtype),
                "actual_model_device": str(runtime.model.device),
                "processor_class": type(runtime.processor).__name__,
                "processor_source": processor_file,
                "processor_source_sha256": sha256_file(processor_file),
                "attention": self.attention_state,
                "actual_attention_configurations": attention_configurations(runtime.model),
                "compile_setup_ms": self.compile_setup_ms,
                "runtime": runtime.snapshot()}

    def close(self):
        self.runtime.close()


def parsed_output(raw):
    from utils.qwen_response import parse_qwen_text
    cleaned = re.sub(r"<\|(?:im_end|endoftext|eot_id)\|>", "", raw).strip()
    parsed = parse_qwen_text(cleaned)
    return {"language": parsed.language, "text": parsed.text, "has_speech": parsed.has_speech}


def error_details(exc):
    # TorchAir lazy import wraps its real ModuleNotFoundError. Preserve causes
    # rather than reporting only the asynchronous wrapper's generic message.
    chain, seen, cause = [], set(), exc
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        chain.append({"error_type": type(cause).__name__, "error": str(cause)})
        cause = cause.__cause__ if cause.__cause__ is not None else cause.__context__
    return {"error_type": type(exc).__name__, "error": str(exc), "error_chain": chain}


def summarize(records):
    steady = [record for record in records if record["phase"] == "steady" and record["status"] == "ok"]
    times = sorted(record["elapsed_ms"] for record in steady)
    audio = sum(record["audio_seconds"] for record in steady)
    total_ms = sum(times)
    return {"successful_batches": len(steady),
            "failed_batches": sum(record["status"] == "failed" for record in records),
            "steady_batch_p50_ms": statistics.median(times) if times else None,
            "steady_batch_p95_ms": times[max(0, math.ceil(len(times) * .95) - 1)] if times else None,
            "steady_audio_seconds": audio, "steady_total_inference_ms": total_ms,
            "steady_rtf": total_ms / 1000 / audio if audio else None,
            "steady_audio_seconds_per_second": audio / (total_ms / 1000) if total_ms else None}


def run_benchmark(samples, *, candidate, batch_size, bucket_seconds, warmup, repeats,
                  runner, report, clock=time.perf_counter, checkpoint=None, should_stop=None):
    """Own the model through load/execution/report failures; no network services."""
    report.update({"records": [], "close": {"status": "pending"}, "load": {"status": "pending"}})
    checkpoint = checkpoint or (lambda: None)
    should_stop = should_stop or (lambda: False)
    batches = ([[(index, sample)] for index, sample in enumerate(samples)] if candidate == "eager"
               else length_batches(samples, batch_size, bucket_seconds))

    def save():
        report["summary"] = summarize(report["records"])
        checkpoint()

    try:
        started = clock()
        try:
            runner.load()
            report["load"] = {"status": "ok", "elapsed_ms": (clock() - started) * 1000}
            report["model"] = runner.metadata()
        except Exception as exc:
            report["load"] = {"status": "failed", "elapsed_ms": (clock() - started) * 1000,
                              **error_details(exc)}
            raise
        for phase, count in (("warmup", warmup), ("steady", repeats)):
            for iteration in range(count):
                for batch_index, pairs in enumerate(batches):
                    if should_stop():
                        raise InterruptedError("stop requested; completed work will be closed before exit")
                    selected = [sample for _, sample in pairs]
                    record = {"phase": phase, "iteration": iteration, "batch_index": batch_index,
                              "sample_ids": [sample.id for sample in selected],
                              "original_indices": [index for index, _ in pairs],
                              "audio_seconds": sum(sample.seconds for sample in selected),
                              "raw_output_stage": "decoder_with_controls" if candidate == "batch" else "runtime_after_end_control_strip"}
                    started = clock()
                    try:
                        outputs, masks = runner.infer(selected, native_batch=candidate == "batch")
                        if len(outputs) != len(selected):
                            raise ValueError("inference result count mismatch")
                        record["outputs"] = [
                            {"sample_id": sample.id, "original_index": index,
                             "raw": raw, "parsed": parsed_output(raw),
                             "raw_contract_ok": not raw.strip() or bool(re.search(r"language\s+[^<]*<asr_text>", raw, re.I))}
                            for (index, sample), raw in zip(pairs, outputs)]
                        if not all(item["raw_contract_ok"] for item in record["outputs"]):
                            raise ValueError("raw output lacks language/asr_text prefix; cannot classify as no-speech")
                        record.update({"status": "ok", "elapsed_ms": (clock() - started) * 1000,
                                       "masks": masks})
                    except Exception as exc:
                        record.update({"status": "failed", "elapsed_ms": (clock() - started) * 1000,
                                       **error_details(exc),
                                       "error_scope": "batch; no member is reported successful"})
                    report["records"].append(record)
                    save()
        save()
    finally:
        started = clock()
        try:
            runner.close()
            report["close"] = {"status": "ok", "elapsed_ms": (clock() - started) * 1000}
        except BaseException as exc:
            report["close"] = {"status": "failed", "elapsed_ms": (clock() - started) * 1000,
                               **error_details(exc)}
            raise
        finally:
            save()
    return report


def model_files(model_dir):
    """Hash small metadata; weight names/sizes do not pretend to be a revision."""
    root, files = Path(model_dir), {}
    if root.is_dir():
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix in {".json", ".jinja", ".safetensors", ".bin"}:
                entry = {"bytes": path.stat().st_size}
                if path.stat().st_size < 16 * 1024 * 1024:
                    entry["sha256"] = sha256_file(path)
                files[path.name] = entry
    return {"path": str(root.resolve()), "files": files,
            "weight_hash_policy": "large weights name/size only; supply revision independently"}


def environment_metadata():
    packages = {}
    for name in ("torch", "torch-npu", "transformers", "numpy", "torchair", "vllm", "vllm-ascend"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    bundled = None
    try:
        path = Path(importlib.metadata.distribution("torch-npu").locate_file("torch_npu/dynamo/torchair/__init__.py"))
        if path.is_file():
            bundled = {"path": str(path), "sha256": sha256_file(path),
                       "note": "bundled source exists; this is not evidence of import/graph execution"}
    except importlib.metadata.PackageNotFoundError:
        pass
    cann = Path("/usr/local/Ascend/ascend-toolkit/latest/opp/version.info")
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "packages": packages,
            "torchair_bundled_in_torch_npu": bundled,
            "cann_opp_version_info": cann.read_text() if cann.is_file() else None,
            "ASCEND_VISIBLE_DEVICES": os.environ.get("ASCEND_VISIBLE_DEVICES"),
            "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("ASCEND_RT_VISIBLE_DEVICES")}


def validate_device(device):
    # This task uses host physical card7. A mapped/container benchmark needs its
    # own reviewed entry point instead of assuming logical7 still means card7.
    if device != "npu:7":
        raise ValueError("this host benchmark is restricted to npu:7")
    for variable in ("ASCEND_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES"):
        if os.environ.get(variable):
            raise ValueError(f"{variable} is set; host physical-card mapping must be unmasked")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate", choices=("eager", "batch"), default="eager")
    parser.add_argument("--compile", choices=("none", "torchair"), default="none", dest="compile_mode")
    parser.add_argument("--attention", choices=("default", "eager"), default="default")
    parser.add_argument("--model-dir", default="/home/model/Qwen3-ASR-1.7B-hf")
    parser.add_argument("--model-revision", default=None, help="traceable upstream revision; not inferred from directory name")
    parser.add_argument("--device", default="npu:7")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--bucket-seconds", type=float, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    validate_device(args.device)
    if min(args.repeats, args.batch_size, args.max_new_tokens) < 1 or args.warmup < 0:
        parser.error("repeats/batch-size/max-new-tokens must be positive; warmup >=0")
    if not math.isfinite(args.bucket_seconds) or args.bucket_seconds <= 0:
        parser.error("bucket-seconds must be finite and positive")
    samples = load_samples(args.manifest)
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists; preserve prior evidence with a new run path")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"run_started_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
              "configuration": vars(args), "environment": environment_metadata(),
              "model_files": model_files(args.model_dir), "declared_model_revision": args.model_revision,
              "samples": [sample.metadata() for sample in samples],
              "scope": "offline; synchronized device completion; shared card; no quality claim"}

    def checkpoint():
        temporary = output.with_suffix(output.suffix + ".partial")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
        temporary.replace(output)

    runner = CandidateRunner(args.model_dir, args.device, args.dtype, args.max_new_tokens, args.compile_mode, args.attention)
    failed = False
    stop_signals = []
    previous_handlers = {}

    def request_stop(signum, frame):
        # Signals cannot safely interrupt owned NPU work or its cleanup. Repeated
        # requests are recorded and propagate only after synchronous close.
        stop_signals.append(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_stop)
    try:
        run_benchmark(samples, candidate=args.candidate, batch_size=args.batch_size,
                      bucket_seconds=args.bucket_seconds, warmup=args.warmup, repeats=args.repeats,
                      runner=runner, report=report, checkpoint=checkpoint,
                      should_stop=lambda: bool(stop_signals))
    except BaseException as exc:
        report["fatal_error"] = error_details(exc)
        failed = True
    finally:
        report["stop_signals"] = stop_signals
        report["stop_requested"] = bool(stop_signals)
        checkpoint()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    print(json.dumps({"report": str(output.resolve()), "summary": report.get("summary"),
                      "close": report.get("close"), "fatal_error": report.get("fatal_error")}))
    if stop_signals:
        return 128 + stop_signals[0]
    return 1 if failed or report["summary"]["failed_batches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
