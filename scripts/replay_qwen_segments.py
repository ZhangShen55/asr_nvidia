#!/usr/bin/env python3
"""Verify fixed VAD PCM archive and emit model-free reproducible replay inputs.

No model import or NPU access. The output pins language and generation settings;
consumers must use the project's independent FP32 frontend, never pre-pad PCM.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def build(manifest_path, samples_path, *, language=None, max_new_tokens=256, indices=None):
    manifest_path, samples_path = Path(manifest_path), Path(samples_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("sample_rate") != 16000:
        raise ValueError("archive must use 16000Hz")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive integer")
    rows = []
    seen = set()
    with np.load(samples_path, allow_pickle=False) as archive:
        for sample in manifest["samples"]:
            index = sample["index"]
            if indices is not None and index not in indices:
                continue
            identity = sample.get("sample_id", str(index))
            if identity in seen:
                raise ValueError(f"duplicate sample identity: {identity}")
            seen.add(identity)
            audio = archive[sample["audio_key"]]
            if audio.ndim != 1 or audio.dtype != np.dtype("float32") or not np.isfinite(audio).all():
                raise ValueError(f"invalid float32 mono PCM: {identity}")
            digest = hashlib.sha256(audio.tobytes()).hexdigest()
            if digest != sample["pcm_sha256"]:
                raise ValueError(f"PCM SHA mismatch: {identity}")
            if audio.size != sample["end_sample"] - sample["start_sample"]:
                raise ValueError(f"PCM sample count mismatch: {identity}")
            if abs(audio.size / 16000 - sample["duration_s"]) > 1/16000:
                raise ValueError(f"PCM duration mismatch: {identity}")
            rows.append(dict(sample, request_id=sample["source_id"], segment_id=identity,
                             language=language, pcm_samples=int(audio.size), verified_pcm_sha256=digest))
    if indices is not None and {x["index"] for x in rows} != set(indices):
        raise ValueError("requested index is absent from manifest")
    return {"manifest": str(manifest_path.resolve()), "samples_npz": str(samples_path.resolve()),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "samples_sha256": hashlib.sha256(samples_path.read_bytes()).hexdigest(),
            "sample_rate": 16000, "sample_count": len(rows), "samples": rows,
            "generation": {"max_new_tokens": max_new_tokens, "do_sample": False,
                           "eos_token_id": [151643, 151645]},
            "frontend_mode": "independent_single_fp32_then_collate",
            "historical_replay": {"exact_old_service_batch_restored": False,
                                  "reason": "本工具仅校验固定新128段；缺少历史慢批逐成员PCM/提示映射，不能宣称恢复历史批或闭环历史根因。"}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("samples", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--language", default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--indices", type=int, nargs="+")
    a = p.parse_args()
    report = build(a.manifest, a.samples, language=a.language, max_new_tokens=a.max_new_tokens, indices=a.indices)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"sample_count": report["sample_count"], "output": str(a.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
