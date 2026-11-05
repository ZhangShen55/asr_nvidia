#!/usr/bin/env python
"""Reproducible isolated FunASR FSMN-VAD vs FireRedVAD runner.

The script intentionally does not import the API service or any downstream
model.  It writes one JSON object per audio/backend, so interrupted full runs
can be resumed safely with ``--resume``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import wave
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO_ROOT = ROOT / "test/audio_file/音频文件"
DEFAULT_OUT = ROOT / "test/artifacts/fireredvad"


def wav_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as w:
        rate, channels, width, frames = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
    return {"sample_rate": rate, "channels": channels, "bit_depth": width * 8, "frames": frames, "duration_s": frames / rate}


def sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def manifest(audio_root: Path, out: Path) -> list[dict]:
    rows = []
    for p in sorted(audio_root.rglob("*.wav")):
        info = wav_info(p)
        rows.append({"path": str(p.relative_to(audio_root)), "absolute_path": str(p), "sha256": sha256(p), **info})
    (out / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def parse_funasr(result) -> list[dict]:
    if isinstance(result, list):
        result = result[0] if result else {}
    stamps = result.get("value", []) if isinstance(result, dict) else []
    out = []
    for item in stamps:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            # FunASR returns milliseconds.
            start, end = float(item[0]) / 1000, float(item[1]) / 1000
        elif isinstance(item, dict):
            start, end = float(item.get("start", 0)), float(item.get("end", 0))
        else:
            continue
        if end > start:
            out.append({"start_s": start, "end_s": end})
    return out


def load_funasr(model_dir: str, device: str, kwargs: dict):
    from funasr import AutoModel
    return AutoModel(model=model_dir, device=device, disable_update=True, disable_pbar=True, vad_kwargs=kwargs)


def build_backend(backend: str, args):
    if backend == "fireredvad":
        from fireredvad import FireRedVad, FireRedVadConfig
        cfg = FireRedVadConfig(
            use_gpu=args.firered_use_gpu, speech_threshold=args.speech_threshold,
            smooth_window_size=args.smooth_window_size, min_speech_frame=args.min_speech_frame,
            max_speech_frame=args.max_speech_frame, min_silence_frame=args.min_silence_frame,
            merge_silence_frame=args.merge_silence_frame, extend_speech_frame=args.extend_speech_frame,
            chunk_max_frame=args.chunk_max_frame)
        return FireRedVad.from_pretrained(args.firered_model_dir, cfg)
    if backend == "funasr":
        return load_funasr(args.funasr_model_dir, args.funasr_device, {
            "max_single_segment_time": args.max_single_segment_time_ms,
            "max_end_silence_time": args.max_end_silence_time_ms,
            "speech_noise_thres": args.speech_noise_thres,
            "decibel_thres": args.decibel_thres, "snr_thres": args.snr_thres})
    raise ValueError(backend)


def run_backend(backend: str, model, path: Path, args) -> tuple[list[dict], dict]:
    started = time.perf_counter()
    if backend == "fireredvad":
        result, _ = model.detect(str(path))
        segments = [{"start_s": float(a), "end_s": float(b)} for a, b in result.get("timestamps", []) if b > a]
    elif backend == "funasr":
        segments = parse_funasr(model.generate(input=str(path), cache={}, vad_kwargs={}))
    else:
        raise ValueError(backend)
    elapsed = time.perf_counter() - started
    return segments, {"elapsed_s": elapsed, "realtime_factor": elapsed / max(wav_info(path)["duration_s"], 1e-9)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT / "results")
    ap.add_argument("--backend", choices=["funasr", "fireredvad", "both"], default="both")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--firered-model-dir", default=str(DEFAULT_OUT / "model/VAD"))
    ap.add_argument("--firered-use-gpu", action="store_true")
    ap.add_argument("--speech-threshold", type=float, default=0.4)
    ap.add_argument("--smooth-window-size", type=int, default=5)
    ap.add_argument("--min-speech-frame", type=int, default=20)
    ap.add_argument("--max-speech-frame", type=int, default=2000)
    ap.add_argument("--min-silence-frame", type=int, default=20)
    ap.add_argument("--merge-silence-frame", type=int, default=0)
    ap.add_argument("--extend-speech-frame", type=int, default=0)
    ap.add_argument("--chunk-max-frame", type=int, default=30000)
    ap.add_argument("--funasr-model-dir", default=str(ROOT / "model/speech_fsmn_vad_zh-cn-16k-common-pytorch"))
    ap.add_argument("--funasr-device", default="cpu")
    ap.add_argument("--max-single-segment-time-ms", type=int, default=30000)
    ap.add_argument("--max-end-silence-time-ms", type=int, default=800)
    ap.add_argument("--speech-noise-thres", type=float, default=0.6)
    ap.add_argument("--decibel-thres", type=float, default=-100.0)
    ap.add_argument("--snr-thres", type=float, default=-100.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = manifest(args.audio_root, args.out.parent)
    if args.limit:
        rows = rows[: args.limit]
    backends = [args.backend] if args.backend != "both" else ["funasr", "fireredvad"]
    models = {backend: build_backend(backend, args) for backend in backends}
    for row in rows:
        for backend in backends:
            key = hashlib.sha1((backend + "\0" + row["path"]).encode()).hexdigest()[:16]
            target = args.out / f"{key}.json"
            if args.resume and target.exists():
                continue
            record = {"backend": backend, "path": row["path"], "sha256": row["sha256"], "duration_s": row["duration_s"], "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            try:
                segments, timing = run_backend(backend, models[backend], Path(row["absolute_path"]), args)
                record.update({"status": "ok", "segments": segments, "segment_count": len(segments), **timing})
            except Exception as exc:
                record.update({"status": "error", "error": repr(exc), "traceback": traceback.format_exc()})
            target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"backend": backend, "path": row["path"], "status": record["status"], "elapsed_s": record.get("elapsed_s"), "error": record.get("error")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
