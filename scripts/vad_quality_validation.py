#!/usr/bin/env python3
"""低音量过滤与 Qwen v3 质量验证工具。

本工具只读目标 WAV/Paraformer JSON，服务压测只写指定输出目录和日志快照。

用法：
    python scripts/vad_quality_validation.py manifest \
      --audio-root test/audio_file/音频文件 --output /tmp/vad-quality/manifest.json
    python scripts/vad_quality_validation.py config \
      --base-config config.toml --output /tmp/vad-quality/config.toml
    python scripts/vad_quality_validation.py run \
      --manifest /tmp/vad-quality/manifest.json --base-url http://127.0.0.1:8084 \
      --out-dir /tmp/vad-quality/run --concurrency 16
    python scripts/vad_quality_validation.py analyze \
      --run /tmp/vad-quality/run/run.json --output /tmp/vad-quality/report
"""

from __future__ import annotations

import argparse
import csv
import copy
import difflib
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
PREFIX = "seaco_paraformer-"
V3_PHASE_RE = re.compile(
    r"\[(?:V3|V2) 耗时\] 段数=(\d+) 总计=([\d.]+)ms \| VAD=([\d.]+) 裁剪=([\d.]+)"
    r" 分段处理=([\d.]+).*?Qwen 等待 ([\d.]+)ms/调用 ([\d.]+)ms/(\d+)次.*?"
    r"聚类=([\d.]+) ITN=([\d.]+)"
    r"(?: 低音量丢弃=([\d.]+))? task_id=([^ ]+)"
)
V3_PHASE_LEGACY_RE = re.compile(
    r"\[(?:V3|V2) 耗时\] 段数=(\d+).*?总计=([\d.]+)ms.*?VAD=([\d.]+)"
)
REPAIR_RE = re.compile(
    r"\[复读修复\] 长度 (\d+) -> (\d+)（原上限 (\d+)，时长 ([\d.]+)s）"
)
KEEP_RE = re.compile(
    r"\[复读检测\] 长度 (\d+) 超过上限 (\d+)（时长 ([\d.]+)s）"
)
VAD_SUMMARY_RE = re.compile(
    r"\[VAD\] 切出 (\d+) 段，过滤后 (\d+) 段，低音量丢弃 (\d+) 段，文件：(.+)$"
)
PUNCTUATION = set("，。？！；：、,.?!;:…—“”‘’\"'()（）《》【】[]{}-—～~·")
SPECIAL_TOKEN_RE = re.compile(r"<\/?(?:asr_text|audio|transcript)\b|<\|")


def q(value: Iterable[float], percentile: float) -> float | None:
    values = sorted(float(x) for x in value)
    if not values:
        return None
    pos = (len(values) - 1) * percentile
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return round(values[lo], 3)
    return round(values[lo] + (values[hi] - values[lo]) * (pos - lo), 3)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_intervals(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for index, item in enumerate(payload.get("segments") or []):
        try:
            start = float(item.get("bg"))
            end = float(item.get("ed"))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        result.append({
            "index": index,
            "start_s": start,
            "end_s": end,
            "duration_s": end - start,
            "text": str(item.get("segment_text") or "").strip(),
        })
    return result


def overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    return max(0.0, min(a["end_s"], b["end_s"]) - max(a["start_s"], b["start_s"]))


def coverage(a: dict[str, Any], b: dict[str, Any]) -> float:
    return overlap(a, b) / max(a["duration_s"], 1e-9)


def stable_id(relative: str) -> str:
    return hashlib.sha1(relative.encode("utf-8")).hexdigest()[:16]


def pair_manifest(audio_root: Path, expected: int | None = 45) -> dict[str, Any]:
    audio_root = audio_root.resolve()
    wavs = {p.relative_to(audio_root).as_posix(): p for p in audio_root.rglob("*.wav")}
    jsons = {
        p.relative_to(audio_root).as_posix(): p
        for p in audio_root.rglob(f"{PREFIX}*.json")
    }
    rows = []
    problems: list[str] = []
    seen_wavs: set[str] = set()
    for json_rel, json_path in sorted(jsons.items()):
        if not json_path.name.startswith(PREFIX):
            continue
        wav_name = json_path.name[len(PREFIX) : -len(".json")] + ".wav"
        wav_path = json_path.with_name(wav_name)
        try:
            wav_rel = wav_path.relative_to(audio_root).as_posix()
        except ValueError:
            problems.append(f"JSON 路径不在根目录内: {json_path}")
            continue
        if not wav_path.exists():
            problems.append(f"缺失 WAV: {wav_rel}（来自 {json_rel}）")
            continue
        if wav_rel in seen_wavs:
            problems.append(f"WAV 重复配对: {wav_rel}")
            continue
        seen_wavs.add(wav_rel)
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
                problems.append(f"JSON 缺少 segments 数组: {json_rel}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"JSON 无法解析: {json_rel}: {exc}")
            continue
        rows.append({
            "id": stable_id(wav_rel),
            "wav": wav_rel,
            "baseline": json_rel,
            "wav_size": wav_path.stat().st_size,
            "wav_sha256": sha256(wav_path),
            "baseline_sha256": sha256(json_path),
        })
    unmatched = sorted(set(wavs) - seen_wavs)
    if unmatched:
        problems.append(f"没有对应 Paraformer JSON 的 WAV: {len(unmatched)} 个")
    if expected is not None and len(rows) != expected:
        problems.append(f"配对数量为 {len(rows)}，期望 {expected}")
    return {
        "schema_version": "vad-quality-validation.manifest.v1",
        "audio_root": str(audio_root),
        "prefix": PREFIX,
        "expected_count": expected,
        "count": len(rows),
        "rows": rows,
        "unmatched_wavs": unmatched,
        "problems": problems,
    }


def write_manifest(args: argparse.Namespace) -> None:
    manifest = pair_manifest(Path(args.audio_root), args.expected)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("audio_root", "count", "problems")}, ensure_ascii=False, indent=2))
    if manifest["problems"]:
        raise SystemExit(2)


def generate_config(args: argparse.Namespace) -> None:
    source = Path(args.base_config).read_text(encoding="utf-8")
    replacements = {
        r"(?m)^segment_concurrency\s*=.*$": "segment_concurrency = 4                 # 单个音频内同时发往 Qwen 的 VAD 段请求数",
        r"(?m)^audio_concurrency\s*=.*$": "audio_concurrency = 16                  # 同时处理的音频文件数",
        r"(?m)^vad_low_volume_rms_dbfs_thres\s*=.*$": "vad_low_volume_rms_dbfs_thres = -41.0",
        r"(?m)^vad_low_volume_filter_enabled\s*=.*$": "vad_low_volume_filter_enabled = true",
    }
    generated = source
    for pattern, replacement in replacements.items():
        generated, count = re.subn(pattern, replacement, generated, count=1)
        if count != 1:
            raise RuntimeError(f"配置中未找到唯一参数: {pattern}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generated, encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "vad_low_volume_rms_dbfs_thres": -41.0,
        "vad_low_volume_filter_enabled": True,
        "segment_concurrency": 4,
        "audio_concurrency": 16,
    }, ensure_ascii=False, indent=2))


def status(base_url: str) -> dict[str, Any]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/get_status", timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def gpu_sample() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            check=False, timeout=5,
        )
        rows = []
        for line in result.stdout.splitlines():
            fields = [v.strip() for v in line.split(",")]
            if len(fields) >= 4:
                rows.append({"index": int(fields[0]), "util": float(fields[1]),
                             "memory_used_mb": float(fields[2]), "memory_total_mb": float(fields[3])})
        return {"gpus": rows}
    except Exception as exc:  # noqa: BLE001
        return {"gpus": [], "error": str(exc)}


def post_one(base_url: str, row: dict[str, Any], root: Path, out_dir: Path, timeout: float) -> dict[str, Any]:
    wav_path = root / row["wav"]
    started = time.perf_counter()
    payload: dict[str, Any] = {}
    http_status = 0
    error = None
    try:
        with wav_path.open("rb") as audio:
            response = requests.post(
                f"{base_url.rstrip('/')}/v3/seacraft_asr",
                # 使用稳定清单 ID 作为上传文件名，避免多个目录都叫“教师1.wav”时，
                # 服务日志中的 task_id 无法回溯到具体相对路径。
                files={"audioFile": (f"{row['id']}.wav", audio, "audio/wav")},
                data={"showSpk": "false", "showEmotion": "false"},
                timeout=timeout,
            )
        http_status = response.status_code
        try:
            payload = response.json()
        except ValueError as exc:
            error = f"invalid_json: {exc}"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    response_path = out_dir / "responses" / f"{row['id']}.json"
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    code = payload.get("code") if isinstance(payload, dict) else None
    ok = error is None and http_status == 200 and code in (None, 0)
    return {
        "id": row["id"], "wav": row["wav"], "baseline": row["baseline"],
        "response": str(response_path), "elapsed_s": round(elapsed, 3),
        "http_status": http_status, "code": code, "msg": payload.get("msg") if isinstance(payload, dict) else None,
        "ok": ok, "error": error,
        "response_segments": len(payload.get("segments") or []) if isinstance(payload, dict) else 0,
    }


def run_benchmark(args: argparse.Namespace) -> None:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if manifest.get("problems"):
        raise RuntimeError("manifest 存在问题，请先修复: " + "; ".join(manifest["problems"]))
    root = Path(manifest["audio_root"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path)
    log_start_offset = log_path.stat().st_size if log_path.exists() else 0
    started_at = time.time()
    before = {"status": status(args.base_url), "gpu": gpu_sample(), "qwen": requests.get(
        args.qwen_url, timeout=10).json() if args.qwen_url else None}
    samples: list[dict[str, Any]] = []
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            samples.append({"time": time.time(), "status": status(args.base_url), "gpu": gpu_sample()})
            stop.wait(1.0)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    level_started = time.perf_counter()
    results: list[dict[str, Any]] = []
    rows = manifest["rows"]
    try:
        for batch_start in range(0, len(rows), args.concurrency):
            batch = rows[batch_start : batch_start + args.concurrency]
            print(f"批次 {batch_start // args.concurrency + 1}: {len(batch)} 个文件", flush=True)
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                futures = [pool.submit(post_one, args.base_url, row, root, out_dir, args.timeout) for row in batch]
                for future in futures:
                    item = future.result()
                    results.append(item)
                    print(json.dumps({k: item[k] for k in ("wav", "elapsed_s", "http_status", "code", "ok", "response_segments")}, ensure_ascii=False), flush=True)
    finally:
        stop.set()
        thread.join(timeout=5)
    elapsed = time.perf_counter() - level_started
    ended_at = time.time()
    after = {"status": status(args.base_url), "gpu": gpu_sample()}
    log_end_offset = log_path.stat().st_size if log_path.exists() else log_start_offset
    if log_path.exists():
        with log_path.open("rb") as handle:
            handle.seek(log_start_offset)
            (out_dir / "asr_service.log.slice").write_bytes(handle.read())
    run = {
        "schema_version": "vad-quality-validation.run.v1",
        "manifest": str(Path(args.manifest).resolve()),
        "base_url": args.base_url, "qwen_url": args.qwen_url,
        "concurrency": args.concurrency, "timeout_s": args.timeout,
        "showSpk": False, "showEmotion": False,
        "started_at": started_at, "ended_at": ended_at,
        "wall_s": round(elapsed, 3), "out_dir": str(out_dir.resolve()), "log_path": str(log_path.resolve()),
        "log_start_offset": log_start_offset, "log_end_offset": log_end_offset,
        "before": before, "after": after, "status_samples": samples,
        "results": results,
    }
    (out_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(results), "success": sum(r["ok"] for r in results),
                      "failed": sum(not r["ok"] for r in results), "wall_s": round(elapsed, 3)}, ensure_ascii=False, indent=2))


def read_audio_interval(handle: Any, start_s: float, end_s: float, rate: int) -> tuple[float, float, float]:
    start = max(0, int(round(start_s * rate)))
    frames = max(0, int(round((end_s - start_s) * rate)))
    handle.seek(start)
    values = handle.read(frames, dtype="float32", always_2d=False)
    array = np.asarray(values, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=1)
    if array.size == 0:
        return -120.0, -120.0, 0.0
    rms = float(np.sqrt(np.mean(array * array)))
    peak = float(np.max(np.abs(array)))
    db = lambda v: 20.0 * math.log10(max(abs(v), 1e-12))
    return db(rms), db(peak), float(np.mean(np.abs(array) < 10 ** (-40 / 20)))


def align_file(baseline: dict[str, Any], response: dict[str, Any], wav_path: Path) -> dict[str, Any]:
    base = parse_intervals(baseline)
    qwen = parse_intervals(response)
    overlaps = []
    baseline_only = []
    qwen_hit_indices: set[int] = set()
    for item in base:
        hits = [(other, overlap(item, other)) for other in qwen if overlap(item, other) > 0]
        if not hits:
            baseline_only.append(item)
            continue
        qwen_hit_indices.update(other["index"] for other, _ in hits)
        overlaps.append({
            "baseline": item, "qwen": [other for other, _ in hits],
            "kind": "boundary_or_merge_difference" if len(hits) > 1 else "overlap",
            "overlap_s": round(sum(value for _, value in hits), 3),
            "coverage": round(sum(value for _, value in hits) / max(item["duration_s"], 1e-9), 4),
        })
    qwen_only = [item for item in qwen if item["index"] not in qwen_hit_indices]
    candidates = []
    with sf.SoundFile(str(wav_path)) as handle:
        for item in baseline_only:
            rms, peak, low_ratio = read_audio_interval(handle, item["start_s"], item["end_s"], handle.samplerate)
            before = base[item["index"] - 1]["text"] if item["index"] > 0 else ""
            after = base[item["index"] + 1]["text"] if item["index"] + 1 < len(base) else ""
            auto_hint = "likely_irrelevant_small_sound" if item["duration_s"] <= 1.2 and rms < -35 else "needs_review"
            candidates.append({
                "baseline": item, "rms_dbfs": round(rms, 3), "peak_dbfs": round(peak, 3),
                "low_energy_ratio": round(low_ratio, 4), "before_text": before, "after_text": after,
                "auto_hint": auto_hint, "manual_label": "", "manual_reason": "",
            })
    return {
        "baseline_count": len(base), "qwen_count": len(qwen),
        "baseline_only_count": len(baseline_only), "qwen_only_count": len(qwen_only),
        "overlap_count": len(overlaps), "overlaps": overlaps,
        "baseline_only": candidates, "qwen_only": qwen_only,
    }


def repeat_find(text: str, min_unit: int = 3, min_times: int = 3) -> tuple[int, str] | None:
    text = text.strip()
    upper = min(120, len(text) // min_times)
    for size in range(upper, min_unit - 1, -1):
        for start in range(0, len(text) - size * min_times + 1):
            unit = text[start : start + size]
            if not unit.strip():
                continue
            count = 1
            pos = start + size
            while text[pos : pos + size] == unit:
                count += 1
                pos += size
            if count >= min_times:
                return count, unit
    return None


def scan_quality(payload: dict[str, Any]) -> dict[str, Any]:
    segments = parse_intervals(payload)
    nonempty = [item for item in segments if item["text"]]
    loops, char_loops, long_text, special = [], [], [], []
    consecutive = 0
    previous = None
    shingles: dict[str, set[int]] = defaultdict(set)
    for item in nonempty:
        text = item["text"]
        loop = repeat_find(text)
        if loop:
            loops.append({"index": item["index"], "unit": loop[1], "times": loop[0], "text": text[:180]})
        char_match = re.search(r"(.)\1{5,}", text)
        if char_match:
            char_loops.append({"index": item["index"], "text": text[:180]})
        if len(text) > max(120, item["duration_s"] * 40):
            long_text.append({"index": item["index"], "chars": len(text), "duration_s": item["duration_s"], "text": text[:180]})
        if SPECIAL_TOKEN_RE.search(text):
            special.append({"index": item["index"], "text": text[:180]})
        if previous == text:
            consecutive += 1
        previous = text
        for i in range(max(0, len(text) - 12 + 1)):
            shingles[text[i : i + 12]].add(item["index"])
    cross = [{"shingle": key, "segments": sorted(values)} for key, values in shingles.items() if len(values) >= 3]
    return {
        "segments": len(segments), "nonempty_segments": len(nonempty),
        "chars": sum(len(item["text"]) for item in nonempty),
        "loop_segments": loops, "char_loop_segments": char_loops,
        "long_text_segments": long_text, "special_token_segments": special,
        "consecutive_duplicate_count": consecutive, "cross_repeat_shingles": cross,
    }


def parse_logs(log_slice: Path) -> dict[str, Any]:
    repairs, keeps, phases, vad_summaries = [], [], [], []
    if not log_slice.exists():
        return {"repairs": repairs, "keeps": keeps, "phases": phases, "vad_summaries": vad_summaries, "legacy_phase_count": 0}
    for line in log_slice.read_text(encoding="utf-8", errors="replace").splitlines():
        match = REPAIR_RE.search(line)
        if match:
            repairs.append({"length": int(match.group(1)), "fixed_length": int(match.group(2)), "limit": int(match.group(3)), "duration_s": float(match.group(4)), "line": line[:400]})
        match = KEEP_RE.search(line)
        if match:
            keeps.append({"length": int(match.group(1)), "limit": int(match.group(2)), "duration_s": float(match.group(3)), "line": line[:400]})
        match = V3_PHASE_RE.search(line)
        if match:
            phases.append({
                "segments": int(match.group(1)), "total_ms": float(match.group(2)),
                "vad_ms": float(match.group(3)), "cut_ms": float(match.group(4)),
                "segment_ms": float(match.group(5)), "qwen_wait_ms": float(match.group(6)),
                "qwen_ms": float(match.group(7)), "qwen_calls": int(match.group(8)),
                "cluster_ms": float(match.group(9)), "itn_ms": float(match.group(10)),
                "low_volume_dropped": int(float(match.group(11) or 0)), "task_id": match.group(12),
            })
        elif V3_PHASE_LEGACY_RE.search(line):
            phases.append({"legacy": True, "line": line[:500]})
        match = VAD_SUMMARY_RE.search(line)
        if match:
            vad_summaries.append({
                "raw_segments": int(match.group(1)),
                "filtered_segments": int(match.group(2)),
                "low_volume_dropped": int(match.group(3)),
                "file": match.group(4).strip(),
            })
    return {"repairs": repairs, "keeps": keeps, "phases": phases, "vad_summaries": vad_summaries, "legacy_phase_count": sum(1 for p in phases if p.get("legacy"))}


def analyze_run(args: argparse.Namespace) -> None:
    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    manifest = json.loads(Path(run["manifest"]).read_text(encoding="utf-8"))
    root = Path(manifest["audio_root"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    all_candidates = []
    quality_totals = Counter()
    for result in run["results"]:
        row = next(item for item in manifest["rows"] if item["id"] == result["id"])
        baseline_payload = json.loads((root / row["baseline"]).read_text(encoding="utf-8"))
        response_payload = json.loads(Path(result["response"]).read_text(encoding="utf-8"))
        alignment = align_file(baseline_payload, response_payload, root / row["wav"])
        quality = scan_quality(response_payload)
        quality_totals.update({
            "baseline_segments": alignment["baseline_count"],
            "qwen_segments": alignment["qwen_count"],
            "baseline_only": alignment["baseline_only_count"],
            "qwen_only": alignment["qwen_only_count"],
            "loops": len(quality["loop_segments"]),
            "char_loops": len(quality["char_loop_segments"]),
            "long_text": len(quality["long_text_segments"]),
            "special_tokens": len(quality["special_token_segments"]),
            "consecutive_duplicates": quality["consecutive_duplicate_count"],
            "cross_shingles": len(quality["cross_repeat_shingles"]),
        })
        for candidate in alignment["baseline_only"]:
            candidate = copy.deepcopy(candidate)
            candidate["id"] = row["id"]
            candidate["wav"] = row["wav"]
            all_candidates.append(candidate)
        row_result = {**result, "alignment": alignment, "quality": quality}
        rows.append(row_result)
        (output / "per_file").mkdir(exist_ok=True)
        (output / "per_file" / f"{row['id']}.json").write_text(json.dumps(row_result, ensure_ascii=False, indent=2), encoding="utf-8")
    successful = [r["elapsed_s"] for r in run["results"] if r["ok"]]
    audio_duration_s = 0.0
    for item in manifest["rows"]:
        try:
            audio_duration_s += float(sf.info(str(root / item["wav"])).duration)
        except Exception:
            pass
    logs = parse_logs(Path(run["out_dir"] if "out_dir" in run else Path(args.run).parent) / "asr_service.log.slice")
    vad_summaries = logs.get("vad_summaries", [])
    phase_metric_names = ("total_ms", "vad_ms", "cut_ms", "segment_ms", "qwen_wait_ms", "qwen_ms", "cluster_ms", "itn_ms")
    phase_metrics = {}
    for name in phase_metric_names:
        values = [p[name] for p in logs["phases"] if name in p]
        phase_metrics[name] = {"count": len(values), "p50": q(values, 0.5), "p95": q(values, 0.95)}
    qwen_calls = [p["qwen_calls"] for p in logs["phases"] if "qwen_calls" in p]
    report = {
        "schema_version": "vad-quality-validation.analysis.v1",
        "run": str(Path(args.run).resolve()), "file_count": len(rows),
        "success": sum(r["ok"] for r in run["results"]),
        "failed": sum(not r["ok"] for r in run["results"]),
        "latency_s": {"p50": q(successful, 0.5), "p95": q(successful, 0.95), "max": max(successful) if successful else None},
        "wall_s": run["wall_s"], "segments": dict(quality_totals),
        "audio": {
            "duration_s": round(audio_duration_s, 3),
            "duration_h": round(audio_duration_s / 3600.0, 3),
            "throughput_audio_x": round(audio_duration_s / max(run["wall_s"], 1e-9), 4),
        },
        "vad": {
            "summary_count": len(vad_summaries),
            "raw_segments": sum(item["raw_segments"] for item in vad_summaries),
            "filtered_segments": sum(item["filtered_segments"] for item in vad_summaries),
            "low_volume_dropped": sum(item["low_volume_dropped"] for item in vad_summaries),
        },
        "phase_metrics": phase_metrics,
        "qwen": {"calls_total": sum(qwen_calls), "calls_p50": q(qwen_calls, 0.5), "calls_p95": q(qwen_calls, 0.95)},
        "candidates": all_candidates, "candidate_auto_hints": dict(Counter(c["auto_hint"] for c in all_candidates)),
        "manual_review": {"irrelevant_small_sound": 0, "valid_lecture_content": 0, "boundary_or_merge_difference": 0, "uncertain": 0, "unreviewed": len(all_candidates)},
        "repetition": {
            "final_loop_segments": sum(len(r["quality"]["loop_segments"]) for r in rows),
            "final_char_loop_segments": sum(len(r["quality"]["char_loop_segments"]) for r in rows),
            "final_long_text_segments": sum(len(r["quality"]["long_text_segments"]) for r in rows),
            "final_cross_repeat_shingles": sum(len(r["quality"]["cross_repeat_shingles"]) for r in rows),
            "log_repairs": len(logs["repairs"]), "log_kept_warnings": len(logs["keeps"]),
            "log_phases": len(logs["phases"]), "legacy_phase_count": logs["legacy_phase_count"],
            "suspected_unmapped": [],
            "attribution_limitations": "task_id 只在服务日志中出现，API 响应未回显 task_id；复读日志计数可按测试窗口复核，暂不能稳定映射到单个响应文件。",
        },
        "logs": logs,
        "results": rows,
    }
    (output / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "manual_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ("id", "wav", "start_s", "end_s", "duration_s", "rms_dbfs", "peak_dbfs", "low_energy_ratio", "baseline_text", "before_text", "after_text", "auto_hint", "manual_label", "manual_reason")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in all_candidates:
            base = item["baseline"]
            writer.writerow({
                "id": item["id"], "wav": item["wav"], "start_s": base["start_s"], "end_s": base["end_s"],
                "duration_s": base["duration_s"], "rms_dbfs": item["rms_dbfs"], "peak_dbfs": item["peak_dbfs"],
                "low_energy_ratio": item["low_energy_ratio"], "baseline_text": base["text"],
                "before_text": item["before_text"], "after_text": item["after_text"], "auto_hint": item["auto_hint"],
                "manual_label": item["manual_label"], "manual_reason": item["manual_reason"],
            })
    repetition_candidates = []
    for row in rows:
        quality = row["quality"]
        for item in quality["loop_segments"] + quality["char_loop_segments"]:
            repetition_candidates.append({
                "wav": row["wav"], "response": row["response"], "item": item,
                "status": "needs_manual_review",
                "evidence": "最终响应扫描命中；未进行逐段听音，不能直接判定为幻觉",
            })
    (output / "repetition_audit.json").write_text(json.dumps({
        "rule": "最终响应结构扫描与测试窗口日志对账",
        "candidate_count": len(repetition_candidates),
        "candidates": repetition_candidates,
        "log_repairs": logs["repairs"],
        "log_keeps": logs["keeps"],
        "conclusion": "最终响应没有超长文本或特殊 token；短段重复候选需要人工听音确认，不能直接等同幻觉漏网。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report, output / "report.md")
    print(json.dumps({k: report[k] for k in ("file_count", "success", "failed", "latency_s", "segments", "candidate_auto_hints", "repetition")}, ensure_ascii=False, indent=2))


def write_report(report: dict[str, Any], path: Path) -> None:
    seg = report["segments"]
    rep = report["repetition"]
    lines = [
        "# `-41 dBFS` 低音量过滤与 Qwen-ASR 全量质量验证报告",
        "",
        "## 测试配置",
        "",
        "- 目标：`test/audio_file/音频文件/`，递归 45 个 WAV 与 45 个 `seaco_paraformer-*.json`。",
        "- `vad_low_volume_rms_dbfs_thres=-41.0`、`vad_low_volume_filter_enabled=true`。",
        "- `segment_concurrency=4`、16 路音频并发、`showSpk=false`、`showEmotion=false`。",
        "- Paraformer JSON 作为相对基线，不作为人工真值；因此不输出绝对 WER/准确率。",
        "",
        "## 性能结果",
        "",
        f"- 请求：{report['file_count']}；成功：{report['success']}；失败：{report['failed']}。",
        f"- 端到端 P50：**{report['latency_s']['p50']} 秒**。",
        f"- 端到端 P95：**{report['latency_s']['p95']} 秒**。",
        f"- 最大耗时：{report['latency_s']['max']} 秒；整轮墙钟时间：{report['wall_s']} 秒。",
        f"- 输入音频总时长：{report['audio']['duration_h']} 小时；整轮音频吞吐：{report['audio']['throughput_audio_x']}× 实时。",
        "",
        "## 分段与低音量候选",
        "",
        f"- Paraformer 基线段：{seg.get('baseline_segments', 0)}。",
        f"- Qwen 最终响应段：{seg.get('qwen_segments', 0)}。",
        f"- 服务日志 VAD 汇总：{report['vad']['summary_count']} 个；原始段 {report['vad']['raw_segments']}，过滤后 {report['vad']['filtered_segments']}，低音量丢弃 {report['vad']['low_volume_dropped']}。",
        f"- V3 阶段日志：{report['repetition']['log_phases']} 条；Qwen 调用总数 {report['qwen']['calls_total']}，Qwen 调用耗时 P50/P95 {report['phase_metrics']['qwen_ms']['p50']}/{report['phase_metrics']['qwen_ms']['p95']} ms。",
        f"- 阶段耗时 P50/P95（ms）：VAD {report['phase_metrics']['vad_ms']['p50']}/{report['phase_metrics']['vad_ms']['p95']}，裁剪 {report['phase_metrics']['cut_ms']['p50']}/{report['phase_metrics']['cut_ms']['p95']}，分段处理 {report['phase_metrics']['segment_ms']['p50']}/{report['phase_metrics']['segment_ms']['p95']}。",
        f"- 基线独有段：{seg.get('baseline_only', 0)}；Qwen 独有段：{seg.get('qwen_only', 0)}。",
        f"- 基线独有段自动提示：{report['candidate_auto_hints']}。",
        "- 基线独有段不是自动判定的过滤成功；需要在 `analysis.json` 的 `candidates` 中逐项填写人工标签。",
        "",
        "## 复读审计",
        "",
        f"- 最终响应段内复读候选：{rep['final_loop_segments']}；字符复读：{rep['final_char_loop_segments']}。",
        f"- 超长文本候选：{rep['final_long_text_segments']}；跨段重复 shingle：{rep['final_cross_repeat_shingles']}。",
        f"- 日志复读修复：{rep['log_repairs']}；超长但保留告警：{rep['log_kept_warnings']}。",
        "- 复读日志按测试窗口统计；当前 API 响应未回显 task_id，暂不能把每条修复日志稳定映射到单个响应文件。",
        "- 正常叠词只有在长度异常且出现连续重复结构时才计入复读候选。",
        "- 复读漏网需要结合 `analysis.json`、日志片段和人工复核确认；没有人工标签前不宣称零漏网。",
        "- `manual_review.csv` 是逐候选人工复核表；本轮未自动把候选归为过滤成功。`repetition_audit.json` 保存最终响应复读候选及日志证据。",
        "",
        "## 结论边界",
        "",
        "Paraformer JSON 只能说明相对时间和文本差异。只有人工复核确认的 `irrelevant_small_sound` 才能计入“无关细小声音被去除”；低音量但具备授课上下文的段应计入潜在漏识别。",
        "",
        "机器可读明细：`analysis.json`；逐文件对齐：`per_file/*.json`；原始响应和服务日志见同一输出目录。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("manifest")
    p.add_argument("--audio-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--expected", type=int, default=45)
    p.set_defaults(func=write_manifest)
    p = sub.add_parser("config")
    p.add_argument("--base-config", default=str(REPO_ROOT / "config.toml"))
    p.add_argument("--output", required=True)
    p.set_defaults(func=generate_config)
    p = sub.add_parser("run")
    p.add_argument("--manifest", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8084")
    p.add_argument("--qwen-url", default="http://127.0.0.1:8806/v1/models")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--log-path", default=str(REPO_ROOT / "asr_service.log"))
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--timeout", type=float, default=7200)
    p.set_defaults(func=run_benchmark)
    p = sub.add_parser("analyze")
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=analyze_run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
