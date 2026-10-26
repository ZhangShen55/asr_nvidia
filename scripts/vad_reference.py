"""Reference-audio VAD regression helpers and optional threshold scanner.

The supplied annotation file is a human-edited JSON-like document, so this module
deliberately parses its segment objects without rewriting the source fixture.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import wave
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

import numpy as np

# Allow the documented ``python scripts/vad_reference.py`` invocation to import
# repository packages when the script directory is placed first on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import settings
from core.vad import VadSegment, _parse_vad_result
from core.vad_pool import VadPool
from core.vad_volume import compute_volume_stats, percentile_summary

DEFAULT_AUDIO = Path("test/audio_file/理论计算机科学导论_时间am9.55_191803.wav")
DEFAULT_ANNOTATION = Path("test/test_result/理论计算机科学导论_时间am9.55_191803.json")


def parse_reference_annotations(path: str | Path = DEFAULT_ANNOTATION) -> list[dict]:
    """Parse marked segments from the human-edited reference document."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("细小声音"):
            # The marker is inside the object, after its opening brace. Insert a
            # private JSON field immediately after that brace.
            for index in range(len(cleaned) - 1, -1, -1):
                if cleaned[index].strip() == "{":
                    indent = cleaned[index][: len(cleaned[index]) - len(cleaned[index].lstrip())] + "    "
                    cleaned.insert(index + 1, f'{indent}"__small_noise__": true,')
                    break
            continue
        # Human notes such as "转写错误应该是 ..." are intentionally skipped.
        if stripped and not stripped.startswith(('"', "{", "}", "[", "]", ",")):
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)
    # A few hand-edited objects omit the comma before the bg field.
    text = re.sub(
        r'("segment_text"\s*:\s*"(?:[^"\\]|\\.)*")\s*(?="bg")',
        r"\1,\n",
        text,
    )
    payload = json.loads(text)
    segments = []
    for segment in payload.get("segments", []):
        try:
            item = {
                "bg": float(segment["bg"]),
                "ed": float(segment["ed"]),
                "segment_text": str(segment.get("segment_text", "")),
                "small_noise": bool(segment.get("__small_noise__", False)),
            }
        except (KeyError, TypeError, ValueError):
            continue
        segments.append(item)
    return segments


def overlap_ratio(candidate: VadSegment | Sequence[float], marked: Sequence[float]) -> float:
    """Return candidate overlap divided by marked interval duration."""
    start, end = (candidate.start_s, candidate.end_s) if isinstance(candidate, VadSegment) else candidate
    marked_start, marked_end = float(marked[0]), float(marked[1])
    duration = max(0.0, marked_end - marked_start)
    if duration <= 0:
        return 0.0
    return max(0.0, min(float(end), marked_end) - max(float(start), marked_start)) / duration


def interval_hit(candidates: Iterable[VadSegment | Sequence[float]], marked: Sequence[float], threshold: float = 0.5) -> bool:
    """A marked interval is detected when any VAD span covers at least 50%."""
    return any(overlap_ratio(candidate, marked) >= threshold for candidate in candidates)


def _read_reference_audio(path: Path) -> tuple[np.ndarray, int]:
    """读取参考音频，不修改源文件；优先使用 soundfile，兼容 PCM WAV。"""
    try:
        import soundfile as sf

        audio, rate = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return np.asarray(audio, dtype=np.float32), int(rate)
    except Exception:
        with wave.open(str(path), "rb") as stream:
            rate = stream.getframerate()
            frames = stream.readframes(stream.getnframes())
            width = stream.getsampwidth()
            dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width, np.int16)
            audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
            if width == 1:
                audio = (audio - 128.0) / 128.0
            else:
                audio /= float(2 ** (8 * width - 1))
            return audio, int(rate)


def _audio_diagnostics(path: Path, annotations: list[dict]) -> dict:
    marked_rms: list[float] = []
    other_rms: list[float] = []
    marked_peak: list[float] = []
    other_peak: list[float] = []
    details: list[dict] = []
    try:
        audio, rate = _read_reference_audio(path)
        for index, item in enumerate(annotations):
            stats = compute_volume_stats(audio, rate, item["bg"], item["ed"])
            previous = annotations[index - 1] if index else None
            following = annotations[index + 1] if index + 1 < len(annotations) else None
            detail = {
                **item,
                "duration_s": round(stats.duration_s, 6),
                "frame_count": stats.frame_count,
                "rms_dbfs": round(stats.rms_dbfs, 4),
                "peak_dbfs": round(stats.peak_dbfs, 4),
                "noise_floor_dbfs": round(stats.noise_floor_dbfs, 4),
                "snr_db": round(stats.snr_db, 4),
                "low_energy_frame_ratio": round(stats.low_energy_frame_ratio, 4),
                "valid_speech_frame_count": stats.valid_speech_frame_count,
                "context": {
                    "previous_bg": previous.get("bg") if previous else None,
                    "previous_ed": previous.get("ed") if previous else None,
                    "next_bg": following.get("bg") if following else None,
                    "next_ed": following.get("ed") if following else None,
                },
            }
            details.append(detail)
            (marked_rms if item["small_noise"] else other_rms).append(stats.rms_dbfs)
            (marked_peak if item["small_noise"] else other_peak).append(stats.peak_dbfs)
    except (OSError, wave.Error):
        rate = 0
        audio = np.asarray([], dtype=np.float32)
    marked_details = [item for item in details if item["small_noise"]]
    unmarked_details = [item for item in details if not item["small_noise"]]
    return {
        "audio_duration_s": round(len(audio) / rate, 6) if rate else None,
        "sample_rate": rate,
        "marked_intervals": marked_details,
        "unmarked_intervals": unmarked_details,
        "marked_count": len(marked_details),
        "unmarked_count": len(unmarked_details),
        "marked_rms_dbfs": percentile_summary(marked_rms),
        "unmarked_rms_dbfs": percentile_summary(other_rms),
        "marked_peak_dbfs": percentile_summary(marked_peak),
        "unmarked_peak_dbfs": percentile_summary(other_peak),
        "marked_rms_median_dbfs": round(median(marked_rms), 2) if marked_rms else None,
        "unmarked_rms_median_dbfs": round(median(other_rms), 2) if other_rms else None,
        "marked_peak_median_dbfs": round(median(marked_peak), 2) if marked_peak else None,
        "unmarked_peak_median_dbfs": round(median(other_peak), 2) if other_peak else None,
    }


def assign_frame_segments(frame_records: Iterable[dict], segments: Iterable[VadSegment] = ()) -> list[dict]:
    """规范化 10 ms 诊断帧并记录其最终 VAD 段归属。"""
    spans = list(segments)
    normalized = []
    for index, record in enumerate(frame_records):
        start = float(record.get("start_s", record.get("frame_index", index) * 0.01))
        end = float(record.get("end_s", start + 0.01))
        owner = None
        for segment_index, segment in enumerate(spans):
            if min(end, segment.end_s) > max(start, segment.start_s):
                owner = segment_index
                break
        normalized.append({**record, "frame_index": int(record.get("frame_index", index)), "start_s": start, "end_s": end, "segment_index": owner})
    return normalized


def summarize_frame_diagnostics(frame_records: Iterable[dict], segments: Iterable[VadSegment] = ()) -> dict:
    frames = assign_frame_segments(frame_records, segments)
    posterior = [float(item["posterior_delta"]) for item in frames if item.get("posterior_delta") is not None]
    energy = [float(item["log_energy"]) for item in frames if item.get("log_energy") is not None]
    return {
        "frame_duration_ms": 10,
        "frames": frames,
        "frame_count": len(frames),
        "posterior_delta": percentile_summary(posterior),
        "log_energy": percentile_summary(energy),
        "speech_frame_count": sum(str(item.get("state", "")).lower() == "speech" for item in frames),
    }


def collect_vad_frame_diagnostics(frame_records: Iterable[dict], segments: Iterable[VadSegment] = ()) -> dict:
    """诊断入口的统一适配器。

    FunASR 版本之间返回的字段名略有差异，入口接受已采集的模型帧记录，
    统一为 frame_index/start_s/end_s/posterior_delta/log_energy/state，并把
    每帧映射到最终 VAD 段。真实模型钩子和合成测试都使用此契约。
    """
    normalized = []
    for index, record in enumerate(frame_records):
        record = dict(record)
        posterior = record.get("posterior_delta", record.get("score_delta"))
        energy = record.get("log_energy", record.get("decibel"))
        normalized.append({
            "frame_index": int(record.get("frame_index", index)),
            "start_s": float(record.get("start_s", index * 0.01)),
            "end_s": float(record.get("end_s", (index + 1) * 0.01)),
            "posterior_delta": float(posterior) if posterior is not None else None,
            "log_energy": float(energy) if energy is not None else None,
            "state": str(record.get("state", record.get("frame_state", "unknown"))),
        })
    return summarize_frame_diagnostics(normalized, segments)


def generate_threshold_candidates(
    marked_values: Iterable[float],
    unmarked_values: Iterable[float],
    *,
    coarse_step: float = 5.0,
    fine_step: float = 0.5,
    lower: float = -80.0,
    upper: float = 0.0,
) -> dict:
    """由两类响度分位数生成粗扫和交叠边界附近的细扫候选。"""
    marked = np.asarray(list(marked_values), dtype=float)
    unmarked = np.asarray(list(unmarked_values), dtype=float)
    if not marked.size:
        return {"coarse": [], "fine": [], "all": []}
    raw = list(np.percentile(marked, [50, 75, 90, 95]))
    if unmarked.size:
        raw += list(np.percentile(unmarked, [1, 5, 10, 25]))
    coarse = sorted({round(float(np.clip(value, lower, upper)), 4) for value in raw})
    if coarse_step > 0:
        coarse = sorted({round(round(value / coarse_step) * coarse_step, 4) for value in coarse})
    boundary_low = max(lower, min(float(np.min(marked)), float(np.min(unmarked)) if unmarked.size else float(np.min(marked))))
    boundary_high = min(upper, max(float(np.max(marked)), float(np.max(unmarked)) if unmarked.size else float(np.max(marked))))
    fine = []
    if fine_step > 0 and boundary_high >= boundary_low:
        fine = [round(float(value), 4) for value in np.arange(boundary_low, boundary_high + fine_step / 2, fine_step)]
    all_values = sorted(set(coarse) | set(fine))
    return {"coarse": coarse, "fine": fine, "all": all_values, "step": fine_step}


def evaluate_threshold(
    threshold: float,
    marked_intervals: Sequence[dict],
    unmarked_intervals: Sequence[dict],
    candidates: Iterable[VadSegment] = (),
) -> dict:
    """按段级响度和最终 VAD 区间同时评价一个候选。"""
    marked = list(marked_intervals)
    unmarked = list(unmarked_intervals)
    marked_suppressed = sum(float(item.get("rms_dbfs", 0.0)) < threshold for item in marked)
    unmarked_retained = sum(float(item.get("rms_dbfs", 0.0)) >= threshold for item in unmarked)
    segments = list(candidates)
    hits = sum(interval_hit(segments, (item["bg"], item["ed"])) for item in marked)
    metrics = {
        "rms_threshold_dbfs": float(threshold),
        "marked_suppression_rate": round(marked_suppressed / len(marked), 4) if marked else 0.0,
        "unmarked_retention_rate": round(unmarked_retained / len(unmarked), 4) if unmarked else 0.0,
        "marked_frame_suppression_rate": round(marked_suppressed / len(marked), 4) if marked else 0.0,
        "marked_segment_hit_rate": round(hits / len(marked), 4) if marked else 0.0,
        "vad_segment_count": len(segments),
        "qwen_request_count": len(segments),
        "duration_filter_used": False,
    }
    metrics["passes_targets"] = (
        metrics["marked_suppression_rate"] >= 0.95
        and metrics["unmarked_retention_rate"] >= 0.95
        and metrics["marked_segment_hit_rate"] == 0.0
    )
    return metrics


def evaluate_sound_rule_candidates(
    records: Sequence[dict],
    *,
    rms_thresholds: Iterable[float],
    snr_thresholds: Iterable[float] = (),
    low_energy_ratio_thresholds: Iterable[float] = (),
    continuous_valid_frame_thresholds: Iterable[int] = (),
    target_suppression_rate: float = 0.95,
    target_retention_rate: float = 0.95,
) -> dict:
    """评价声音组合规则；没有同时满足目标时明确返回无解。"""
    marked = [item for item in records if item.get("small_noise")]
    unmarked = [item for item in records if not item.get("small_noise")]
    snr_values = list(snr_thresholds) or [-100.0]
    ratio_values = list(low_energy_ratio_thresholds) or [1.0]
    continuous_values = list(continuous_valid_frame_thresholds) or [0]
    results = []
    for rms in rms_thresholds:
        for snr in snr_values:
            for ratio in ratio_values:
                for continuous in continuous_values:
                    def rejected(item):
                        return (
                            float(item.get("rms_dbfs", -120)) < rms
                            or (snr > -100 and float(item.get("snr_db", -120)) < snr)
                            or (ratio < 1.0 and float(item.get("low_energy_frame_ratio", 0)) >= ratio)
                            or (continuous > 0 and int(item.get("valid_speech_frame_count", 0)) < continuous)
                        )
                    suppressed = sum(rejected(item) for item in marked)
                    retained = sum(not rejected(item) for item in unmarked)
                    result = {
                        "rms_threshold_dbfs": float(rms),
                        "snr_threshold_db": float(snr),
                        "low_energy_frame_ratio_threshold": float(ratio),
                        "continuous_valid_frame_threshold": int(continuous),
                        "marked_suppression_rate": round(suppressed / len(marked), 4) if marked else 0.0,
                        "unmarked_retention_rate": round(retained / len(unmarked), 4) if unmarked else 0.0,
                        "duration_filter_used": False,
                    }
                    result["passes_targets"] = (
                        result["marked_suppression_rate"] >= target_suppression_rate
                        and result["unmarked_retention_rate"] >= target_retention_rate
                    )
                    results.append(result)
    feasible = [item for item in results if item["passes_targets"]]
    feasible.sort(key=lambda item: (-item["unmarked_retention_rate"], item["rms_threshold_dbfs"]))
    return {
        "status": "feasible" if feasible else "无可行声音组合规则",
        "candidates": results,
        "recommended": feasible[0] if feasible else None,
        "next_step": None if feasible else "继续细扫 RMS/SNR/低能量帧比例；仍无解时才评估时长兜底",
    }


def evaluate_duration_fallback(
    records: Sequence[dict],
    duration_thresholds: Iterable[float],
    *,
    sound_rule_result: dict,
) -> dict:
    """仅在声音组合规则无解时计算时长兜底，输出独立误伤指标。"""
    if sound_rule_result.get("recommended") is not None:
        return {"status": "blocked", "reason": "声音规则已有可行解", "duration_filter_used": False}
    marked = [item for item in records if item.get("small_noise")]
    unmarked = [item for item in records if not item.get("small_noise")]
    results = []
    for threshold in duration_thresholds:
        def duration(item):
            if item.get("duration_s") is not None:
                return float(item["duration_s"])
            return float(item.get("ed", 0.0)) - float(item.get("bg", 0.0))

        rejected_marked = sum(duration(item) < threshold for item in marked)
        rejected_unmarked = sum(duration(item) < threshold for item in unmarked)
        results.append({
            "duration_threshold_s": float(threshold),
            "marked_suppression_rate": rejected_marked / len(marked) if marked else 0.0,
            "unmarked_false_drop_rate": rejected_unmarked / len(unmarked) if unmarked else 0.0,
            "duration_filter_used": True,
        })
    return {"status": "evaluated", "candidates": results, "duration_filter_used": True}


def recognized_content_for_intervals(annotation_items: Sequence[dict], response_segments: Sequence[dict]) -> list[dict]:
    """把最终响应中与标注区间重叠的文本归档，并区分口气词/有效词。"""
    output = []
    for item in annotation_items:
        overlaps = []
        for segment in response_segments:
            try:
                start, end = float(segment["bg"]), float(segment["ed"])
            except (KeyError, TypeError, ValueError):
                continue
            if min(end, item["ed"]) > max(start, item["bg"]):
                overlaps.append({"bg": start, "ed": end, "segment_text": str(segment.get("segment_text", ""))})
        effective = [entry for entry in overlaps if entry["segment_text"].strip()]
        output.append({
            "bg": item["bg"],
            "ed": item["ed"],
            "small_noise": bool(item.get("small_noise")),
            "segments": overlaps,
            "classification": "no_content" if not effective else (
                "interjection_only" if all(len(entry["segment_text"].strip()) <= 4 for entry in effective) else "valid_words"
            ),
        })
    return output


async def scan_thresholds(
    audio_path: str | Path = DEFAULT_AUDIO,
    annotation_path: str | Path = DEFAULT_ANNOTATION,
    thresholds: Iterable[float] = (0.3, 0.6, 0.8),
    pool_factory=VadPool,
) -> dict:
    """Run the optional long test and return a JSON-serializable report."""
    annotations = parse_reference_annotations(annotation_path)
    marked = [item for item in annotations if item["small_noise"]]
    unmarked = [item for item in annotations if not item["small_noise"]]
    report = {
        "schema_version": "vad-low-volume-calibration.v1",
        "audio": str(audio_path),
        "annotation": str(annotation_path),
        "thresholds": [],
        "vad": {"segment_count": None, "marked_interval_hits": None, "qwen_request_count": None},
        "recognized_content": [],
    }
    report["annotation_count"] = len(marked)
    report["diagnostics"] = _audio_diagnostics(Path(audio_path), annotations)
    marked_values = [item.get("rms_dbfs", -100.0) for item in report["diagnostics"]["marked_intervals"]]
    unmarked_values = [item.get("rms_dbfs", -100.0) for item in report["diagnostics"]["unmarked_intervals"]]
    report["candidate_search"] = generate_threshold_candidates(marked_values, unmarked_values)
    for threshold in thresholds:
        started = time.perf_counter()
        pool = pool_factory(vad_kwargs={
            "max_single_segment_time": settings.qwen_vad_max_single_segment_time_ms,
            "max_end_silence_time": settings.qwen_vad_max_end_silence_time_ms,
            "speech_noise_thres": float(threshold),
        })
        try:
            raw = await pool.detect(str(audio_path))
            candidates = _parse_vad_result(raw)
        finally:
            await pool.stop()
        marked_hits = sum(interval_hit(candidates, (item["bg"], item["ed"])) for item in marked)
        unmarked_hits = sum(interval_hit(candidates, (item["bg"], item["ed"])) for item in unmarked)
        report["thresholds"].append({
            "threshold": float(threshold),
            "marked_hit_rate": round(marked_hits / len(marked), 4) if marked else 0.0,
            "marked_suppression_rate": round(1 - marked_hits / len(marked), 4) if marked else 0.0,
            "unmarked_retention_rate": round(unmarked_hits / len(unmarked), 4) if unmarked else 0.0,
            "unmarked_false_positive_rate": round(unmarked_hits / len(unmarked), 4) if unmarked else 0.0,
            "filtered_segment_count": len(candidates),
            "qwen_request_count": len(candidates),
            "elapsed_s": round(time.perf_counter() - started, 3),
            "duration_filter_used": False,
        })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="扫描细小声音参考音频的 VAD 阈值")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.3, 0.6, 0.8])
    parser.add_argument("--target-suppression-rate", type=float, default=0.95)
    parser.add_argument("--target-retention-rate", type=float, default=0.95)
    parser.add_argument("--coarse-step", type=float, default=5.0)
    parser.add_argument("--fine-step", type=float, default=0.5)
    parser.add_argument("--rms-lower", type=float, default=-80.0)
    parser.add_argument("--rms-upper", type=float, default=0.0)
    parser.add_argument("--response-json", type=Path, help="可选：最终接口响应 JSON，用于标注区间识别内容归档")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = asyncio.run(scan_thresholds(args.audio, args.annotation, args.thresholds))
    report["targets"] = {
        "marked_suppression_rate": args.target_suppression_rate,
        "unmarked_retention_rate": args.target_retention_rate,
    }
    report["candidate_search"]["coarse_step"] = args.coarse_step
    report["candidate_search"]["fine_step"] = args.fine_step
    report["candidate_search"]["rms_range_dbfs"] = [args.rms_lower, args.rms_upper]
    if args.response_json:
        response = json.loads(args.response_json.read_text(encoding="utf-8"))
        response_segments = response.get("segments", []) if isinstance(response, dict) else []
        report["recognized_content"] = recognized_content_for_intervals(
            parse_reference_annotations(args.annotation), response_segments
        )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
