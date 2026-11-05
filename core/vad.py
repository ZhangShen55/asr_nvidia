"""v3 链路的本地 VAD 切分：进程池客户端 + 分段解析。

FSMN-VAD 实例与 worker 进程的生命周期在 :mod:`core.vad_pool`，本模块只保留
分段结果的解析、过滤与日志，是纯数据处理层。
"""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import List

import numpy as np

from core.config import settings
from core.vad_pool import get_vad_pool
from core.vad_volume import filter_low_volume_segments

logger = logging.getLogger(__name__)


# 每个请求保存最近一次 VAD 的段统计，供 V3 总耗时日志关联。
# ContextVar 保证并发请求之间不会互相覆盖。
_VAD_STATS: ContextVar[dict] = ContextVar("vad_stats", default={})


@dataclass(frozen=True)
class VadSegment:
    """一个语音段的起止时间（秒）。"""

    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def _parse_vad_result(result) -> List[VadSegment]:
    """从 FunASR VAD 输出中解析出毫秒区间列表。"""
    if not result:
        return []
    payload = result[0]
    if isinstance(payload, dict):
        spans = payload.get("value") or []
    elif isinstance(payload, list) and payload and isinstance(payload[0], list):
        spans = payload[0]
    else:
        spans = []

    segments: List[VadSegment] = []
    for span in spans:
        try:
            start_ms, end_ms = float(span[0]), float(span[1])
        except (TypeError, ValueError, IndexError):
            continue
        if end_ms <= start_ms:
            continue
        segments.append(VadSegment(start_s=start_ms / 1000.0, end_s=end_ms / 1000.0))
    return segments


def filter_short_segments(
    segments: List[VadSegment],
    min_duration_s: float | None = None,
) -> List[VadSegment]:
    """过滤掉时长过短的语音段。"""
    threshold = settings.vad_min_segment_duration_s if min_duration_s is None else min_duration_s
    return [seg for seg in segments if seg.duration_s >= threshold]


def _audio_duration_s(audio, sample_rate: int) -> float:
    """Return the duration of a mono/channel-first audio value in seconds."""
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    array = np.asarray(audio)
    if array.ndim > 1:
        samples = array.shape[-1]
    else:
        samples = array.size
    return float(samples) / float(sample_rate) if sample_rate else 0.0


def _in_filter_scope(segment: VadSegment, duration_s: float, after_start: float, before_end: float) -> bool:
    """Check whether a segment is in the configured beginning/end windows."""
    if after_start < 0 and before_end < 0:
        return True
    if after_start == 0 and before_end == 0:
        return True
    in_start = after_start >= 0 and segment.start_s < after_start * 60.0
    in_end = before_end >= 0 and segment.end_s > max(0.0, duration_s - before_end * 60.0)
    return in_start or in_end


async def detect_segments(
    audio_path: str,
    *,
    audio_tensor=None,
    sample_rate: int = 0,
) -> List[VadSegment]:
    """对音频做 VAD 切分，返回按时间排序的语音段列表（已过滤过短段）。

    音频路径由主进程统一落盘后交给 worker 直接读取，进程间只传路径。
    """
    result = await get_vad_pool().detect(audio_path)
    segments = _parse_vad_result(result)
    filtered = filter_short_segments(segments)
    short_filtered_count = len(segments) - len(filtered)
    dropped = []
    range_candidates = 0
    decibel_dropped = 0
    snr_dropped = 0
    rms_dropped = 0
    volume_filter_started = time.perf_counter()
    decibel_threshold = settings.vad_segment_decibel_thres
    snr_threshold = settings.vad_snr_thres
    rms_threshold = settings.vad_low_volume_rms_dbfs_thres
    decibel_enabled = settings.vad_decibel_thres_filter_enabled and decibel_threshold > -100.0
    snr_enabled = settings.vad_snr_thres_filter_enabled and snr_threshold > -100.0
    rms_enabled = settings.vad_low_volume_filter_enabled and rms_threshold > -100.0
    segment_filters_enabled = decibel_enabled or snr_enabled or rms_enabled
    if segment_filters_enabled and filtered:
        if audio_tensor is None or not sample_rate:
            try:
                from utils.audio_utils import read_audio_array

                audio_tensor, sample_rate = read_audio_array(audio_path)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"VAD 段级过滤无法读取音频: {audio_path}: {exc}") from exc
        duration_s = _audio_duration_s(audio_tensor, sample_rate)
        after_start = settings.vad_after_start
        before_end = settings.vad_before_end
        scoped = []
        out_of_scope = []
        for segment in filtered:
            if _in_filter_scope(segment, duration_s, after_start, before_end):
                scoped.append(segment)
            else:
                out_of_scope.append(segment)
        range_candidates = len(scoped)
        scoped, dropped = filter_low_volume_segments(
            scoped,
            audio_tensor,
            sample_rate,
            enabled=True,
            rms_dbfs_thres=rms_threshold,
            snr_thres=snr_threshold,
            low_energy_dbfs_thres=decibel_threshold,
            frame_sample_ratio=settings.vad_decibel_frame_sample_ratio,
            low_energy_frame_ratio_thres=settings.vad_decibel_low_frame_ratio_thres,
            decibel_enabled=decibel_enabled,
            rms_enabled=rms_enabled,
            snr_enabled=snr_enabled,
        )
        filtered = sorted([*out_of_scope, *scoped], key=lambda item: item.start_s)
        for segment, stats, reason in dropped:
            decibel_dropped += int("decibel" in reason)
            snr_dropped += int("snr" in reason)
            rms_dropped += int("rms_dbfs" in reason)
            logger.info(
                "[VAD] 丢弃段 reason=%s bg=%.3f ed=%.3f rms=%.2f peak=%.2f snr=%.2f low_ratio=%.3f sampled=%d",
                reason,
                segment.start_s,
                segment.end_s,
                stats.rms_dbfs,
                stats.peak_dbfs,
                stats.snr_db,
                stats.low_energy_frame_ratio,
                stats.sampled_frame_count,
            )
    volume_filter_ms = (time.perf_counter() - volume_filter_started) * 1000
    logger.info(
        f"[VAD] 切出 {len(segments)} 段，过滤后 {len(filtered)} 段，"
        f"短段丢弃 {short_filtered_count} 段，范围内候选 {range_candidates} 段，"
        f"段级丢弃 {len(dropped)} 段(decibel={decibel_dropped},snr={snr_dropped},rms={rms_dropped})，"
        f"文件：{audio_path}"
    )
    _VAD_STATS.set({
        "raw_segments": len(segments),
        "short_filtered": short_filtered_count,
        "low_volume_dropped": len(dropped),
        "range_candidates": range_candidates,
        "decibel_dropped": decibel_dropped,
        "snr_dropped": snr_dropped,
        "rms_dropped": rms_dropped,
        "kept_segments": len(filtered),
        "low_volume_filter_ms": volume_filter_ms,
    })
    if segment_filters_enabled:
        logger.info(
            "[VAD] 段级过滤耗时=%.0fms，候选段=%d，丢弃=%d",
            volume_filter_ms,
            range_candidates,
            len(dropped),
        )
    return filtered


def get_vad_stats() -> dict:
    """返回当前请求最近一次 VAD 统计，供诊断日志使用。"""
    return dict(_VAD_STATS.get())


__all__ = [
    "VadSegment",
    "detect_segments",
    "filter_short_segments",
    "get_vad_stats",
]
