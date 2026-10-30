"""v2 链路的本地 VAD 切分：进程池客户端 + 分段解析。

FSMN-VAD 实例与 worker 进程的生命周期在 :mod:`core.vad_pool`，本模块只保留
分段结果的解析、过滤与日志，是纯数据处理层。
"""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import List

from core.config import settings
from core.vad_pool import get_vad_pool
from core.vad_volume import filter_low_volume_segments

logger = logging.getLogger(__name__)


# 每个请求保存最近一次 VAD 的段统计，供 V2 总耗时日志关联。
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
    threshold = settings.qwen_min_segment_duration_s if min_duration_s is None else min_duration_s
    return [seg for seg in segments if seg.duration_s >= threshold]


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
    volume_filter_started = time.perf_counter()
    if settings.qwen_vad_low_volume_filter_enabled and audio_tensor is not None and sample_rate:
        filtered, dropped = filter_low_volume_segments(
            filtered,
            audio_tensor,
            sample_rate,
            enabled=True,
            rms_dbfs_thres=settings.qwen_vad_low_volume_rms_dbfs_thres,
            snr_thres=settings.qwen_vad_snr_thres,
        )
        for segment, stats, reason in dropped:
            logger.info(
                "[VAD] 丢弃低音量段 reason=%s bg=%.3f ed=%.3f rms=%.2f peak=%.2f snr=%.2f",
                reason,
                segment.start_s,
                segment.end_s,
                stats.rms_dbfs,
                stats.peak_dbfs,
                stats.snr_db,
            )
    volume_filter_ms = (time.perf_counter() - volume_filter_started) * 1000
    logger.info(
        f"[VAD] 切出 {len(segments)} 段，过滤后 {len(filtered)} 段，"
        f"低音量丢弃 {len(dropped)} 段，"
        f"文件：{audio_path}"
    )
    _VAD_STATS.set({
        "raw_segments": len(segments),
        "short_filtered": short_filtered_count,
        "low_volume_dropped": len(dropped),
        "kept_segments": len(filtered),
        "low_volume_filter_ms": volume_filter_ms,
    })
    if settings.qwen_vad_low_volume_filter_enabled:
        logger.info(
            "[VAD] 低音量过滤耗时=%.0fms，候选段=%d，丢弃=%d",
            volume_filter_ms,
            len(filter_short_segments(segments)),
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
