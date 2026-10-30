"""声音特征与低音量 VAD 段过滤。

FunASR 的 log 能量阈值和用户常用的 dBFS 不是同一单位。本模块把段级
音频特征集中计算，供运行时过滤和参考音频标定共用，避免两套统计口径。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np


def _as_mono_array(audio) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=0) if array.shape[0] <= 8 else array.mean(axis=-1)
    return array.reshape(-1)


def dbfs(value: float) -> float:
    return float(20.0 * np.log10(max(abs(float(value)), 1e-12)))


@dataclass(frozen=True)
class VolumeStats:
    start_s: float
    end_s: float
    duration_s: float
    rms_dbfs: float
    peak_dbfs: float
    noise_floor_dbfs: float
    snr_db: float
    frame_count: int
    low_energy_frame_ratio: float
    valid_speech_frame_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_volume_stats(
    audio,
    sample_rate: int,
    start_s: float = 0.0,
    end_s: float | None = None,
    *,
    frame_ms: int = 10,
    low_energy_dbfs_thres: float = -40.0,
) -> VolumeStats:
    """计算一个区间的 RMS/峰值、估计 SNR 和 10 ms 帧统计。"""
    array = _as_mono_array(audio)
    rate = max(1, int(sample_rate))
    start = max(0, int(round(float(start_s) * rate)))
    stop = len(array) if end_s is None else min(len(array), int(round(float(end_s) * rate)))
    chunk = array[start:max(start, stop)]
    duration = len(chunk) / rate
    if len(chunk) == 0:
        return VolumeStats(float(start_s), float(end_s or start_s), duration, -120.0, -120.0, -120.0, 0.0, 0, 1.0, 0)

    frame_size = max(1, int(round(rate * frame_ms / 1000.0)))
    frame_count = int(np.ceil(len(chunk) / frame_size))
    padded_size = frame_count * frame_size
    if padded_size != len(chunk):
        padded = np.pad(chunk, (0, padded_size - len(chunk)))
    else:
        padded = chunk
    frames = padded.reshape(frame_count, frame_size)
    # 这里原来是“每个 10ms frame 一个 Python/NumPy 调用”。长音频有数十万
    # 个 frame，多个 VAD 请求并行时会把 CPU 时间放大到分钟级；一次 reshape
    # 后沿 frame 维度计算，结果口径保持一致但不再逐帧调用 Python。
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1, dtype=np.float32))
    frame_dbfs = 20.0 * np.log10(np.maximum(frame_rms, 1e-12))
    rms = float(np.sqrt(np.mean(chunk * chunk)))
    peak = float(np.max(np.abs(chunk)))
    noise_floor = float(np.percentile(frame_dbfs, 10))
    segment_dbfs = dbfs(rms)
    snr = segment_dbfs - noise_floor
    low_ratio = float(np.mean(frame_dbfs < float(low_energy_dbfs_thres)))
    valid = int(np.count_nonzero(frame_dbfs >= float(low_energy_dbfs_thres)))
    return VolumeStats(
        start_s=float(start_s),
        end_s=float(end_s if end_s is not None else start_s + duration),
        duration_s=duration,
        rms_dbfs=segment_dbfs,
        peak_dbfs=dbfs(peak),
        noise_floor_dbfs=noise_floor,
        snr_db=snr,
        frame_count=len(frames),
        low_energy_frame_ratio=low_ratio,
        valid_speech_frame_count=valid,
    )


def low_volume_reason(
    stats: VolumeStats,
    *,
    rms_dbfs_thres: float = -100.0,
    snr_thres: float = -100.0,
) -> str | None:
    """返回低音量丢弃原因；小于等于 -100 的门限表示该规则关闭。"""
    reasons = []
    if float(rms_dbfs_thres) > -100.0 and stats.rms_dbfs < float(rms_dbfs_thres):
        reasons.append("rms_dbfs")
    if float(snr_thres) > -100.0 and stats.snr_db < float(snr_thres):
        reasons.append("snr")
    return "low_volume:" + ",".join(reasons) if reasons else None


def filter_low_volume_segments(
    segments: Sequence,
    audio,
    sample_rate: int,
    *,
    enabled: bool,
    rms_dbfs_thres: float = -100.0,
    snr_thres: float = -100.0,
    low_energy_dbfs_thres: float = -100.0,
) -> tuple[list, list[tuple[object, VolumeStats, str]]]:
    """按段级声音特征过滤，返回 (保留段, [(丢弃段, 特征, 原因)])。"""
    if not enabled:
        return list(segments), []
    kept = []
    dropped = []
    energy_threshold = (
        float(low_energy_dbfs_thres)
        if float(low_energy_dbfs_thres) > -100.0
        else float(rms_dbfs_thres)
    )
    # 音频张量到 numpy/单声道的转换只做一次；此前每个分段都会重复执行。
    audio_array = _as_mono_array(audio)
    for segment in segments:
        stats = compute_volume_stats(
            audio_array,
            sample_rate,
            segment.start_s,
            segment.end_s,
            low_energy_dbfs_thres=energy_threshold,
        )
        reason = low_volume_reason(
            stats,
            rms_dbfs_thres=rms_dbfs_thres,
            snr_thres=snr_thres,
        )
        if reason:
            dropped.append((segment, stats, reason))
        else:
            kept.append(segment)
    return kept, dropped


def percentile_summary(values: Iterable[float]) -> dict[str, float | None]:
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        return {key: None for key in ("p10", "p50", "p90", "p99")}
    return {
        key: round(float(np.percentile(values, percentile)), 4)
        for key, percentile in (("p10", 10), ("p50", 50), ("p90", 90), ("p99", 99))
    }


__all__ = [
    "VolumeStats",
    "compute_volume_stats",
    "filter_low_volume_segments",
    "low_volume_reason",
    "percentile_summary",
]
