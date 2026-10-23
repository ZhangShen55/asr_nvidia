"""v2 链路的本地 VAD 切分。"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import List

from core.config import settings
from core.models import _resolve_device, device_allocated_mb, record_model_memory

logger = logging.getLogger(__name__)

# FSMN-VAD 模型实例内部带状态缓存，并发调用同一实例会导致数据错乱，
# 因此用实例池隔离：每个实例同一时刻只服务一个请求。
_vad_instances: List = []
_vad_busy: List[bool] = []
_vad_pool_lock = asyncio.Lock()
_vad_semaphore = asyncio.Semaphore(max(1, settings.qwen_vad_concurrency))


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


async def _acquire_vad_model():
    """从实例池取一个空闲的 VAD 模型实例。"""
    await _vad_semaphore.acquire()
    async with _vad_pool_lock:
        for index, busy in enumerate(_vad_busy):
            if not busy:
                _vad_busy[index] = True
                return index, _vad_instances[index]
        model = _create_vad_model()
        _vad_instances.append(model)
        _vad_busy.append(True)
        return len(_vad_instances) - 1, model


async def _release_vad_model(index: int) -> None:
    async with _vad_pool_lock:
        if 0 <= index < len(_vad_busy):
            _vad_busy[index] = False
    _vad_semaphore.release()


def _create_vad_model():
    """新建一个 FSMN-VAD 模型实例。"""
    from funasr import AutoModel

    device = _resolve_device(settings.model_device_vad)
    before = device_allocated_mb(device)
    model = AutoModel(
        model=settings.vad_model_dir,
        device=device,
        vad_kwargs={
            "max_single_segment_time": settings.qwen_vad_max_single_segment_time_ms,
            "max_end_silence_time": settings.qwen_vad_max_end_silence_time_ms,
        },
        disable_update=True,
        disable_pbar=True,
    )
    allocated = device_allocated_mb(device) - before
    record_model_memory("vad", device, allocated, instance_added=True)
    logger.info(f"[模型] FSMN-VAD 实例已加载到 {device}，显存占用约 {allocated:.1f} MB")
    return model


async def detect_segments(audio_path: str) -> List[VadSegment]:
    """对音频做 VAD 切分，返回按时间排序的语音段列表（已过滤过短段）。"""
    index, model = await _acquire_vad_model()
    try:
        result = await asyncio.to_thread(
            model.generate,
            input=audio_path,
            # funasr 的 FsmnVADStreaming.inference 使用可变默认参数 cache={}，
            # 并发调用会共享同一份内部状态，必须显式传入独立的 cache。
            cache={},
            vad_kwargs={
                "max_single_segment_time": settings.qwen_vad_max_single_segment_time_ms,
                "max_end_silence_time": settings.qwen_vad_max_end_silence_time_ms,
            },
        )
    finally:
        await _release_vad_model(index)
    segments = _parse_vad_result(result)
    segments.sort(key=lambda seg: seg.start_s)
    filtered = filter_short_segments(segments)
    logger.info(
        f"[VAD] 切出 {len(segments)} 段，过滤后 {len(filtered)} 段，"
        f"文件：{audio_path}"
    )
    return filtered


__all__ = [
    "VadSegment",
    "detect_segments",
    "filter_short_segments",
]
