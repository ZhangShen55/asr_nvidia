"""有界的请求级阶段遥测。

遥测只保存最近一批请求的轻量统计，不保存音频、转写正文或模型对象。写入由
线程锁保护，状态接口只做快照复制，避免诊断采样阻塞 ASR 主流程。
"""

from __future__ import annotations

import threading
import time
import uuid
import math
from copy import deepcopy
from collections import deque
from datetime import datetime, timezone
from typing import Any, Mapping
from contextvars import ContextVar


_MAX_EVENTS = 4096
_LOCK = threading.RLock()
_EVENTS: deque[dict[str, Any]] = deque(maxlen=_MAX_EVENTS)
_ACTIVE: dict[str, dict[str, Any]] = {}
_REQUEST_ID: ContextVar[str | None] = ContextVar("active_asr_request", default=None)
_ENHANCEMENT_PHASES: ContextVar[dict | None] = ContextVar("asr_enhancement_phases", default=None)
# 仅允许固定模型/事件维度，不能把段ID、模型对象或正文放进请求统计。
_ENHANCEMENT_COUNTS = ("nn_calls", "nn_success", "nn_empty", "nn_failed", "nn_cancelled", "duration_skips", "load_failures")
_ENHANCEMENT_TIMES = ("load_ms", "load_work_ms", "load_dispatch_wait_ms", "load_lock_wait_ms")
_ENHANCEMENT_PHASE_KEYS = frozenset(
    f"{prefix}_{key}" for prefix in ("spk", "emo")
    for key in _ENHANCEMENT_COUNTS + _ENHANCEMENT_TIMES
) | {"model_load_lock_wait_ms"}
_RECENT_PHASE_KEYS = _ENHANCEMENT_PHASE_KEYS | {
    "audio_wait_ms", "vad_ms", "vad_wait_ms", "vad_worker_ms", "low_volume_filter_ms",
    "cut_sum_ms", "cut_wall_ms", "cut_cpu_ms", "cut_dispatch_wait_ms", "pending_wait_ms",
    "segment_ms", "qwen_wait_ms", "qwen_ms", "qwen_calls", "spk_wait_ms", "spk_ms", "spk_calls",
    "emo_wait_ms", "emo_ms", "emo_calls", "cluster_ms", "itn_ms", "cleanup_ms",
}
_MAX_ACTIVE = 256
_ACTIVE_STAGES = {"admission", "decode", "model_loading", "vad", "segments", "cluster", "itn", "cleanup"}
_PROGRESS_KEYS = {"segments_total", "segments_done", "qwen_done", "spk_done", "emo_done", "cut_done"}


def begin_request(request_id: str, started_at: float) -> None:
    _REQUEST_ID.set(request_id)
    # 子分支继承同一个有界字典；end_request后的finally仍需把统计写入完成事件。
    _ENHANCEMENT_PHASES.set({"request_id": request_id, "phases": {}})
    now = time.perf_counter()
    with _LOCK:
        # 正常准入上限远小于此值；即使遭遇大量并发拒绝也不创建无界诊断记录。
        if len(_ACTIVE) >= _MAX_ACTIVE:
            oldest = next(iter(_ACTIVE))
            _ACTIVE.pop(oldest, None)
        _ACTIVE[request_id] = {"request_id": request_id, "received_utc": datetime.now(timezone.utc).isoformat(),
                               "started_monotonic": started_at, "updated_monotonic": now,
                               "stage": "admission", "progress": {}}


def update_active(stage: str | None = None, *, increment: str | None = None, **progress) -> None:
    request_id = _REQUEST_ID.get()
    if request_id is None:
        return
    with _LOCK:
        entry = _ACTIVE.get(request_id)
        if entry is None:
            return
        if stage in _ACTIVE_STAGES:
            entry["stage"] = stage
        if increment in _PROGRESS_KEYS:
            entry["progress"][increment] = entry["progress"].get(increment, 0) + 1
        for key, value in progress.items():
            if key in _PROGRESS_KEYS and type(value) is int and value >= 0:
                entry["progress"][key] = value
        entry["updated_monotonic"] = time.perf_counter()


def end_request(request_id: str) -> None:
    with _LOCK:
        _ACTIVE.pop(request_id, None)


def record_enhancement_phase(key: str, value: float) -> None:
    """累加有限增强指标；此处不采样设备、不触发模型或保存逐段历史。"""
    if key not in _ENHANCEMENT_PHASE_KEYS or type(value) not in (int, float):
        return
    if not math.isfinite(value) or value < 0:
        return
    current = _ENHANCEMENT_PHASES.get()
    if current is None or current["request_id"] != _REQUEST_ID.get():
        return
    with _LOCK:
        phases = current["phases"]
        phases[key] = phases.get(key, 0.0) + value


def enhancement_phases() -> dict[str, float]:
    current = _ENHANCEMENT_PHASES.get()
    if current is None:
        return {}
    with _LOCK:
        return dict(current["phases"])


def active_snapshot() -> dict:
    now = time.perf_counter()
    with _LOCK:
        active = deepcopy(list(_ACTIVE.values()))
    for entry in active:
        entry["age_s"] = max(0.0, now - entry["started_monotonic"])
        entry["no_progress_s"] = max(0.0, now - entry["updated_monotonic"])
    return {"limit": _MAX_ACTIVE, "count": len(active), "requests": active}
_ADMISSION_REASONS = (
    "total_full", "spk_emo_full", "waiting_full", "wait_timeout",
    "role_disabled", "emotion_disabled", "both_disabled", "cancelled", "qwen_disabled",
)
_ADMISSION_OUTCOMES = ("admitted", "rejected", "timeout", "cancelled", "not_attempted")


def _admission_values(admission: Mapping[str, Any]) -> dict[str, Any]:
    """只保留有限分类与容量数字，不接受请求正文或任意维度键。"""
    capacity = {}
    source = admission.get("snapshot") or {}
    for group, keys in (
        ("audioConcurrency", ("configured_total", "active_total", "waiting_total")),
        ("spkEmoConcurrency", ("configured", "active", "waiting")),
    ):
        values = source.get(group, {})
        capacity[group] = {key: values[key] for key in keys if type(values.get(key)) is int}
    return {
        "category": admission.get("category") if admission.get("category") in ("A", "B") else None,
        "outcome": admission.get("outcome") if admission.get("outcome") in _ADMISSION_OUTCOMES else None,
        "reason": admission.get("reason") if admission.get("reason") in _ADMISSION_REASONS else None,
        "snapshot": capacity,
    }


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 3)


def new_request_id() -> str:
    return str(uuid.uuid4())


def _phase_summary(events: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(event.get("phases", {}).get(key, 0.0)) for event in events]
    values = [value for value in values if value >= 0]
    return {
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "max_ms": round(max(values), 3) if values else None,
    }


def record_request(
    *,
    request_id: str,
    task_id: str | None,
    filename: str | None,
    started_at: float,
    ended_at: float | None = None,
    status: str = "completed",
    http_status: int | None = 200,
    code: int | None = None,
    retry_count: int = 0,
    valid_segments: int | None = None,
    phases: Mapping[str, Any] | None = None,
    error: str | None = None,
    admission: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from telemetry_runtime import record_request as _record_request

    phase_values = dict(phases or {})
    current = _ENHANCEMENT_PHASES.get()
    if current is not None and current["request_id"] == request_id:
        phase_values.update(enhancement_phases())
    return _record_request(
        _EVENTS, _LOCK,
        request_id=request_id, task_id=task_id, filename=filename,
        started_at=started_at, ended_at=ended_at, status=status,
        http_status=http_status, code=code, retry_count=retry_count,
        valid_segments=valid_segments, phases=phase_values, error=error,
        admission=admission,
    )


def snapshot(*, include_events: bool = True) -> dict[str, Any]:
    with _LOCK:
        events = deepcopy(list(_EVENTS))
    completed = [event for event in events if event.get("status") == "completed"]
    failed = [event for event in events if event.get("status") != "completed"]
    duration_values = [float(event.get("duration_ms", 0.0)) for event in events]
    result: dict[str, Any] = {
        "available": True,
        "buffer_limit": _MAX_EVENTS,
        "events": len(events),
        "active": active_snapshot(),
        "completed": len(completed),
        "failed": len(failed),
        "duration_ms": {
            "p50_ms": _percentile(duration_values, 0.50),
            "p95_ms": _percentile(duration_values, 0.95),
            "max_ms": round(max(duration_values), 3) if duration_values else None,
        },
        "phases_ms": {
            key: _phase_summary(events, key)
            for key in (
                "audio_wait_ms", "vad_ms", "cut_sum_ms", "cut_wall_ms", "segment_ms", "qwen_wait_ms",
                "qwen_ms", "spk_wait_ms", "spk_ms", "emo_wait_ms", "emo_ms",
                "cluster_ms", "itn_ms", "cleanup_ms", "cut_cpu_ms", "cut_dispatch_wait_ms", "pending_wait_ms",
                "vad_wait_ms", "vad_worker_ms", "low_volume_filter_ms",
            )
        },
        "last_error": failed[len(failed) - 1].get("error") if failed else None,
        "enhancements": {
            prefix: {
                "requests_observed": sum(any(key.startswith(prefix + "_") and key in _ENHANCEMENT_PHASE_KEYS
                                             for key in event.get("phases", {})) for event in events),
                "counts": {key: sum(event.get("phases", {}).get(f"{prefix}_{key}", 0)
                                    for event in events) for key in _ENHANCEMENT_COUNTS},
                "timings_ms": {
                    key: _phase_summary([event for event in events
                                         if f"{prefix}_{key}" in event.get("phases", {})], f"{prefix}_{key}")
                    for key in _ENHANCEMENT_TIMES
                },
            }
            for prefix in ("spk", "emo")
        },
        # 独立于完整recent事件：最多8条、固定数值维度，足以关联A6/B4且不回传正文。
        "recent_stages": [
            {"request_id": event["request_id"][:128] if isinstance(event.get("request_id"), str) else None,
             "filename": event["filename"][:512] if isinstance(event.get("filename"), str) else None,
             "started_monotonic": event.get("started_monotonic"),
             "ended_monotonic": event.get("ended_monotonic"), "status": event.get("status"),
             "phases": {key: value for key, value in event.get("phases", {}).items()
                        if key in _RECENT_PHASE_KEYS and type(value) in (int, float) and math.isfinite(value)}}
            for event in events[-8:]
        ],
        "admission": {
            "window": "retained_events",
            "by_class": {
                category: sum(event.get("admission", {}).get("category") == category for event in events)
                for category in ("A", "B")
            },
            "rejections_by_reason": {
                reason: sum(
                    event.get("admission", {}).get("reason") == reason
                    and event.get("admission", {}).get("outcome") in ("rejected", "timeout", "cancelled")
                    for event in events
                )
                for reason in _ADMISSION_REASONS
            },
        },
    }
    if include_events:
        result["recent"] = events[max(0, len(events) - 20):]
    return result


def reset_for_tests() -> None:
    with _LOCK:
        _EVENTS.clear()
        _ACTIVE.clear()
    _ENHANCEMENT_PHASES.set(None)


__all__ = ["new_request_id", "record_request", "reset_for_tests", "snapshot", "begin_request",
           "update_active", "end_request", "active_snapshot", "record_enhancement_phase", "enhancement_phases"]
