"""有界的请求级阶段遥测。

遥测只保存最近一批请求的轻量统计，不保存音频、转写正文或模型对象。写入由
线程锁保护，状态接口只做快照复制，避免诊断采样阻塞 ASR 主流程。
"""

from __future__ import annotations

import threading
import time
import uuid
from copy import deepcopy
from collections import deque
from datetime import datetime, timezone
from typing import Any, Mapping


_MAX_EVENTS = 4096
_LOCK = threading.RLock()
_EVENTS: deque[dict[str, Any]] = deque(maxlen=_MAX_EVENTS)
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
    ended = time.perf_counter() if ended_at is None else ended_at
    phase_values = {
        str(key): round(float(value), 3)
        for key, value in (phases or {}).items()
        if isinstance(value, (int, float))
    }
    event = {
        "request_id": request_id,
        "task_id": task_id,
        "filename": filename,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "duration_ms": round(max(0.0, ended - started_at) * 1000, 3),
        "status": status,
        "http_status": http_status,
        "code": code,
        "retry_count": int(retry_count),
        "valid_segments": valid_segments,
        "phases": phase_values,
        "error": error,
    }
    if admission is not None:
        event["admission"] = _admission_values(admission)
    with _LOCK:
        _EVENTS.append(event)
    return deepcopy(event)


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
                "cluster_ms", "itn_ms", "cleanup_ms",
            )
        },
        "last_error": failed[-1].get("error") if failed else None,
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
        result["recent"] = events[-20:]
    return result


def reset_for_tests() -> None:
    with _LOCK:
        _EVENTS.clear()


__all__ = ["new_request_id", "record_request", "reset_for_tests", "snapshot"]
