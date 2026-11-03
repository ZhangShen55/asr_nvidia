"""纯 Python 请求遥测写入 helper。

复杂的 Mapping/Union 参数来自 FastAPI 请求上下文；把事件归一化和 deque
写入放在 Python helper 中，避免 Cython 扩展在请求完成时处理嵌套临时对象。
"""

import time
from copy import deepcopy
from datetime import datetime, timezone


_OUTCOMES = ("admitted", "rejected", "timeout", "cancelled", "not_attempted")
_REASONS = (
    "total_full", "spk_emo_full", "waiting_full", "wait_timeout",
    "role_disabled", "emotion_disabled", "both_disabled", "cancelled", "qwen_disabled",
)


def _admission_values(admission):
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
        "outcome": admission.get("outcome") if admission.get("outcome") in _OUTCOMES else None,
        "reason": admission.get("reason") if admission.get("reason") in _REASONS else None,
        "snapshot": capacity,
    }


def record_request(events, lock, *, request_id, task_id, filename, started_at,
                   ended_at=None, status="completed", http_status=200, code=None,
                   retry_count=0, valid_segments=None, phases=None, error=None,
                   admission=None):
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
    with lock:
        events.append(event)
    return deepcopy(event)
