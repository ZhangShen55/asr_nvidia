"""测试侧诊断闸门；不参与业务请求超时或修改业务错误码。"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) and value >= 0 for value in ordered):
        raise ValueError("elapsed必须是非负有限秒数")
    if not 0 <= q <= 1:
        raise ValueError("quantile必须在[0,1]")
    position = (len(ordered) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


@dataclass(frozen=True)
class DiagnosticTrigger:
    reason: str
    observed_monotonic: float
    request_id: str | None
    observed_s: float
    threshold_s: float


class PerformanceWatchdog:
    def __init__(self, wall_timeout_s: float = 420, no_progress_s: float = 120,
                 *, clock: Callable[[], float] = time.monotonic):
        if not all(math.isfinite(v) and v > 0 for v in (wall_timeout_s, no_progress_s)):
            raise ValueError("诊断门槛必须为有限正秒数")
        self.wall_timeout_s = wall_timeout_s
        self.no_progress_s = no_progress_s
        self.clock = clock
        self._token = None
        self._last_progress = clock()
        self._had_active = False
        self.trigger: DiagnosticTrigger | None = None

    def sample(self, completed_elapsed_s: Sequence[float], active: Mapping[str, float],
               progress_token=None) -> DiagnosticTrigger | None:
        """active的值为同一客户端时钟的开始时间，严格超过门槛才触发。"""
        if self.trigger:
            return self.trigger
        now = self.clock()
        reason = request_id = None
        observed = threshold = 0.0
        if active:
            oldest = min(active, key=active.get)
            age = now - active[oldest]
            if age > self.wall_timeout_s:
                reason, request_id = "inflight_wall_timeout", oldest
                observed, threshold = age, self.wall_timeout_s
        if not reason:
            for name, q in (("p50_wall_timeout", .5), ("p95_wall_timeout", .95)):
                value = percentile(completed_elapsed_s, q)
                if value is not None and value > self.wall_timeout_s:
                    reason, observed, threshold = name, value, self.wall_timeout_s
                    break
        # 只比较阶段进度标识，不能把不断增长的age/内存采样误当作进展。
        token = (tuple(sorted(active)), len(completed_elapsed_s), progress_token)
        if token != self._token or not self._had_active:
            self._token, self._last_progress = token, now
        elif active and now - self._last_progress > self.no_progress_s and not reason:
            reason = "no_progress"
            observed, threshold = now - self._last_progress, self.no_progress_s
        self._had_active = bool(active)
        if reason:
            self.trigger = DiagnosticTrigger(reason, now, request_id, observed, threshold)
        return self.trigger
