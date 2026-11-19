import math
import pytest
from core.performance_watchdog import PerformanceWatchdog, percentile


def test_linear_quantiles_and_empty():
    assert percentile([], .95) is None
    assert percentile([1, 2, 3, 4], .5) == 2.5
    assert percentile([1, 2, 3, 4], .95) == pytest.approx(3.85)
    with pytest.raises(ValueError):
        percentile([math.inf], .5)


def test_exact_boundary_and_inflight_without_completions():
    now = [0.0]
    watchdog = PerformanceWatchdog(no_progress_s=1000, clock=lambda: now[0])
    watchdog.sample([], {"long": 0})
    now[0] = 420
    assert watchdog.sample([], {"long": 0}) is None
    now[0] = 420.01
    trigger = watchdog.sample([], {"long": 0})
    assert trigger.reason == "inflight_wall_timeout" and trigger.request_id == "long"
    assert watchdog.sample([], {}) is trigger


def test_completed_p95_and_p50():
    watchdog = PerformanceWatchdog(clock=lambda: 1000)
    assert watchdog.sample([1, 450], {}).reason == "p95_wall_timeout"
    assert PerformanceWatchdog().sample([450, 451], {}).reason == "p50_wall_timeout"


def test_no_progress_and_real_stage_progress():
    now = [0.0]
    watchdog = PerformanceWatchdog(no_progress_s=10, clock=lambda: now[0])
    watchdog.sample([], {"r": 0}, (0,))
    now[0] = 9
    assert watchdog.sample([], {"r": 0}, (1,)) is None
    now[0] = 19
    assert watchdog.sample([], {"r": 0}, (1,)) is None
    now[0] = 19.01
    assert watchdog.sample([], {"r": 0}, (1,)).reason == "no_progress"


@pytest.mark.parametrize("value", [0, -1, math.inf, math.nan])
def test_invalid_threshold(value):
    with pytest.raises(ValueError):
        PerformanceWatchdog(wall_timeout_s=value)
