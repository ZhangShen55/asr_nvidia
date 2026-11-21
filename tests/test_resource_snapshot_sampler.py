"""CPU memory-ASGI/controlled threads only; no NPU collector or listening socket."""
import asyncio
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
import tempfile
import threading
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from core.resource_diagnostics import ResourceSnapshotSampler
from core.config import Settings
from core import status
from core.qwen_asr_queue import AudioSlotManager
from api.routes import status as route
from tests.test_status_sampling import patch_assembly, STATS, RESOURCE


async def spin_until(predicate):
    for _ in range(1000):
        if predicate(): return
        await asyncio.sleep(.001)
    raise AssertionError("controlled CPU task did not progress")


def test_http_immediate_unknown_singleflight_and_live_counts_while_collector_blocked():
    async def scenario():
        started, release = threading.Event(), threading.Event()
        calls = []
        def collect():
            calls.append(threading.get_ident()); started.set()
            assert release.wait(3), "test did not release collector"
            return RESOURCE
        sampler = ResourceSnapshotSampler(collector=collect)
        app = FastAPI(); app.state.run_start_time = datetime.utcnow(); app.state.resource_sampler = sampler
        app.include_router(route.router)
        manager = AudioSlotManager(8, spk_emo_capacity=4)
        leases = []
        try:
            with ExitStack() as stack:
                patch_assembly(stack, threading.get_ident())
                stack.enter_context(patch.object(status, "get_slot_manager", return_value=manager))
                stack.enter_context(patch.object(route, "read_stats", return_value=STATS))
                stack.enter_context(patch.object(status, "status_snapshot", side_effect=AssertionError("GET ran heavy sync collector")))
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://cpu") as client:
                    first = (await asyncio.wait_for(client.get("/get_status"), .5)).json()
                    assert first["resourceDiagnostics"]["available"] is False
                    assert first["resourceDiagnostics"]["sampling"]["age_s"] is None
                    assert first["resourceDiagnostics"]["sampling"]["in_progress"] is True
                    await spin_until(started.is_set)
                    leases = [await manager.acquire("B"), await manager.acquire("A")]
                    replies = await asyncio.wait_for(asyncio.gather(*[client.get("/get_status") for _ in range(20)]), .5)
                    assert not release.is_set() and len(calls) == 1
                    assert all(row.json()["audioConcurrency"]["active_total"] == 2 and row.json()["spkEmoConcurrency"]["active"] == 1 for row in replies)
                    release.set()
                    await spin_until(lambda: not sampler._in_progress)
                    complete = (await client.get("/get_status")).json()
                    assert complete["resourceDiagnostics"]["timestamp"] == "test-time"
                    assert complete["resourceDiagnostics"]["sampling"]["has_snapshot"] is True
        finally:
            release.set()
            for lease in leases: await manager.release(lease)
            await sampler.aclose()
    asyncio.run(scenario())


def test_ttl_is_on_demand_failure_preserves_old_snapshot_and_no_unqueried_timer():
    async def scenario():
        clock = [0.0]; calls = []
        def collect():
            calls.append(len(calls) + 1)
            if len(calls) == 2: raise OSError("controlled collector failure")
            return {"available": True, "timestamp": str(len(calls))}
        sampler = ResourceSnapshotSampler(interval_s=1, collector=collect, clock=lambda: clock[0])
        await asyncio.sleep(.01); assert calls == []
        assert sampler.snapshot()["available"] is False
        await spin_until(lambda: not sampler._in_progress)
        cached = sampler.snapshot(); assert cached["timestamp"] == "1" and cached["sampling"]["age_s"] == 0
        cached["timestamp"] = "external mutation"
        assert sampler.snapshot()["timestamp"] == "1" and len(calls) == 1
        clock[0] = 2
        stale = sampler.snapshot(); assert stale["timestamp"] == "1" and stale["sampling"]["in_progress"]
        await spin_until(lambda: not sampler._in_progress)
        failed = sampler.snapshot()
        assert failed["timestamp"] == "1" and failed["sampling"]["age_s"] == 2
        assert "controlled collector failure" in failed["sampling"]["last_error"]
        clock[0] = 10
        await asyncio.sleep(.01); assert len(calls) == 2  # no GET means no new I/O, even long after TTL.
        sampler.snapshot(); await spin_until(lambda: not sampler._in_progress)
        assert sampler.snapshot()["timestamp"] == "3" and sampler.snapshot()["sampling"]["last_error"] is None
        await sampler.aclose()
    asyncio.run(scenario())


def test_first_failure_stays_unknown_and_close_waits_actual_thread_despite_repeat_cancel():
    async def scenario():
        failed = ResourceSnapshotSampler(collector=lambda: (_ for _ in ()).throw(RuntimeError("first failure")))
        failed.snapshot(); await spin_until(lambda: not failed._in_progress)
        state = failed.snapshot()
        assert state["available"] is False and state["sampling"]["age_s"] is None
        assert "first failure" in state["sampling"]["last_error"]
        await failed.aclose()
        started, release, ended = threading.Event(), threading.Event(), threading.Event()
        def collect():
            started.set(); assert release.wait(3); ended.set(); return {"available": True}
        sampler = ResourceSnapshotSampler(collector=collect)
        sampler.snapshot(); await spin_until(started.is_set)
        closing = asyncio.create_task(sampler.aclose())
        await asyncio.sleep(0); closing.cancel(); await asyncio.sleep(0); closing.cancel(); await asyncio.sleep(0)
        assert not closing.done() and not ended.is_set()
        assert sampler.snapshot()["sampling"]["closing"] is True
        another = asyncio.create_task(sampler.aclose()); release.set()
        with pytest.raises(asyncio.CancelledError): await closing
        await another
        assert ended.is_set() and sampler.snapshot()["sampling"]["closed"]
        assert sampler.snapshot()["sampling"]["attempts"] == 1
    asyncio.run(scenario())


def test_independent_apps_loops_never_share_task_cache_or_restart_closed_sampler():
    sampler = ResourceSnapshotSampler(collector=lambda: {"timestamp": "first"})
    async def first():
        sampler.snapshot(); await spin_until(lambda: not sampler._in_progress)
        await sampler.aclose()
        assert sampler.snapshot()["timestamp"] == "first"
    asyncio.run(first())
    async def second():
        with pytest.raises(RuntimeError, match="跨app事件循环"): sampler.snapshot()
        fresh = ResourceSnapshotSampler(collector=lambda: {"timestamp": "second"})
        assert fresh.snapshot()["sampling"]["has_snapshot"] is False
        await spin_until(lambda: not fresh._in_progress)
        assert fresh.snapshot()["timestamp"] == "second"
        await fresh.aclose()
    asyncio.run(second())


def test_main_lifespan_has_no_collector_io_without_get_and_closes_owned_sampler():
    import main
    async def scenario():
        with ExitStack() as stack:
            stack.enter_context(patch.object(main.settings, "validate_model_backends"))
            stack.enter_context(patch.object(main, "validate_npu_runtime", return_value={}))
            for name in ("_set_process_name", "setup_logging", "cleanup_stale_service_cache", "reset_stats"):
                stack.enter_context(patch.object(main, name))
            for name in ("start_vad_pool", "stop_vad_pool", "preload_qwen_client", "close_v3_model_pools", "close_qwen_client"):
                stack.enter_context(patch.object(main, name, AsyncMock()))
            collector = stack.enter_context(patch("core.resource_diagnostics.status_snapshot", side_effect=AssertionError("idle app did I/O")))
            app = main.create_app()
            assert app.state.resource_sampler is None
            async with main.lifespan(app):
                assert app.state.resource_sampler._task is None
            assert app.state.resource_sampler._closed
            collector.assert_not_called()
    asyncio.run(scenario())


@pytest.mark.parametrize("literal", ["true", "false", '"1"', "0", "-1", "0.099", "60.01", "nan", "inf", "-inf"])
def test_invalid_interval_rejected_before_resource_or_model_start(tmp_path, literal):
    path = tmp_path / "config.toml"; path.write_text(f"[status]\nresource_sample_interval_s={literal}\n")
    with pytest.raises(ValueError, match="resource_sample_interval_s"): Settings(config_path=str(path)).validate_model_backends()


@pytest.mark.parametrize("literal,expected", [("0.1", .1), ("1", 1.0), ("60", 60.0)])
def test_interval_boundaries_and_default_are_typed(tmp_path, literal, expected):
    path = tmp_path / "config.toml"; path.write_text("")
    assert Settings(config_path=str(path)).resource_sample_interval_s == 1.0
    path.write_text(f"[status]\nresource_sample_interval_s={literal}\n")
    value = Settings(config_path=str(path)).resource_sample_interval_s
    assert value == expected and type(value) is float


def test_worker_npu_hbm_identity_is_owned_without_relabeling_parent_allocator_or_external():
    raw = {"services": {"backend": [{"pid": 123}]}, "npu": {"available": True, "compute_apps": [
        {"pid": 123, "memory_mb": 100}, {"pid": 987, "memory_mb": 500, "process_identity": {"start_ticks": "10", "create_time": 100.0}},
        {"pid": 1601397, "memory_mb": 900, "process_identity": {"start_ticks": "999", "create_time": 1.0}}], "other_compute_apps": [{"pid": 987}, {"pid": 1601397}]}}
    qwen = {"execution_mode": "process", "model_workers": [{"worker_pid": 987, "worker_start_ticks": "10", "worker_create_time": 100.0}]}
    with patch.object(status.os, "getpid", return_value=123):
        result = status._classify_npu_apps(raw, qwen)
    assert [r["pid"] for r in result["service_compute_apps"]] == [123, 987]
    assert result["qwen_compute_apps"][0]["memory_mb"] == 500
    assert [r["pid"] for r in result["other_compute_apps"]] == [1601397]
    assert raw["npu"]["other_compute_apps"] == [{"pid": 987}, {"pid": 1601397}]
    raw["npu"]["compute_apps"][1]["process_identity"] = {"start_ticks": "11", "create_time": 101.0}
    with patch.object(status.os, "getpid", return_value=123):
        unverified = status._classify_npu_apps(raw, qwen)
    assert not unverified["qwen_compute_apps"]
    assert [row["pid"] for row in unverified["unverified_compute_apps"]] == [987]
    assert [row["pid"] for row in unverified["other_compute_apps"]] == [1601397]


def test_two_app_samplers_on_one_loop_have_independent_cache_and_close():
    async def scenario():
        clock = [0.0]; calls = [0, 0]
        def collector(index):
            calls[index] += 1
            return {"timestamp": f"app-{index}-{calls[index]}"}
        apps = [FastAPI(), FastAPI()]
        for index, app in enumerate(apps):
            app.state.resource_sampler = ResourceSnapshotSampler(collector=lambda index=index: collector(index), clock=lambda: clock[0])
        a, b = [app.state.resource_sampler for app in apps]
        a.snapshot(); b.snapshot()
        await spin_until(lambda: not a._in_progress and not b._in_progress)
        assert a.snapshot()["timestamp"] == "app-0-1" and b.snapshot()["timestamp"] == "app-1-1"
        await a.aclose(); clock[0] = 2
        assert a.snapshot()["sampling"]["attempts"] == 1
        b.snapshot(); await spin_until(lambda: not b._in_progress)
        assert b.snapshot()["timestamp"] == "app-1-2" and calls == [1, 2]
        await b.aclose()
    asyncio.run(scenario())


def test_invalid_interval_lifespan_fails_before_sampler_preflight_or_models(tmp_path):
    import main
    path = tmp_path / "invalid.toml"; path.write_text("[status]\nresource_sample_interval_s=nan\n")
    async def scenario():
        with patch.object(main, "settings", Settings(config_path=str(path))), \
             patch.object(main, "ResourceSnapshotSampler", side_effect=AssertionError("must validate before sampler construction")) as sampler, \
             patch.object(main, "validate_npu_runtime") as preflight, \
             patch.object(main, "start_vad_pool", AsyncMock()) as vad:
            with pytest.raises(ValueError, match="resource_sample_interval_s"):
                async with main.lifespan(main.create_app()): pass
            sampler.assert_not_called(); preflight.assert_not_called(); vad.assert_not_awaited()
    asyncio.run(scenario())


def test_attempt_timing_includes_dispatch_wall_and_failure_does_not_reset_cache_age():
    async def scenario():
        now = [10.0]; outcomes = ["ok", "fail"]
        def collect():
            now[0] += 3.0
            if outcomes.pop(0) == "fail": raise OSError("timed failure")
            return {"timestamp": "success"}
        sampler = ResourceSnapshotSampler(collector=collect, clock=lambda: now[0])
        initial = sampler.snapshot()["sampling"]
        assert initial["last_started_monotonic_s"] == 10 and initial["last_finished_monotonic_s"] is None
        await spin_until(lambda: not sampler._in_progress)
        success = sampler.snapshot()["sampling"]
        assert success["last_finished_monotonic_s"] == 13 and success["last_elapsed_s"] == 3 and success["age_s"] == 0
        now[0] = 15; sampler.snapshot(); await spin_until(lambda: not sampler._in_progress)
        failed = sampler.snapshot()["sampling"]
        assert failed["last_started_monotonic_s"] == 15 and failed["last_finished_monotonic_s"] == 18
        assert failed["last_elapsed_s"] == 3 and failed["age_s"] == 5
        assert "timed failure" in failed["last_error"]
        await sampler.aclose()
    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["thread", "process"])
def test_shutdown_publishes_sampler_closing_but_models_close_while_collector_is_blocked(mode):
    import main
    from unittest.mock import PropertyMock
    async def scenario():
        started, release, ended = threading.Event(), threading.Event(), threading.Event()
        models_closed = asyncio.Event()
        def collect():
            started.set(); assert release.wait(3); ended.set(); return {}
        async def model_close(): models_closed.set()
        context = None
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(Settings, "qwen_execution_mode", new_callable=PropertyMock, return_value=mode))
                stack.enter_context(patch.object(main.settings, "validate_model_backends"))
                stack.enter_context(patch.object(main, "validate_npu_runtime", return_value={}))
                for name in ("_set_process_name", "setup_logging", "cleanup_stale_service_cache", "reset_stats"):
                    stack.enter_context(patch.object(main, name))
                for name in ("start_vad_pool", "stop_vad_pool", "preload_qwen_client", "close_qwen_client"):
                    stack.enter_context(patch.object(main, name, AsyncMock()))
                stack.enter_context(patch.object(main, "close_v3_model_pools", model_close))
                stack.enter_context(patch("core.resource_diagnostics.status_snapshot", side_effect=collect))
                app = main.create_app(); context = main.lifespan(app)
                await context.__aenter__(); app.state.resource_sampler.snapshot()
                await spin_until(started.is_set)
                closing = asyncio.create_task(context.__aexit__(None, None, None))
                await asyncio.wait_for(models_closed.wait(), .5)
                assert app.state.resource_sampler._closing and not ended.is_set() and not closing.done()
                release.set(); await closing
                assert ended.is_set() and app.state.resource_sampler._closed
        finally: release.set()
    asyncio.run(scenario())


def test_unknown_or_disappeared_nonworker_pid_is_unverified_not_external():
    raw = {"npu": {"compute_apps": [{"pid": 600, "process_identity": {"pid": 600, "available": False}}, {"pid": 601}]}}
    with patch.object(status.os, "getpid", return_value=123):
        result = status._classify_npu_apps(raw, {"model_workers": []})
    assert not result["other_compute_apps"] and not result["service_compute_apps"]
    assert [row["pid"] for row in result["unverified_compute_apps"]] == [600, 601]
