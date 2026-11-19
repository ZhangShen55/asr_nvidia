"""Heavy status collection leaves the event loop free; model snapshots stay on it."""
import asyncio
from contextlib import ExitStack
from datetime import datetime
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from api.routes import status as route
from core import status


STATS = {"offline": 7, "online": 1, "total_failed_tasks": 2,
         "processing_task_ids": ["task"], "processing_task_names": {"task": "sample.wav"}}
RESOURCE = {"timestamp": "test-time", "backend": {"available": True, "pid": 123},
            "host": {"available": True}, "services": {"backend": [{"pid": 123}]}}


def patch_assembly(stack, event_loop_thread):
    def on_loop(value):
        def snapshot(*args, **kwargs):
            assert threading.get_ident() == event_loop_thread
            return value
        return snapshot
    stack.enter_context(patch.object(status, "get_slot_manager", on_loop(SimpleNamespace(snapshot=lambda: {
        "audioConcurrency": {"active_total": 1}, "spkEmoConcurrency": {"active": 0}}))))
    stack.enter_context(patch.object(status, "get_local_model_status", on_loop({"speech_campplus": {"loaded": False}})))
    stack.enter_context(patch.object(status, "get_qwen_status", on_loop({"loaded": False})))
    stack.enter_context(patch.object(status, "get_live_allocator_status", on_loop({"available": False, "reason": "npu_not_initialized"})))
    stack.enter_context(patch.object(status, "get_vad_pool", on_loop(SimpleNamespace(stats=lambda: {"busy": 0}))))
    stack.enter_context(patch.object(status, "request_telemetry_snapshot", on_loop({})))


def app():
    application = FastAPI()
    application.state.run_start_time = datetime.utcnow()
    application.include_router(route.router)
    return application


def assert_business_fields(payload):
    assert payload["status"] == "living"
    assert payload["totalHaveDoneProcessTasks"] == 8
    assert payload["offlineDone"] == 7
    assert payload["totalFailedTasks"] == 2
    assert payload["processingTaskList"] == ["sample.wav"]
    assert payload["localModels"] == {"speech_campplus": {"loaded": False}}
    assert payload["qwenModel"] == {"loaded": False}
    assert payload["vadPool"] == {"busy": 0}
    assert payload["resourceDiagnostics"]["backend"]["npu_allocator"]["reason"] == "npu_not_initialized"


def test_direct_sync_assembly_keeps_sampler_compatibility_and_accepts_empty_snapshot():
    with ExitStack() as stack:
        patch_assembly(stack, threading.get_ident())
        sampler = stack.enter_context(patch.object(status, "status_snapshot", return_value=RESOURCE))
        payload = status.build_status_payload(STATS, datetime.utcnow())
        sampler.assert_called_once_with()
        assert_business_fields(payload)
        sampler.reset_mock()
        payload = status.build_status_payload(STATS, datetime.utcnow(), resource_snapshot={})
        sampler.assert_not_called()
        assert_business_fields(payload)


def test_direct_sync_sampler_failure_is_reported_without_losing_business_fields():
    with ExitStack() as stack:
        patch_assembly(stack, threading.get_ident())
        stack.enter_context(patch.object(status, "status_snapshot", side_effect=OSError("/proc unavailable")))
        payload = status.build_status_payload(STATS, datetime.utcnow())
    assert_business_fields(payload)
    assert payload["resourceDiagnostics"]["available"] is False
    assert "/proc unavailable" in payload["resourceDiagnostics"]["reason"]



def test_http_without_lifespan_returns_honest_unknown_without_starting_sampler():
    async def scenario():
        with ExitStack() as stack:
            patch_assembly(stack, threading.get_ident())
            stack.enter_context(patch.object(route, "read_stats", return_value=STATS))
            stack.enter_context(patch.object(status, "status_snapshot", side_effect=AssertionError("GET sampled heavy resources")))
            async with AsyncClient(transport=ASGITransport(app=app()), base_url="http://fixture") as client:
                response = await client.get("/get_status")
        payload = response.json()
        assert_business_fields(payload)
        assert payload["resourceDiagnostics"]["available"] is False
        assert payload["resourceDiagnostics"]["sampling"]["lifecycle"] == "not_initialized"
    asyncio.run(scenario())
