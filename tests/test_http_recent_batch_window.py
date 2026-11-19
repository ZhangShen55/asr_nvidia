"""Bounded HTTP history preserves complete records and direct snapshot compatibility."""
import asyncio
import io
import threading
import wave
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.routes.status as status_route
import core.qwen_asr_client as client_module
import core.status as status_module
from core.qwen_asr_client import QwenAsrClient
from core.qwen_batch_scheduler import BatchScheduler


def make_scheduler(record_count=64):
    async def unexpected_execute(*args):
        raise AssertionError("Snapshot tests must not run inference")

    scheduler = BatchScheduler(
        unexpected_execute, workers=1, batch_size=128, wait_ms=10,
        audio_frame_budget=115200, token_budget=80000, max_new_tokens=256,
        capacity=256, duration_ratio=32,
    )
    for batch_id in range(1, record_count + 1):
        members = [{"request_id": "audio-1", "segment_index": batch_id}]
        timings = {
            "generate_wall_ms": 1,
            "single_processor_shapes": [{"input_features": [1, 128, 900]}],
        }
        scheduler._recent.append({
            "batch_id": batch_id,
            "members": members,
            "dispatched_members": members,
            "dispatch_attempts": [{
                "attempt": 1, "input_members": members, "members": members,
                "runtime_timings": timings,
            }],
            "runtime_timings": timings,
            "status": "completed",
        })
        scheduler._sizes[1] = scheduler._sizes.get(1, 0) + 1
    scheduler._batch_sequence = record_count
    scheduler._publish_state()
    return scheduler


def test_default_full64_and_limited8_keep_every_retained_record_field():
    scheduler = make_scheduler()
    full = scheduler.snapshot()
    limited = scheduler.snapshot(recent_limit=8)
    assert len(full["recent_batches"]) == 64
    assert "history" not in full
    assert limited["recent_batches"] == full["recent_batches"][56:]
    assert {
        key: value for key, value in limited.items()
        if key not in {"recent_batches", "history"}
    } == {key: value for key, value in full.items() if key != "recent_batches"}
    history = limited["history"]
    assert history["history_limit"] == 8
    assert history["stored_limit"] == 64
    assert history["available_batch_id_range"] == [1, 64]
    assert history["published_batch_id_range"] == [57, 64]
    assert history["truncated"] is True
    assert history["omitted_available_count"] == 56
    assert history["published_noncontiguous"] is False


@pytest.mark.parametrize("limited", [False, True])
def test_nested_snapshot_mutation_cannot_change_internal_history(limited):
    scheduler = make_scheduler()
    snapshot = scheduler.snapshot(recent_limit=8) if limited else scheduler.snapshot()
    record = snapshot["recent_batches"][0]
    record["runtime_timings"]["single_processor_shapes"][0]["input_features"][1] = 999
    record["members"][0]["request_id"] = "changed"
    for original in scheduler.snapshot()["recent_batches"]:
        assert original["members"][0]["request_id"] == "audio-1"
        assert original["runtime_timings"]["single_processor_shapes"][0]["input_features"] == [1, 128, 900]


@pytest.mark.parametrize("bad_limit", [True, False, 0, -1, 65, 8.0, "8"])
def test_invalid_history_limit_is_rejected(bad_limit):
    with pytest.raises(ValueError):
        make_scheduler().snapshot(recent_limit=bad_limit)


def test_empty_history_and_gaps_have_explicit_metadata():
    history = make_scheduler(0).snapshot(recent_limit=8)["history"]
    assert history["published_count"] == 0
    assert history["published_batch_id_range"] is None
    assert history["truncated"] is False
    scheduler = make_scheduler(3)
    scheduler._recent[1]["batch_id"] = 5
    scheduler._publish_state()
    assert scheduler.snapshot(recent_limit=8)["history"]["published_noncontiguous"] is True



def test_history_metadata_exposes_records_older_than_stored_capacity():
    scheduler = make_scheduler(68)
    full = scheduler.snapshot()
    history = scheduler.snapshot(recent_limit=8)["history"]
    assert len(full["recent_batches"]) == 64
    assert history["available_batch_id_range"] == [5, 68]
    assert history["published_batch_id_range"] == [61, 68]
    assert history["older_than_available_count"] == 4
    assert history["omitted_available_count"] == 56
    assert history["truncated"] is True


def test_concurrent_read_and_whole_state_publication_are_consistent():
    scheduler = make_scheduler()
    errors = []

    def publish():
        try:
            for batch_id in range(65, 129):
                scheduler._recent.append({"batch_id": batch_id, "members": [], "status": "completed"})
                scheduler._sizes[1] += 1
                scheduler._batch_sequence = batch_id
                scheduler._publish_state()
        except BaseException as error:
            errors.append(error)

    publisher = threading.Thread(target=publish)
    publisher.start()
    for _ in range(30):
        snapshot = scheduler.snapshot(recent_limit=8)
        assert len(snapshot["recent_batches"]) == snapshot["history"]["published_count"] <= 8
    publisher.join()
    assert not errors


def test_client_and_getter_keep_full_history_by_default(monkeypatch):
    class FakeRuntime:
        failure = None

        def __init__(self, *args):
            pass

        def transcribe_batch(self, *args):
            raise AssertionError("Snapshot test must not run inference")

        def snapshot(self):
            return {"loaded": True, "failure": None}

    client = QwenAsrClient(
        runtime_factory=FakeRuntime, execution_mode="thread", local_instance_count=1,
    )
    client._batch_scheduler = make_scheduler()
    assert len(client.snapshot()["batch_scheduler"]["recent_batches"]) == 64
    assert len(client.snapshot(recent_batch_limit=8)["batch_scheduler"]["recent_batches"]) == 8
    monkeypatch.setattr(client_module, "_client_instance", client)
    assert len(client_module.get_qwen_status()["batch_scheduler"]["recent_batches"]) == 64
    assert len(client_module.get_qwen_status(recent_batch_limit=8)["batch_scheduler"]["recent_batches"]) == 8


def test_status_builder_preserves_default_getter_call_and_forwards_optional_limit(monkeypatch):
    calls = []

    class StopAfterGetter(Exception):
        pass

    def getter(**kwargs):
        calls.append(kwargs)
        raise StopAfterGetter

    monkeypatch.setattr(status_module, "get_slot_manager", lambda: SimpleNamespace(snapshot=lambda: {}))
    monkeypatch.setattr(status_module, "get_qwen_status", getter)
    with pytest.raises(StopAfterGetter):
        status_module.build_status_payload({}, datetime.utcnow())
    with pytest.raises(StopAfterGetter):
        status_module.build_status_payload({}, datetime.utcnow(), qwen_recent_batch_limit=8)
    assert calls == [{}, {"recent_batch_limit": 8}]


def test_actual_http_route_returns_json_response_without_recursive_encoder(monkeypatch):
    scheduler = make_scheduler()
    calls = []

    def build_payload(stats, started, **kwargs):
        calls.append(kwargs)
        return {
            "qwenModel": {"batch_scheduler": scheduler.snapshot(recent_limit=kwargs["qwen_recent_batch_limit"])},
            "old": "retained",
            "sampling": kwargs["resource_snapshot"],
        }

    def unexpected_encoder(*args, **kwargs):
        raise AssertionError("JSONResponse must bypass FastAPI's recursive encoder")

    monkeypatch.setattr(status_route, "build_status_payload", build_payload)
    monkeypatch.setattr(status_route, "read_stats", lambda: {})
    monkeypatch.setattr(status_route, "unknown_resource_snapshot", lambda reason: {"available": False, "reason": reason})
    monkeypatch.setattr("fastapi.routing.jsonable_encoder", unexpected_encoder)
    app = FastAPI()
    app.state.run_start_time = datetime.utcnow()
    app.include_router(status_route.router)
    with TestClient(app) as client:
        response = client.get("/get_status")
    payload = response.json()
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert payload["old"] == "retained"
    assert len(payload["qwenModel"]["batch_scheduler"]["recent_batches"]) == 8
    assert calls[0]["qwen_recent_batch_limit"] == 8
    assert payload["sampling"]["sampling"]["lifecycle"] == "not_initialized"


def test_limited_snapshot_does_not_release_cancelled_work_before_actual_done():
    async def scenario():
        started = asyncio.Event()
        done = asyncio.Event()

        async def execute(members, batch_id):
            started.set()
            await done.wait()
            return ["fake"] * len(members)

        scheduler = BatchScheduler(
            execute, workers=1, batch_size=128, wait_ms=1,
            audio_frame_budget=115200, token_budget=80000, max_new_tokens=256,
            capacity=256, duration_ratio=32,
        )
        stream = io.BytesIO()
        with wave.open(stream, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 4800)
        task = asyncio.create_task(scheduler.submit(stream.getvalue(), None, request_id="audio", segment_index=0))
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        assert not task.done()
        assert scheduler.snapshot(recent_limit=8)["running_members"] == 1
        done.set()
        await asyncio.gather(task, return_exceptions=True)
        await scheduler.close()
        snapshot = scheduler.snapshot(recent_limit=8)
        assert snapshot["inflight"] == snapshot["waiting"] == snapshot["running_members"] == 0
        assert scheduler._slots._value == 256

    asyncio.run(asyncio.wait_for(scenario(), 2))
