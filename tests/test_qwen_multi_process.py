"""Real CPU spawn, shared batching and complete ownership for multiple model workers."""
import asyncio
import io
import json
import os
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

import core.qwen_asr_client as client_module
from core.config import Settings
from core.qwen_asr_client import QwenAsrClient, QwenAsrUnavailableError
from core.qwen_asr_process import LocalQwenProcessRuntime
from core.qwen_asr_queue import RequestConcurrency

pytestmark = pytest.mark.skipif(not Path("/proc/stat").exists(), reason="worker identity needs Linux /proc")


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


async def until(predicate, timeout=10):
    async def poll():
        while not predicate():
            await asyncio.sleep(.005)
    await asyncio.wait_for(poll(), timeout)


def make_client(tmp_path, count, options=None, **kwargs):
    paths = []
    configurations = []
    for index in range(count):
        path = tmp_path / f"worker-{index}.json"
        event_path = tmp_path / f"events-{index}.jsonl"
        config = {"events": str(event_path), "instance_index": index}
        config.update((options or {}).get(index, {}))
        path.write_text(json.dumps(config))
        paths.append(event_path)
        configurations.append(path)
    created = []

    def factory(model_dir, device, dtype, max_new_tokens):
        runtime = LocalQwenProcessRuntime(
            str(configurations[len(created)]), device, dtype, max_new_tokens,
            _runtime_factory="tests.fake_qwen_multi:CpuRuntime",
            _startup_timeout_s=10, _close_timeout_s=2,
        )
        created.append(runtime)
        return runtime

    parameters = dict(
        generation_mode="legacy",
        local_model_dir=str(tmp_path / "configured-model"), local_device="npu:7",
        local_dtype="bfloat16", local_instance_count=count, max_new_tokens=256,
        execution_mode="process", runtime_factory=factory, retry_count=0,
        request_timeout_s=10, max_inference_batch_size=128 // count,
        batch_wait_timeout_ms=10, max_batch_audio_frames=115200,
        max_batch_tokens=80000, batch_max_duration_ratio=32,
    )
    parameters.update(kwargs)
    return QwenAsrClient(**parameters), paths


def worker_states(client):
    return [runtime.snapshot()["model_worker"] for runtime in client._runtimes]


def assert_closed(client, normal=True):
    for state in worker_states(client):
        assert state["resource_reaped"] is True
        assert state["worker_alive"] is False
        if normal:
            assert state["normal_exit_ok"] is True
            assert state["worker_exitcode"] == (0 if state["started"] else None)
            assert state["forced_cleanup"] is False
    assert client._available.qsize() == len(client._runtimes)
    assert not any(client._busy)
    snapshot = client._batch_scheduler.snapshot()
    assert snapshot["inflight"] == snapshot["running_members"] == snapshot["waiting"] == 0


def wav_bytes(code):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(struct.pack("<h", code) + b"\x00\x00" * 4799)
    return stream.getvalue()


@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_process_configuration_accepts_explicit_instance_count(tmp_path, count):
    path = tmp_path / "config.toml"
    path.write_text(f'[qwen_asr]\nexecution_mode="process"\nlocal_instance_count={count}\n')
    settings = Settings(str(path))
    assert settings.qwen_execution_mode == "process"
    assert settings.qwen_local_instance_count == count


@pytest.mark.parametrize("count", [2, 4, 8])
def test_eight_simultaneous_discoveries_load_every_owned_worker_once(tmp_path, count):
    async def scenario():
        gate = tmp_path / "load-release"
        client, paths = make_client(
            tmp_path, count, {index: {"load_gate": str(gate)} for index in range(count)},
            model="explicit-name",
        )
        tasks = [asyncio.create_task(client.discover_model()) for _ in range(8)]
        try:
            await until(lambda: all(any(row["event"] == "load_started" for row in events(path)) for path in paths))
            assert client._discovered_model is None
            assert all(client._busy)
            gate.touch()
            assert await asyncio.gather(*tasks) == ["explicit-name"] * 8
            states = worker_states(client)
            assert len({state["worker_pid"] for state in states}) == count
            assert len({state["launch_id"] for state in states}) == count
            assert all(runtime.loaded and not runtime.failure for runtime in client._runtimes)
            assert all(sum(row["event"] == "load_done" for row in events(path)) == 1 for path in paths)
            for state in states:
                assert state["ready_verified"] is True
                assert state["worker_create_time"] > 0 and state["worker_start_ticks"]
                if os.getenv("ASR_EXPECT_PROCESS_ENTRY_SO") == "1":
                    assert state["worker_entry_module_file"].endswith(".so")
                    assert Path(state["worker_entry_module_file"]).is_relative_to(Path(os.environ["ASR_EXPECT_PROCESS_STAGE_ROOT"]))
        finally:
            gate.touch()
            await asyncio.gather(*tasks, return_exceptions=True)
            await client.aclose()
        assert_closed(client)
    asyncio.run(asyncio.wait_for(scenario(), 20))


def test_partial_discovery_failure_never_marks_whole_pool_discovered(tmp_path):
    async def scenario():
        client, _ = make_client(tmp_path, 2, {0: {"unavailable": True}})
        try:
            results = await asyncio.gather(*(client.discover_model() for _ in range(8)), return_exceptions=True)
            assert all(isinstance(result, QwenAsrUnavailableError) for result in results)
            assert client._discovered_model is None
            assert client._available.qsize() == 2
            assert not any(client._busy)
        finally:
            await client.aclose()
        assert_closed(client)
    asyncio.run(asyncio.wait_for(scenario(), 15))


def test_slow_worker_does_not_block_sibling_and_cancel_member_keeps_ownership(tmp_path):
    async def scenario():
        gate = tmp_path / "worker-zero-release"
        client, paths = make_client(tmp_path, 2, {0: {"gate": str(gate)}})
        concurrency = RequestConcurrency(16, 128)
        semaphores = [concurrency.new_segment_semaphore() for _ in range(8)]
        held = set()
        tasks = {}

        async def submit(owner, index):
            code = owner * 1000 + index + 1
            await concurrency.acquire(semaphores[owner])
            held.add(code)
            try:
                return await client.transcribe_bytes(
                    wav_bytes(code), language="bad" if (owner, index) == (0, 1) else None,
                    request_id=f"audio-{owner}", segment_index=index,
                )
            finally:
                held.remove(code)
                await concurrency.release(semaphores[owner])

        try:
            await client.discover_model()
            for owner in range(8):
                for index in range(32):
                    tasks[owner, index] = asyncio.create_task(submit(owner, index))
            await until(lambda: sum(row["event"] == "started" for row in events(paths[1])) >= 2)
            first_slow = next(row for row in events(paths[0]) if row["event"] == "started")
            code = first_slow["members"][0]
            owner, index = (code - 1) // 1000, (code - 1) % 1000
            cancelled = tasks[owner, index]
            cancelled.cancel()
            await asyncio.sleep(0)
            cancelled.cancel()
            await asyncio.sleep(.005)
            assert not cancelled.done() and code in held
            assert not any(row["event"] == "read_complete" for row in events(paths[0]))
            gate.touch()
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            assert sum(isinstance(value, asyncio.CancelledError) for value in results) == 1
            assert sum(getattr(value, "code", None) == 4009 for value in results) == 1
            assert sum(isinstance(value, str) for value in results) == 254
            records = client._batch_scheduler.snapshot()["recent_batches"]
            instances = {record["runtime_instance"]["instance_index"] for record in records}
            assert instances == {0, 1}
            for record in records:
                attempt = record["dispatch_attempts"][0]
                identity = attempt["runtime_instance"]
                expected = worker_states(client)[identity["instance_index"]]
                assert identity["worker_pid"] == expected["worker_pid"]
                assert identity["launch_id"] == expected["launch_id"]
                assert identity["worker_start_ticks"] == expected["worker_start_ticks"]
                assert identity["last_rpc_id"] >= 2
                assert record["runtime_instance"] == identity
            assert not held
            assert concurrency._global._value == 128
            assert all(semaphore._value == 16 for semaphore in semaphores)
        finally:
            gate.touch()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            await client.aclose()
        assert_closed(client)
    asyncio.run(asyncio.wait_for(scenario(), 20))


def test_one_unrecoverable_child_does_not_respawn_or_quarantine_sibling(tmp_path):
    async def scenario():
        client, _ = make_client(tmp_path, 2, {0: {"die_on_transcribe": True}})
        try:
            await client.discover_model()
            results = await asyncio.gather(*(
                client.transcribe_bytes(wav_bytes(index + 1), request_id=f"audio-{index % 8}", segment_index=index)
                for index in range(128)
            ), return_exceptions=True)
            assert sum(isinstance(value, str) for value in results) == 64
            assert sum(getattr(value, "code", None) == 4011 for value in results) == 64
            before = worker_states(client)
            assert before[0]["resource_reaped"] and before[0]["worker_exitcode"] == 17
            assert client._runtimes[0].failure and not client._runtimes[1].failure
            cached_name = client._discovered_model
            assert cached_name
            assert await client.discover_model() == cached_name
            assert client.snapshot()["capacity"] == 1
            assert client.snapshot()["configured_instances"] == 2
            assert isinstance(await client.transcribe_bytes(wav_bytes(1000), request_id="later", segment_index=0), str)
            assert worker_states(client)[0]["worker_pid"] == before[0]["worker_pid"]
        finally:
            with pytest.raises(RuntimeError):
                await client.aclose()
        assert_closed(client, normal=False)
        assert worker_states(client)[1]["normal_exit_ok"] is True
        with pytest.raises(RuntimeError):
            await client.aclose()
    asyncio.run(asyncio.wait_for(scenario(), 20))


def test_normal_closes_start_concurrently_after_all_leases_drained(tmp_path, monkeypatch):
    async def scenario():
        gates = [tmp_path / f"close-{index}" for index in range(2)]
        client, paths = make_client(tmp_path, 2, {index: {"close_gate": str(gates[index])} for index in range(2)})
        close = None
        try:
            await client.discover_model()
            close = asyncio.create_task(client.aclose())
            await until(lambda: all(any(row["event"] == "close_started" for row in events(path)) for path in paths), timeout=.8)
            assert not close.done()
            for gate in gates:
                gate.touch()
            await close
        finally:
            for gate in gates:
                gate.touch()
            if close is not None:
                await asyncio.gather(close, return_exceptions=True)
            await client.aclose()
        assert_closed(client)
    monkeypatch.setattr(client_module, "settings", SimpleNamespace(qwen_process_shutdown_grace_s=1.5))
    asyncio.run(asyncio.wait_for(scenario(), 15))


def test_close_error_is_aggregated_without_skipping_healthy_sibling(tmp_path):
    async def scenario():
        client, paths = make_client(tmp_path, 2, {0: {"close_error": True}, 1: {"close_delay_s": .05}})
        await client.discover_model()
        with pytest.raises(RuntimeError):
            await client.aclose()
        assert all(any(row["event"] == "close_started" for row in events(path)) for path in paths)
        assert_closed(client, normal=False)
        assert worker_states(client)[0]["normal_exit_ok"] is False
        assert worker_states(client)[1]["normal_exit_ok"] is True
        with pytest.raises(RuntimeError):
            await client.aclose()
    asyncio.run(asyncio.wait_for(scenario(), 15))


def test_repeated_cancel_of_discovery_waits_all_started_loads(tmp_path):
    async def scenario():
        gate = tmp_path / "load-release"
        client, paths = make_client(tmp_path, 2, {index: {"load_gate": str(gate)} for index in range(2)})
        tasks = [asyncio.create_task(client.discover_model()) for _ in range(8)]
        try:
            await until(lambda: all(any(row["event"] == "load_started" for row in events(path)) for path in paths))
            tasks[0].cancel()
            await asyncio.sleep(0)
            tasks[0].cancel()
            await asyncio.sleep(.01)
            assert not tasks[0].done() and all(client._busy)
            gate.touch()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            assert sum(isinstance(value, asyncio.CancelledError) for value in results) == 1
            assert sum(isinstance(value, str) for value in results) == 7
            assert client._available.qsize() == 2
            assert not client._discovering_all and client._waiting == 0
        finally:
            gate.touch()
            await asyncio.gather(*tasks, return_exceptions=True)
            await client.aclose()
        assert_closed(client)
    asyncio.run(asyncio.wait_for(scenario(), 20))


def test_shared_shutdown_grace_and_repeated_cancel_reap_all_active_children(tmp_path, monkeypatch):
    async def scenario():
        gate = tmp_path / "never-forward-release"
        client, paths = make_client(tmp_path, 2, {index: {"gate": str(gate)} for index in range(2)})
        tasks = []
        close = None
        try:
            await client.discover_model()
            tasks = [asyncio.create_task(client.transcribe_bytes(
                wav_bytes(index + 1), request_id=f"audio-{index % 8}", segment_index=index,
            )) for index in range(128)]
            await until(lambda: all(any(row["event"] == "started" for row in events(path)) for path in paths))
            close = asyncio.create_task(client.aclose())
            await asyncio.sleep(.02)
            close.cancel()
            await asyncio.sleep(0)
            close.cancel()
            assert not close.done()
            await asyncio.gather(close, return_exceptions=True)
            await asyncio.gather(*tasks, return_exceptions=True)
            with pytest.raises(RuntimeError):
                await client.aclose()
            assert_closed(client, normal=False)
            assert all(state["forced_cleanup"] for state in worker_states(client))
            assert client._shutdown_state["phase"] == "failed"
        finally:
            gate.touch()
            await asyncio.gather(*tasks, return_exceptions=True)
            if close is not None:
                await asyncio.gather(close, return_exceptions=True)
            try:
                await client.aclose()
            except RuntimeError:
                pass
    monkeypatch.setattr(client_module, "settings", SimpleNamespace(qwen_process_shutdown_grace_s=.2))
    asyncio.run(asyncio.wait_for(scenario(), 20))


@pytest.mark.parametrize("count", [2, 4, 8])
def test_true_parallel_capped_batches_use_every_worker_under_16_128_credits(tmp_path, count):
    async def scenario():
        gate = tmp_path / "all-forward-release"
        client, paths = make_client(tmp_path, count, {index: {"gate": str(gate)} for index in range(count)})
        concurrency = RequestConcurrency(16, 128)
        semaphores = [concurrency.new_segment_semaphore() for _ in range(8)]
        tasks = []

        async def submit(owner, index):
            await concurrency.acquire(semaphores[owner])
            try:
                return await client.transcribe_bytes(
                    wav_bytes(owner * 1000 + index + 1),
                    request_id=f"audio-{owner}", segment_index=index,
                )
            finally:
                await concurrency.release(semaphores[owner])

        try:
            await client.discover_model()
            tasks = [asyncio.create_task(submit(owner, index)) for owner in range(8) for index in range(16)]
            await until(lambda: all(any(row["event"] == "started" for row in events(path)) for path in paths))
            calls = [next(row for row in events(path) if row["event"] == "started") for path in paths]
            assert all(len(call["members"]) == 128 // count for call in calls)
            assert len({code for call in calls for code in call["members"]}) == 128
            assert concurrency._global._value == 0
            assert all(semaphore._value == 0 for semaphore in semaphores)
            assert not any(any(row["event"] == "read_complete" for row in events(path)) for path in paths)
            gate.touch()
            assert all(isinstance(value, str) for value in await asyncio.gather(*tasks))
            records = client._batch_scheduler.snapshot()["recent_batches"]
            assert len(records) == count
            assert {record["runtime_instance"]["instance_index"] for record in records} == set(range(count))
            assert all(record["actual_dispatched_size"] == 128 // count for record in records)
            assert concurrency._global._value == 128
            assert all(semaphore._value == 16 for semaphore in semaphores)
        finally:
            gate.touch()
            await asyncio.gather(*tasks, return_exceptions=True)
            await client.aclose()
        assert_closed(client)
    asyncio.run(asyncio.wait_for(scenario(), 20))


def test_batch_cap_is_explicit_and_is_not_silently_split_from_instance_count(tmp_path):
    async def scenario():
        client, _ = make_client(tmp_path, 2, max_inference_batch_size=128)
        assert client._batch_scheduler.snapshot()["configured_batch_size"] == 128
        assert all(not state["started"] for state in worker_states(client))
        await client.aclose()
        assert_closed(client)
    asyncio.run(scenario())


def test_close_during_parallel_discovery_waits_loads_and_releases_all_indices(tmp_path):
    async def scenario():
        gate = tmp_path / "load-release"
        client, paths = make_client(tmp_path, 2, {index: {"load_gate": str(gate)} for index in range(2)})
        discoveries = [asyncio.create_task(client.discover_model()) for _ in range(8)]
        close = None
        try:
            await until(lambda: all(any(row["event"] == "load_started" for row in events(path)) for path in paths))
            close = asyncio.create_task(client.aclose())
            await asyncio.sleep(.01)
            assert not close.done() and all(client._busy)
            gate.touch()
            results = await asyncio.gather(*discoveries, return_exceptions=True)
            assert all(isinstance(value, QwenAsrUnavailableError) for value in results)
            await close
            assert client._discovered_model is None
            assert not client._discovering_all and client._waiting == 0
        finally:
            gate.touch()
            await asyncio.gather(*discoveries, return_exceptions=True)
            if close is not None:
                await asyncio.gather(close, return_exceptions=True)
            await client.aclose()
        assert_closed(client)
    asyncio.run(asyncio.wait_for(scenario(), 20))


def test_previously_discovered_pool_rejects_new_audio_when_all_children_fail(tmp_path):
    async def scenario():
        client, _ = make_client(tmp_path, 2, {index: {"die_on_transcribe": True} for index in range(2)})
        try:
            await client.discover_model()
            assert client._discovered_model
            results = await asyncio.gather(*(
                client.transcribe_bytes(wav_bytes(index + 1), request_id=f"audio-{index % 8}", segment_index=index)
                for index in range(128)
            ), return_exceptions=True)
            assert all(getattr(value, "code", None) == 4011 for value in results)
            states = worker_states(client)
            assert all(state["worker_exitcode"] == 17 and state["resource_reaped"] for state in states)
            with pytest.raises(QwenAsrUnavailableError):
                await client.discover_model()
            assert client.snapshot()["capacity"] == 0
            assert [state["worker_pid"] for state in worker_states(client)] == [state["worker_pid"] for state in states]
        finally:
            with pytest.raises(RuntimeError):
                await client.aclose()
        assert_closed(client, normal=False)
    asyncio.run(asyncio.wait_for(scenario(), 20))
