"""正式共享增强池的CPU替身契约；不加载模型或调用设备。"""
from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import torch

from api.routes import asr_qwen_common as pipeline
from core import models, request_telemetry as telemetry
from core.config import settings
from core.models import ModelPool
from core.qwen_asr_queue import RequestConcurrency
from core.segment_pipeline import PendingSegmentBudget
from core.vad import VadSegment


async def until(predicate):
    async def observe():
        while not predicate():
            await asyncio.sleep(.001)
    await asyncio.wait_for(observe(), 5)


@pytest.fixture(autouse=True)
def isolated_pool_status():
    saved = deepcopy(models.local_model_status)
    saved_models = models._model_spk_v3, models._model_emotion_v3
    telemetry.reset_for_tests()
    with patch.dict(settings._cfg, {
        "features": {"open_v2_role": True, "open_v2_emotion": True},
        "model_device_pool": {"speech_campplus_execution_mode": "thread"},
    }):
        yield
    models.local_model_status.clear()
    models.local_model_status.update(saved)
    models._model_spk_v3, models._model_emotion_v3 = saved_models
    telemetry.reset_for_tests()


class FakeRunner:
    execution_mode = "thread"
    healthy = True

    def __init__(self, name, index):
        self.name, self.index = name, index

    def close(self):
        pass


class FakePool(ModelPool):
    def __init__(self, name, count=1, *, build=None):
        super().__init__(name, "unused", devices_getter=lambda: ["cpu"] * count,
                         prefetch_depth_getter=lambda: 0)
        self.builds = []
        self.build = build

    def _build_instance(self, device):
        index = len(self.builds)
        self.builds.append(device)
        if self.build is not None:
            self.build(index)
        return FakeRunner(self.name, index), device, 0.0


@pytest.mark.parametrize("qwen_count", [1, 6])
def test_independent_pool_capacity_and_actual_instance_coverage(qwen_count):
    async def scenario():
        cam, emo = FakePool("speech_campplus", 2), FakePool("emotion2vec", 1)
        busy = asyncio.Event()
        finish = asyncio.Event()
        try:
            with patch.dict(settings._cfg, {"qwen_asr": {"local_instance_count": qwen_count}}):
                await cam.ensure_loaded()
                await emo.ensure_loaded()
                async def hold_emo():
                    async with emo.lease():
                        busy.set()
                        await finish.wait()
                        emo.record_inference(0, 1)
                        emo.record_outcome("success")
                task = asyncio.create_task(hold_emo())
                await busy.wait()
                # emo忙时两个CAM实例仍能同时取得，不能互借或按Qwen数复制。
                async with cam.lease():
                    async with cam.lease():
                        cam.record_inference(0, 1)
                        cam.record_outcome("empty")
                    cam.record_inference(0, 2)
                    cam.record_outcome("success")
                assert cam.configured_instances == 2 and emo.configured_instances == 1
                assert cam.builds == ["cpu", "cpu"] and emo.builds == ["cpu"]
                cam.refresh_status()
                status = models.local_model_status["speech_campplus"]
                assert status["called_instances"] == 2
                assert [row["calls"] for row in status["instance_metrics"]] == [1, 1]
                assert status["inference_outcomes"] == {"success": 1, "empty": 1, "failed": 0, "cancelled": 0}
                assert status["unclassified_calls"] == status["unattributed_calls"] == 0
                assert models.local_model_status["emotion2vec"]["busy"] == 1
                finish.set()
                await task
        finally:
            finish.set()
            if "task" in locals():
                await asyncio.gather(task, return_exceptions=True)
            await asyncio.gather(cam.close(), emo.close())
    asyncio.run(scenario())


def test_cold_loading_is_not_repeated_for_waiting_request_and_records_lock_wait():
    async def scenario():
        entered, release = threading.Event(), threading.Event()
        def build(index):
            if index == 0:
                entered.set()
                assert release.wait(5)
        cam = FakePool("speech_campplus", 2, build=build)
        emo = FakePool("emotion2vec", 1)
        second_entered = asyncio.Event()
        events = {}
        async def request(request_id):
            telemetry.begin_request(request_id, time.perf_counter())
            if request_id == "load-second":
                second_entered.set()
            try:
                await models.load_v3_models_if_needed(need_role=True)
            finally:
                telemetry.end_request(request_id)
                events[request_id] = telemetry.record_request(
                    request_id=request_id, task_id=None, filename=None, started_at=0, ended_at=1)
        first = second = None
        try:
            with patch.dict(models._MODEL_POOLS, {"speech_campplus": cam, "emotion2vec": emo}), \
                 patch.object(models, "_model_lock", asyncio.Lock()):
                first = asyncio.create_task(request("load-first"))
                await until(entered.is_set)
                second = asyncio.create_task(request("load-second"))
                await second_entered.wait()
                await asyncio.sleep(.005)
                assert not second.done()
                release.set()
                await asyncio.gather(first, second)
                assert cam.builds == ["cpu", "cpu"] and emo.builds == []
                assert events["load-first"]["phases"]["spk_load_ms"] > 0
                assert events["load-first"]["phases"]["spk_load_work_ms"] > 0
                assert events["load-second"]["phases"]["model_load_lock_wait_ms"] > 0
                assert "spk_load_ms" not in events["load-second"]["phases"]
                rows = models.local_model_status["speech_campplus"]["instance_metrics"]
                assert [row["load_attempts"] for row in rows] == [1, 1]
                assert all(row["calls"] == 0 for row in rows)
        finally:
            release.set()
            await asyncio.gather(*(task for task in (first, second) if task is not None), return_exceptions=True)
            await asyncio.gather(cam.close(), emo.close())
    asyncio.run(scenario())


async def run_pipeline(*, sample_counts=(16000,), slow=None, cancel=False, failure=None, empty=None):
    pools = {name: FakePool(name) for name in ("speech_campplus", "emotion2vec")}
    calls = {name: [] for name in pools}
    started = {name: threading.Event() for name in pools}
    nn_release = threading.Event()
    qwen_release, qwen_started = asyncio.Event(), asyncio.Event()
    qwen_calls = []
    budget = PendingSegmentBudget(16, 8 * 1024 * 1024)
    concurrency = RequestConcurrency(16, 96)
    request_id = "actual-three-branches"
    telemetry.begin_request(request_id, time.perf_counter())

    def generate(model, device, **kwargs):
        assert device == "cpu"
        calls[model.name].append(len(kwargs["input"]))
        started[model.name].set()
        if slow == model.name:
            assert nn_release.wait(5)
        if failure == model.name:
            raise RuntimeError("controlled NN failure")
        if empty == model.name:
            return [{}]
        if model.name == "speech_campplus":
            return [{"spk_embedding": torch.ones(192)}]
        return [{"labels": ["NEUTRAL"], "scores": [1.0]}]

    class Client:
        supports_batch_metadata = True
        async def transcribe_bytes(self, data, **kwargs):
            qwen_calls.append(kwargs["segment_index"])
            qwen_started.set()
            if slow is not None:
                await qwen_release.wait()
            return "language Chinese<asr_text>测试内容"

    cursor = 0
    segments = []
    for size in sample_counts:
        segments.append(VadSegment(cursor / 16000, (cursor + size) / 16000))
        cursor += size
    request = SimpleNamespace(showSpk=True, showEmotion=True, showRoleIdentify=False, language="auto")
    ctx = SimpleNamespace(request=request, audio_tensor=torch.zeros((1, cursor)), sample_rate=16000,
                          tmp_paths=[], tmp_path="unused", task_id=request_id, filename="cpu.wav",
                          audio_total_s=cursor / 16000, load_audio_time_ms=0)
    task = None
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.dict(settings._cfg, {"itn": {"enabled": False},
                                "vad": {"min_segment_duration_s": .3}}))
            stack.enter_context(patch.object(pipeline, "validate_v3_capabilities"))
            stack.enter_context(patch.object(pipeline, "load_v3_models_if_needed", AsyncMock()))
            stack.enter_context(patch.object(pipeline, "detect_segments", AsyncMock(return_value=segments)))
            stack.enter_context(patch.object(pipeline, "get_qwen_client", AsyncMock(return_value=Client())))
            stack.enter_context(patch.object(pipeline, "release_qwen_request", AsyncMock()))
            stack.enter_context(patch.object(pipeline, "get_request_concurrency", return_value=concurrency))
            stack.enter_context(patch.object(pipeline, "get_pending_budget", return_value=budget))
            stack.enter_context(patch.object(pipeline, "get_model_pool", side_effect=lambda name: pools[name]))
            stack.enter_context(patch.object(pipeline, "generate_local", side_effect=generate))
            stack.enter_context(patch.object(pipeline, "update_stat"))
            task = asyncio.create_task(pipeline.run_qwen_asr(ctx))
            if slow is not None:
                # Qwen故意不返回；两个NN仍必须已进入，证明不是Qwen完成后串行分发。
                await until(lambda: qwen_started.is_set() and all(event.is_set() for event in started.values()))
                assert concurrency._global._value == 95
                qwen_release.set()
                await until(lambda: concurrency._global._value == 96)
                assert budget.snapshot()["active"] == 1 and not task.done()
                if cancel:
                    task.cancel()
                    await asyncio.sleep(0)
                    task.cancel()
                    await asyncio.sleep(0)
                    assert not task.done() and budget.snapshot()["active"] == 1
                nn_release.set()
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    await task
                result = None
            else:
                result = await task
            assert budget.snapshot()["active"] == budget.snapshot()["audio_bytes"] == 0
            assert concurrency._global._value == 96
            assert all(not lock.locked() for pool in pools.values() for lock in pool.locks)
            return result, calls, telemetry.enhancement_phases(), {
                name: deepcopy(models.local_model_status[name]) for name in pools}, qwen_calls
    finally:
        nn_release.set()
        qwen_release.set()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await asyncio.gather(*(pool.close() for pool in pools.values()))
        telemetry.end_request(request_id)


@pytest.mark.parametrize("slow", ["speech_campplus", "emotion2vec"])
@pytest.mark.parametrize("cancel", [False, True])
def test_three_tasks_overlap_qwen_credit_and_pcm_have_independent_terminals(slow, cancel):
    result, calls, phases, status, _ = asyncio.run(run_pipeline(slow=slow, cancel=cancel))
    assert calls == {"speech_campplus": [16000], "emotion2vec": [16000]}
    assert phases["spk_nn_calls"] == phases["emo_nn_calls"] == 1
    prefix = "spk" if slow == "speech_campplus" else "emo"
    assert phases[f"{prefix}_nn_cancelled" if cancel else f"{prefix}_nn_success"] == 1
    assert all(row["called_instances"] == 1 for row in status.values())
    if not cancel:
        assert result["segments"][0]["role"] == "spk_0"
        assert result["segments"][0]["emotion"] == "平淡"


def test_filter_and_emo_closed_sample_boundaries_and_skip_accounting():
    sizes = (4799, 4800, 4801, 15999, 16000, 16001, 160000, 160001)
    result, calls, phases, status, qwen_calls = asyncio.run(run_pipeline(sample_counts=sizes))
    assert len(result["segments"]) == 7 and sorted(qwen_calls) == list(range(7))
    assert Counter(calls["speech_campplus"]) == Counter(sizes[1:])
    assert Counter(calls["emotion2vec"]) == Counter([16000, 16001, 160000])
    assert phases["spk_nn_success"] == phases["spk_nn_calls"] == 7
    assert phases["emo_nn_success"] == phases["emo_nn_calls"] == 3
    assert phases["emo_duration_skips"] == 4
    assert status["emotion2vec"]["calls"] == 3
    assert status["emotion2vec"]["duration_skips"] == 4
    assert result["segments"][0]["bg"] == f"{4799 / 16000:.2f}"


def test_duration_skip_does_not_load_or_lease_any_model_instance():
    async def scenario():
        pool = FakePool("emotion2vec", 2)
        telemetry.begin_request("skip-only", 0)
        try:
            with patch.object(pipeline, "get_model_pool", return_value=pool), \
                 patch.object(pipeline, "generate_local", side_effect=AssertionError("skip cannot invoke NN")):
                assert await pipeline._segment_emotion(torch.zeros(4800).numpy(), 16000, .3) == "平淡"
            assert pool.builds == pool.models == []
            assert models.local_model_status["emotion2vec"]["configured_instances"] == 2
            assert models.local_model_status["emotion2vec"]["calls"] == 0
            assert models.local_model_status["emotion2vec"]["duration_skips"] == 1
            assert telemetry.enhancement_phases() == {"emo_duration_skips": 1}
        finally:
            telemetry.end_request("skip-only")
            await pool.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("name", ["speech_campplus", "emotion2vec"])
@pytest.mark.parametrize("outcome", ["failed", "empty"])
def test_nn_failure_or_empty_is_not_success_and_keeps_other_branch(name, outcome):
    result, _, phases, status, _ = asyncio.run(run_pipeline(**{
        "failure" if outcome == "failed" else "empty": name}))
    prefix = "spk" if name == "speech_campplus" else "emo"
    peer = "emo" if prefix == "spk" else "spk"
    assert phases[f"{prefix}_nn_calls"] == phases[f"{prefix}_nn_{outcome}"] == 1
    assert phases.get(f"{prefix}_nn_success", 0) == 0
    assert phases[f"{peer}_nn_success"] == 1
    assert status[name]["inference_outcomes"][outcome] == 1
    assert result["segments"][0]["role" if prefix == "spk" else "emotion"] is None


def test_request_enhancement_counts_remain_bounded_isolated_and_survive_end_request():
    async def scenario():
        async def request(request_id, count):
            telemetry.begin_request(request_id, 0)
            telemetry.record_enhancement_phase("emo_duration_skips", count)
            telemetry.record_enhancement_phase("emo_load_ms", 2)
            telemetry.record_enhancement_phase("arbitrary_segment_body", 999)
            telemetry.record_enhancement_phase("emo_nn_success", float("nan"))
            await asyncio.sleep(0)
            telemetry.end_request(request_id)
            return telemetry.record_request(request_id=request_id, task_id=None, filename=None,
                                            started_at=0, ended_at=1, phases={"qwen_ms": 1})
        first, second = await asyncio.gather(request("one", 1), request("two", 2))
        assert first["phases"]["emo_duration_skips"] == 1
        assert second["phases"]["emo_duration_skips"] == 2
        assert "arbitrary_segment_body" not in first["phases"]
        assert "emo_nn_success" not in first["phases"]
        summary = telemetry.snapshot()["enhancements"]["emo"]
        assert summary["counts"]["duration_skips"] == 3
        assert summary["counts"]["nn_calls"] == summary["counts"]["nn_success"] == 0
        assert summary["requests_observed"] == 2
    asyncio.run(scenario())


def test_recent_stage_window_is_bounded_and_never_contains_text_or_arbitrary_dimensions():
    for index in range(12):
        telemetry.begin_request(f"round-{index}", index)
        telemetry.record_enhancement_phase("spk_nn_calls", 1)
        telemetry.end_request(f"round-{index}")
        telemetry.record_request(request_id=f"round-{index}", task_id=None,
            filename=f"round-{index}.wav", started_at=index, ended_at=index + 1,
            phases={"itn_ms": 2, "private_body_dimension": 999, "qwen_ms": float("inf")},
            error="transcript must not be present in stage window")
    value = telemetry.snapshot(include_events=False)
    assert "recent" not in value
    rows = value["recent_stages"]
    assert len(rows) == 8
    assert [row["request_id"] for row in rows] == [f"round-{index}" for index in range(4, 12)]
    assert all(row["phases"] == {"itn_ms": 2.0, "spk_nn_calls": 1.0} for row in rows)
    assert all(set(row) == {"request_id", "filename", "started_monotonic", "ended_monotonic", "status", "phases"} for row in rows)
