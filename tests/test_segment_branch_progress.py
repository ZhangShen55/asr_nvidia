"""CPU-only branch-terminal telemetry; no real models, device backend or service."""
import asyncio
from collections import Counter
from contextlib import ExitStack, asynccontextmanager
from types import SimpleNamespace
import threading
import time
from unittest.mock import AsyncMock, patch

import pytest
import torch

from api.routes import asr_qwen_common as pipeline
from core.config import settings
from core.qwen_asr_queue import RequestConcurrency
from core.request_telemetry import active_snapshot, begin_request, end_request
from core.segment_pipeline import PendingSegmentBudget
from core.vad import VadSegment


def progress(request_id):
    rows = active_snapshot()['requests']
    return next(row['progress'] for row in rows if row['request_id'] == request_id)


async def until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(.001)
    await asyncio.wait_for(wait(), 5)


class FakePool:
    def __init__(self, name):
        self.name = name
        self.busy = 0
        self.timings = []

    @asynccontextmanager
    async def lease(self):
        self.busy += 1
        try:
            yield SimpleNamespace(name=self.name), 'cpu'
        finally:
            self.busy -= 1

    def record_inference(self, wait_ms, infer_ms):
        self.timings.append((wait_ms, infer_ms))


async def run_case(*, slow='speech_campplus', cancel=False, duration=1.0, failure=None):
    request_id = f'branch-{slow}-{cancel}-{duration}-{failure}'
    begin_request(request_id, time.perf_counter())
    released = threading.Event()
    started = {name:threading.Event() for name in ('speech_campplus', 'emotion2vec')}
    done = {name:threading.Event() for name in started}
    calls = Counter()
    pools = {name:FakePool(name) for name in started}
    budget = PendingSegmentBudget(16, 4*1024*1024)
    concurrency = RequestConcurrency(16, 128)
    stats = {}

    def generate(model, device, **kwargs):
        assert device == 'cpu'
        name = model.name
        calls[name] += 1
        started[name].set()
        if name == slow:
            assert released.wait(5), 'CPU test gate was not released'
        done[name].set()
        if name == failure:
            raise RuntimeError('controlled '+name+' failure')
        return [{'spk_embedding':torch.ones(192)}] if name == 'speech_campplus' else [{'labels':['NEUTRAL'], 'scores':[1.0]}]

    class Client:
        supports_batch_metadata = True
        async def transcribe_bytes(self, data, **kwargs):
            return 'language Chinese<asr_text>测试内容'

    request = SimpleNamespace(showSpk=True, showEmotion=True, showRoleIdentify=False, language='auto')
    ctx = SimpleNamespace(request=request, audio_tensor=torch.zeros((1, round(duration*16000))), sample_rate=16000,
        tmp_paths=[], tmp_path='unused', task_id=request_id, filename='cpu.wav', audio_total_s=duration, load_audio_time_ms=0)
    async def invoke():
        try:
            return await pipeline.run_qwen_asr(ctx)
        finally:
            stats.update(pipeline.get_phase_stats())
    task = None
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.dict(settings._cfg, {'itn':{'enabled':False}}))
            stack.enter_context(patch.object(pipeline, 'validate_v3_capabilities'))
            stack.enter_context(patch.object(pipeline, 'load_v3_models_if_needed', AsyncMock()))
            stack.enter_context(patch.object(pipeline, 'detect_segments', AsyncMock(return_value=[VadSegment(0, duration)])))
            stack.enter_context(patch.object(pipeline, 'get_qwen_client', AsyncMock(return_value=Client())))
            stack.enter_context(patch.object(pipeline, 'get_request_concurrency', return_value=concurrency))
            stack.enter_context(patch.object(pipeline, 'get_pending_budget', return_value=budget))
            stack.enter_context(patch.object(pipeline, 'get_model_pool', side_effect=lambda name:pools[name]))
            stack.enter_context(patch.object(pipeline, 'generate_local', side_effect=generate))
            stack.enter_context(patch.object(pipeline, 'update_stat'))
            task = asyncio.create_task(invoke())
            if slow is not None:
                fast_field = 'emo_done' if slow == 'speech_campplus' else 'spk_done'
                slow_field = 'spk_done' if slow == 'speech_campplus' else 'emo_done'
                await until(lambda:started[slow].is_set() and progress(request_id).get(fast_field) == 1
                            and progress(request_id).get('qwen_done') == 1)
                assert progress(request_id).get(slow_field, 0) == 0
                assert progress(request_id).get('segments_done', 0) == 0
                assert not task.done() and not done[slow].is_set()
                assert budget.snapshot()['active'] == 1 and pools[slow].busy == 1
                assert concurrency._global._value == 128
                if cancel:
                    task.cancel()
                    await asyncio.sleep(0)
                    task.cancel()
                    await asyncio.sleep(0)
                    assert not task.done()
                    assert progress(request_id).get(slow_field, 0) == 0
                    assert progress(request_id).get('segments_done', 0) == 0
                    assert budget.snapshot()['active'] == 1 and pools[slow].busy == 1
                released.set()
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    await task
                result = None
            else:
                result = await task
                assert len(result['segments']) == 1
            completed = progress(request_id)
            assert {key:completed.get(key, 0) for key in ('qwen_done', 'spk_done', 'emo_done', 'segments_done')} == {
                'qwen_done':1, 'spk_done':1, 'emo_done':1, 'segments_done':1}
            assert budget.snapshot()['active'] == budget.snapshot()['audio_bytes'] == 0
            assert concurrency._global._value == 128
            assert all(pool.busy == 0 for pool in pools.values())
            assert stats['qwen_calls'] == stats['spk_calls'] == 1
            assert stats['emo_calls'] == (0 if duration < 1 else 1)
            assert calls['speech_campplus'] == 1 and calls['emotion2vec'] == (0 if duration < 1 else 1)
            assert len(pools['speech_campplus'].timings) == 1
            assert len(pools['emotion2vec'].timings) == (0 if duration < 1 else 1)
            if duration < 1:
                assert result['segments'][0]['emotion'] == '平淡'
    finally:
        released.set()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        end_request(request_id)
        assert not any(row['request_id'] == request_id for row in active_snapshot()['requests'])


@pytest.mark.parametrize('slow', ['speech_campplus', 'emotion2vec'])
def test_fast_branch_finishes_before_slow_peer_join_and_qwen_releases(slow):
    asyncio.run(run_case(slow=slow))


@pytest.mark.parametrize('slow', ['speech_campplus', 'emotion2vec'])
def test_repeated_cancel_counts_started_thread_only_after_actual_done(slow):
    asyncio.run(run_case(slow=slow, cancel=True))


@pytest.mark.parametrize('failure', ['speech_campplus', 'emotion2vec'])
def test_branch_failure_counts_terminal_once_without_changing_calls(failure):
    asyncio.run(run_case(slow=None, failure=failure))


def test_emotion_duration_skip_counts_terminal_without_model_call_or_join_wait():
    asyncio.run(run_case(duration=.5))


def test_task_cancelled_before_wrapper_entry_does_not_count_or_create_model_coroutine():
    async def scenario():
        request_id = 'never-entered'
        begin_request(request_id, time.perf_counter())
        branch = AsyncMock()
        try:
            task = asyncio.create_task(pipeline._segment_branch_with_progress(branch, completion_field='spk_done'))
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert branch.await_count == 0
            assert progress(request_id).get('spk_done', 0) == 0
        finally:
            end_request(request_id)
    asyncio.run(scenario())


def test_entered_branch_cancelled_while_waiting_pool_counts_terminal_not_inference():
    async def scenario():
        request_id = 'entered-pool-wait'
        begin_request(request_id, time.perf_counter())
        stats = pipeline.reset_phase_stats()
        entered = asyncio.Event()
        never = asyncio.Event()
        @asynccontextmanager
        async def lease():
            entered.set()
            await never.wait()
            yield object(), 'cpu'
        try:
            with patch.object(pipeline, 'validate_v3_capabilities'), \
                 patch.object(pipeline, 'get_model_pool', return_value=SimpleNamespace(lease=lease)), \
                 patch.object(pipeline, 'generate_local', side_effect=AssertionError('no lease, no inference')) as generate:
                task = asyncio.create_task(pipeline._segment_branch_with_progress(
                    pipeline._speaker_embedding, [], 16000, completion_field='spk_done'))
                await entered.wait()
                assert progress(request_id).get('spk_done', 0) == 0
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert progress(request_id).get('spk_done', 0) == 1
                assert progress(request_id).get('segments_done', 0) == 0
                assert stats['spk_calls'] == 0 and generate.call_count == 0
        finally:
            end_request(request_id)
    asyncio.run(scenario())
