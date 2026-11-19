import asyncio
from contextlib import ExitStack, asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import torch

from api.routes import asr_qwen_common as pipeline
from core.async_utils import await_completion
from core.config import Settings, settings
from core.qwen_asr_queue import RequestConcurrency
from core.segment_pipeline import PendingSegmentBudget, gather_owned_branches, run_bounded_segments
from core.request_telemetry import begin_request, end_request, update_active, active_snapshot
from core.vad import VadSegment, filter_short_segments


def test_pending_budget_fifo_cancel_bytes_and_idempotent_release():
    async def scenario():
        budget = PendingSegmentBudget(2, 100)
        first = await budget.acquire(80)
        blocked = asyncio.create_task(budget.acquire(30))
        later = asyncio.create_task(budget.acquire(10))
        await asyncio.sleep(0)
        assert not blocked.done() and not later.done()
        blocked.cancel()
        with pytest.raises(asyncio.CancelledError):
            await blocked
        second = await later
        assert budget.snapshot()['active'] == 2
        assert budget.snapshot()['audio_bytes'] == 90
        await budget.release(first)
        await budget.release(first)
        await budget.release(second)
        assert budget.snapshot()['active'] == budget.snapshot()['audio_bytes'] == 0
        with pytest.raises(ValueError):
            await budget.acquire(101)
    asyncio.run(scenario())


def test_thousands_of_segments_have_bounded_workers_and_stable_order():
    async def scenario():
        active = peak = 0
        async def process(index, value):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1
            return (index, value)
        result = await run_bounded_segments(list(range(3000)), process, 7)
        assert peak == 7
        assert result == list(enumerate(range(3000)))
    asyncio.run(scenario())


def test_cancel_waits_all_owned_branches_before_pending_release():
    async def scenario():
        started = asyncio.Event()
        finish = asyncio.Event()
        stopped = []
        async def busy():
            started.set()
            try:
                await await_completion(finish.wait())
            finally:
                stopped.append(True)
        task = asyncio.create_task(gather_owned_branches([busy(), asyncio.sleep(30)]))
        await started.wait()
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        assert not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped == [True]
    asyncio.run(scenario())


def test_short_sample_boundaries_and_embeddings_enter_clustering():
    segments = [VadSegment(2.0, 2.299), VadSegment(2.0, 2.3), VadSegment(2.0, 2.301)]
    assert filter_short_segments(segments, 0.3) == segments[1:]
    rows = [pipeline.SegmentOutcome(0, segments[1], embedding=torch.tensor([1.0, 0.0])),
            pipeline.SegmentOutcome(1, VadSegment(3.0, 4.0), embedding=torch.tensor([0.0, 1.0]))]
    labels = pipeline.cluster_speakers(rows, threshold=0.3)
    assert labels[0] != labels[1]


def test_short_segments_never_enter_any_model_or_response():
    from tests.test_v3_contract_snapshot import run_contract
    async def scenario():
        result, _, speaker, emotion, calls = await run_contract(
            flags={'showSpk':True,'showEmotion':True},
            segments=[VadSegment(0.0,0.299),VadSegment(0.3,0.6)])
        assert len(result['segments']) == 1
        assert result['segments'][0]['bg'] == '0.30'
        assert result['segments'][0]['ed'] == '0.60'
        assert len(calls) == speaker.await_count == emotion.await_count == 1
    asyncio.run(scenario())


def test_emo_one_and_ten_second_sample_boundaries():
    @asynccontextmanager
    async def lease():
        yield object(), 'cpu'
    pool = SimpleNamespace(lease=lease)
    async def scenario():
        with patch.object(pipeline, 'validate_v3_capabilities'), patch.object(pipeline, 'get_model_pool', return_value=pool), \
             patch.object(pipeline, 'generate_local', return_value=[{'labels':['NEUTRAL'], 'scores':[1.0]}]) as infer:
            for ms in (299, 300, 301, 999, 1000, 1001, 9999, 10000, 10001):
                before = infer.call_count
                await pipeline._segment_emotion(torch.zeros(ms * 16).numpy(), 16000, ms / 1000)
                assert infer.call_count-before == int(1000 <= ms <= 10000)
    asyncio.run(scenario())


def test_slow_cam_does_not_hold_completed_qwen_credit():
    async def scenario():
        gate = asyncio.Event()
        qwen_calls = []
        class Client:
            supports_batch_metadata = True
            async def transcribe_bytes(self, data, **kwargs):
                qwen_calls.append(kwargs['segment_index'])
                await asyncio.sleep(0)
                return 'language Chinese<asr_text>测试内容'
        async def speaker(*args):
            await gate.wait()
            return torch.ones(192)
        original_run_in_thread = pipeline.run_in_thread
        async def delayed_dispatch(function, *args, **kwargs):
            # 调度等待故意比纯切片长，验证cut_cpu不再把线程队列算成操作时间。
            await asyncio.sleep(0.005)
            return await original_run_in_thread(function, *args, **kwargs)
        request = SimpleNamespace(showSpk=True, showEmotion=False, showRoleIdentify=False, language='auto')
        ctx = SimpleNamespace(request=request, audio_tensor=torch.zeros((1,16000*12)), sample_rate=16000,
                              tmp_paths=[], tmp_path='unused', task_id='slow-cam', filename='fake.wav',
                              audio_total_s=12, load_audio_time_ms=0)
        concurrency = RequestConcurrency(2, 2)
        with ExitStack() as stack:
            stack.enter_context(patch.dict(settings._cfg, {'itn':{'enabled':False},
                 'qwen_asr':{'pending_segment_limit':16,'pending_audio_bytes_limit':2097152}}))
            stack.enter_context(patch.object(pipeline,'validate_v3_capabilities'))
            stack.enter_context(patch.object(pipeline,'load_v3_models_if_needed',AsyncMock()))
            stack.enter_context(patch.object(pipeline,'detect_segments',AsyncMock(return_value=[VadSegment(i,i+1) for i in range(12)])))
            stack.enter_context(patch.object(pipeline,'get_qwen_client',AsyncMock(return_value=Client())))
            stack.enter_context(patch.object(pipeline,'get_request_concurrency',return_value=concurrency))
            stack.enter_context(patch.object(pipeline,'_speaker_embedding',speaker))
            stack.enter_context(patch.object(pipeline,'run_in_thread',delayed_dispatch))
            stack.enter_context(patch.object(pipeline,'update_stat'))
            async def run_with_stats():
                result = await pipeline.run_qwen_asr(ctx)
                return result, pipeline.get_phase_stats()
            task = asyncio.create_task(run_with_stats())
            for _ in range(1000):
                if len(qwen_calls) == 12 and concurrency._global._value == 2:
                    break
                await asyncio.sleep(0.001)
            assert len(qwen_calls) == 12
            assert concurrency._global._value == 2
            assert not task.done()
            gate.set()
            result, phases = await task
            assert len(result['segments']) == 12
            assert sorted(qwen_calls) == list(range(12))
            assert phases['cut_dispatch_wait_ms'] >= 50
            assert phases['cut_cpu_ms'] < phases['cut_sum_ms']
    asyncio.run(scenario())


def test_active_progress_contains_only_bounded_numeric_metadata():
    async def scenario():
        begin_request('test-active', 0.0)
        try:
            update_active('segments', segments_total=3, audio_body='must-not-be-stored')
            update_active(increment='qwen_done')
            entry = next(r for r in active_snapshot()['requests'] if r['request_id']=='test-active')
            assert entry['stage'] == 'segments'
            assert entry['progress'] == {'segments_total':3, 'qwen_done':1}
        finally:
            end_request('test-active')
    asyncio.run(scenario())


@pytest.mark.parametrize('cancel_request', [False, True])
def test_slow_enhanced_branch_only_backpressures_at_global_pending_budget(cancel_request):
    """300段使旧32个join worker先卡住；PCM预算256才是允许的反压点。"""
    async def scenario():
        enhanced_release = asyncio.Event()
        qwen_calls = []
        budget = PendingSegmentBudget(256, 32 * 1024 * 1024)
        concurrency = RequestConcurrency(16, 128)
        class Client:
            supports_batch_metadata = True
            async def transcribe_bytes(self, data, **kwargs):
                qwen_calls.append(kwargs['segment_index'])
                await asyncio.sleep(0)
                return 'language Chinese<asr_text>测试内容'
        async def speaker(*args):
            # 模拟已启动的设备工作，取消也必须等真实工作收尾。
            await await_completion(enhanced_release.wait())
            return torch.ones(192)
        request = SimpleNamespace(showSpk=True, showEmotion=False, showRoleIdentify=False, language='auto')
        ctx = SimpleNamespace(request=request, audio_tensor=torch.zeros((1,16000)), sample_rate=16000,
                              tmp_paths=[], tmp_path='unused', task_id='pending-300', filename='fake.wav',
                              audio_total_s=300, load_audio_time_ms=0)
        with ExitStack() as stack:
            stack.enter_context(patch.dict(settings._cfg, {'itn':{'enabled':False},
                 'qwen_asr':{'pending_segment_limit':256,'pending_audio_bytes_limit':32*1024*1024}}))
            stack.enter_context(patch.object(pipeline,'get_pending_budget',return_value=budget))
            stack.enter_context(patch.object(pipeline,'validate_v3_capabilities'))
            stack.enter_context(patch.object(pipeline,'load_v3_models_if_needed',AsyncMock()))
            stack.enter_context(patch.object(pipeline,'detect_segments',AsyncMock(return_value=[VadSegment(0,1)]*300)))
            stack.enter_context(patch.object(pipeline,'get_qwen_client',AsyncMock(return_value=Client())))
            stack.enter_context(patch.object(pipeline,'get_request_concurrency',return_value=concurrency))
            stack.enter_context(patch.object(pipeline,'_speaker_embedding',speaker))
            stack.enter_context(patch.object(pipeline,'update_stat'))
            task = asyncio.create_task(pipeline.run_qwen_asr(ctx))
            try:
                async def reached_budget():
                    while len(qwen_calls) < 256:
                        await asyncio.sleep(.001)
                await asyncio.wait_for(reached_budget(), 5)
                await asyncio.sleep(.02)
                assert len(qwen_calls) == 256
                assert budget.snapshot()['active'] == 256
                assert concurrency._global._value == 128
                if cancel_request:
                    task.cancel()
                    await asyncio.sleep(.01)
                    task.cancel()
                    await asyncio.sleep(.01)
                    assert not task.done()
                enhanced_release.set()
                if cancel_request:
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    assert len(qwen_calls) == 256
                else:
                    await task
                    assert sorted(qwen_calls) == list(range(300))
                assert budget.snapshot()['active'] == budget.snapshot()['audio_bytes'] == 0
                assert concurrency._global._value == 128
            finally:
                enhanced_release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    asyncio.run(scenario())


def test_failed_outcomes_do_not_retain_pcm_or_exception_chain_after_budget_release():
    import gc
    import weakref
    from core.qwen_asr_client import QwenAsrCallError
    class Payload:
        def __init__(self):
            self.pcm = bytearray(131072)
    async def scenario():
        budget = PendingSegmentBudget(2, 262144)
        refs = []
        async def fail(index, _):
            lease = await budget.acquire(131072)
            try:
                payload = Payload()
                refs.append(weakref.ref(payload))
                await asyncio.sleep(0)
                try:
                    raise ValueError('inner failure')
                except ValueError as cause:
                    raise QwenAsrCallError('retained error contract') from cause
            finally:
                await budget.release(lease)
        outcomes = await run_bounded_segments(list(range(100)), fail, 2)
        gc.collect()
        assert all(ref() is None for ref in refs)
        assert budget.snapshot()['active'] == budget.snapshot()['audio_bytes'] == 0
        assert len(outcomes) == 100
        assert all(isinstance(error, QwenAsrCallError) and error.code == 4012
                   and error.message == 'retained error contract' for error in outcomes)
        assert all(error.__traceback__ is error.__cause__ is error.__context__ is None for error in outcomes)
    asyncio.run(scenario())


def test_eof_short_clip_skips_all_models_and_response_after_actual_cut():
    from tests.test_v3_contract_snapshot import run_contract
    async def scenario():
        result, _, speaker, emotion, calls = await run_contract(
            flags={'showSpk':True,'showEmotion':True},
            segments=[VadSegment(0,1),VadSegment(2.8,3.2)])
        assert len(result['segments']) == 1
        assert sorted(calls) == [(0,'auto')]
        assert speaker.await_count == emotion.await_count == 1
        result, _, speaker, emotion, calls = await run_contract(
            flags={'showSpk':True,'showEmotion':True}, segments=[VadSegment(2.8,3.2)])
        assert result['code'] == 4008
        assert not calls and not speaker.await_count and not emotion.await_count
    asyncio.run(scenario())


def test_emotion_duration_uses_actual_pcm_samples():
    @asynccontextmanager
    async def lease():
        yield object(), 'cpu'
    pool = SimpleNamespace(lease=lease)
    async def scenario():
        with patch.object(pipeline, 'validate_v3_capabilities'), patch.object(pipeline, 'get_model_pool', return_value=pool), \
             patch.object(pipeline, 'generate_local', return_value=[{'labels':['NEUTRAL'], 'scores':[1.0]}]) as infer:
            for samples, duration, expected in [(15999,1.0,0),(16000,.999,1),(160000,10.001,1),(160001,10.0,0)]:
                before = infer.call_count
                assert await pipeline._segment_emotion(torch.zeros(samples).numpy(),16000,duration) == '平淡'
                assert infer.call_count-before == expected
    asyncio.run(scenario())
