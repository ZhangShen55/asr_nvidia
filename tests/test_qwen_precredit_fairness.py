"""CPU regression for precut Qwen credits; no model or NPU execution."""
import asyncio
from collections import Counter
from contextlib import ExitStack
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import pytest
import torch
from api.routes import asr_qwen_common as candidate
from core.config import settings
from core.qwen_asr_queue import RequestConcurrency
from core.segment_pipeline import PendingSegmentBudget
from core.vad import VadSegment
from core.async_utils import await_completion
OWNER = ContextVar('fairness_request', default='unknown')

async def until(predicate):

    async def loop():
        while not predicate():
            await asyncio.sleep(0.001)
    await asyncio.wait_for(loop(), 10)

async def scenario(module):
    qwen_release = asyncio.Event()
    enhanced_release = asyncio.Event()
    offered = Counter()
    done = Counter()
    lease_owners = {}
    vad_ready = set()

    class Budget(PendingSegmentBudget):

        async def acquire(self, size):
            lease = await super().acquire(size)
            lease_owners[lease] = OWNER.get()
            return lease

        async def release(self, lease):
            await super().release(lease)
            lease_owners.pop(lease, None)
    budget = Budget(256, 32 * 1024 * 1024)
    concurrency = RequestConcurrency(16, 128)

    class Client:
        supports_batch_metadata = True

        async def transcribe_bytes(self, data, **kwargs):
            owner = kwargs['request_id']
            offered[owner] += 1
            await qwen_release.wait()
            done[owner] += 1
            return 'language Chinese<asr_text>测试内容'

    async def speaker(*args):
        await await_completion(enhanced_release.wait())
        return torch.ones(192)

    async def detect(*args, **kwargs):
        vad_ready.add(OWNER.get())
        return [VadSegment(0, 1)] * 256

    def ctx(rid):
        return SimpleNamespace(request=SimpleNamespace(showSpk=True, showEmotion=False, showRoleIdentify=False, language='auto'), audio_tensor=torch.zeros((1, 16000)), sample_rate=16000, tmp_paths=[], tmp_path='unused', task_id=rid, filename=rid + '.wav', audio_total_s=256, load_audio_time_ms=0)

    async def request(rid):
        token = OWNER.set(rid)
        try:
            return await module.run_qwen_asr(ctx(rid))
        finally:
            OWNER.reset(token)

    def snapshot():
        return {'pending_by_request': dict(Counter(lease_owners.values())), 'qwen_offered_by_request': dict(offered), 'qwen_done_by_request': dict(done), 'global_qwen_credit_available': concurrency._global._value, 'pending': budget.snapshot(), 'vad_ready': sorted(vad_ready)}
    tasks = []
    with ExitStack() as stack:
        stack.enter_context(patch.dict(settings._cfg, {'itn': {'enabled': False}, 'qwen_asr': {'pending_segment_limit': 256, 'pending_audio_bytes_limit': 32 * 1024 * 1024}}))
        for key, value in [('get_pending_budget', lambda: budget), ('validate_v3_capabilities', lambda *a: None), ('load_v3_models_if_needed', AsyncMock()), ('detect_segments', detect), ('get_qwen_client', AsyncMock(return_value=Client())), ('get_request_concurrency', lambda: concurrency), ('_speaker_embedding', speaker), ('update_stat', lambda *a, **k: None)]:
            stack.enter_context(patch.object(module, key, value))
        tasks.append(asyncio.create_task(request('r0')))
        try:
            await until(lambda: offered['r0'] == 16 and budget.snapshot()['active'] == 16)
            first = snapshot()
            tasks.extend((asyncio.create_task(request('r' + str(i))) for i in range(1, 8)))
            await until(lambda: len(vad_ready) == 8 and sum(offered.values()) == 128)
            staggered = snapshot()
            qwen_release.set()
            await until(lambda: sum(done.values()) == 256 and budget.snapshot()['active'] == 256)
            for _ in range(20):
                await asyncio.sleep(0)
            after_qwen = snapshot()
            return {'first_only': first, 'eight_vad_ready_qwen_blocked': staggered, 'qwen_released_cam_still_slow': after_qwen}
        finally:
            for task in tasks:
                task.cancel()
            qwen_release.set()
            enhanced_release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            assert budget.snapshot()['active'] == budget.snapshot()['audio_bytes'] == 0
            assert concurrency._global._value == 128

def test_staggered_eight_request_current_api_fills_128_before_true_pcm_backpressure():

    async def check():
        proposed = await scenario(candidate)
        first = proposed['first_only']
        assert first['pending_by_request'] == {'r0': 16}
        ready = proposed['eight_vad_ready_qwen_blocked']
        assert ready['pending_by_request'] == {f'r{i}': 16 for i in range(8)}
        assert ready['qwen_offered_by_request'] == {f'r{i}': 16 for i in range(8)}
        full = proposed['qwen_released_cam_still_slow']
        assert full['pending']['active'] == 256
        assert all((full['qwen_done_by_request'].get(f'r{i}', 0) >= 16 for i in range(8)))
    asyncio.run(check())

@pytest.mark.parametrize('failure', ['client_error', 'client_cancel', 'parse_error'])
def test_precredit_releases_once_on_client_discovery_cancel_or_parse(failure):

    async def check():
        concurrency = RequestConcurrency(16, 128)
        sem = concurrency.new_segment_semaphore()
        await concurrency.acquire(sem)
        credit = candidate._QwenCredit(concurrency, sem)
        entered = asyncio.Event()

        async def client():
            entered.set()
            if failure == 'client_error':
                raise RuntimeError('getclient failed')
            if failure == 'client_cancel':
                await asyncio.Future()
            return SimpleNamespace(transcribe_bytes=AsyncMock(return_value='raw'))
        with patch.object(candidate, 'get_qwen_client', client), patch.object(candidate, 'parse_qwen_text', side_effect=ValueError('parse failed')):
            task = asyncio.create_task(candidate._transcribe_segment(b'wav', 'auto', concurrency, segment_semaphore=sem, qwen_credit=credit))
            await entered.wait()
            if failure == 'client_cancel':
                task.cancel()
            with pytest.raises((RuntimeError, ValueError, asyncio.CancelledError)):
                await task
        await credit.release()
        await credit.release()
        assert concurrency._global._value == 128 and sem._value == 16
    asyncio.run(check())

@pytest.mark.parametrize('early_exit', ['cut_error', 'actual_eof_short', 'cut_cancel'])
def test_precut_credit_and_pending_released_before_any_qwen_branch(early_exit):

    async def check():
        concurrency = RequestConcurrency(16, 128)
        budget = PendingSegmentBudget(256, 32 * 1024 * 1024)
        started = asyncio.Event()
        never = asyncio.Event()
        ctx = SimpleNamespace(request=SimpleNamespace(showSpk=False, showEmotion=False, showRoleIdentify=False, language='auto'), audio_tensor=torch.zeros((1, 4000 if early_exit == 'actual_eof_short' else 16000)), sample_rate=16000, tmp_paths=[], tmp_path='unused', task_id='early', filename='early.wav', audio_total_s=1, load_audio_time_ms=0)
        original = candidate.run_in_thread

        async def delayed(function, *args, **kwargs):
            started.set()
            await never.wait()
            return await original(function, *args, **kwargs)
        client = AsyncMock(side_effect=AssertionError('early exit must not start Qwen'))
        with ExitStack() as stack:
            for key, value in [('get_pending_budget', lambda: budget), ('validate_v3_capabilities', lambda *a: None), ('load_v3_models_if_needed', AsyncMock()), ('detect_segments', AsyncMock(return_value=[VadSegment(0, 1)])), ('get_qwen_client', client), ('get_request_concurrency', lambda: concurrency), ('update_stat', lambda *a, **kw: None)]:
                stack.enter_context(patch.object(candidate, key, value))
            if early_exit == 'cut_error':
                stack.enter_context(patch.object(candidate, 'encode_audio_clip_wav', side_effect=RuntimeError('cut failed')))
            if early_exit == 'cut_cancel':
                stack.enter_context(patch.object(candidate, 'run_in_thread', delayed))
            task = asyncio.create_task(candidate.run_qwen_asr(ctx))
            if early_exit == 'cut_cancel':
                await started.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                await task
        assert client.await_count == 0
        assert budget.snapshot()['active'] == budget.snapshot()['audio_bytes'] == 0
        assert concurrency._global._value == 128
    asyncio.run(check())
