"""真实v3路由的owner接线合约；所有模型/音频/统计IO均为局部替身。"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from api.routes import asr_v3 as route
from core.models import ModelUnavailableError
from core.qwen_asr_client import QwenAsrUnavailableError
from entity.data import AsrRequestParams, get_asr_params


def request():
    return AsrRequestParams(audioFile=SimpleNamespace(filename='arbitrary-new-file.wav'),
                            wordTimestamps=False, showSpk=True, showEmotion=True)


@pytest.fixture
def wired(monkeypatch):
    events = []
    records = []
    ctx = SimpleNamespace(task_id='display-task-only', filename='arbitrary-new-file.wav',
                          tmp_paths=['private-owned.wav'], request_id='')
    lease = SimpleNamespace(snapshot={'active': 1}, request_id='request-42')

    class Client:
        async def ensure_owner_pool(self): events.append(('ensure',))

    client = Client()

    class Slots:
        def snapshot(self): return {'active': 0}
        def bind_owner_pool(self, value):
            assert value is client
            events.append(('bind',))
        async def acquire(self, category, **kwargs):
            events.append(('acquire', category, kwargs))
            return lease
        async def release(self, value):
            assert value is lease
            events.append(('release',))

    slots = Slots()
    settings = SimpleNamespace(qwen_asr_enabled=True, qwen_generation_mode='continuous', qwen_retry_count=2)
    monkeypatch.setattr(route, 'settings', settings)
    monkeypatch.setattr(route, 'get_slot_manager', lambda: slots)
    monkeypatch.setattr(route, 'new_request_id', lambda: 'request-42')
    async def get_client():
        events.append(('client',))
        return client
    monkeypatch.setattr(route, 'get_qwen_client', get_client)
    def validate(role, emotion):
        events.append(('capabilities', role, emotion))
    monkeypatch.setattr(route, 'validate_v3_capabilities', validate)
    async def prepare(req, unique_tmp=False):
        assert unique_tmp
        events.append(('prepare',))
        return None, ctx
    monkeypatch.setattr(route, 'prepare_asr_context', prepare)
    response = dict(code=0, msg='success', text='保留正文。', segments=[dict(bg=0.00, ed=1.00,
                    segment_text='保留正文。', role='教师', emotion='平淡')],
                    processing_status='completed', segment_warnings=[])
    async def run(value):
        assert value is ctx
        events.append(('run', value.request_id))
        return response
    monkeypatch.setattr(route, 'run_qwen_asr', run)
    def cleanup(paths):
        assert paths == ctx.tmp_paths
        events.append(('cleanup',))
    monkeypatch.setattr(route, 'cleanup_temp_files', cleanup)
    monkeypatch.setattr(route, 'add_processing_task', lambda rid, name: events.append(('registered', rid)))
    monkeypatch.setattr(route, 'remove_processing_task', lambda rid: events.append(('removed', rid)))
    monkeypatch.setattr(route, 'update_fail_task', lambda: events.append(('failed',)))
    monkeypatch.setattr(route, 'record_request', lambda **row: records.append(row))
    for name in ('begin_request', 'update_active', 'end_request'):
        monkeypatch.setattr(route, name, lambda *a, **kw: None)
    return SimpleNamespace(events=events, records=records, ctx=ctx, lease=lease,
                           client=client, slots=slots, settings=settings, response=response)


def test_http_route_binds_before_admission_and_keeps_original_fields(wired):
    async def run():
        app = FastAPI()
        app.include_router(route.router)
        app.dependency_overrides[get_asr_params] = request
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as http:
            response = await http.post('/v3/seacraft_asr')
        assert response.status_code == 200
        assert response.json() == wired.response
    asyncio.run(run())
    assert wired.events == [('capabilities', True, True), ('client',), ('ensure',), ('bind',),
        ('acquire', 'B', {'request_id': 'request-42'}), ('registered', 'request-42'),
        ('prepare',), ('run', 'request-42'), ('cleanup',), ('release',), ('removed', 'request-42')]
    assert wired.records[0]['request_id'] == 'request-42'
    assert wired.records[0]['task_id'] == 'display-task-only'
    assert wired.ctx.request_id != wired.ctx.task_id


def test_disabled_enhancement_precedes_client_loading_and_admission(wired, monkeypatch):
    def unavailable(*_):
        wired.events.append(('capability_rejected',))
        raise ModelUnavailableError('emotion2vec', reason='emotion_disabled')
    monkeypatch.setattr(route, 'validate_v3_capabilities', unavailable)
    value = asyncio.run(route.api_asr_v3(request()))
    assert value['code'] == 4016
    assert wired.events == [('capability_rejected',), ('failed',)]
    assert wired.records[0]['admission']['reason'] == 'emotion_disabled'


def test_unavailable_owner_pool_keeps_4011_and_does_not_prepare(wired, monkeypatch):
    async def unavailable():
        wired.events.append(('ensure_failed',))
        raise QwenAsrUnavailableError()
    monkeypatch.setattr(wired.client, 'ensure_owner_pool', unavailable)
    value = asyncio.run(route.api_asr_v3(request()))
    assert value == {'code': 4011, 'msg': QwenAsrUnavailableError.message}
    assert not any(row[0] in {'bind', 'acquire', 'prepare', 'release'} for row in wired.events)
    assert wired.records[0]['admission']['reason'] == 'qwen_unavailable'


def test_prepare_error_after_grant_returns_lease_once(wired, monkeypatch):
    async def invalid(*a, **kw):
        wired.events.append(('prepare_failed',))
        return {'code': 4002, 'msg': '读取音频文件发生错误'}, None
    monkeypatch.setattr(route, 'prepare_asr_context', invalid)
    value = asyncio.run(route.api_asr_v3(request()))
    assert value['code'] == 4002
    assert wired.events.count(('release',)) == 1
    assert wired.events[-2:] == [('release',), ('removed', 'request-42')]
    assert not any(row[0] in {'cleanup', 'run'} for row in wired.events)


def test_cancel_during_prepare_after_grant_returns_lease(wired, monkeypatch):
    async def run():
        preparing = asyncio.Event()
        async def blocked(*a, **kw):
            preparing.set()
            await asyncio.Event().wait()
        monkeypatch.setattr(route, 'prepare_asr_context', blocked)
        task = asyncio.create_task(route.api_asr_v3(request()))
        await asyncio.wait_for(preparing.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
    asyncio.run(run())
    assert wired.events.count(('release',)) == 1
    assert wired.records[0]['status'] == 'cancelled'


def test_cancel_before_owner_grant_does_not_release_unowned_lease(wired, monkeypatch):
    async def run():
        waiting = asyncio.Event()
        async def blocked(*a, **kw):
            waiting.set()
            await asyncio.Event().wait()
        monkeypatch.setattr(wired.slots, 'acquire', blocked)
        task = asyncio.create_task(route.api_asr_v3(request()))
        await asyncio.wait_for(waiting.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
    asyncio.run(run())
    assert not any(row[0] in {'release', 'prepare', 'registered'} for row in wired.events)
    assert wired.records[0]['admission']['outcome'] == 'cancelled'


def test_repeat_cancel_waits_for_safe_owner_release_after_cleanup(wired, monkeypatch):
    async def run():
        infer = asyncio.Event()
        releasing = asyncio.Event()
        safe = asyncio.Event()
        async def blocked(ctx):
            infer.set()
            await asyncio.Event().wait()
        async def release(value):
            assert value is wired.lease
            wired.events.append(('release_started',))
            releasing.set()
            await safe.wait()
            wired.events.append(('release_done',))
        monkeypatch.setattr(route, 'run_qwen_asr', blocked)
        monkeypatch.setattr(wired.slots, 'release', release)
        task = asyncio.create_task(route.api_asr_v3(request()))
        await asyncio.wait_for(infer.wait(), 1)
        task.cancel()
        await asyncio.wait_for(releasing.wait(), 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert ('cleanup',) in wired.events
        assert ('removed', 'request-42') not in wired.events
        safe.set()
        with pytest.raises(asyncio.CancelledError): await task
    asyncio.run(run())
    assert wired.events.count(('release_started',)) == 1
    assert wired.events.index(('cleanup',)) < wired.events.index(('release_started',))
    assert wired.events[-2:] == [('release_done',), ('removed', 'request-42')]


def test_explicit_legacy_keeps_category_only_admission(wired):
    wired.settings.qwen_generation_mode = 'legacy'
    value = asyncio.run(route.api_asr_v3(request()))
    assert value == wired.response
    assert ('acquire', 'B', {}) in wired.events
    assert not any(row[0] in {'client', 'ensure', 'bind'} for row in wired.events)
    assert ('run', 'request-42') in wired.events


def test_lifespan_binds_same_preloaded_pool_before_ready(monkeypatch, tmp_path):
    import main
    events = []
    client = object()
    slots = SimpleNamespace(bind_owner_pool=lambda value: events.append(('bind', value)))
    cfg = SimpleNamespace(qwen_audio_spk_emo_concurrency=4,
        validate_model_backends=lambda: events.append(('validate',)),
        log_path=str(tmp_path/'service.log'), resource_sample_interval_s=1.,
        qwen_execution_mode='process', qwen_generation_mode='continuous',
        model_device_speech_campplus_execution_mode='thread',
        qwen_preload_on_startup=True, qwen_asr_enabled=True)
    monkeypatch.setattr(main, 'settings', cfg)
    monkeypatch.setattr(main, 'validate_npu_runtime', lambda cfg: {'fixture': True})
    monkeypatch.setattr(main, '_set_process_name', lambda: None)
    monkeypatch.setattr(main, 'setup_logging', lambda *_: None)
    monkeypatch.setattr(main, 'cleanup_stale_service_cache', lambda: None)
    monkeypatch.setattr(main, 'reset_stats', lambda: None)
    class Sampler:
        def __init__(self, **kw): pass
        async def aclose(self): events.append(('sampler_close',))
    monkeypatch.setattr(main, 'ResourceSnapshotSampler', Sampler)
    async def vad(): events.append(('vad',))
    async def preload(): events.append(('preload',))
    async def get_client():
        events.append(('client',))
        return client
    monkeypatch.setattr(main, 'start_vad_pool', vad)
    monkeypatch.setattr(main, 'preload_qwen_client', preload)
    monkeypatch.setattr(main, 'get_qwen_client', get_client)
    monkeypatch.setattr(main, 'get_slot_manager', lambda: slots)
    closes = {}
    for name in ('stop_vad_pool', 'close_qwen_client', 'close_v3_model_pools'):
        closes[name] = AsyncMock()
        monkeypatch.setattr(main, name, closes[name])
    async def run():
        async with main.lifespan(main.create_app()):
            assert events == [('validate',), ('vad',), ('preload',), ('client',), ('bind', client)]
            events.append(('ready',))
    asyncio.run(run())
    assert events.index(('bind', client)) < events.index(('ready',))
    for close in closes.values(): close.assert_awaited_once()


def test_prepare_exception_after_owner_grant_still_releases(wired, monkeypatch):
    async def prepare(*a, **kw):
        raise RuntimeError('preparation fixture failure')
    monkeypatch.setattr(route, 'prepare_asr_context', prepare)
    with pytest.raises(RuntimeError, match='preparation fixture failure'):
        asyncio.run(route.api_asr_v3(request()))
    assert wired.events.count(('release',)) == 1
    assert wired.records[0]['status'] == 'failed'


@pytest.mark.parametrize('kind,code', [('total', 4013), ('enhancement', 4017), ('timeout', 4010)])
def test_admission_errors_keep_existing_code_and_skip_prepare(wired, monkeypatch, kind, code):
    from core.qwen_asr_queue import QueueFullError, SpkEmoQueueFullError, QueueTimeoutError
    error = (QueueFullError(active=6, capacity=6, snapshot={'active': 6}) if kind == 'total' else
             SpkEmoQueueFullError(active=4, capacity=4, snapshot={'active': 4}) if kind == 'enhancement' else
             QueueTimeoutError(snapshot={'waiting': 1}))
    async def reject(*a, **kw): raise error
    monkeypatch.setattr(wired.slots, 'acquire', reject)
    value = asyncio.run(route.api_asr_v3(request()))
    assert value == {'code': code, 'msg': error.message}
    assert not any(row[0] in {'prepare', 'release', 'registered'} for row in wired.events)
    assert wired.records[0]['admission']['reason'] == error.reason
