"""CPU-only coverage of enhancement startup, degradation and reuse."""
import asyncio
from types import SimpleNamespace

import pytest

import core.models as models
from core.models import ModelPool, ModelUnavailableError
from test_qwen_startup_preload import patch_lifespan, until


def pools(monkeypatch, *, cam=True, emo=True, devices=(4, 1), failures=()):
    made = []
    monkeypatch.setattr(models.settings, '_cfg', dict(models.settings._cfg,
        model_device_pool={'speech_campplus_execution_mode': 'thread'},
        features={'open_v2_role': cam, 'open_v2_emotion': emo}))
    def make(name, count, enabled):
        pool = ModelPool(name, '/fake', devices_getter=lambda: ['cpu'] * count,
                         enabled_getter=lambda: enabled)
        def build(device):
            index = sum(n == name for n in made)
            made.append(name)
            if (name, index) in failures:
                raise RuntimeError('synthetic loading failure')
            return SimpleNamespace(generate=lambda **_: []), device, 0.0
        monkeypatch.setattr(pool, '_build_instance', build)
        return pool
    ps = {'speech_campplus': make('speech_campplus', devices[0], cam),
          'emotion2vec': make('emotion2vec', devices[1], emo)}
    monkeypatch.setattr(models, '_MODEL_POOLS', ps)
    monkeypatch.setattr(models, '_model_lock', asyncio.Lock())
    return ps, made


@pytest.mark.parametrize('cam,emo,devices,expected', [
    (True, True, (4, 1), ['speech_campplus'] * 4 + ['emotion2vec']),
    (False, True, (4, 1), ['emotion2vec']),
    (True, False, (4, 1), ['speech_campplus'] * 4),
    (True, True, (0, 0), []),
])
def test_startup_arrays_switches_and_reuse(monkeypatch, cam, emo, devices, expected):
    async def run():
        ps, made = pools(monkeypatch, cam=cam, emo=emo, devices=devices)
        try:
            await models.preload_v3_model_pools()
            assert made == expected
            if cam and emo and all(devices):
                for _ in range(2):
                    await models.load_v3_models_if_needed(True, True)
                    for pool in ps.values():
                        async with pool.lease():
                            pass
            for _ in range(2):
                models.get_local_model_status()
            assert made == expected
        finally:
            await models.close_v3_model_pools()
    asyncio.run(run())


@pytest.mark.parametrize('all_fail', [False, True])
def test_load_failure_preserves_other_pool_and_original_errors(monkeypatch, all_fail):
    async def run():
        failures = [('speech_campplus', i) for i in range(4 if all_fail else 1)]
        ps, made = pools(monkeypatch, failures=failures)
        try:
            await models.preload_v3_model_pools()
            assert made == ['speech_campplus'] * 4 + ['emotion2vec']
            assert sum(ps['speech_campplus']._is_healthy(m) for m in ps['speech_campplus'].models) == (0 if all_fail else 3)
            await models.load_v3_models_if_needed(False, True)
            if all_fail:
                with pytest.raises(ModelUnavailableError):
                    await models.load_v3_models_if_needed(True, False)
            else:
                await models.load_v3_models_if_needed(True, False)
            assert len(made) == 5
        finally:
            await models.close_v3_model_pools()
    asyncio.run(run())


@pytest.mark.parametrize('finish', ['success', 'failure', 'cancel'])
def test_lifespan_waits_for_enhancements_and_closes_partial_resources(monkeypatch, finish):
    async def run():
        main, events = patch_lifespan(monkeypatch)
        entered, release = asyncio.Event(), asyncio.Event()
        async def preload():
            events.append('enhancements_started')
            entered.set()
            await release.wait()
            if finish == 'failure':
                raise RuntimeError('synthetic startup error')
            events.append('enhancements_loaded')
        monkeypatch.setattr(main, 'preload_v3_model_pools', preload)
        async def serve():
            async with main.lifespan(main.create_app()):
                events.append('ready')
        task = asyncio.create_task(serve())
        await asyncio.wait_for(entered.wait(), 3)
        assert 'ready' not in events and not task.done()
        assert events[:3] == ['vad_started', 'preload_started', 'preload_done']
        if finish == 'cancel':
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            release.set()
            if finish == 'failure':
                with pytest.raises(RuntimeError, match='synthetic startup error'):
                    await task
            else:
                await task
        assert ('ready' in events) == (finish == 'success')
        assert all(n in events for n in ['vad_closed', 'models_closed', 'qwen_closed', 'sampler_closed'])
    asyncio.run(run())
