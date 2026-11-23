"""CPU-only startup contract: no device imports, model weights, or real services."""
import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.config import Settings
from core.qwen_asr_client import QwenAsrClient, QwenAsrUnavailableError


def configured(tmp_path, text):
    path = tmp_path / 'preload.toml'
    path.write_text(text)
    return Settings(str(path))


def test_preload_omission_retains_lazy_default(tmp_path):
    assert configured(tmp_path, '[qwen_asr]\ngeneration_mode="legacy"\n').qwen_preload_on_startup is False


@pytest.mark.parametrize('value, expected', [('true', True), ('false', False)])
def test_preload_accepts_only_actual_booleans(tmp_path, value, expected):
    assert configured(tmp_path, f'[qwen_asr]\npreload_on_startup={value}\n').qwen_preload_on_startup is expected


@pytest.mark.parametrize('value', ['"true"', '"false"', '0', '1', '1.0', '[]', '{}'])
def test_bad_preload_rejected_even_if_qwen_disabled(tmp_path, value):
    settings = configured(tmp_path, f'[qwen_asr]\nenabled=false\npreload_on_startup={value}\n')
    with pytest.raises(ValueError, match='preload_on_startup'):
        settings.validate_model_backends()


@pytest.mark.parametrize('name, b_limit', [('config.qwen128.toml', 4), ('config.qwen128-b8.toml', 8)])
def test_profiles_separate_complete_audio_from_logical_and_physical_segments(name, b_limit):
    settings = Settings(str(Path(__file__).resolve().parents[1] / 'test' / name))
    settings._cfg['qwen_asr']['generation_mode'] = 'legacy'
    settings.validate_model_backends()
    assert settings.qwen_preload_on_startup is True
    assert settings.qwen_execution_mode == 'process'
    assert settings.qwen_local_instance_count == 1
    assert settings.qwen_segment_concurrency == 16
    assert settings.qwen_request_concurrency == 128
    assert settings.qwen_max_inference_batch_size == 128
    assert settings.qwen_audio_concurrency == 8
    assert settings.qwen_audio_spk_emo_concurrency == b_limit
    assert settings.qwen_process_shutdown_grace_s == 120
    assert settings.vad_device == ['cpu'] * 8


class CpuLoadRuntime:
    def __init__(self, *args):
        self.loaded = False
        self.failure = None
        self.load_calls = 0
        self.closed = False
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.fail_load = False

    def ensure_loaded(self):
        self.load_calls += 1
        self.started.set()
        assert self.release.wait(5), 'CPU gate was never released'
        if self.fail_load:
            self.failure = 'synthetic load error'
            raise RuntimeError(self.failure)
        self.loaded = True

    def close(self):
        self.closed = True
        self.loaded = False


def cpu_client(count=2, mode='thread'):
    return QwenAsrClient(generation_mode='legacy', model='display-name', local_model_dir='/not-a-real-model',
                         local_device='npu:7', local_dtype='bfloat16',
                         local_instance_count=count, execution_mode=mode,
                         runtime_factory=CpuLoadRuntime, request_timeout_s=10,
                         retry_count=0, max_inference_batch_size=1)


async def until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(.005)
    await asyncio.wait_for(poll(), 3)


@pytest.mark.parametrize('mode', ['thread', 'process'])
def test_preload_ignores_display_name_and_loads_every_instance_once_without_generation(mode):
    async def run():
        client = cpu_client(mode=mode)
        try:
            assert not any(runtime.loaded for runtime in client._runtimes)
            assert await client.preload() == 'display-name'
            assert await client.preload() == 'display-name'
            assert all(runtime.loaded and runtime.load_calls == 1 for runtime in client._runtimes)
            assert client._available.qsize() == 2 and not any(client._busy)
            assert client._batch_scheduler.snapshot()['inflight'] == 0
        finally:
            await client.aclose()
        assert all(runtime.closed for runtime in client._runtimes)
    asyncio.run(run())


def test_repeated_cancel_does_not_return_loading_lease_or_close_before_actual_completion():
    async def run():
        client = cpu_client()
        for runtime in client._runtimes:
            runtime.release.clear()
        preload = asyncio.create_task(client.preload())
        close = None
        try:
            await until(lambda: all(runtime.started.is_set() for runtime in client._runtimes))
            preload.cancel()
            await asyncio.sleep(.01)
            preload.cancel()
            close = asyncio.create_task(client.aclose())
            await asyncio.sleep(.02)
            assert not preload.done() and not close.done()
            assert client._available.qsize() == 0 and all(client._busy)
            assert not any(runtime.closed for runtime in client._runtimes)
            for runtime in client._runtimes:
                runtime.release.set()
            with pytest.raises(asyncio.CancelledError):
                await preload
            await close
            await client.aclose()
            assert client._available.qsize() == 2 and not any(client._busy)
            assert all(runtime.closed for runtime in client._runtimes)
        finally:
            for runtime in client._runtimes:
                runtime.release.set()
            await asyncio.gather(preload, return_exceptions=True)
            await client.aclose()
    asyncio.run(run())


def test_partial_load_failure_drains_other_instance_and_keeps_discovery_unpublished():
    async def run():
        client = cpu_client()
        client._runtimes[0].fail_load = True
        client._runtimes[1].release.clear()
        preload = asyncio.create_task(client.preload())
        try:
            await until(lambda: all(runtime.started.is_set() for runtime in client._runtimes))
            await asyncio.sleep(.02)
            assert not preload.done() and client._available.qsize() == 0
            assert client._discovered_model is None
            client._runtimes[1].release.set()
            with pytest.raises(QwenAsrUnavailableError):
                await preload
            assert client._available.qsize() == 2 and not any(client._busy)
        finally:
            client._runtimes[1].release.set()
            await asyncio.gather(preload, return_exceptions=True)
            await client.aclose()
        assert all(runtime.closed for runtime in client._runtimes)
    asyncio.run(run())


def patch_lifespan(monkeypatch, *, enabled=True, preload=True, fail=False, block=None):
    import main
    events = []
    settings = SimpleNamespace(qwen_preload_on_startup=preload, qwen_asr_enabled=enabled,
                               qwen_generation_mode='legacy',
                               resource_sample_interval_s=1, qwen_execution_mode='thread',
                               qwen_audio_spk_emo_concurrency=4, log_path='unused',
                               validate_model_backends=lambda: None)
    monkeypatch.setattr(main, 'settings', settings)
    for name in ['_set_process_name', 'cleanup_stale_service_cache', 'reset_stats']:
        monkeypatch.setattr(main, name, lambda: None)
    monkeypatch.setattr(main, 'setup_logging', lambda *_: None)
    monkeypatch.setattr(main, 'validate_npu_runtime', lambda *_: {'CPU': True})
    class Sampler:
        def __init__(self, **_):
            pass
        async def aclose(self):
            events.append('sampler_closed')
    monkeypatch.setattr(main, 'ResourceSnapshotSampler', Sampler)
    async def start():
        events.append('vad_started')
    async def load():
        events.append('preload_started')
        if block is not None:
            await block.wait()
        if fail:
            raise QwenAsrUnavailableError()
        events.append('preload_done')
    async def stop():
        events.append('vad_closed')
    async def models():
        events.append('models_closed')
    async def qwen():
        events.append('qwen_closed')
    monkeypatch.setattr(main, 'start_vad_pool', start)
    monkeypatch.setattr(main, 'preload_qwen_client', load)
    monkeypatch.setattr(main, 'stop_vad_pool', stop)
    monkeypatch.setattr(main, 'close_v3_model_pools', models)
    monkeypatch.setattr(main, 'close_qwen_client', qwen)
    return main, events


@pytest.mark.parametrize('preload, enabled', [(False, True), (True, False)])
def test_disabled_preload_or_disabled_qwen_preserves_lazy_startup(monkeypatch, preload, enabled):
    main, events = patch_lifespan(monkeypatch, preload=preload, enabled=enabled)
    async def run():
        async with main.lifespan(main.create_app()):
            events.append('ready')
    asyncio.run(run())
    assert events[0:2] == ['vad_started', 'ready']
    assert 'preload_started' not in events
    assert all(name in events for name in ['vad_closed', 'models_closed', 'qwen_closed', 'sampler_closed'])


def test_lifespan_readiness_waits_for_preload_and_reuses_normal_cleanup(monkeypatch):
    async def run():
        gate = asyncio.Event()
        main, events = patch_lifespan(monkeypatch, block=gate)
        async def serve():
            async with main.lifespan(main.create_app()):
                events.append('ready')
        task = asyncio.create_task(serve())
        await until(lambda: 'preload_started' in events)
        assert 'ready' not in events and not task.done()
        gate.set()
        await task
        assert events[:4] == ['vad_started', 'preload_started', 'preload_done', 'ready']
        assert all(name in events for name in ['vad_closed', 'models_closed', 'qwen_closed', 'sampler_closed'])
    asyncio.run(run())


def test_lifespan_load_failure_never_ready_but_closes_every_owner(monkeypatch):
    main, events = patch_lifespan(monkeypatch, fail=True)
    app = main.create_app()
    async def run():
        with pytest.raises(QwenAsrUnavailableError):
            async with main.lifespan(app):
                events.append('ready')
    asyncio.run(run())
    assert 'ready' not in events
    assert all(name in events for name in ['vad_closed', 'models_closed', 'qwen_closed', 'sampler_closed'])
    assert app.state.shutdown_failed is False


def test_lifespan_startup_cancel_still_closes_all_owners(monkeypatch):
    async def run():
        gate = asyncio.Event()
        main, events = patch_lifespan(monkeypatch, block=gate)
        async def serve():
            async with main.lifespan(main.create_app()):
                events.append('ready')
        task = asyncio.create_task(serve())
        await until(lambda: 'preload_started' in events)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert 'ready' not in events
        assert all(name in events for name in ['vad_closed', 'models_closed', 'qwen_closed', 'sampler_closed'])
    asyncio.run(run())


def test_global_preload_calls_existing_client_without_replacing_it(monkeypatch):
    import core.qwen_asr_client as module
    calls = []
    class Client:
        async def preload(self):
            calls.append('preload')
            return 'configured-model'
    client = Client()
    async def get():
        calls.append('get')
        return client
    monkeypatch.setattr(module, 'get_qwen_client', get)
    assert asyncio.run(module.preload_qwen_client()) == 'configured-model'
    assert calls == ['get', 'preload']
