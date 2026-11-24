"""宿主验收控制器的归属边界；全部Docker/NPU接口均使用替身。"""
import json
from types import SimpleNamespace

import pytest
from scripts import validate_source_container as validator

IMAGE = 'sha256:'+'a'*64
ID = 'b'*64
TASK = 'owned-container-test'


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.fixture(autouse=True)
def isolated_shared_identity(monkeypatch):
    monkeypatch.setattr(validator, 'shared_identity', lambda: {
        'available': True, 'alive': True, 'birth': 1, 'listening_ports': [('127.0.0.1', 8007)]})


def inspected(*, task=TASK, image=IMAGE):
    return {'Id': ID, 'Image': image, 'State': {'Running': True, 'Pid': 123},
            'Config': {'Labels': {'asr.build.task': task}},
            'NetworkSettings': {'Ports': {'9000/tcp': [{'HostIp': '127.0.0.1', 'HostPort': '18083'}]}}}


def args(tmp_path):
    return SimpleNamespace(container='named-container', expected_image_id=IMAGE, task_id=TASK,
                           output=str(tmp_path/'result'), config=str(tmp_path/'config.toml'),
                           base_url='http://127.0.0.1:18083', manifest='unused', clip='unused')


@pytest.mark.anyio
@pytest.mark.parametrize('bad', ['task', 'image'])
async def test_mistaken_shared_container_is_never_stopped(tmp_path, monkeypatch, bad):
    calls = []
    item = inspected(task='shared' if bad == 'task' else TASK,
                     image='sha256:'+'c'*64 if bad == 'image' else IMAGE)
    def fake_command(argv, **kw):
        calls.append(argv)
        assert argv[:2] == ['docker', 'inspect']
        return json.dumps([item])
    monkeypatch.setattr(validator, 'command', fake_command)
    async def background(): return {}
    monkeypatch.setattr(validator, 'background_snapshot', background)
    assert await validator.execute(args(tmp_path)) == 1
    assert calls == [['docker', 'inspect', 'named-container']]


def test_matching_identity_gets_stop_eligibility_before_fallible_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(validator, 'command', lambda *_a, **_k: json.dumps([inspected()]))
    monkeypatch.setattr(validator.DockerOwner, 'register', lambda _self: pytest.fail('constructor must not register'))
    owner = validator.DockerOwner('named', tmp_path, expected_image_id=IMAGE, task_id=TASK, base_url='http://127.0.0.1:18083')
    assert owner.verified_owned and owner.id == ID and owner.image == IMAGE


@pytest.mark.anyio
@pytest.mark.parametrize('failing_stage', ['registration', 'evidence'])
async def test_verified_owned_is_stopped_even_if_early_evidence_fails(tmp_path, monkeypatch, failing_stage):
    monkeypatch.setattr(validator, 'command', lambda *_a, **_k: json.dumps([inspected()]))
    def register(_self):
        if failing_stage == 'registration': raise OSError('registration unavailable')
    monkeypatch.setattr(validator.DockerOwner, 'register', register)
    def write(*_a, **_k): raise OSError('disk evidence unavailable')
    monkeypatch.setattr(validator, 'write_json', write)
    cleaned = []
    def cleanup(owner, *_a):
        cleaned.append((owner.id, owner.verified_owned))
        return {'passed': True}
    monkeypatch.setattr(validator, 'cleanup', cleanup)
    async def background(): return {}
    monkeypatch.setattr(validator, 'background_snapshot', background)
    assert await validator.execute(args(tmp_path)) == 1
    assert cleaned == [(ID, True)]


def test_cleanup_registration_failure_still_attempts_normal_stop(monkeypatch):
    calls = []
    def command(argv, **_kw):
        calls.append(argv)
        if argv[1] == 'inspect':
            return json.dumps([{'State': {'Running': False, 'ExitCode': 0, 'OOMKilled': False, 'Error': ''}}])
        return ID
    def register(): raise OSError('cannot enumerate proc')
    owner = SimpleNamespace(verified_owned=True, id=ID, last=0, register=register, pids={})
    monkeypatch.setattr(validator, 'command', command)
    monkeypatch.setattr(validator, 'stopped_cache', lambda _id: {'passed': True})
    shared = {'available': True, 'alive': True, 'birth': 1, 'listening_ports': [('127.0.0.1', 8007)]}
    monkeypatch.setattr(validator, 'shared_identity', lambda: shared)
    def no_connection(*_a, **_kw): raise OSError('not listening')
    monkeypatch.setattr(validator.socket, 'create_connection', no_connection)
    clock = iter(range(0, 600, 6))
    monkeypatch.setattr(validator.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(validator.time, 'sleep', lambda _n: None)
    result = validator.cleanup(owner, 'http://127.0.0.1:18083', shared)
    assert ['docker', 'stop', '--time', '240', ID] in calls
    assert result['passed'] is False and 'registration_error' in result['errors'][0]


def test_cleanup_unverified_owner_cannot_stop(monkeypatch):
    monkeypatch.setattr(validator, 'command', lambda *_a, **_k: pytest.fail('unverified owner must not run Docker'))
    result = validator.cleanup(SimpleNamespace(verified_owned=False), 'http://127.0.0.1:18083', {})
    assert result['not_stopped'] is True


@pytest.mark.anyio
@pytest.mark.parametrize('url', ['http://127.0.0.1:8007', 'http://192.168.1.2:18083', 'https://127.0.0.1:18083', 'http://localhost:18083'])
async def test_wrong_or_ambiguous_endpoint_never_requests_or_stops(tmp_path, monkeypatch, url):
    calls = []
    def fake_command(argv, **kw):
        calls.append(argv)
        assert argv[:2] == ['docker', 'inspect']
        return json.dumps([inspected()])
    monkeypatch.setattr(validator, 'command', fake_command)
    async def background(): return {}
    monkeypatch.setattr(validator, 'background_snapshot', background)
    options = args(tmp_path); options.base_url = url
    assert await validator.execute(options) == 1
    assert calls == [['docker', 'inspect', 'named-container']]


@pytest.mark.anyio
async def test_repeated_cancel_during_stop_keeps_cleanup_and_final_result(tmp_path, monkeypatch):
    import asyncio
    import threading
    monkeypatch.setattr(validator, 'command', lambda *_a, **_k: json.dumps([inspected()]))
    monkeypatch.setattr(validator.DockerOwner, 'register', lambda _self: None)
    def invalid_config(_path): raise ValueError('intentional early validation failure')
    monkeypatch.setattr(validator, 'config_contract', invalid_config)
    started, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    def cleanup(owner, *_a):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        return {'passed': True, 'container': owner.id}
    monkeypatch.setattr(validator, 'cleanup', cleanup)
    async def background(): return {}
    monkeypatch.setattr(validator, 'background_snapshot', background)
    monkeypatch.setattr(validator, 'device7_hbm', lambda _s: {'available': True})
    operation = asyncio.create_task(validator.execute(args(tmp_path)))
    await asyncio.wait_for(started.wait(), 3)
    operation.cancel(); await asyncio.sleep(0)
    operation.cancel(); await asyncio.sleep(0)
    release.set()
    assert await asyncio.wait_for(operation, 5) == 1
    result = json.loads((tmp_path/'result/result.json').read_text())
    assert result['cleanup']['container'] == ID and result['cleanup']['passed']
    assert len(result['finalizer_cancellations']) == 2


@pytest.mark.anyio
async def test_final_snapshot_failure_does_not_hide_cleanup_result(tmp_path, monkeypatch):
    monkeypatch.setattr(validator, 'command', lambda *_a, **_k: json.dumps([inspected()]))
    monkeypatch.setattr(validator.DockerOwner, 'register', lambda _self: None)
    def invalid_config(_path): raise ValueError('intentional early validation failure')
    monkeypatch.setattr(validator, 'config_contract', invalid_config)
    monkeypatch.setattr(validator, 'cleanup', lambda owner, *_a: {'passed': True, 'container': owner.id})
    async def failed_background(): raise OSError('resource sampler unavailable')
    monkeypatch.setattr(validator, 'background_snapshot', failed_background)
    assert await validator.execute(args(tmp_path)) == 1
    result = json.loads((tmp_path/'result/result.json').read_text())
    assert result['cleanup']['passed'] and result['final_snapshot_error']
