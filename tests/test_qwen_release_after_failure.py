"""Terminal dedup cleanup must not mask an owned inference failure."""
from concurrent.futures import Future
from types import SimpleNamespace
import pytest
import core.qwen_continuous_process as mod


def runtime_with_terminal():
    runtime=mod.LocalQwenContinuousProcessRuntime('/tmp/no-cpu-model','npu:7')
    future=Future();future.set_exception(RuntimeError('original4011/4012'))
    runtime._futures[('request','0')]=future
    return runtime,future


@pytest.mark.parametrize('condition',['closed','closing','failure','dead','connection_none','connection_closed'])
def test_terminal_cleanup_skips_unavailable_pipe_and_preserves_original(monkeypatch,condition):
    runtime,future=runtime_with_terminal()
    runtime._connection=SimpleNamespace(closed=False)
    runtime._process=SimpleNamespace(is_alive=lambda:True)
    if condition=='closed':runtime._closed=True
    elif condition=='closing':runtime._explicit_closing=True
    elif condition=='failure':runtime._failed('original4011/4012')
    elif condition=='dead':runtime._process=SimpleNamespace(is_alive=lambda:False)
    elif condition=='connection_none':runtime._connection=None
    else:runtime._connection.closed=True
    monkeypatch.setattr(mod,'_send',lambda *a:pytest.fail('terminal unavailable cleanup attempted Pipe write'))
    runtime.release_request('request')
    assert not runtime._futures
    assert str(future.exception())=='original4011/4012'


def test_pending_future_never_cleared_even_when_closing(monkeypatch):
    runtime=mod.LocalQwenContinuousProcessRuntime('/tmp/no-cpu-model','npu:7')
    future=Future();runtime._futures[('request','0')]=future;runtime._explicit_closing=True
    monkeypatch.setattr(mod,'_send',lambda *a:pytest.fail('active cleanup wrote Pipe'))
    with pytest.raises(RuntimeError,match='active'):runtime.release_request('request')
    assert runtime._futures[('request','0')] is future and not future.done()


def test_no_owned_entries_does_not_start_or_contact_worker(monkeypatch):
    runtime=mod.LocalQwenContinuousProcessRuntime('/tmp/no-cpu-model','npu:7')
    monkeypatch.setattr(runtime,'_start',lambda:pytest.fail('cleanup spawned worker'))
    monkeypatch.setattr(mod,'_send',lambda *a:pytest.fail('empty cleanup wrote Pipe'))
    runtime.release_request('unknown')
    assert runtime._process is None and runtime._connection is None


def test_healthy_terminal_request_still_releases_worker_dedup(monkeypatch):
    runtime,future=runtime_with_terminal()
    runtime._process=SimpleNamespace(is_alive=lambda:True)
    runtime._connection=SimpleNamespace(closed=False)
    messages=[];monkeypatch.setattr(mod,'_send',lambda conn,msg:messages.append(msg))
    runtime.release_request('request')
    assert not runtime._futures
    assert messages[0]['operation']=='release_request'
    assert str(future.exception())=='original4011/4012'


@pytest.mark.parametrize('error_type',[BrokenPipeError,EOFError,AttributeError])
def test_reap_race_does_not_mask_original_terminal_failure(monkeypatch,error_type):
    runtime,future=runtime_with_terminal()
    runtime._process=SimpleNamespace(is_alive=lambda:True)
    runtime._connection=SimpleNamespace(closed=False)
    def failed_send(*args):
        runtime._closed=True;runtime._connection=None
        raise error_type('reaped during release')
    monkeypatch.setattr(mod,'_send',failed_send)
    runtime.release_request('request')
    assert not runtime._futures
    assert str(future.exception())=='original4011/4012'


def test_unrelated_healthy_channel_fault_is_not_silently_ignored(monkeypatch):
    runtime,future=runtime_with_terminal()
    runtime._process=SimpleNamespace(is_alive=lambda:True)
    runtime._connection=SimpleNamespace(closed=False)
    def failed_send(*args):raise BrokenPipeError('unexpected healthy-channel fault')
    monkeypatch.setattr(mod,'_send',failed_send)
    with pytest.raises(BrokenPipeError):runtime.release_request('request')
    assert runtime._futures[('request','0')] is future
