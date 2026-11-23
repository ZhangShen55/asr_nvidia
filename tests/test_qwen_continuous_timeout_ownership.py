import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

from core import qwen_asr_client as candidate

class BackendCancelled(RuntimeError): pass


def test_timeout_keeps_primary_cause_after_backend_cancel_ack():
    async def scenario():
        client = object.__new__(candidate.QwenAsrClient)
        client.request_timeout_s = -1
        future = asyncio.get_running_loop().create_future()
        calls = []
        class Runtime:
            def cancel_segment(self, request, segment):
                calls.append((request, segment))
                future.set_exception(BackendCancelled('cancelled'))
        member = SimpleNamespace(request_id='r', segment_index='s', cancelled=False)
        with pytest.raises(TimeoutError) as captured:
            await client._await_sequence(Runtime(), member, future)
        assert captured.value.reason == 'request_timeout'
        assert isinstance(captured.value.__cause__, BackendCancelled)
        assert calls == [('r', 's')]
        assert future.done()
    asyncio.run(scenario())


def test_caller_cancel_stays_cancel_after_backend_cancel_error():
    async def scenario():
        client = object.__new__(candidate.QwenAsrClient)
        client.request_timeout_s = 100
        future = asyncio.get_running_loop().create_future()
        class Runtime:
            def cancel_segment(self, request, segment):
                future.set_exception(BackendCancelled('cancelled'))
        member = SimpleNamespace(request_id='r', segment_index='s', cancelled=True)
        with pytest.raises(asyncio.CancelledError):
            await client._await_sequence(Runtime(), member, future)
        assert future.done()
    asyncio.run(scenario())


def test_non_timeout_preserves_backend_failure():
    async def scenario():
        client = object.__new__(candidate.QwenAsrClient)
        client.request_timeout_s = 100
        future = asyncio.get_running_loop().create_future()
        original = BackendCancelled('length_limit')
        future.set_exception(original)
        member = SimpleNamespace(request_id='r', segment_index='s', cancelled=False)
        with pytest.raises(BackendCancelled) as captured:
            await client._await_sequence(None, member, future)
        assert captured.value is original
    asyncio.run(scenario())


def test_mapping_records_reason_and_type_without_text(caplog):
    async def scenario():
        client = object.__new__(candidate.QwenAsrClient)
        client.generation_mode = 'continuous'
        client._closing = False
        class Scheduler:
            async def submit(self, *args, **kwargs):
                error = TimeoutError('PRIVATE_TRANSCRIPT_MUST_NOT_BE_LOGGED')
                error.reason = 'request_timeout'
                raise error
        client._batch_scheduler = Scheduler()
        with pytest.raises(candidate.QwenAsrCallError) as captured:
            await client.transcribe_bytes(b'fake', request_id='r', segment_index=1)
        assert captured.value.reason == 'request_timeout'
        assert captured.value.backend_error_type == 'TimeoutError'
        assert captured.value.code == 4012
    asyncio.run(scenario())
    assert 'reason=request_timeout' in caplog.text
    assert 'type=TimeoutError' in caplog.text
    assert 'PRIVATE_TRANSCRIPT' not in caplog.text


def test_blocking_submit_keeps_loop_live_and_cancel_owns_late_acceptance():
    import threading
    from concurrent.futures import Future
    async def scenario():
        entered, allow_send, allow_cleanup = threading.Event(), threading.Event(), threading.Event()
        submitted = Future()
        class Runtime:
            loaded = True
            failure = None
            def submit(self, *args, **kwargs):
                entered.set()
                assert allow_send.wait(5)
                return submitted
            def cancel_segment(self, request, segment):
                assert allow_cleanup.wait(5)
                submitted.set_exception(BackendCancelled('cancelled'))
        client = object.__new__(candidate.QwenAsrClient)
        client._runtimes = [Runtime()]
        # The production client has already completed pool discovery and audio
        # admission before a sequence reaches this send/cancel ownership probe.
        client._pool_initialized = True
        client._owner_lock = threading.RLock()
        client._request_owners = {'r': 0}
        client._owner_requests = {0: 'r'}
        client._discovery_lock = asyncio.Lock()
        client._closing = False
        client.request_timeout_s = 120
        member = SimpleNamespace(request_id='r',segment_index='s',cancelled=False,audio=b'pcm',language=None)
        task = asyncio.create_task(client._execute_sequence(member))
        try:
            for _ in range(100):
                if entered.is_set(): break
                await asyncio.sleep(.001)
            assert entered.is_set()
            ticks = 0
            for _ in range(5):
                await asyncio.sleep(.001)
                ticks += 1
            assert ticks == 5 and not task.done()
            task.cancel()
            await asyncio.sleep(.005)
            assert not task.done()
            allow_send.set()
            await asyncio.sleep(.005)
            task.cancel()
            assert not task.done()
            allow_cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert submitted.done()
        finally:
            allow_send.set(); allow_cleanup.set()
    asyncio.run(scenario())


def test_release_pipe_io_is_owned_and_does_not_block_loop():
    import threading
    async def scenario():
        entered, finish = threading.Event(), threading.Event()
        class Client:
            def release_request(self, request_id):
                entered.set()
                assert finish.wait(5)
        candidate._client_instance = Client()
        task = asyncio.create_task(candidate.release_qwen_request('r'))
        try:
            while not entered.is_set():
                await asyncio.sleep(.001)
            task.cancel()
            await asyncio.sleep(.005)
            assert not task.done()
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            finish.set()
            candidate._client_instance = None
    asyncio.run(scenario())
