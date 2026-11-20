"""CPU contracts: fake futures verify scheduling, never claim NPU throughput."""
import asyncio
import io
import wave
from concurrent.futures import Future
import pytest
from core.qwen_continuous_scheduler import ContinuousScheduler
from core.qwen_asr_client import QwenAsrClient


def wav():
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as audio:
        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(16000)
        audio.writeframes(b'\0\0' * 8000)
    return buffer.getvalue()


def scheduler(execute, capacity=2):
    return ContinuousScheduler(execute, capacity=capacity, audio_frame_budget=115200,
                               token_budget=80000, max_new_tokens=256)


def test_completion_refills_while_long_member_is_running():
    async def scenario():
        events = []
        long_done, short_done = asyncio.Event(), asyncio.Event()
        async def execute(member):
            events.append(member.segment_index)
            if member.segment_index == 'long':
                await long_done.wait()
            elif member.segment_index == 'short':
                await short_done.wait()
            return member.segment_index
        owner = scheduler(execute)
        long = asyncio.create_task(owner.submit(wav(), None, request_id='a', segment_index='long'))
        short = asyncio.create_task(owner.submit(wav(), None, request_id='a', segment_index='short'))
        while len(events) < 2:
            await asyncio.sleep(0)
        short_done.set()
        assert await short == 'short'
        assert not long.done()
        assert await owner.submit(wav(), None, request_id='a', segment_index='next') == 'next'
        assert not long.done()
        assert owner.snapshot()['peak_inflight'] == 2
        long_done.set()
        await long
        await owner.close()
        assert owner.snapshot()['inflight'] == 0
    asyncio.run(scenario())


def test_cancel_does_not_release_until_backend_cleanup():
    async def scenario():
        entered, cleanup = asyncio.Event(), asyncio.Event()
        async def execute(member):
            entered.set()
            while not member.cancelled:
                await asyncio.sleep(0)
            await cleanup.wait()
            return 'discard'
        owner = scheduler(execute, 1)
        first = asyncio.create_task(owner.submit(wav(), None, request_id='a', segment_index=1))
        await entered.wait()
        first.cancel()
        await asyncio.sleep(0)
        assert owner.snapshot()['inflight'] == 1
        assert not first.done()
        first.cancel()
        cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await owner.close()
        assert owner.snapshot()['inflight'] == 0
    asyncio.run(scenario())


class Runtime:
    def __init__(self, *args):
        self.loaded = True
        self.failure = None
        self.calls = []
        self.futures = {}
    def ensure_loaded(self): pass
    def snapshot(self):
        return dict(loaded=self.loaded, failure=None)
    def submit(self, audio, language, *, request_id, segment_id):
        self.calls.append((request_id, segment_id))
        future = Future()
        self.futures[segment_id] = future
        return future
    def cancel_segment(self, request_id, segment_id):
        self.futures[segment_id].set_exception(RuntimeError('cancelled after safe step'))
    def close(self): pass


def test_client_does_not_lease_whole_model_and_cancellation_is_owned():
    async def scenario():
        client = QwenAsrClient(runtime_factory=Runtime, execution_mode='process',
                               generation_mode='continuous', local_instance_count=1)
        runtime = client._runtimes[0]
        long = asyncio.create_task(client.transcribe_bytes(wav(), request_id='a', segment_index=1))
        short = asyncio.create_task(client.transcribe_bytes(wav(), request_id='b', segment_index=2))
        while len(runtime.calls) < 2:
            await asyncio.sleep(0)
        runtime.futures['2'].set_result('short')
        assert await short == 'short'
        assert not long.done()
        long.cancel()
        with pytest.raises(asyncio.CancelledError):
            await long
        await client.aclose()
    asyncio.run(scenario())


def test_retries_only_explicit_unaccepted_capacity_and_never_execution_error():
    class Rejected(RuntimeError):
        accepted = False
        reason = 'capacity_rejected'
    class RejectRuntime(Runtime):
        def submit(self, *args, **kwargs):
            self.calls.append(kwargs['segment_id'])
            future = Future()
            future.set_exception(Rejected())
            return future
    async def scenario():
        client = QwenAsrClient(runtime_factory=RejectRuntime, execution_mode='process',
                               generation_mode='continuous', local_instance_count=1)
        with pytest.raises(Exception):
            await client.transcribe_bytes(wav(), request_id='a', segment_index=1)
        assert len(client._runtimes[0].calls) == 4
        await client.aclose()
        class FailedRuntime(Runtime):
            def submit(self, *args, **kwargs):
                self.calls.append(kwargs['segment_id'])
                future = Future(); future.set_exception(TimeoutError('execution unknown'))
                return future
        client = QwenAsrClient(runtime_factory=FailedRuntime, execution_mode='process',
                               generation_mode='continuous', local_instance_count=1)
        with pytest.raises(Exception):
            await client.transcribe_bytes(wav(), request_id='a', segment_index=1)
        assert len(client._runtimes[0].calls) == 1
        await client.aclose()
    asyncio.run(scenario())


def test_eight_audio_windows_and_tail_remain_bounded():
    from core.qwen_asr_queue import RequestConcurrency
    async def scenario():
        windows = RequestConcurrency(16, 128)
        audio_counts = [0] * 8
        peak = [0] * 8
        global_count = 0
        global_peak = 0
        permits = asyncio.Event()
        async def execute(member):
            nonlocal global_count, global_peak
            audio = int(member.request_id)
            audio_counts[audio] += 1
            global_count += 1
            peak[audio] = max(peak[audio], audio_counts[audio])
            global_peak = max(global_peak, global_count)
            await permits.wait()
            audio_counts[audio] -= 1
            global_count -= 1
            return 'ok'
        owner = scheduler(execute, 128)
        async def audio(index, count):
            sem = windows.new_segment_semaphore()
            async def segment(number):
                await windows.acquire(sem)
                try:
                    return await owner.submit(wav(), None, request_id=str(index), segment_index=number)
                finally:
                    await windows.release(sem)
            await asyncio.gather(*(segment(number) for number in range(count)))
        tasks = [asyncio.create_task(audio(index, 17)) for index in range(8)]
        while global_count < 128:
            await asyncio.sleep(0)
        assert audio_counts == [16] * 8
        permits.set()
        await asyncio.gather(*tasks)
        assert global_peak == 128
        assert peak == [16] * 8
        await audio(0, 17)
        assert peak[0] == 16
        assert owner.snapshot()['inflight'] == 0
        await owner.close()
    asyncio.run(scenario())


def test_pipeline_request_release_waits_owned_cancellation_cleanup():
    from unittest.mock import patch
    from api.routes import asr_qwen_common as pipeline
    from tests.test_v3_contract_snapshot import run_contract
    from core.segment_pipeline import run_bounded_segments
    from core.async_utils import await_completion
    async def scenario():
        entered, cleanup = asyncio.Event(), asyncio.Event()
        active = 0
        released = []
        async def segments(rows, unused_process, workers, **kwargs):
            async def process(index, row):
                nonlocal active
                active += 1
                entered.set()
                try:
                    await await_completion(cleanup.wait())
                finally:
                    active -= 1
            return await run_bounded_segments(rows, process, 2, **kwargs)
        async def release(request_id):
            assert active == 0
            released.append(request_id)
        with patch.object(pipeline, 'run_bounded_segments', segments), patch.object(pipeline, 'release_qwen_request', release):
            task = asyncio.create_task(run_contract())
            await entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            assert not released
            task.cancel()
            cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert released == ['contract-task']
    asyncio.run(scenario())


def test_pipeline_request_release_after_inference_failure_and_first_cause():
    from unittest.mock import patch
    from api.routes import asr_qwen_common as pipeline
    from tests.test_v3_contract_snapshot import run_contract
    from core.qwen_asr_client import QwenAsrCallError
    async def scenario():
        released = []
        async def release(request_id):
            released.append(request_id)
        with patch.object(pipeline, 'release_qwen_request', release):
            result, *_ = await run_contract(error=QwenAsrCallError())
        assert result['code'] == 4012
        assert released == ['contract-task']
    asyncio.run(scenario())


def test_fail_fast_preserves_primary_failure_and_drains_siblings():
    from core.segment_pipeline import run_bounded_segments
    from core.async_utils import await_completion
    async def scenario():
        ready, fail, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
        active = 0
        async def process(index, value):
            nonlocal active
            active += 1
            try:
                if index == 0:
                    await fail.wait()
                    raise RuntimeError('primary')
                ready.set()
                await await_completion(finish.wait())
            finally:
                active -= 1
        task = asyncio.create_task(run_bounded_segments([0,1,2], process, 2, fail_fast=True))
        await ready.wait()
        fail.set()
        await asyncio.sleep(0)
        assert not task.done()
        finish.set()
        outcomes = await task
        assert active == 0
        assert isinstance(outcomes[0], RuntimeError)
        assert str(outcomes[0]) == 'primary'
    asyncio.run(scenario())
