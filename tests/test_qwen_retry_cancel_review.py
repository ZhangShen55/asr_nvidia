"""只用CPU桩：失败backoff期间的取消不可进入下个物理attempt。"""
import asyncio
import pytest

from core.qwen_asr_client import QwenAsrClient
from tests.test_qwen_batch_scheduler import wav


@pytest.mark.parametrize('cancel_all', [False, True])
def test_retry_refilters_cancelled_members_and_preserves_result_owner(cancel_all):
    async def scenario():
        class Runtime:
            def __init__(self, *args):
                self.failure = None
                self.calls = []
            def transcribe_batch(self, audio, languages):
                self.calls.append(list(languages))
                if len(self.calls) == 1:
                    raise RuntimeError('intentional first attempt failure')
                # 实际runtime合法成员索引属于当前attempt，必须正确映射到幸存者。
                self.last_batch_timings = {'valid_member_indices':list(range(len(audio))), 'batch_size':len(audio)}
                return [f'language {language}<asr_text>{language}' for language in languages]
            def snapshot(self):
                return {'loaded':True,'loading':False,'failure':None}
            def close(self):
                pass
        client = QwenAsrClient(generation_mode="legacy", execution_mode="thread", runtime_factory=Runtime, retry_count=1, local_instance_count=1,
                               max_inference_batch_size=2, batch_wait_timeout_ms=10)
        first = asyncio.create_task(client.transcribe_bytes(wav(), language='Chinese',request_id='a',segment_index=0))
        second = asyncio.create_task(client.transcribe_bytes(wav(), language='English',request_id='b',segment_index=1))
        try:
            async def failed_once():
                while not client._execution_failures:
                    await asyncio.sleep(.001)
            await asyncio.wait_for(failed_once(), 1)
            first.cancel()
            if cancel_all:
                second.cancel()
            outcomes = await asyncio.gather(first, second, return_exceptions=True)
            assert isinstance(outcomes[0], asyncio.CancelledError)
            if cancel_all:
                assert isinstance(outcomes[1], asyncio.CancelledError)
                assert client._runtimes[0].calls == [['Chinese','English']]
            else:
                assert outcomes[1] == 'language English<asr_text>English'
                assert client._runtimes[0].calls == [['Chinese','English'],['English']]
                recent = client._batch_scheduler.snapshot()['recent_batches'][-1]
                assert [a['size'] for a in recent['dispatch_attempts']] == [2,1]
                assert recent['selected_size'] == 2
                assert recent['actual_dispatched_size'] == 1
                assert recent['dispatched_members'] == [{'request_id':'b','segment_index':1}]
            assert client._batch_scheduler.snapshot()['inflight'] == 0
            assert not any(client._busy)
        finally:
            first.cancel();second.cancel()
            await asyncio.gather(first, second, return_exceptions=True)
            await client.aclose()
    asyncio.run(scenario())


def test_scheduler_error_copy_strips_chain_without_touching_active_original():
    from core.qwen_batch_scheduler import detached_exception
    from core.qwen_asr_client import QwenAsrCallError
    try:
        try:
            raise ValueError('original cause')
        except ValueError as cause:
            raise QwenAsrCallError('test message') from cause
    except QwenAsrCallError as original:
        traceback, cause, context = original.__traceback__, original.__cause__, original.__context__
        first, second = detached_exception(original), detached_exception(original)
        assert first is not second and first is not original
        assert first.__traceback__ is first.__cause__ is first.__context__ is None
        assert second.__traceback__ is second.__cause__ is second.__context__ is None
        assert first.code == 4012 and first.message == 'test message'
        assert original.__traceback__ is traceback and original.__cause__ is cause and original.__context__ is context


@pytest.mark.parametrize('cancel_all', [False, True])
def test_actual_runtime_failed_decode_keeps_attempt_timing_ownership_on_retry_cancel(cancel_all):
    from unittest.mock import patch
    from tests.test_qwen_batch_runtime import ClientLanguageIsolationTests
    async def scenario():
        runtime, _ = ClientLanguageIsolationTests().runtime()
        generate_calls = []
        generate = runtime.model.generate
        def capture_generate(**kwargs):
            generate_calls.append(kwargs['input_ids'].shape[0])
            return generate(**kwargs)
        runtime.model.generate = capture_generate
        decode_calls = []
        def decode(ids, **kwargs):
            decode_calls.append(ids.shape[0])
            if len(decode_calls) == 1:
                raise RuntimeError('first read/decode fails after actual generate')
            return ['language English<asr_text>survivor'] * ids.shape[0]
        runtime.processor.batch_decode = decode
        client = QwenAsrClient(generation_mode="legacy", execution_mode="thread", runtime_factory=lambda *_:runtime, retry_count=1, local_instance_count=1,
                               max_inference_batch_size=2, batch_wait_timeout_ms=10)
        first = asyncio.create_task(client.transcribe_bytes(wav(),language='auto',request_id='a',segment_index=0))
        second = asyncio.create_task(client.transcribe_bytes(wav(),language='en',request_id='b',segment_index=1))
        try:
            with patch('core.qwen_asr_runtime.move_inputs',side_effect=lambda inputs,**kwargs:inputs):
                async def failed_once():
                    while not client._execution_failures:
                        await asyncio.sleep(.001)
                await asyncio.wait_for(failed_once(),1)
                first.cancel()
                if cancel_all:
                    second.cancel()
                results = await asyncio.gather(first,second,return_exceptions=True)
            assert isinstance(results[0],asyncio.CancelledError)
            batch = client._batch_scheduler.snapshot()['recent_batches'][-1]
            attempts = batch['dispatch_attempts']
            assert batch['selected_size'] == 2
            assert attempts[0]['status'] == 'failed'
            assert attempts[0]['actual_dispatched_size'] == 2
            assert attempts[0]['members'] == [{'request_id':'a','segment_index':0},{'request_id':'b','segment_index':1}]
            first_timing = attempts[0]['runtime_timings']
            assert first_timing['valid_member_indices'] == [0,1]
            assert first_timing['processor_shapes']['input_ids'] == [2,2]
            assert first_timing['generate_started'] is first_timing['generate_completed'] is True
            assert first_timing['generate_wall_ms'] >= 0
            if cancel_all:
                assert isinstance(results[1],asyncio.CancelledError)
                assert generate_calls == [2] and decode_calls == [2]
                assert len(attempts) == 1
                assert batch['actual_dispatched_size'] == batch['size'] == 2
                assert batch['cancelled_before_retry'] == 2
                assert batch['retry_skipped_all_cancelled'] is True
                assert batch['runtime_timings']['valid_member_indices'] == [0,1]
            else:
                assert results[1] == 'language English<asr_text>survivor'
                assert generate_calls == [2,1] and decode_calls == [2,1]
                assert len(attempts) == 2 and attempts[1]['status'] == 'completed'
                assert attempts[1]['actual_dispatched_size'] == 1
                assert attempts[1]['members'] == [{'request_id':'b','segment_index':1}]
                assert attempts[1]['runtime_timings']['valid_member_indices'] == [0]
                assert batch['dispatched_members'] == [{'request_id':'b','segment_index':1}]
                assert batch['cancelled_before_retry'] == 1
                assert batch['retry_skipped_all_cancelled'] is False
            assert client._batch_scheduler.snapshot()['inflight'] == 0
        finally:
            first.cancel();second.cancel()
            await asyncio.gather(first,second,return_exceptions=True)
            await client.aclose()
    asyncio.run(scenario())
