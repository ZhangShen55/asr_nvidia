"""Warnings preserve original segment identity without weakening fatal failures."""
import asyncio
import copy
from unittest.mock import AsyncMock, patch

from api.routes import asr_qwen_common as pipeline
from core.config import settings
from core.qwen_asr_client import QwenAsrCallError
from tests.test_v3_contract_snapshot import run_contract


def structured(text, reason=None, action='repetition_collapsed', recovered=True):
    return dict(text=text, warning=None if reason is None else dict(
        reason=reason, action=action, recovered=recovered),
        recovered=recovered, original_raw='PRIVATE_ORIGINAL_RAW_MUST_NOT_LEAK')


async def run_continuous(**kwargs):
    cfg = dict(settings._cfg.get('qwen_asr', {}), generation_mode='continuous')
    with patch.dict(settings._cfg, {'qwen_asr': cfg}), patch.object(pipeline, 'release_qwen_request', AsyncMock()):
        return await run_contract(**kwargs)


def test_mixed_recovered_warning_preserves_boundaries_sort_and_continues():
    async def scenario():
        result, _, _, _, calls = await run_continuous(raw_texts=[
            structured('language Chinese<asr_text>可信内容。', 'repetition_stop'),
            structured('language English<asr_text>Hello class.')])
        assert len(calls) == 2
        assert len(result['segments']) == 2
        assert result['segments'][0]['bg'] == '0.00'
        assert result['segments'][0]['ed'] == '1.00'
        assert result['processing_status'] == 'completed_with_warnings'
        assert result['segment_warnings'] == [dict(segment_index=0,bg='0.00',ed='1.00',
            reason='repetition_stop',action='repetition_collapsed',recovered=True)]
        assert 'PRIVATE_ORIGINAL_RAW' not in str(result)
        assert 'original_raw' not in str(result)
    asyncio.run(scenario())


def test_all_unrecoverable_is_explicit_completed_with_warnings_not_no_speech():
    async def scenario():
        result, _, _, _, calls = await run_continuous(raw_texts=[
            structured('language None<asr_text>', 'length_limit','unrecoverable',False),
            structured('language None<asr_text>', 'invalid_output_prefix','unrecoverable',False)])
        assert len(calls) == 2
        assert 'code' not in result
        assert result['segments'] == []
        assert result['text'] == ''
        assert result['processing_status'] == 'completed_with_warnings'
        assert [row['segment_index'] for row in result['segment_warnings']] == [0,1]
        assert [row['bg'] for row in result['segment_warnings']] == ['0.00','1.50']
        assert [row['ed'] for row in result['segment_warnings']] == ['1.00','3.00']
        assert all(not row['recovered'] for row in result['segment_warnings'])
        assert 'PRIVATE_ORIGINAL_RAW' not in str(result)
    asyncio.run(scenario())


def test_normal_continuous_adds_complete_status_without_changing_legacy():
    async def scenario():
        legacy, *_ = await run_contract()
        normal, *_ = await run_continuous(raw_texts=[
            structured('language Chinese<asr_text>，今天学习。'),
            structured('language English<asr_text>Hello class.')])
        assert 'processing_status' not in legacy
        assert 'segment_warnings' not in legacy
        assert normal.pop('processing_status') == 'complete'
        assert normal.pop('segment_warnings') == []
        assert normal == legacy
    asyncio.run(scenario())


def test_fatal_inference_error_is_not_downgraded_to_warning():
    async def scenario():
        result, *_ = await run_continuous(error=QwenAsrCallError('fatal device error'))
        assert result['code'] == 4012
        assert 'processing_status' not in result
        assert 'segment_warnings' not in result
        for reason in ('request_timeout', 'cancelled', 'prefill_error', 'decode_error'):
            result, *_ = await run_continuous(raw_texts=[
                structured('language None<asr_text>',reason,'unrecoverable',False),
                structured('language Chinese<asr_text>正常。')])
            assert result['code'] == 4012
            assert 'processing_status' not in result
    asyncio.run(scenario())
