"""384是有预算的实际批，不把逻辑窗口或PCM等待量冒充执行量。全为CPU。"""
import asyncio
from collections import Counter
from copy import deepcopy
import io
from pathlib import Path
import wave

import pytest

from core.config import Settings
from core.qwen_asr_queue import RequestConcurrency
from core.qwen_batch_scheduler import BatchScheduler
from core.segment_pipeline import PendingSegmentBudget

ROOT = Path(__file__).resolve().parents[1]


def config(tmp_path, literal):
    p = tmp_path / 'candidate.toml'
    p.write_text('[qwen_asr]\ngeneration_mode="legacy"\nmax_inference_batch_size=' + literal + '\n')
    return Settings(str(p))


@pytest.mark.parametrize('literal, expected', [('1', 1), ('128', 128), ('129', 129), ('256', 256), ('384', 384)])
def test_config_supports_measured_384_bound_without_changing_default(tmp_path, literal, expected):
    settings = config(tmp_path, literal)
    settings.validate_model_backends()
    assert settings.qwen_max_inference_batch_size == expected


@pytest.mark.parametrize('literal', ['0', '385', '512', 'true', 'false', '1.5', '\"384\"'])
def test_bad_or_unmeasured_physical_batch_rejected_before_loading(tmp_path, literal):
    with pytest.raises(ValueError, match='max_inference_batch_size'):
        config(tmp_path, literal).validate_model_backends()


@pytest.mark.parametrize('name', ['config.toml', 'docker/config.toml'])
def test_shipped_templates_use_formal_multi_owner_profile(name):
    settings = Settings(str(ROOT / name))
    settings.validate_model_backends()
    assert settings.qwen_generation_mode == 'continuous'
    assert settings.qwen_local_instance_count == 4
    assert settings.qwen_segment_concurrency == 16
    assert settings.qwen_request_concurrency == 64
    assert settings.qwen_max_inference_batch_size == 16
    assert settings.qwen_preload_on_startup is True
    qwen = settings._cfg['qwen_asr']
    expected = dict(max_batch_audio_frames=14400,
                    max_batch_tokens=10000,
                    pending_segment_limit=256,
                    pending_audio_bytes_limit=268435456)
    assert {k: qwen[k] for k in expected} == expected
    assert not {'audio_concurrency', 'request_concurrency', 'max_inference_batch_size',
                'continuous_prefill_batch_size'}.intersection(qwen)


def test_384_profiles_have_room_beyond_384_and_only_B_admission_differs():
    configurations = []
    for name, limit in [('config.qwen384.toml', 4), ('config.qwen384-b8.toml', 8)]:
        cfg = Settings(str(ROOT / 'test' / name))
        # 384静态批是历史legacy候选，不继承新的continuous缺省。
        cfg._cfg['qwen_asr']['generation_mode'] = 'legacy'
        cfg.validate_model_backends()
        assert cfg.qwen_local_instance_count == 1
        assert cfg.qwen_execution_mode == 'process'
        assert cfg.qwen_preload_on_startup is True
        assert cfg.qwen_segment_concurrency == 48
        assert cfg.qwen_request_concurrency == cfg.qwen_max_inference_batch_size == 384
        assert cfg.qwen_audio_concurrency == 8 and cfg.qwen_audio_spk_emo_concurrency == limit
        assert cfg.qwen_max_batch_audio_frames == 345600
        assert cfg.qwen_max_batch_tokens == 240000
        assert cfg.qwen_pending_segment_limit == 512
        assert cfg.qwen_pending_audio_bytes_limit == 268435456
        assert cfg.vad_device == ['cpu'] * 8
        assert cfg.model_device_speech_campplus_devices == ['npu:7'] * 8
        assert cfg.model_device_emotion2vec_devices == ['npu:7'] * 8
        assert cfg.speech_campplus_batch_size == cfg.emotion2vec_batch_size == 1
        body = deepcopy(cfg._cfg)
        body['qwen_asr'].pop('audio_spk_emo_concurrency')
        configurations.append(body)
    assert configurations[0] == configurations[1]


def wav(seconds=.3):
    output = io.BytesIO()
    with wave.open(output, 'wb') as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b'\0\0' * round(seconds * 16000))
    return output.getvalue()


def scheduler(execute, **changes):
    args = dict(workers=1, batch_size=384, wait_ms=100,
                audio_frame_budget=345600, token_budget=240000,
                max_new_tokens=256, capacity=512, duration_ratio=32)
    args.update(changes)
    return BatchScheduler(execute, **args)


async def until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(.001)
    await asyncio.wait_for(poll(), 3)


def test_eight_owners_each48_execute_one384_and_extra_segments_wait_for_done():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        seen = []
        async def execute(members, _):
            seen.append([(m.request_id, m.segment_index) for m in members])
            if len(seen) == 1:
                entered.set()
                await release.wait()
            return [f'{m.request_id}:{m.segment_index}' for m in members]
        sched = scheduler(execute)
        gate = RequestConcurrency(48, 384)
        semaphores = {str(i): gate.new_segment_semaphore() for i in range(9)}
        budget = PendingSegmentBudget(512, 268435456)
        audio = wav()
        async def submit(owner, index):
            sem = semaphores[owner]
            await gate.acquire(sem)
            lease = None
            try:
                lease = await budget.acquire(4800 * 4)
                return await sched.submit(audio, None, request_id=owner, segment_index=index)
            finally:
                if lease is not None:
                    await budget.release(lease)
                await gate.release(sem)
        tasks = [asyncio.create_task(submit(str(r), i)) for r in range(8) for i in range(48)]
        extras = []
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert len(seen[0]) == 384
            assert Counter(r for r, _ in seen[0]) == {str(r): 48 for r in range(8)}
            assert [r for r, _ in seen[0][:8]] == [str(r) for r in range(8)]
            assert budget.snapshot()['active'] == 384
            assert sched.snapshot()['running_members'] == 384
            extras = [asyncio.create_task(submit('0', 48)), asyncio.create_task(submit('8', 0))]
            await asyncio.sleep(.01)
            assert all(not t.done() for t in extras)
            assert sched.snapshot()['inflight'] == 384
            release.set()
            outputs = await asyncio.gather(*tasks)
            assert outputs == [f'{r}:{i}' for r in range(8) for i in range(48)]
            assert await asyncio.gather(*extras) == ['0:48', '8:0']
            assert budget.snapshot()['active'] == budget.snapshot()['audio_bytes'] == 0
            assert gate._global._value == 384
            assert all(s._value == 48 for s in semaphores.values())
            assert sched.snapshot()['inflight'] == sched.snapshot()['running_members'] == 0
        finally:
            release.set()
            await asyncio.gather(*tasks, *extras, return_exceptions=True)
            await sched.close()
    asyncio.run(run())


def test_old_256_pending_capacity_cannot_be_reported_as_real384():
    async def run():
        batches = []
        async def execute(members, _):
            batches.append(len(members))
            return [m.segment_index for m in members]
        sched = scheduler(execute, capacity=256, wait_ms=10)
        try:
            audio = wav()
            result = await asyncio.gather(*(sched.submit(audio, None, request_id=str(r), segment_index=i)
                                          for r in range(8) for i in range(48)))
            assert result == list(range(48)) * 8
            assert sum(batches) == 384 and max(batches) <= 256
            assert 384 not in batches
        finally:
            await sched.close()
    asyncio.run(run())


@pytest.mark.parametrize('smaller', ['frames', 'tokens'])
def test_long_inputs_split_at_independent_frame_or_token_budget(smaller):
    async def run():
        batches = []
        async def execute(members, _):
            batches.append(len(members))
            return [m.segment_index for m in members]
        changes = {'audio_frame_budget': 172800} if smaller == 'frames' else {'token_budget': 120000}
        sched = scheduler(execute, **changes)
        try:
            # 801有效帧向900补齐；每成员估算token为ceil(900/8)+256+256=625。
            audio = wav(8.01)
            result = await asyncio.gather(*(sched.submit(audio, None, request_id=str(r), segment_index=i)
                                          for r in range(8) for i in range(48)))
            assert result == list(range(48)) * 8
            assert batches == [192, 192]
        finally:
            await sched.close()
    asyncio.run(run())


def test_512_PCM_slots_are_bounded_and_refill_immediately():
    async def run():
        budget = PendingSegmentBudget(512, 268435456)
        leases = [await budget.acquire(256) for _ in range(512)]
        extra = asyncio.create_task(budget.acquire(256))
        try:
            await until(lambda: budget.snapshot()['waiting'] == 1)
            assert not extra.done()
            assert budget.snapshot()['active'] == budget.snapshot()['peak'] == 512
            await budget.release(leases.pop())
            leases.append(await asyncio.wait_for(extra, 1))
            assert budget.snapshot()['active'] == 512
        finally:
            extra.cancel()
            await asyncio.gather(extra, return_exceptions=True)
            for lease in leases:
                await budget.release(lease)
        assert budget.snapshot()['active'] == budget.snapshot()['audio_bytes'] == 0
    asyncio.run(run())


def test_PCM_byte_budget_can_backpressure_before512_slots_without_deadlock():
    async def run():
        size = 4800 * 4
        budget = PendingSegmentBudget(512, size * 2)
        first, second = await budget.acquire(size), await budget.acquire(size)
        third = asyncio.create_task(budget.acquire(size))
        try:
            await until(lambda: budget.snapshot()['waiting'] == 1)
            assert budget.snapshot()['active'] == 2 and not third.done()
            await budget.release(first)
            first = await asyncio.wait_for(third, 1)
        finally:
            third.cancel()
            await asyncio.gather(third, return_exceptions=True)
            await budget.release(first)
            await budget.release(second)
        assert budget.snapshot()['active'] == budget.snapshot()['audio_bytes'] == 0
    asyncio.run(run())


def test_repeated_cancel_in384_keeps_execution_until_real_done():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        async def execute(members, _):
            entered.set()
            await release.wait()
            return [m.segment_index for m in members]
        sched = scheduler(execute)
        audio = wav()
        tasks = [asyncio.create_task(sched.submit(audio, None, request_id=str(i % 8), segment_index=i))
                 for i in range(384)]
        try:
            await asyncio.wait_for(entered.wait(), 3)
            tasks[0].cancel()
            await asyncio.sleep(0)
            tasks[0].cancel()
            await asyncio.sleep(.01)
            assert not tasks[0].done()
            assert sched.snapshot()['running_members'] == sched.snapshot()['inflight'] == 384
            release.set()
            result = await asyncio.gather(*tasks, return_exceptions=True)
            assert isinstance(result[0], asyncio.CancelledError)
            assert result[1:] == list(range(1, 384))
            assert sched.snapshot()['inflight'] == sched.snapshot()['running_members'] == 0
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await sched.close()
    asyncio.run(run())
