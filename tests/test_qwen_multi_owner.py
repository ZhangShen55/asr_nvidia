"""CPU owner doubles only. Parent runs these under its device/weight guard."""
import asyncio
from concurrent.futures import Future
from functools import wraps
from types import SimpleNamespace

import pytest

import core.qwen_asr_client as module
from core.qwen_asr_client import QwenAsrClient, QwenAsrUnavailableError
from core.qwen_asr_queue import AudioSlotManager, QueueFullError, SpkEmoQueueFullError
from core.qwen_asr_runtime import LocalModelUnavailable


def run_async(function):
    """Keep normal pytest fixtures without requiring pytest-asyncio."""
    @wraps(function)
    def synchronous(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return synchronous


class OwnerDouble:
    def __init__(self, index, events, fail=False, **budgets):
        self.index, self.events, self.fail = index, events, fail
        self.budgets = budgets
        self.loaded = False
        self.failure = None
        self.device = 'npu:7'
        self.closing = False
        self.closed = False
        self.close_calls = 0
        self.futures = {}
        self.submissions = []
        self.auto_finish = True
        self.alive = True

    def ensure_loaded(self):
        self.events.append(('load', self.index))
        if self.fail:
            self.failure = 'fake owned load failure'
            raise LocalModelUnavailable(self.failure)
        self.loaded = True

    def submit(self, audio, language=None, *, request_id, segment_id):
        key = (str(request_id), str(segment_id))
        if key in self.futures:
            return self.futures[key]
        self.submissions.append(key)
        future = Future()
        if self.auto_finish:
            future.set_result({'text': f'language English<asr_text>owner {self.index}',
                               'warning': None, 'recovered': False, 'original_raw': 'private'})
        self.futures[key] = future
        return future

    def request_drained(self, request_id):
        return all(value.done() for key, value in self.futures.items() if key[0] == str(request_id))

    def release_request(self, request_id):
        if not self.request_drained(request_id):
            raise RuntimeError('active owned sequence')
        for key in list(self.futures):
            if key[0] == str(request_id):
                self.futures.pop(key)

    def begin_closing(self):
        self.closing = True

    def close(self):
        if self.closed:
            return
        if any(not f.done() for f in self.futures.values()):
            raise RuntimeError('premature owned close')
        self.close_calls += 1
        self.closed = True
        self.alive = False
        self.loaded = False
        self.events.append(('close', self.index))

    def snapshot(self):
        return {'loaded': self.loaded, 'loading': False, 'failure': self.failure,
                'memory_allocated_mb': 0, 'memory_reserved_mb': 0,
                'continuous': {'active_sequences': sum(not f.done() for f in self.futures.values())},
                'model_worker': {'worker_pid': 10000+self.index, 'worker_start_ticks': str(self.index+1),
                                 'worker_alive': self.alive,
                                 'closing': self.closing, 'resource_reaped': self.closed}}


def make_client(monkeypatch, n=3, s=2, fail=None):
    settings = SimpleNamespace(qwen_segment_concurrency=s, qwen_request_concurrency=n*s,
        qwen_pending_segment_limit=256, qwen_max_batch_audio_frames=14400,
        qwen_max_batch_tokens=10000, qwen_continuous_prefill_batch_size=min(s,16),
        qwen_process_shutdown_grace_s=2, qwen_owner_threads=1)
    monkeypatch.setattr(module, 'settings', settings)
    owners, events = [], []
    def factory(*args, **kwargs):
        owner = OwnerDouble(len(owners), events, len(owners) == fail, **kwargs)
        owners.append(owner)
        return owner
    client = QwenAsrClient(local_model_dir='/tmp/no-owner-weights', local_device='npu:7',
        local_dtype='bfloat16', local_instance_count=n, max_new_tokens=256,
        execution_mode='process', generation_mode='continuous', runtime_factory=factory,
        max_inference_batch_size=s, max_batch_audio_frames=14400, max_batch_tokens=10000)
    return client, owners, events


@run_async
async def test_sequential_preload_and_no_status_lazy_start(monkeypatch):
    client, owners, events = make_client(monkeypatch)
    assert client.owner_capacity_snapshot()['healthy'] == 0
    assert client.snapshot()['owner_pool']['initialized'] is False
    assert events == []
    await client.preload()
    assert events == [('load',0),('load',1),('load',2)]
    await client.ensure_owner_pool()
    assert len(events) == 3
    assert client.owner_capacity_snapshot()['available'] == 3
    assert client._batch_scheduler.capacity == 6
    await client.aclose()
    assert all(owner.closed for owner in owners)


@run_async
async def test_partial_preload_failure_rolls_back_even_unstarted_owners(monkeypatch):
    client, owners, events = make_client(monkeypatch, fail=1)
    with pytest.raises(QwenAsrUnavailableError):
        await client.preload()
    assert [item for item in events if item[0]=='load'] == [('load',0),('load',1)]
    assert all(owner.closed for owner in owners)
    with pytest.raises(QwenAsrUnavailableError):
        await client.ensure_owner_pool()
    assert client.owner_capacity_snapshot()['startup_failed']
    assert client.owner_capacity_snapshot()['healthy'] == 0
    await client.aclose()


@run_async
async def test_dynamic_request_binding_reuse_and_dedup_is_not_audio_release(monkeypatch):
    client, owners, events = make_client(monkeypatch,n=2)
    slots = AudioSlotManager(2,spk_emo_capacity=4)
    slots.bind_owner_pool(client)
    first = await slots.acquire('A',request_id='random-first-same-file')
    second = await slots.acquire('A',request_id='random-second-same-file')
    assert first.owner_index != second.owner_index
    member = SimpleNamespace(request_id=first.request_id,segment_index='17',audio=b'pcm',
                             language=None,cancelled=False)
    value = await client._execute_sequence(member)
    assert str(first.owner_index) in value['text']
    client.release_request(first.request_id)
    assert client.owner_capacity_snapshot()['leased'] == 2
    assert slots.active == 2
    await slots.release(first)
    third = await slots.acquire('A',request_id='unrelated-next-file')
    assert third.owner_index == first.owner_index
    await slots.release(first)  # old duplicate cannot unbind the next occupant
    assert client._request_owners[third.request_id] == third.owner_index
    await slots.release(second);await slots.release(third)
    assert slots.active == 0 and client.owner_capacity_snapshot()['leased'] == 0
    await client.aclose()


@run_async
async def test_health_drop_retains_live_lease_and_never_migrates_its_request(monkeypatch):
    client, owners, _ = make_client(monkeypatch,n=3)
    slots = AudioSlotManager(3);slots.bind_owner_pool(client)
    leases=[await slots.acquire('A',request_id=f'r{i}') for i in range(3)]
    bad=owners[leases[0].owner_index]
    future=Future();bad.futures[('r0','0')]=future
    bad.failure='owned worker died';bad.loaded=False
    assert client.owner_capacity_snapshot()['healthy'] == 2
    assert client.owner_capacity_snapshot()['leased'] == 3
    assert slots.snapshot()['audioConcurrency']['active_total'] == 3
    with pytest.raises(RuntimeError,match='active'):
        await slots.release(leases[0])
    assert slots.active == 3
    assert client._request_runtime('r0') is bad
    await slots.release(leases[1])
    replacement=await slots.acquire('A',request_id='healthy-free-while-bad-still-closing')
    assert replacement.owner_index != leases[0].owner_index
    assert slots.active==3 > client.owner_capacity_snapshot()['healthy']
    await slots.release(replacement)
    future.set_exception(LocalModelUnavailable('owned death confirmed'))
    await slots.release(leases[0])
    for lease in leases[1:]:await slots.release(lease)
    bad2=owners[1];bad2.failure='also dead';bad2.loaded=False
    bad3=owners[2];bad3.failure='also dead';bad3.loaded=False
    with pytest.raises(QwenAsrUnavailableError):
        await slots.acquire('A',request_id='must-not-restart')
    assert len([event for event in bad.events if event[0]=='load']) == 3
    await client.aclose()


@run_async
async def test_total_and_b_capacity_share_atomic_owner_grant(monkeypatch):
    client, owners, _ = make_client(monkeypatch,n=6)
    slots=AudioSlotManager(6,spk_emo_capacity=4);slots.bind_owner_pool(client)
    leases=[await slots.acquire('B',request_id=f'b{i}') for i in range(4)]
    leases.append(await slots.acquire('A',request_id='a0'))
    with pytest.raises(SpkEmoQueueFullError):await slots.acquire('B',request_id='fifth-b')
    leases.append(await slots.acquire('A',request_id='a1'))
    with pytest.raises(QueueFullError) as exc:await slots.acquire('B',request_id='full-total')
    assert exc.value.code==4013
    assert len({lease.owner_index for lease in leases})==6
    for lease in leases:await slots.release(lease)
    for owner in owners[3:]:owner.failure='dead';owner.loaded=False
    assert slots.snapshot()['spkEmoConcurrency']['effective']==3
    await client.aclose()


@run_async
async def test_waiting_cancel_and_grant_cancel_do_not_leak_owner(monkeypatch):
    client, _, _=make_client(monkeypatch,n=1)
    slots=AudioSlotManager(1,wait_timeout_s=5);slots.bind_owner_pool(client)
    first=await slots.acquire('A',request_id='first')
    waiting=asyncio.create_task(slots.acquire('A',request_id='withdraw'))
    await asyncio.sleep(0);await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):await waiting
    assert slots.active==1 and slots.waiting==0
    raced=asyncio.create_task(slots.acquire('A',request_id='race'))
    await asyncio.sleep(0);await asyncio.sleep(0)
    slots._queues['A'][0].future.add_done_callback(lambda _:raced.cancel())
    await slots.release(first)
    with pytest.raises(asyncio.CancelledError):await raced
    assert slots.active==0 and client.owner_capacity_snapshot()['leased']==0
    await client.aclose()


@run_async
async def test_shutdown_waits_audio_cleanup_and_repeated_close_is_idempotent(monkeypatch):
    client, owners, _=make_client(monkeypatch,n=2)
    slots=AudioSlotManager(2);slots.bind_owner_pool(client)
    lease=await slots.acquire('A',request_id='cleanup-still-running')
    close=asyncio.create_task(client.aclose())
    await asyncio.sleep(0);await asyncio.sleep(0)
    assert not close.done() and not any(owner.closed for owner in owners)
    assert client.owner_capacity_snapshot()['leased']==1
    await slots.release(lease)
    await close;await client.aclose()
    assert all(owner.close_calls==1 for owner in owners)


def test_worker_capacity_and_thread_budget_are_per_owner(monkeypatch):
    import core.qwen_continuous_process as process
    owners=[]
    def factory(*args,**kwargs):
        owner=OwnerDouble(len(owners),[],**kwargs);owners.append(owner);return owner
    monkeypatch.setattr(module,'settings',SimpleNamespace(qwen_segment_concurrency=2,
        qwen_owner_threads=1,qwen_request_concurrency=6,qwen_pending_segment_limit=256,
        qwen_max_batch_audio_frames=14400,qwen_max_batch_tokens=10000,
        qwen_continuous_prefill_batch_size=2))
    monkeypatch.setattr(process,'LocalQwenContinuousProcessRuntime',factory)
    client=QwenAsrClient(local_model_dir='/tmp/no-owner-weights',local_device='npu:7',
        local_dtype='bfloat16',local_instance_count=3,max_new_tokens=256,
        execution_mode='process',generation_mode='continuous',max_inference_batch_size=2)
    assert len(owners)==3
    assert all(owner.budgets['max_active']==2 and owner.budgets['owner_threads']==1 for owner in owners)
    assert all(owner.budgets['max_batch_audio_frames']==14400 and owner.budgets['max_batch_tokens']==10000 for owner in owners)
    assert client._batch_scheduler.capacity==6


@run_async
async def test_two_owners_each_hold_s16_without_a_shared_input_budget_barrier(monkeypatch):
    import io
    import wave
    client, owners, _=make_client(monkeypatch,n=2,s=16)
    slots=AudioSlotManager(2);slots.bind_owner_pool(client)
    leases=[await slots.acquire('A',request_id=f'audio-{i}') for i in range(2)]
    for owner in owners:owner.auto_finish=False
    stream=io.BytesIO()
    with wave.open(stream,'wb') as writer:
        writer.setnchannels(1);writer.setsampwidth(2);writer.setframerate(16000)
        writer.writeframes(b'\0'*16000)
    jobs=[asyncio.create_task(client.transcribe_bytes(stream.getvalue(),request_id=lease.request_id,
            segment_index=str(index))) for lease in leases for index in range(16)]
    try:
        for _ in range(500):
            if all(len(owner.futures)==16 for owner in owners):break
            await asyncio.sleep(.001)
        assert [len(owner.futures) for owner in owners]==[16,16]
        assert client._batch_scheduler.snapshot()['running_members']==32
        assert all(not value.done() for owner in owners for value in owner.futures.values())
        for owner in owners:
            for value in owner.futures.values():
                value.set_result({'text':'language English<asr_text>result','warning':None})
        assert len(await asyncio.gather(*jobs))==32
        assert client._batch_scheduler.snapshot()['peak_inflight']==32
    finally:
        for owner in owners:
            for value in owner.futures.values():
                if not value.done():value.set_result({'text':'language None<asr_text>','warning':None})
        await asyncio.gather(*jobs,return_exceptions=True)
        for lease in leases:await slots.release(lease)
        await client.aclose()


@run_async
async def test_cancel_during_owner_birth_waits_thread_before_partial_rollback(monkeypatch):
    import threading
    client, owners, _=make_client(monkeypatch,n=3)
    entered=threading.Event();finish=threading.Event()
    original=owners[1].ensure_loaded
    def slow_birth():
        entered.set()
        assert finish.wait(2)
        original()
    owners[1].ensure_loaded=slow_birth
    preload=asyncio.create_task(client.preload())
    while not entered.is_set():await asyncio.sleep(0)
    preload.cancel()
    finish.set()
    with pytest.raises(asyncio.CancelledError):await preload
    assert all(owner.closed for owner in owners)
    assert client.owner_capacity_snapshot()['startup_failed']
    assert not owners[2].loaded
    await client.aclose()


def test_owner_threads_128_boundary_is_constructor_only_no_spawn():
    from core.qwen_continuous_process import LocalQwenContinuousProcessRuntime
    runtime=LocalQwenContinuousProcessRuntime('/tmp/no-owner-weights','npu:7',owner_threads=128)
    assert runtime._process is None and not runtime.loaded
    assert runtime._launch_config['continuous_options']['owner_threads']==128
    runtime.close()
    with pytest.raises(ValueError):
        LocalQwenContinuousProcessRuntime('/tmp/no-owner-weights','npu:7',owner_threads=129)


@run_async
async def test_explicit_dead_worker_cache_is_not_healthy_without_failure_sideeffect(monkeypatch):
    client,owners,_=make_client(monkeypatch,n=2)
    await client.preload()
    owners[0].alive=False
    assert owners[0].loaded and owners[0].failure is None
    assert client.owner_capacity_snapshot()['healthy']==1
    assert client.snapshot()['capacity']==1
    assert client.try_acquire_owner('healthy-only')==1
    await client.prepare_owner_release('healthy-only')
    client.release_owner_nowait('healthy-only')
    await client.aclose()


@run_async
async def test_idle_slot_manager_rebinds_only_after_old_pool_normal_close(monkeypatch):
    old,old_owners,_=make_client(monkeypatch,n=2)
    slots=AudioSlotManager(2);slots.bind_owner_pool(old)
    lease=await slots.acquire('A',request_id='first-lifespan')
    await slots.release(lease);await old.aclose()
    new,new_owners,_=make_client(monkeypatch,n=2)
    slots.bind_owner_pool(new)
    second=await slots.acquire('A',request_id='second-lifespan')
    assert slots._owner_pool is new and new_owners[second.owner_index] is not old_owners[second.owner_index]
    assert all(owner.closed for owner in old_owners)
    with pytest.raises(RuntimeError):slots.bind_owner_pool(old)
    assert new._request_owners[second.request_id]==second.owner_index
    await slots.release(second);await new.aclose()


@run_async
async def test_idle_but_live_pool_cannot_be_rebound(monkeypatch):
    old,_,_=make_client(monkeypatch,n=2)
    new,_,_=make_client(monkeypatch,n=2)
    slots=AudioSlotManager(2);slots.bind_owner_pool(old)
    await old.preload()
    assert slots.active==0 and slots.waiting==0
    with pytest.raises(RuntimeError,match='live or closing'):slots.bind_owner_pool(new)
    assert slots._owner_pool is old
    await old.aclose();await new.aclose()


@run_async
async def test_no_activity_does_not_allow_rebind_while_close_is_still_running(monkeypatch):
    import threading
    old,owners,_=make_client(monkeypatch,n=1)
    new,_,_=make_client(monkeypatch,n=1)
    slots=AudioSlotManager(1);slots.bind_owner_pool(old)
    await old.preload()
    entered=threading.Event();finish=threading.Event()
    original=owners[0].close
    def slow_close():
        entered.set()
        assert finish.wait(2)
        original()
    owners[0].close=slow_close
    closing=asyncio.create_task(old.aclose())
    while not entered.is_set():await asyncio.sleep(0)
    try:
        assert slots.active==0 and slots.waiting==0
        with pytest.raises(RuntimeError,match='live or closing'):slots.bind_owner_pool(new)
    finally:
        finish.set()
    await closing
    slots.bind_owner_pool(new)
    assert slots._owner_pool is new
    await new.aclose()


@run_async
async def test_all_owners_die_after_ensure_but_before_admission_lock(monkeypatch):
    client,owners,_=make_client(monkeypatch,n=1)
    slots=AudioSlotManager(1);slots.bind_owner_pool(client)
    await client.preload()
    ensured=asyncio.Event()
    original=client.ensure_owner_pool
    async def ensure_then_notify():
        value=await original()
        ensured.set()
        return value
    client.ensure_owner_pool=ensure_then_notify
    await slots._lock.acquire()
    caller=asyncio.create_task(slots.acquire('A',request_id='dies-before-slot-lock'))
    await ensured.wait()
    assert client.owner_capacity_snapshot()['healthy']==1
    owners[0].failure='died after discovery';owners[0].loaded=False;owners[0].alive=False
    slots._lock.release()
    with pytest.raises(QwenAsrUnavailableError) as error:
        await caller
    assert error.value.code==4011
    assert slots.active==0 and slots.waiting==0
    assert client.owner_capacity_snapshot()['leased']==0
    await client.aclose()
