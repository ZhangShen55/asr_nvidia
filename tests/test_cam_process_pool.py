"""Pool/main integration ownership with CPU stand-ins; never initializes an NPU."""
import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from core import models
from core.async_utils import run_in_thread
from core.local_inference import InstanceFailure
from core.segment_pipeline import PendingSegmentBudget


class FakeProcessRunner:
    execution_mode = 'process'
    effective_backend = 'torch_npu'
    actual_precision = 'fp32'
    math_mode = 'fp32_ops_hf32_disabled'
    healthy = True
    failure = None

    def __init__(self, directory, device, **kwargs):
        self.directory, self.device, self.options = directory, device, kwargs
        self.loaded = self.alive = self.closing = self.closed = False
        self.reaped = True
        self.normal = None
        self.forced = False
        self.calls = self.snapshot_calls = self.request_count = self.close_count = 0
        self.deadline = None
        self.load_gate = self.forward_gate = None
        self.loading_started = threading.Event()
        self.forward_started = threading.Event()
        self.close_started = threading.Event()
        self.forward_done = threading.Event()
        self.on_ensure = None
        self.close_error = None

    def ensure_loaded(self):
        self.loading_started.set()
        if self.on_ensure:
            self.on_ensure()
        if self.load_gate is not None:
            assert self.load_gate.wait(5)
        if self.closing:
            raise InstanceFailure('closing before READY')
        self.loaded = self.alive = True
        self.reaped = False

    def generate(self, *, input, fs=16000):
        self.forward_started.set()
        try:
            if self.forward_gate is not None:
                assert self.forward_gate.wait(5)
            self.calls += 1
            if self.forced:
                raise InstanceFailure('owned worker died')
            return [{'spk_embedding':torch.arange(192,dtype=torch.float32).view(1,192)}]
        finally:
            self.forward_done.set()

    def request_close(self):
        self.request_count += 1
        if self.deadline is None:
            self.deadline = time.monotonic()+self.options.get('shutdown_grace_s',30)
        self.closing = True

    def close(self):
        self.close_started.set()
        self.close_count += 1
        if self.load_gate is not None:
            self.load_gate.set()
        if self.forward_started.is_set() and not self.forward_done.is_set():
            assert self.forward_done.wait(5)
        self.loaded = self.alive = False
        self.closed = self.reaped = True
        self.normal = not self.forced and self.close_error is None
        if self.close_error:
            raise RuntimeError(self.close_error)

    def snapshot(self):
        self.snapshot_calls += 1
        return {'loaded':self.loaded,'worker_alive':self.alive,'closing':self.closing,
                'closed':self.closed,'resource_reaped':self.reaped,
                'normal_exit_ok':self.normal,'forced_cleanup':self.forced,
                'worker_exitcode':7 if self.forced else (0 if self.reaped else None),
                'worker_identity':{'pid':id(self),'create_time':1.0,'start_ticks':'fake','sid':0}}

    @property
    def memory(self):
        return self.snapshot()


def setup_pool(monkeypatch, count=2, depth=0, factory=None):
    created=[]
    def build(*args,**kwargs):
        runner=(factory or FakeProcessRunner)(*args,**kwargs)
        created.append(runner)
        return runner
    monkeypatch.setattr(models,'settings',SimpleNamespace(
        model_device_speech_campplus_execution_mode='process',open_v2_role=True,open_v2_emotion=True,
        model_device_speech_campplus_process_startup_timeout_s=60,
        model_device_speech_campplus_process_shutdown_grace_s=30))
    monkeypatch.setattr(models,'CamProcessRunner',build)
    monkeypatch.setattr(models,'device_allocated_mb',lambda *_:pytest.fail('parent allocator touched'))
    monkeypatch.setattr(models,'_resolve_device',lambda *_:pytest.fail('parent NPU discovery touched'))
    pool=models.ModelPool('speech_campplus','unused',devices_getter=lambda:['npu:0']*count,
                          prefetch_depth_getter=lambda:depth)
    return pool,created


async def eventually(predicate):
    for _ in range(2000):
        if predicate():
            return
        await asyncio.sleep(.001)
    raise AssertionError('CPU gate not reached')


def test_process_branch_registry_precedes_birth_and_parent_npu_is_never_touched(monkeypatch):
    async def scenario():
        pool,created=setup_pool(monkeypatch)
        original=models.CamProcessRunner
        def build(*args,**kwargs):
            runner=original(*args,**kwargs)
            runner.on_ensure=lambda:pytest.fail('unregistered birth') if pool._process_entry(runner) is None else None
            return runner
        monkeypatch.setattr(models,'CamProcessRunner',build)
        assert not created
        await pool.ensure_loaded()
        assert len(created)==2 and all(r.device=='npu:0' for r in created)
        assert all(r.options=={'startup_timeout_s':60,'shutdown_grace_s':30}for r in created)
        status=models.local_model_status['speech_campplus']
        assert status['instances']==status['available_instances']==2
        assert status['allocated_mb'] is None and len(status['model_workers'])==2
        await pool.close()
        assert all(r.reaped and r.normal for r in created)
        with pytest.raises(models.ModelUnavailableError):await pool.ensure_loaded()
    asyncio.run(scenario())


def test_loading_close_registers_and_closes_before_load_lock_without_late_ready(monkeypatch):
    async def scenario():
        gate=threading.Event()
        def factory(*args,**kwargs):
            r=FakeProcessRunner(*args,**kwargs);r.load_gate=gate;return r
        pool,created=setup_pool(monkeypatch,count=8,factory=factory)
        loading=asyncio.create_task(pool.ensure_loaded())
        await eventually(lambda:created and created[0].loading_started.is_set())
        pool.refresh_status();status=models.local_model_status['speech_campplus']
        assert status['instance_details'][0]['health']=='loading'
        assert status['failed_instances']==0 and status['instance_details'][0]['failure'] is None
        closing=asyncio.create_task(pool.close())
        with pytest.raises(models.ModelUnavailableError):await loading
        await closing
        assert len(created)==1 and created[0].close_started.is_set()
        assert created[0].reaped and not created[0].loaded and not pool.models
    asyncio.run(scenario())


def test_pool_close_starts_all_process_watchers_before_waiting_for_active_leases(monkeypatch):
    async def scenario():
        pool,created=setup_pool(monkeypatch,count=8)
        await pool.ensure_loaded()
        entered=asyncio.Event();release=asyncio.Event();count=0
        async def work():
            nonlocal count
            async with pool.lease():
                count+=1
                if count==8:entered.set()
                await release.wait()
        tasks=[asyncio.create_task(work())for _ in range(8)]
        await entered.wait();close=asyncio.create_task(pool.close())
        await eventually(lambda:all(r.close_started.is_set()for r in created))
        assert not close.done() and all(r.request_count==1 for r in created)
        release.set();await asyncio.gather(*tasks);await close
        assert all(r.close_count==1 for r in created)
    asyncio.run(scenario())


def test_repeat_cancel_keeps_lease_and_pcm_until_actual_forward_and_close_finish(monkeypatch):
    async def scenario():
        pool,created=setup_pool(monkeypatch,count=1);await pool.ensure_loaded()
        runner=created[0];runner.forward_gate=threading.Event();budget=PendingSegmentBudget(1,100)
        async def work():
            token=await budget.acquire(10)
            try:
                async with pool.lease()as(runner,_):
                    return await run_in_thread(runner.generate,input=None)
            finally:await budget.release(token)
        task=asyncio.create_task(work());await eventually(runner.forward_started.is_set)
        close=asyncio.create_task(pool.close());await eventually(runner.close_started.is_set)
        task.cancel();await asyncio.sleep(0);task.cancel();close.cancel();await asyncio.sleep(0)
        assert not task.done() and not close.done() and pool.locks[0].locked()
        assert budget.snapshot()['active']==1 and not runner.reaped
        pool.refresh_status();status=models.local_model_status['speech_campplus']
        assert status['pending_close_instances']==1 and status['available_instances']==0
        assert status['instance_details'][0]['health']=='closing'
        assert status['failed_instances']==0 and status['instance_details'][0]['failure'] is None
        runner.forward_gate.set()
        with pytest.raises(asyncio.CancelledError):await task
        with pytest.raises(asyncio.CancelledError):await close
        assert budget.snapshot()['active']==0 and runner.reaped and not pool.models
        pool.refresh_status();status=models.local_model_status['speech_campplus']
        assert status['failed_instances']==0 and status['instance_details'][0]['health']=='closed'
        assert status['instance_details'][0]['failure'] is None
    asyncio.run(scenario())


def test_idle_dead_handle_is_not_healthy_or_available_and_never_respawns(monkeypatch):
    async def scenario():
        pool,created=setup_pool(monkeypatch,count=2);await pool.ensure_loaded()
        created[0].alive=False;created[0].forced=True
        pool.refresh_status();state=models.local_model_status['speech_campplus']
        assert state['instances']==state['available_instances']==1 and state['failed_instances']==1
        async with pool.lease()as(runner,_):assert runner is created[1]
        assert len(created)==2
        with pytest.raises(RuntimeError,match='关闭失败'):await pool.close()
        assert len(models.local_model_status['speech_campplus']['model_workers'])==2
        with pytest.raises(RuntimeError):await pool.close()
        assert created[0].close_count==1
    asyncio.run(scenario())


def test_retired_forced_worker_history_survives_slot_none_and_repeated_close(monkeypatch):
    async def scenario():
        pool,created=setup_pool(monkeypatch,count=1);await pool.ensure_loaded();r=created[0]
        with pytest.raises(InstanceFailure):
            async with pool.lease():
                r.forced=True;r.alive=False;r.reaped=True;r.normal=False
                raise InstanceFailure('controlled owned worker death')
        assert pool.models==[None]
        for _ in range(2):
            with pytest.raises(RuntimeError,match='prior abnormal'):await pool.close()
        assert r.close_count==1 and len(pool._owned_process_entries())==1
        assert models.local_model_status['speech_campplus']['model_workers'][0]['close_error']
    asyncio.run(scenario())


def test_long_backpressure_queue_condition_wakes_do_not_rescan_worker_liveness(monkeypatch):
    async def scenario():
        pool,created=setup_pool(monkeypatch,count=8);await pool.ensure_loaded()
        hold=asyncio.Event();running=0;ready=asyncio.Event()
        async def work():
            nonlocal running
            async with pool.lease():
                running+=1
                if running==8:ready.set()
                await hold.wait()
        tasks=[asyncio.create_task(work())for _ in range(136)]
        await ready.wait();await eventually(lambda:len(pool._admission_waiters)==128)
        before=sum(r.snapshot_calls for r in created)
        for _ in range(8):
            async with pool._condition:pool._condition.notify_all()
            await asyncio.sleep(0)
        assert sum(r.snapshot_calls for r in created)==before
        hold.set();await asyncio.gather(*tasks);await pool.close()
    asyncio.run(scenario())


def test_lifespan_starts_process_pool_close_before_waiting_for_vad_shutdown(monkeypatch):
    import main
    async def scenario():
        pool_started=asyncio.Event();vad_started=asyncio.Event();release=asyncio.Event()
        async def noop():pass
        async def pools():pool_started.set();await release.wait()
        async def vad():
            vad_started.set()
            assert pool_started.is_set()
            await release.wait()
        class Sampler:
            def __init__(self,**kwargs):pass
            async def aclose(self):pass
        cfg=SimpleNamespace(qwen_audio_spk_emo_concurrency=4,validate_model_backends=lambda:None,
                            log_path='unused',resource_sample_interval_s=1,qwen_execution_mode='thread',
                            model_device_speech_campplus_execution_mode='process')
        monkeypatch.setattr(main,'settings',cfg);monkeypatch.setattr(main,'validate_npu_runtime',lambda _:None)
        for name in ['_set_process_name','reset_stats','setup_logging','cleanup_stale_service_cache']:
            monkeypatch.setattr(main,name,lambda *args:None)
        monkeypatch.setattr(main,'ResourceSnapshotSampler',Sampler)
        monkeypatch.setattr(main,'preload_v3_model_pools',noop)
        monkeypatch.setattr(main,'start_vad_pool',noop);monkeypatch.setattr(main,'stop_vad_pool',vad)
        monkeypatch.setattr(main,'close_v3_model_pools',pools);monkeypatch.setattr(main,'close_qwen_client',noop)
        async def application():
            async with main.lifespan(SimpleNamespace(state=SimpleNamespace())):pass
        task=asyncio.create_task(application());await vad_started.wait()
        assert not task.done();release.set();await task
    asyncio.run(scenario())


def test_incomplete_close_never_releases_retired_lease_or_pcm_before_death(monkeypatch):
    async def scenario():
        pool,created=setup_pool(monkeypatch,count=1);await pool.ensure_loaded();runner=created[0]
        def incomplete_close():
            runner.close_started.set()
            raise RuntimeError('test double failed without confirming death')
        runner.close=incomplete_close
        budget=PendingSegmentBudget(1,100)
        async def work():
            lease=await budget.acquire(10)
            try:
                async with pool.lease():raise InstanceFailure('controlled unconfirmed failure')
            finally:await budget.release(lease)
        task=asyncio.create_task(work());await eventually(runner.close_started.is_set)
        assert not task.done() and pool.locks[0].locked() and budget.snapshot()['active']==1
        runner.alive=False;runner.reaped=True;runner.normal=False
        with pytest.raises(InstanceFailure):await task
        assert budget.snapshot()['active']==0
        with pytest.raises(RuntimeError):await pool.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('depth',[0,1,5])
def test_process_handles_preserve_fifo_prefetch_capacity_and_continuous_handoff(monkeypatch,depth):
    async def scenario():
        pool,created=setup_pool(monkeypatch,count=2,depth=depth);await pool.ensure_loaded()
        gate=asyncio.Event();order=[]
        async def work(index):
            async with pool.lease():
                order.append(index)
                if index<2:await gate.wait()
        tasks=[asyncio.create_task(work(i))for i in range(16)]
        await eventually(lambda:len(order)==2 and pool.waiting==2*depth
                         and len(pool._admission_waiters)==14-2*depth)
        gate.set();await asyncio.gather(*tasks);await pool.close()
        assert order==list(range(16)) and all(r.normal for r in created)
    asyncio.run(scenario())


def test_disabled_process_capability_does_not_construct_any_runner(monkeypatch):
    async def scenario():
        pool,created=setup_pool(monkeypatch);pool.enabled_getter=lambda:False
        with pytest.raises(models.ModelUnavailableError):await pool.ensure_loaded()
        assert not created and not pool._owned_process_entries()
        await pool.close()
    asyncio.run(scenario())


def test_real_spawn_pool_returns_original_tensor_and_normally_reaps_children(monkeypatch,tmp_path):
    from core.cam_process import CamProcessRunner
    import os
    import numpy as np
    async def scenario():
        directory=tmp_path/'normal';directory.mkdir()
        (directory/'config.yaml').write_text('{}')
        (directory/'campplus_cn_en_common.pt').write_bytes(b'explicit CPU factory fixture only')
        def factory(path,device,**kwargs):
            return CamProcessRunner(path,device,_factory_spec='tests.fake_cam_runner:Runner',**kwargs)
        pool,created=setup_pool(monkeypatch,count=2,factory=factory);pool.model_dir=str(directory)
        try:
            await pool.ensure_loaded()
            identities=[r.snapshot()['worker_identity'] for r in created]
            assert len({x['pid']for x in identities})==2 and all(x['pid']!=os.getpid()for x in identities)
            for _ in range(4):
                async with pool.lease()as(runner,device):
                    result=await run_in_thread(runner.generate,input=np.zeros(8000,dtype=np.float32))
                    value=result[0]['spk_embedding']
                    assert value.shape==(1,192) and value.dtype==torch.float32 and value.device.type=='cpu'
            assert models.local_model_status['speech_campplus']['instances']==2
        finally:await pool.close()
        assert all(r.snapshot()['resource_reaped'] and r.snapshot()['normal_exit_ok']for r in created)
    asyncio.run(scenario())


def test_real_spawn_repeat_cancel_and_pool_close_hold_pcm_until_done(monkeypatch,tmp_path):
    from core.cam_process import CamProcessRunner
    import numpy as np
    async def scenario():
        directory=tmp_path/'slow';directory.mkdir()
        (directory/'config.yaml').write_text('{}')
        (directory/'campplus_cn_en_common.pt').write_bytes(b'explicit CPU factory fixture only')
        def factory(path,device,**kwargs):
            return CamProcessRunner(path,device,_factory_spec='tests.fake_cam_runner:Runner',**kwargs)
        pool,created=setup_pool(monkeypatch,count=1,factory=factory);pool.model_dir=str(directory)
        await pool.ensure_loaded();budget=PendingSegmentBudget(1,100)
        async def work():
            token=await budget.acquire(10)
            try:
                async with pool.lease()as(runner,_):
                    return await run_in_thread(runner.generate,input=np.zeros(8000,dtype=np.float32))
            finally:await budget.release(token)
        task=closing=None
        try:
            task=asyncio.create_task(work());await eventually(lambda:created[0].snapshot()['active'])
            closing=asyncio.create_task(pool.close())
            await eventually(lambda:created[0].snapshot()['closing'])
            task.cancel();await asyncio.sleep(0);task.cancel();closing.cancel()
            assert not task.done() and budget.snapshot()['active']==1 and pool.locks[0].locked()
            with pytest.raises(asyncio.CancelledError):await task
            with pytest.raises(asyncio.CancelledError):await closing
            state=created[0].snapshot()
            assert budget.snapshot()['active']==0 and state['resource_reaped'] and state['normal_exit_ok']
        finally:
            if task is not None:
                if not task.done():task.cancel()
                await asyncio.gather(task,return_exceptions=True)
            if closing is not None:await asyncio.gather(closing,return_exceptions=True)
            await pool.close()
    asyncio.run(scenario())


def test_all_idle_dead_processes_reject_capability_before_pipeline_work(monkeypatch):
    async def scenario():
        pool,created=setup_pool(monkeypatch,count=1);await pool.ensure_loaded()
        monkeypatch.setattr(models,'_MODEL_POOLS',{'speech_campplus':pool})
        monkeypatch.setattr(models,'_model_lock',asyncio.Lock())
        created[0].alive=False
        with pytest.raises(models.ModelUnavailableError)as error:
            await models.load_v3_models_if_needed(True,False)
        assert error.value.code==4015 and len(created)==1
        with pytest.raises(RuntimeError):await pool.close()
    asyncio.run(scenario())


def test_close_before_lazy_first_use_is_terminal_and_cannot_spawn_afterwards(monkeypatch):
    async def scenario():
        pool,created=setup_pool(monkeypatch)
        await pool.close()
        with pytest.raises(models.ModelUnavailableError):await pool.ensure_loaded()
        assert not created and pool._closing
    asyncio.run(scenario())


def test_process_hot_handoffs_only_observe_selected_worker(monkeypatch):
    """128次交接不再额外制造6×128×8次全池OS快照。"""
    async def scenario():
        pool, created = setup_pool(monkeypatch, count=8)
        await pool.ensure_loaded()
        for runner in created:
            runner.snapshot_calls = 0

        async def one():
            async with pool.lease():
                await asyncio.sleep(0)
                pool.record_inference(0, 1)

        await asyncio.gather(*(one() for _ in range(128)))
        assert sum(r.snapshot_calls for r in created) == 128
        assert all(r.snapshot_calls == 16 for r in created)
        entry = models.local_model_status['speech_campplus']
        assert entry['calls'] == 128 and entry['busy'] == entry['waiting'] == 0
        assert entry['available_instances'] == 8
        pool.refresh_status()
        assert sum(r.snapshot_calls for r in created) == 136
        assert all(r.snapshot_calls == 17 for r in created)
        await pool.close()
    asyncio.run(scenario())


def test_known_dead_candidate_is_not_resurrected_by_hot_refresh(monkeypatch):
    async def scenario():
        pool, created = setup_pool(monkeypatch, count=2)
        await pool.ensure_loaded()
        created[0].alive = False
        assert pool._available_index() == 1
        assert pool._health_cache[id(created[0])] is False
        observed = created[0].snapshot_calls
        for _ in range(12):
            pool.refresh_status(observe_process=False)
        assert created[0].snapshot_calls == observed
        assert not pool._is_healthy(created[0])
        pool.refresh_status()
        entry = models.local_model_status['speech_campplus']
        assert entry['failed_instances'] == 1 and entry['available_instances'] == 1
        created[1].alive = False
        async def rejected():
            async with pool.lease():
                pytest.fail('dead runner granted')
        with pytest.raises(models.ModelUnavailableError):
            await asyncio.wait_for(rejected(), timeout=1)
        for _ in range(12):
            pool.refresh_status(observe_process=False)
        assert not any(pool._is_healthy(r) for r in created)
        assert len(created) == 2 and pool.calls == 0
        pool.refresh_status()
        assert models.local_model_status['speech_campplus']['available_instances'] == 0
        assert models.local_model_status['speech_campplus']['failed_instances'] == 2
        # 已知异常死亡不能在close时被洗成正常退出。
        with pytest.raises(RuntimeError, match='prior abnormal cleanup'):
            await pool.close()
    asyncio.run(scenario())


def test_process_fault_publishes_failed_cache_before_error_returns(monkeypatch):
    """process分支的隔离/close错误在返回调用方前已写入直接缓存。"""
    class FatalRunner(FakeProcessRunner):
        def generate(self, **kwargs):
            self.forward_started.set()
            self.forward_done.set()
            self.forced = True
            raise InstanceFailure('controlled CAM child failure')

    async def scenario():
        pool, created = setup_pool(monkeypatch, count=1, factory=FatalRunner)
        await pool.ensure_loaded()
        with pytest.raises(InstanceFailure, match='controlled CAM child failure'):
            async with pool.lease() as (model, device):
                assert device == 'npu:0'
                await run_in_thread(model.generate, input=[], fs=16000)
        # 不调用get_local_model_status/refresh掩盖热路径留下的陈旧计数。
        entry = models.local_model_status['speech_campplus']
        assert entry['failed_instances'] == 1
        assert entry['busy'] == entry['waiting'] == entry['available_instances'] == 0
        assert entry['instances'] == 0 and pool.models == [None]
        assert len(entry['model_workers']) == 1
        worker = entry['model_workers'][0]
        assert worker['resource_reaped'] and worker['forced_cleanup']
        assert worker['close_error'] and worker['normal_exit_ok'] is False
        assert created[0].close_count == 1 and created[0].reaped
        with pytest.raises(RuntimeError):
            await pool.close()
        assert models.local_model_status['speech_campplus']['failed_instances'] == 1
    asyncio.run(scenario())


def test_fault_close_repeated_cancel_keeps_ownership_and_publishes_failed_cache(monkeypatch):
    close_gate = threading.Event()

    class HeldFaultRunner(FakeProcessRunner):
        def generate(self, **kwargs):
            self.forward_started.set()
            self.forward_done.set()
            self.forced = True
            raise InstanceFailure('controlled fault before held close')

        def close(self):
            self.close_started.set()
            assert close_gate.wait(5)
            super().close()

    async def scenario():
        pool, created = setup_pool(monkeypatch, count=1, factory=HeldFaultRunner)
        await pool.ensure_loaded()
        runner = created[0]
        budget = PendingSegmentBudget(1, 100)

        async def work():
            token = await budget.acquire(10)
            try:
                async with pool.lease() as (model, _):
                    await run_in_thread(model.generate, input=[], fs=16000)
            finally:
                await budget.release(token)

        task = asyncio.create_task(work())
        try:
            await eventually(runner.close_started.is_set)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and pool.locks[0].locked()
            assert budget.snapshot()['active'] == 1 and not runner.reaped
            close_gate.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert runner.reaped and runner.close_count == 1
            assert not pool.locks[0].locked() and budget.snapshot()['active'] == 0
            # 没有额外GET或refresh；重复cancel不能越过故障发布。
            entry = models.local_model_status['speech_campplus']
            assert entry['failed_instances'] == 1
            assert entry['busy'] == entry['waiting'] == entry['available_instances'] == 0
            assert entry['instances'] == 0
            assert entry['model_workers'][0]['resource_reaped']
            assert entry['model_workers'][0]['forced_cleanup']
            assert entry['model_workers'][0]['close_error']
        finally:
            close_gate.set()
            await asyncio.gather(task, return_exceptions=True)
            with pytest.raises(RuntimeError):
                await pool.close()
    asyncio.run(scenario())
