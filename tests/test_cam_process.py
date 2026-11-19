"""Real CPU spawn/IPC tests for formal CAM Tensor, ownership and shutdown semantics."""
import asyncio
import hashlib
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from core.cam_process import CamProcessRunner, _identity, _storage_identity
from core.local_inference import InferenceRunner, InstanceFailure

FACTORY='tests.fake_cam_runner:make'
PCM=np.linspace(.1,.2,4800,dtype=np.float32)


def runner(tmp_path,mode='normal',device='npu:7',**kwargs):
    directory=tmp_path/mode;directory.mkdir(exist_ok=True)
    (directory/'config.yaml').write_text('CPU fixture')
    (directory/'campplus_cn_en_common.pt').write_bytes(b'CPU fixture, no actual weights')
    options={'startup_timeout_s':5,'shutdown_grace_s':1,'_factory_spec':FACTORY};options.update(kwargs)
    return CamProcessRunner(str(directory),device,**options)


def assert_reaped(p,normal):
    s=p.snapshot();assert s['resource_reaped'] and s['normal_exit_ok']is normal
    assert not s['worker_alive'] and p._process is None
    if s['worker_identity']:
        with pytest.raises((FileNotFoundError,ProcessLookupError)):_identity(s['worker_identity']['pid'])


def test_constructor_and_status_are_lazy_no_artifact_io_or_rpc(tmp_path):
    import core.cam_process as mod
    with patch.object(mod,'_model_identity',side_effect=AssertionError('constructor read weights')),patch('multiprocessing.get_context',side_effect=AssertionError('spawned')):
        p=CamProcessRunner(tmp_path,'npu:0',_factory_spec=FACTORY)
        for _ in range(3):
            assert p.snapshot()['worker_identity']is None
            assert p.memory['loaded']is False and p.memory_info['allocator']is None
        assert isinstance(p,InferenceRunner)and p.actual_precision is None
        p.request_close();p.close();p.close();assert_reaped(p,True)


@pytest.mark.parametrize('device',['npu:0','npu:7'])
def test_real_spawn_tensor_contract_artifact_identity_pcm_and_normal_close(tmp_path,device):
    import torch
    p=runner(tmp_path,device=device);p.ensure_loaded()
    try:
        s=p.snapshot();r=s['ready'];assert s['worker_alive']and s['loaded']and p.actual_precision=='fp32'
        assert r['identity']==s['worker_identity']and r['identity']['sid']==os.getsid(0)
        assert r['test_double']and not r['actor']['available']and r['actor']['configured_device']==device
        assert r['worker_entry_identity']['module_origin'].endswith(('.py','.so'))
        for key in ('worker_entry_identity','factory_identity'):
            a=r[key];assert hashlib.sha256(Path(a['module_origin']).read_bytes()).hexdigest()==a['module_sha256']
        assert r['device']==device and r['precision']=='fp32'
        result=p.generate(input=PCM,fs=16000)[0]['spk_embedding']
        assert isinstance(result,torch.Tensor)and result.dtype==torch.float32 and result.device.type=='cpu'and list(result.shape)==[1,192]
        torch.testing.assert_close(result,(torch.arange(192,dtype=torch.float32)+float(PCM[0])).reshape(1,192),rtol=0,atol=0)
        d=p.snapshot()['last_done'];assert d['pcm_sha256']==hashlib.sha256(PCM.tobytes()).hexdigest()
        assert d['samples']==4800 and 'vector_float32_bytes'not in d
        assert p.snapshot()['last_audio_bytes']==PCM.nbytes and p.snapshot()['allocator']['memory_allocated_bytes']is None
    finally:p.close()
    p.close();assert_reaped(p,True)


@pytest.mark.parametrize('kwargs',[{'device':'cpu'},{'device':'cuda:7'},{'device':'npu:-1'},{'device':None},
    {'startup_timeout_s':True},{'startup_timeout_s':float('nan')},{'shutdown_grace_s':0},{'shutdown_grace_s':float('inf')},{'_factory_spec':lambda:None}])
def test_invalid_options_rejected_without_spawn(tmp_path,kwargs):
    options=dict(device='npu:7');options.update(kwargs)
    with patch('multiprocessing.get_context',side_effect=AssertionError('spawned')):
        with pytest.raises(ValueError):CamProcessRunner(tmp_path,**options)


@pytest.mark.parametrize('mode',['startup_crash','startup_error','startup_bad_device','startup_bad_precision','startup_unmarked'])
def test_startup_failure_reaps_and_abnormal_history_is_sticky(tmp_path,mode):
    p=runner(tmp_path,mode)
    with pytest.raises(InstanceFailure):p.ensure_loaded()
    assert_reaped(p,False)
    before=p.snapshot()
    for _ in range(2):
        with pytest.raises(InstanceFailure):p.close()
    assert_reaped(p,False)
    assert p.snapshot()['worker_exitcode']==before['worker_exitcode']
    assert p.snapshot()['abnormal_reasons'][:len(before['abnormal_reasons'])]==before['abnormal_reasons']


def test_ordinary_model_error_retains_healthy_worker_and_next_call_succeeds(tmp_path):
    p=runner(tmp_path,'model_error');p.ensure_loaded();pid=p.snapshot()['worker_identity']['pid']
    try:
        with pytest.raises(RuntimeError,match='ordinary model error')as error:p.generate(input=PCM)
        assert type(error.value)is RuntimeError
        assert p.healthy and p.snapshot()['healthy']and p.snapshot()['worker_alive']and not p.snapshot()['resource_reaped']
        assert p.snapshot()['last_done']['error']['category']=='model'
        assert p.generate(input=PCM)[0]['spk_embedding'].shape==(1,192)
        assert p.snapshot()['worker_identity']['pid']==pid
    finally:p.close()
    assert_reaped(p,True)


def test_instance_failure_retains_original_layer_and_never_reissues_unhealthy_worker(tmp_path):
    p=runner(tmp_path,'instance_error');p.ensure_loaded()
    with pytest.raises(InstanceFailure,match='device resource failure'):p.generate(input=PCM)
    assert not p.healthy and not p.snapshot()['healthy']and p.snapshot()['worker_alive']
    assert not p.snapshot()['resource_reaped']
    with pytest.raises(InstanceFailure):p.generate(input=PCM)
    p.close();assert_reaped(p,True)


@pytest.mark.parametrize('mode',['wrong_shape','nonfinite'])
def test_model_result_error_not_misclassified_as_transport_poison(tmp_path,mode):
    p=runner(tmp_path,mode)
    try:
        with pytest.raises(ValueError):p.generate(input=PCM)
        assert p.healthy and p.snapshot()['worker_alive']and not p.snapshot()['resource_reaped']
    finally:p.close()
    assert_reaped(p,True)


def test_crash_and_idle_death_are_not_reported_as_healthy_available(tmp_path):
    p=runner(tmp_path,'crash')
    with pytest.raises(InstanceFailure):p.generate(input=PCM)
    assert_reaped(p,False)
    q=runner(tmp_path,'idle_exit');q.ensure_loaded()
    deadline=time.monotonic()+3
    while q.snapshot()['worker_alive']is not False and time.monotonic()<deadline:time.sleep(.005)
    assert not q.snapshot()['worker_alive']and not q.snapshot()['healthy']and not q.snapshot()['resource_reaped']
    with pytest.raises(InstanceFailure):q.close()
    assert_reaped(q,False)


def test_rpc_pcm_and_ready_artifact_corruption_poison_until_actual_death(tmp_path):
    for field in ('rpc','pcm_sha256','identity'):
        p=runner(tmp_path);p.ensure_loaded();original=p._wait
        def corrupt(*a,_field=field,**kw):
            r,n=original(*a,**kw);r[_field]='wrong';return r,n
        p._wait=corrupt
        with pytest.raises(InstanceFailure):p.generate(input=PCM)
        assert_reaped(p,False)
    q=runner(tmp_path);original=q._wait
    def wrong_artifact(*a,**kw):
        r,n=original(*a,**kw);r['worker_entry_identity']['module_sha256']='wrong';return r,n
    q._wait=wrong_artifact
    with pytest.raises(InstanceFailure):q.ensure_loaded()
    assert_reaped(q,False)


def test_broken_pipe_is_reaped_before_transport_failure_returns(tmp_path):
    p=runner(tmp_path);p.ensure_loaded();p._pipe.close()
    with pytest.raises(InstanceFailure):p.generate(input=PCM)
    assert_reaped(p,False)


def test_cancel_and_repeat_cancel_wait_done_before_releasing_original_tensor(tmp_path):
    async def scenario():
        from core.async_utils import run_in_thread
        p=runner(tmp_path,'slow');await asyncio.to_thread(p.ensure_loaded)
        task=asyncio.create_task(run_in_thread(p.generate,input=PCM))
        while not p.snapshot()['active']:await asyncio.sleep(.001)
        task.cancel();await asyncio.sleep(.01);task.cancel();await asyncio.sleep(.01)
        assert not task.done()and p.snapshot()['active']and not p.snapshot()['resource_reaped']
        with pytest.raises(asyncio.CancelledError):await task
        assert p.snapshot()['last_done']['ok']and not p.snapshot()['active']
        await asyncio.to_thread(p.close);assert_reaped(p,True)
    asyncio.run(scenario())


def test_request_close_is_nonblocking_uses_one_absolute_grace_and_aborts_active_rpc(tmp_path):
    async def scenario():
        p=runner(tmp_path,'slow',shutdown_grace_s=.04);await asyncio.to_thread(p.ensure_loaded)
        task=asyncio.create_task(asyncio.to_thread(p.generate,input=PCM))
        while not p.snapshot()['active']:await asyncio.sleep(.001)
        started=time.monotonic();p.request_close();deadline=p.snapshot()['close_deadline_monotonic_s']
        assert time.monotonic()-started<.03
        await asyncio.sleep(.015);p.request_close();assert p.snapshot()['close_deadline_monotonic_s']==deadline
        with pytest.raises(InstanceFailure):await asyncio.to_thread(p.close)
        with pytest.raises(InstanceFailure):await task
        assert_reaped(p,False);assert p.snapshot()['forced_cleanup']
    asyncio.run(scenario())


def test_request_close_allows_inflight_done_before_grace_without_poison(tmp_path):
    async def scenario():
        p=runner(tmp_path,'slow');await asyncio.to_thread(p.ensure_loaded)
        await asyncio.to_thread(p.generate,input=PCM)  # warm CPU torch import before timing only the held forward
        task=asyncio.create_task(asyncio.to_thread(p.generate,input=PCM))
        while not p.snapshot()['active']:await asyncio.sleep(.001)
        p.request_close();closing=asyncio.create_task(asyncio.to_thread(p.close))
        assert (await task)[0]['spk_embedding'].shape==(1,192)
        await closing;assert_reaped(p,True)
    asyncio.run(scenario())


def test_startup_timeout_confirms_death(tmp_path):
    p=runner(tmp_path,'startup_slow',startup_timeout_s=.02)
    with pytest.raises(InstanceFailure):p.ensure_loaded()
    assert_reaped(p,False)


def test_legal_ready_already_read_then_close_cannot_late_publish_loaded(tmp_path):
    async def scenario():
        p=runner(tmp_path,shutdown_grace_s=.03);seen=threading.Event();release=threading.Event();original=p._wait
        def gate(*a,**kw):
            r,n=original(*a,**kw)
            if r.get('kind')=='READY':
                seen.set();assert release.wait(3)
            return r,n
        p._wait=gate;starting=asyncio.create_task(asyncio.to_thread(p.ensure_loaded))
        assert await asyncio.to_thread(seen.wait,3)
        try:
            with pytest.raises(InstanceFailure):await asyncio.to_thread(p.close)
            assert_reaped(p,False)
        finally:release.set()
        with pytest.raises(InstanceFailure):await starting
        assert not p.snapshot()['loaded']and p.snapshot()['closed'];assert_reaped(p,False)
    asyncio.run(scenario())


def test_partial_birth_lock_cannot_be_released_as_dead_before_start_returns(tmp_path):
    import multiprocessing.process
    async def scenario():
        p=runner(tmp_path,shutdown_grace_s=.025);born=threading.Event();release=threading.Event()
        original=multiprocessing.process.BaseProcess.start
        def partial(process,*a,**kw):
            result=original(process,*a,**kw);born.set();assert release.wait(3);return result
        with patch.object(multiprocessing.process.BaseProcess,'start',partial):
            starting=asyncio.create_task(asyncio.to_thread(p.ensure_loaded));assert await asyncio.to_thread(born.wait,3)
            p.request_close();closing=asyncio.create_task(asyncio.to_thread(p.close));await asyncio.sleep(.08)
            assert not closing.done()and not p.snapshot()['resource_reaped']and p.snapshot()['birth_in_progress']
            release.set()
            result=await asyncio.gather(starting,closing,return_exceptions=True)
        assert all(isinstance(x,InstanceFailure)for x in result)
        assert_reaped(p,False)
    asyncio.run(scenario())


@pytest.mark.parametrize('mode,code',[('close_error',0),('ack_nonzero',7)])
def test_bad_ack_or_nonzero_exit_is_sticky_across_repeated_close(tmp_path,mode,code):
    p=runner(tmp_path,mode);p.ensure_loaded()
    for _ in range(3):
        with pytest.raises(InstanceFailure):p.close()
        assert_reaped(p,False)
    assert p.snapshot()['worker_exitcode']==code and not p.snapshot()['forced_cleanup']
    assert p.snapshot()['last_close_ack']['ok']is(mode=='ack_nonzero')


def test_bad_close_ack_identity_forces_confirmed_cleanup(tmp_path):
    p=runner(tmp_path);p.ensure_loaded();original=p._wait
    def wrong(*a,**kw):
        r,n=original(*a,**kw)
        if r.get('kind')=='CLOSE_ACK':r['rpc']+=1
        return r,n
    p._wait=wrong
    with pytest.raises(InstanceFailure):p.close()
    assert_reaped(p,False)


def test_close_grace_covers_close_operation_not_only_active_generate(tmp_path):
    p=runner(tmp_path,'close_slow',shutdown_grace_s=.02);p.ensure_loaded()
    with pytest.raises(InstanceFailure):p.close()
    assert_reaped(p,False)
    assert p.snapshot()['forced_cleanup']


def test_concurrent_close_and_repeated_cancel_preserve_abnormal_cleanup(tmp_path):
    async def scenario():
        from core.async_utils import run_in_thread
        p=runner(tmp_path,'slow',shutdown_grace_s=.06);await asyncio.to_thread(p.ensure_loaded)
        work=asyncio.create_task(run_in_thread(p.generate,input=PCM))
        while not p.snapshot()['active']:await asyncio.sleep(.001)
        first=asyncio.create_task(run_in_thread(p.close))
        while not p.snapshot()['closing']:await asyncio.sleep(.001)
        second=asyncio.create_task(asyncio.to_thread(p.close));first.cancel();await asyncio.sleep(.005);first.cancel()
        assert not first.done()
        with pytest.raises(asyncio.CancelledError):await first
        with pytest.raises(InstanceFailure):await second
        with pytest.raises(InstanceFailure):await work
        assert_reaped(p,False)
    asyncio.run(scenario())


def test_storage_metadata_matches_requested_container_npu0_not_host7():
    class Tensor:
        dtype='torch.float32';device='npu:0'
        def is_floating_point(self):return True
        def is_complex(self):return False
    network=SimpleNamespace(named_parameters=lambda:[('w',Tensor())],named_buffers=lambda:[('b',Tensor())])
    result=_storage_identity(SimpleNamespace(model=network),'npu:0')
    assert result['all_floating_fp32']and result['all_parameters_on_device']and result['all_floating_buffers_on_device']
    assert result['dtype_device_counts']['parameter']=={'torch.float32@npu:0':1}
    assert not _storage_identity(SimpleNamespace(model=network),'npu:7')['all_parameters_on_device']


def test_protected_spawn_reports_real_worker_entry_artifact_when_requested(tmp_path):
    stage=os.getenv('ASR_EXPECT_PROCESS_STAGE_ROOT')
    if not stage:pytest.skip('protected-stage-only artifact assertion')
    p=runner(tmp_path)
    try:
        p.ensure_loaded();origin=Path(p.snapshot()['ready']['worker_entry_identity']['module_origin'])
        assert origin.suffix=='.so'and origin.is_relative_to(Path(stage))
    finally:p.close()
    assert_reaped(p,True)


def test_exception_path_successful_fence_keeps_ordinary_model_error_reusable(tmp_path):
    p=runner(tmp_path,'model_error_fenced');p.ensure_loaded()
    try:
        with pytest.raises(RuntimeError,match='ordinary model error')as error:p.generate(input=PCM)
        assert type(error.value)is RuntimeError and p.healthy
        done=p.snapshot()['last_done']
        assert done['execution_drained']and done['error_fence']['attempted']
        assert done['error_fence']['status']=='exception_path_fence_completed'
        assert p.generate(input=PCM)[0]['spk_embedding'].shape==(1,192)
        assert 'error_fence'not in p.snapshot()['last_done']
    finally:p.close()
    assert_reaped(p,True)


def test_exception_path_sync_failure_never_returns_healthy_drained_done(tmp_path):
    p=runner(tmp_path,'sync_failure');p.ensure_loaded()
    with pytest.raises(InstanceFailure):p.generate(input=PCM)
    done=p.snapshot()['last_done']
    assert not done['execution_drained']and done['error_fence']['status']=='exception_path_fence_failed'
    assert done['error']['category']=='instance'
    assert not p.healthy and not p.snapshot()['healthy']
    assert_reaped(p,False)


def test_partial_cancel_two_owned_instances_does_not_affect_other_result(tmp_path):
    async def scenario():
        from core.async_utils import run_in_thread
        a=runner(tmp_path,'slow',device='npu:0');b=runner(tmp_path,'normal',device='npu:7')
        await asyncio.gather(asyncio.to_thread(a.ensure_loaded),asyncio.to_thread(b.ensure_loaded))
        first=asyncio.create_task(run_in_thread(a.generate,input=PCM))
        second=asyncio.create_task(run_in_thread(b.generate,input=PCM+.25))
        while not a.snapshot()['active']:await asyncio.sleep(.001)
        first.cancel();await asyncio.sleep(.005);first.cancel()
        output=(await second)[0]['spk_embedding']
        assert output.shape==(1,192)and abs(output[0,0].item()-.35)<1e-6
        with pytest.raises(asyncio.CancelledError):await first
        assert a.snapshot()['last_done']['ok']and b.snapshot()['last_done']['ok']
        assert a.snapshot()['worker_identity']['pid']!=b.snapshot()['worker_identity']['pid']
        await asyncio.gather(asyncio.to_thread(a.close),asyncio.to_thread(b.close))
        assert_reaped(a,True);assert_reaped(b,True)
    asyncio.run(scenario())


def test_natural_exit_between_alive_and_proc_read_still_confirms_join(tmp_path):
    import core.cam_process as mod
    p=runner(tmp_path,'idle_exit');p.ensure_loaded()
    def disappear(_pid):
        p._process.join(2)
        raise FileNotFoundError('controlled natural exit after is_alive')
    with patch.object(mod,'_identity',side_effect=disappear):
        with pytest.raises(InstanceFailure):p._poison('controlled channel failure')
    assert_reaped(p,False)
    assert p.snapshot()['worker_exitcode']==23


def test_snapshot_identity_unknown_is_not_falsely_alive_or_dead(tmp_path):
    import core.cam_process as mod
    p=runner(tmp_path);p.ensure_loaded()
    try:
        with patch.object(mod,'_identity',side_effect=PermissionError('controlled procfs unknown')):
            assert p.snapshot()['worker_alive']is None
        assert p.snapshot()['worker_alive']is True
    finally:p.close()
    assert_reaped(p,True)


@pytest.mark.parametrize('after_birth',[False,True])
def test_process_start_failure_before_or_after_real_birth_never_leaks_child(tmp_path,after_birth):
    import multiprocessing.process
    p=runner(tmp_path);original=multiprocessing.process.BaseProcess.start
    def fail_start(process,*args,**kwargs):
        if after_birth:original(process,*args,**kwargs)
        raise OSError('controlled Process.start failure '+str(after_birth))
    with patch.object(multiprocessing.process.BaseProcess,'start',fail_start):
        with pytest.raises(InstanceFailure):p.ensure_loaded()
    assert_reaped(p,False)
    if after_birth:assert p.snapshot()['worker_identity']is not None
    else:assert p.snapshot()['worker_identity']is None


def test_natural_exit_between_signal_identity_and_terminate_is_still_joined(tmp_path):
    import multiprocessing.process
    p=runner(tmp_path,'idle_exit');p.ensure_loaded()
    def vanished(process):
        process.join(2)
        raise ProcessLookupError('controlled exit between identity and signal')
    with patch.object(multiprocessing.process.BaseProcess,'terminate',vanished):
        with pytest.raises(InstanceFailure):p._poison('controlled channel failure')
    assert_reaped(p,False)
    assert p.snapshot()['worker_exitcode']==23 and not p.snapshot()['forced_cleanup']


def test_loaded_snapshot_and_memory_never_wait_rpc_or_spawn(tmp_path):
    import core.cam_process as mod
    p=runner(tmp_path);p.ensure_loaded()
    try:
        with patch.object(mod,'_send',side_effect=AssertionError('status RPC')),patch('multiprocessing.get_context',side_effect=AssertionError('status spawn')):
            assert p.snapshot()['worker_alive']is True
            assert p.memory['loaded']and p.memory_info['ready']['precision']=='fp32'
    finally:p.close()
    assert_reaped(p,True)


def test_successful_forward_then_post_metadata_failure_returns_only_error_envelope(tmp_path):
    p=runner(tmp_path,'post_metadata_error');p.ensure_loaded()
    try:
        with pytest.raises(RuntimeError,match='post-forward metadata failure')as error:p.generate(input=PCM)
        assert type(error.value)is RuntimeError
        done=p.snapshot()['last_done']
        assert not done['ok']and done['error']['category']=='model'and done['execution_drained']
        assert not any(k in done for k in ('vector_float32_bytes','result_shape','result_dtype'))
        assert done['error_fence']['status']=='exception_path_fence_completed'
        assert p.healthy and p.snapshot()['worker_alive']and not p.snapshot()['resource_reaped']
        assert p.generate(input=PCM)[0]['spk_embedding'].shape==(1,192)
    finally:p.close()
    assert_reaped(p,True)


@pytest.mark.parametrize('kind',['success_with_error','failure_without_error','failure_with_embedding'])
def test_contradictory_response_envelope_is_protocol_poison(tmp_path,kind):
    p=runner(tmp_path);p.ensure_loaded();original=p._wait
    def corrupt(*args,**kwargs):
        value,n=original(*args,**kwargs)
        if kind=='success_with_error':value['error']={'type':'RuntimeError','message':'contradiction','category':'model'}
        elif kind=='failure_without_error':
            value['ok']=False
            for key in ('vector_float32_bytes','result_shape','result_dtype'):value.pop(key,None)
        else:
            value['ok']=False;value['error']={'type':'RuntimeError','message':'contradiction','category':'model'}
        return value,n
    p._wait=corrupt
    with pytest.raises(InstanceFailure):p.generate(input=PCM)
    assert_reaped(p,False)


def test_explicit_close_before_birth_with_valid_grace_is_normal_with_no_spawn(tmp_path):
    import core.cam_process as mod
    async def scenario():
        p=runner(tmp_path);seen=threading.Event();release=threading.Event();original=mod._model_identity
        def gate(directory):
            value=original(directory);seen.set();assert release.wait(3);return value
        with patch.object(mod,'_model_identity',gate),patch('multiprocessing.get_context',side_effect=AssertionError('must never spawn')):
            starting=asyncio.create_task(asyncio.to_thread(p.ensure_loaded));assert await asyncio.to_thread(seen.wait,3)
            p.request_close();closing=asyncio.create_task(asyncio.to_thread(p.close))
            release.set()
            with pytest.raises(InstanceFailure,match='before process creation'):await starting
            await closing
        assert_reaped(p,True)
        assert not p.snapshot()['abnormal_reasons']and p.snapshot()['worker_identity']is None
        p.close();assert_reaped(p,True)
    asyncio.run(scenario())


def test_before_birth_closing_does_not_wash_a_real_asset_error(tmp_path):
    import core.cam_process as mod
    p=runner(tmp_path);seen=threading.Event();release=threading.Event()
    def failed_assets(_directory):
        seen.set();assert release.wait(3);raise ValueError('controlled genuine asset error')
    async def scenario():
        with patch.object(mod,'_model_identity',failed_assets):
            starting=asyncio.create_task(asyncio.to_thread(p.ensure_loaded));assert await asyncio.to_thread(seen.wait,3)
            p.request_close();release.set()
            with pytest.raises(InstanceFailure,match='genuine asset error'):await starting
        with pytest.raises(InstanceFailure):await asyncio.to_thread(p.close)
        assert_reaped(p,False)
    asyncio.run(scenario())


def test_input_substring_is_not_a_device_failure_when_health_and_fence_succeed(tmp_path):
    p=runner(tmp_path,'input_shape_error');p.ensure_loaded();pid=p.snapshot()['worker_identity']['pid']
    try:
        with pytest.raises(RuntimeError,match='invalid input shape')as error:p.generate(input=PCM)
        assert type(error.value)is RuntimeError
        done=p.snapshot()['last_done']
        assert done['error']['category']=='model'and done['healthy']and done['execution_drained']
        assert done['error_fence']['status']=='exception_path_fence_completed'
        assert p.healthy and p.snapshot()['worker_alive']and not p.snapshot()['abnormal_reasons']
        assert p.generate(input=PCM)[0]['spk_embedding'].shape==(1,192)
        assert p.snapshot()['worker_identity']['pid']==pid
    finally:p.close()
    assert_reaped(p,True)


def test_operator_adaptations_forward_actual_ready_metadata_readonly_without_rpc(tmp_path):
    import core.cam_process as mod
    p=runner(tmp_path)
    assert p.operator_adaptations is None and p.snapshot()['operator_adaptations']is None
    p.ensure_loaded()
    try:
        expected={'campplus_segment_pooling':False,'explicit_CPU_test_double':True}
        with patch.object(mod,'_send',side_effect=AssertionError('metadata RPC')):
            assert p.operator_adaptations==expected
            assert p.snapshot()['ready']['operator_adaptations']==expected
            exposed=p.operator_adaptations;exposed['campplus_segment_pooling']=True
            assert p.operator_adaptations==expected and p.snapshot()['operator_adaptations']==expected
    finally:p.close()
    assert_reaped(p,True)


def test_boot_time_is_lazy_cached_but_pid_birth_identity_is_always_fresh(monkeypatch):
    from core import cam_process
    from collections import Counter
    calls = Counter()
    ticks = iter(('10', '20'))
    def read(path, *args, **kwargs):
        calls[str(path)] += 1
        if str(path) == '/proc/stat':
            return 'cpu 1 2 3\nbtime 1000\n'
        if str(path) == '/proc/123/stat':
            fields = ['S'] + ['0'] * 18 + [next(ticks)]
            return '123 (fake process) ' + ' '.join(fields)
        raise AssertionError(path)
    monkeypatch.setattr(cam_process, '_BOOT_TIME', None)
    monkeypatch.setattr(cam_process.Path, 'read_text', read)
    monkeypatch.setattr(cam_process.os, 'getsid', lambda pid: 55)
    monkeypatch.setattr(cam_process.os, 'sysconf', lambda name: 100)
    assert calls == {}
    first = cam_process._identity(123)
    second = cam_process._identity(123)
    assert first == dict(pid=123, start_ticks='10', create_time=1000.1, sid=55)
    assert second == dict(pid=123, start_ticks='20', create_time=1000.2, sid=55)
    assert calls == {'/proc/stat': 1, '/proc/123/stat': 2}


def test_cached_health_has_no_procfs_observation_or_nested_metadata_alias(monkeypatch):
    from core import cam_process
    runner = cam_process.CamProcessRunner('unused', 'npu:0')
    runner._update(loaded=True, resource_reaped=False, active=True)
    monkeypatch.setattr(cam_process.Path, 'read_text', lambda *_: pytest.fail('hot health read procfs'))
    health = runner.cached_health()
    assert health == dict(loaded=True, closing=False, closed=False, healthy=True, resource_reaped=False)
    assert 'ready' not in health and 'last_done' not in health
    health['loaded'] = False
    assert runner.cached_health()['loaded'] is True
    runner.request_close()
    assert runner.cached_health()['closing'] is True
