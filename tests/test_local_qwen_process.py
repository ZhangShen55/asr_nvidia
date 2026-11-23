"""实际spawn的CPU替身；不导入模型、不启动NPU/HTTP服务。"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import time
from unittest.mock import patch
import wave

import pytest

from core.qwen_asr_process import LocalQwenProcessRuntime, QwenWorkerUnavailableError

pytestmark = pytest.mark.skipif(not Path('/proc/stat').exists(), reason='candidate ownership uses Linux /proc identity')


class CustomForwardError(RuntimeError):
    pass


class CpuRuntime:
    def __init__(self, model_dir, device, dtype, max_new_tokens):
        self.config = json.loads(Path(model_dir).read_text())
        self.loaded = self.loading = False
        self.failure = self.last_batch_timings = None
        self.device, self.dtype = device, dtype
        self.calls = 0
    def event(self, event):
        with Path(self.config['events']).open('a') as handle:
            handle.write(json.dumps({'event':event,'pid':os.getpid(),'call':self.calls,'monotonic_s':time.monotonic()})+'\n')
    def ensure_loaded(self):
        load_gate=self.config.get('load_gate')
        if load_gate:
            self.event('loading')
            while not Path(load_gate).exists():time.sleep(.005)
        if self.config.get('unavailable'):
            from core.qwen_asr_runtime import LocalModelUnavailable
            self.failure='controlled unavailable'
            raise LocalModelUnavailable(self.failure)
        self.loaded = True
    def transcribe_batch(self, audios, languages=None):
        from core.qwen_asr_runtime import LocalUnsupportedLanguage
        self.ensure_loaded();self.calls += 1;self.event('started')
        if audios == [b'DIE'] or self.config.get('die_on_transcribe'):
            os._exit(17)
        gate = self.config.get('gate')
        while gate and not Path(gate).exists():
            time.sleep(.005)
        time.sleep(self.config.get('delay_s',0))
        if self.config.get('forward_error')=='custom':
            raise CustomForwardError('controlled custom forward failure')
        if audios == [b'FAIL'] or self.config.get('forward_error'):
            raise RuntimeError('controlled forward failure')
        values=[]
        for audio, language in zip(audios, languages or [None]*len(audios)):
            if language == 'bad':
                values.append(LocalUnsupportedLanguage('unsupported language'))
            elif language == 'broken':
                try:
                    raise ValueError('nested PCM context')
                except ValueError as cause:
                    error=RuntimeError('member decode failure');error.__cause__=cause;values.append(error)
            else:
                values.append('language Chinese<asr_text>'+audio.decode(errors='replace'))
        valid=[i for i,lang in enumerate(languages or [None]*len(audios)) if lang!='bad']
        self.last_batch_timings={'batch_size':len(valid),'valid_member_indices':valid,
                                 'generate_started':True,'generate_completed':True,'generate_wall_ms':1}
        self.event('read_complete')
        return values[:-1] if self.config.get('short_result') else values
    def snapshot(self):
        return {'loaded':self.loaded,'loading':self.loading,'failure':self.failure,'dtype':self.dtype,
                'device':self.device,'memory_allocated_mb':2 if self.loaded else 0,
                'memory_reserved_mb':3 if self.loaded else 0,'cpu_fake_calls':self.calls,
                'inherited_test_env':os.environ.get('ASR_CANDIDATE_CPU_ENV')}
    def close(self):
        gate=self.config.get('close_gate')
        while gate and not Path(gate).exists():
            time.sleep(.005)
        self.loaded=False
        if self.config.get('close_error'):
            raise RuntimeError('controlled close ACK error')


def candidate(tmp_path, **values):
    config=tmp_path/'fake.json';events=tmp_path/'events.jsonl'
    config.write_text(json.dumps(dict(events=str(events),**values)))
    runtime=LocalQwenProcessRuntime(str(config),'npu:7',_runtime_factory=__name__+':CpuRuntime',_startup_timeout_s=5,_close_timeout_s=.5)
    return runtime,events


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_status_and_close_before_load_do_not_spawn(tmp_path):
    with patch('core.qwen_asr_process.multiprocessing.get_context',side_effect=AssertionError('must stay lazy')):
        runtime,_=candidate(tmp_path)
        assert runtime.snapshot()['model_worker']['started'] is False
        assert not runtime.loaded and runtime.failure is None
        runtime.close();runtime.close()
        assert runtime.snapshot()['model_worker']['cleanup_status']=='passed'
    with pytest.raises(ValueError):
        LocalQwenProcessRuntime('/unused',device='cuda:0')


def test_real_spawn_load_cached_status_inheritance_and_normal_close(tmp_path,monkeypatch):
    monkeypatch.setenv('ASR_CANDIDATE_CPU_ENV','inherited-cann-env-test')
    runtime,_=candidate(tmp_path)
    try:
        runtime.ensure_loaded()
        with patch.object(runtime,'_rpc',side_effect=AssertionError('snapshot must not IPC')):
            state=runtime.snapshot()
        assert state['loaded'] and state['inherited_test_env']=='inherited-cann-env-test'
        metadata=state['model_worker']
        assert metadata['worker_pid'] != os.getpid() and metadata['worker_create_time']>0
        assert metadata['worker_start_ticks'] and metadata['worker_alive']
        assert metadata['ready_verified'] and len(metadata['launch_id'])==32
        assert metadata['launch_config']['device']=='npu:7'
        assert state['model_worker']['worker_allocator']['worker_pid']==metadata['worker_pid']
        assert state['dtype']=='bfloat16' and state['device']=='npu:7'
        assert state['memory_allocated_mb']==2 and state['memory_reserved_mb']==3
    finally:
        runtime.close();runtime.close()
    state=runtime.snapshot()
    assert state['model_worker']['cleanup_status']=='passed'
    assert state['model_worker']['worker_exitcode']==0
    assert not state['model_worker']['worker_alive'] and not state['loaded']


def test_concurrent_parent_threads_have_one_serial_worker_and_keep_members(tmp_path):
    runtime,events=candidate(tmp_path,delay_s=.025)
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            outputs=list(executor.map(lambda i:runtime.transcribe_batch([f'clip-{i}'.encode()],[None]),range(8)))
        assert outputs==[['language Chinese<asr_text>clip-'+str(i)] for i in range(8)]
        rows=read_events(events)
        assert len({row['pid'] for row in rows})==1
        for i in range(8):
            assert rows[2*i]['event']=='started' and rows[2*i+1]['event']=='read_complete'
            if i:
                assert rows[2*i]['monotonic_s']>=rows[2*i-1]['monotonic_s']
        assert runtime.last_batch_timings['batch_size']==1
        metadata=runtime.snapshot()['model_worker']
        assert metadata['last_audio_bytes']==6
        assert metadata['last_request_wire_payload_bytes']>metadata['last_audio_bytes']
        assert metadata['last_response_wire_payload_bytes']>0
        assert metadata['parent_rss_mb']>0 and metadata['worker_allocator']['worker_rss_mb']>0
    finally:
        runtime.close()


def test_member_errors_are_independent_light_values_and_keep_timing_indices(tmp_path):
    from core.qwen_asr_runtime import LocalUnsupportedLanguage
    runtime,_=candidate(tmp_path)
    try:
        values=runtime.transcribe_batch([b'ok',b'bad',b'broken'],[None,'bad','broken'])
        assert values[0]=='language Chinese<asr_text>ok'
        assert isinstance(values[1],LocalUnsupportedLanguage) and isinstance(values[2],RuntimeError)
        assert all(error.__traceback__ is error.__cause__ is error.__context__ is None for error in values[1:])
        assert runtime.last_batch_timings['valid_member_indices']==[0,2]
    finally:
        runtime.close()


def test_worker_death_marks_failure_without_restart_or_fallback(tmp_path):
    runtime,_=candidate(tmp_path)
    runtime.ensure_loaded();pid=runtime.snapshot()['model_worker']['worker_pid']
    try:
        with pytest.raises(QwenWorkerUnavailableError):
            runtime.transcribe_batch([b'DIE'])
        assert runtime.failure and not runtime.snapshot()['loaded']
        assert runtime.snapshot()['model_worker']['worker_pid']==pid
        with pytest.raises(QwenWorkerUnavailableError):
            runtime.ensure_loaded()
    finally:
        with pytest.raises(QwenWorkerUnavailableError):runtime.close()
        with pytest.raises(QwenWorkerUnavailableError):runtime.close()
    assert runtime.snapshot()['model_worker']['worker_alive'] is False
    assert runtime.snapshot()['model_worker']['worker_exitcode']==17
    assert runtime.snapshot()['model_worker']['resource_reaped'] is True
    assert runtime.snapshot()['model_worker']['normal_exit_ok'] is False


def test_hung_close_uses_only_owned_identity_and_bounded_cleanup(tmp_path):
    runtime,_=candidate(tmp_path,close_gate=str(tmp_path/'never-opened-close'))
    runtime.ensure_loaded();pid=runtime.snapshot()['model_worker']['worker_pid']
    with patch('core.qwen_asr_process.os.kill',wraps=os.kill) as kill:
        with pytest.raises(QwenWorkerUnavailableError):runtime.close()
    assert kill.call_args_list and all(call.args[0]==pid for call in kill.call_args_list)
    assert runtime.failure and runtime.snapshot()['model_worker']['cleanup_status']=='failed'
    assert runtime.snapshot()['model_worker']['resource_reaped'] is True
    assert runtime.snapshot()['model_worker']['forced_cleanup'] is True
    assert runtime.snapshot()['model_worker']['worker_alive'] is False
    with pytest.raises(QwenWorkerUnavailableError):runtime.close()


def test_client_repeated_cancel_keeps_busy_until_actual_spawn_worker_reads(tmp_path):
    from core.qwen_asr_client import QwenAsrClient
    gate=tmp_path/'release';config=tmp_path/'fake.json';events=tmp_path/'events.jsonl'
    config.write_text(json.dumps({'events':str(events),'gate':str(gate)}))
    stream=io.BytesIO()
    with wave.open(stream,'wb') as wav:
        wav.setparams((1,2,16000,0,'NONE','not compressed'));wav.writeframes(bytes(16000*2))
    async def scenario():
        client=QwenAsrClient(generation_mode="legacy", local_model_dir=str(config),local_instance_count=1,execution_mode='process',retry_count=0,
            max_inference_batch_size=2,batch_wait_timeout_ms=10,
            runtime_factory=lambda *args:LocalQwenProcessRuntime(*args,_runtime_factory=__name__+':CpuRuntime',
                                                           _startup_timeout_s=5,_close_timeout_s=.5))
        first=asyncio.create_task(client.transcribe_bytes(stream.getvalue(),request_id='a',segment_index=0))
        second=asyncio.create_task(client.transcribe_bytes(stream.getvalue(),request_id='b',segment_index=1))
        try:
            async def started():
                while not read_events(events):
                    await asyncio.sleep(.005)
            await asyncio.wait_for(started(),5)
            first.cancel();await asyncio.sleep(.01);first.cancel();await asyncio.sleep(.01)
            assert not first.done() and client.snapshot()['busy']==1 and client._available.qsize()==0
            gate.write_text('finish')
            outcomes=await asyncio.gather(first,second,return_exceptions=True)
            assert isinstance(outcomes[0],asyncio.CancelledError)
            assert isinstance(outcomes[1],str)
            assert client.snapshot()['busy']==0 and client._available.qsize()==1
            assert client.snapshot()['batch_scheduler']['inflight']==0
        finally:
            gate.write_text('finish');first.cancel();second.cancel()
            await asyncio.gather(first,second,return_exceptions=True)
            await client.aclose()
        state=client._runtimes[0].snapshot()
        assert not state['model_worker']['worker_alive'] and state['model_worker']['cleanup_status']=='passed'
    asyncio.run(scenario())


def test_identity_mismatch_refuses_signals_and_keeps_failure(tmp_path):
    runtime,_=candidate(tmp_path)
    runtime.ensure_loaded()
    try:
        with patch('core.qwen_asr_process._identity',return_value={'pid':-1}), \
             patch('core.qwen_asr_process.os.kill') as kill:
            runtime._reap(force=True,deadline=time.monotonic()+.1)
        kill.assert_not_called()
        assert runtime.failure and runtime.snapshot()['model_worker']['cleanup_status']=='failed'
    finally:
        runtime.close()
    assert runtime.failure and runtime.snapshot()['model_worker']['worker_alive'] is False


@pytest.mark.parametrize('field',['rpc_id','launch_id','member_ids'])
def test_wrong_rpc_launch_or_member_ack_is_rejected_without_restart(tmp_path,field):
    runtime,_=candidate(tmp_path)
    runtime.ensure_loaded();pid=runtime.snapshot()['model_worker']['worker_pid']
    receive=runtime._wait_response
    def corrupt(*args,**kwargs):
        response,size=receive(*args,**kwargs)
        response=dict(response)
        response[field]=[999] if field=='member_ids' else 'wrong-ack'
        return response,size
    try:
        with patch.object(runtime,'_wait_response',side_effect=corrupt):
            with pytest.raises(QwenWorkerUnavailableError):
                runtime.transcribe_batch([b'valid'])
        assert runtime.failure and runtime.snapshot()['model_worker']['worker_pid']==pid
    finally:
        with pytest.raises(QwenWorkerUnavailableError):runtime.close()
    assert runtime.failure and runtime.snapshot()['model_worker']['worker_alive'] is False


def wav_bytes():
    stream=io.BytesIO()
    with wave.open(stream,'wb') as wav:
        wav.setparams((1,2,16000,0,'NONE','not compressed'));wav.writeframes(bytes(16000*2))
    return stream.getvalue()


def test_real_client_dead_worker_is_unavailable4011_and_close_preserves_exit17(tmp_path):
    from core.qwen_asr_client import QwenAsrClient,QwenAsrUnavailableError
    config=tmp_path/'fake.json';config.write_text(json.dumps({'events':str(tmp_path/'events'),'die_on_transcribe':True}))
    async def scenario():
        client=QwenAsrClient(generation_mode="legacy", local_model_dir=str(config),local_instance_count=1,execution_mode='process',retry_count=1,
            runtime_factory=lambda *args:LocalQwenProcessRuntime(*args,_runtime_factory=__name__+':CpuRuntime',
                                                           _startup_timeout_s=5,_close_timeout_s=.5))
        with pytest.raises(QwenAsrUnavailableError) as error:
            await client.transcribe_bytes(wav_bytes())
        assert error.value.code==4011 and client._execution_failures==0
        with pytest.raises(RuntimeError):await client.aclose()
        state=client._runtimes[0].snapshot()['model_worker']
        assert state['resource_reaped'] and not state['normal_exit_ok'] and state['worker_exitcode']==17
        assert state['cleanup_status']=='failed' and not state['worker_alive']
    asyncio.run(scenario())


@pytest.mark.parametrize('options',[{'close_error':True},{'close_gate':'never-opened'}])
def test_real_client_fault_close_raises_aggregate_after_resource_reap(tmp_path,options):
    from core.qwen_asr_client import QwenAsrClient
    config=tmp_path/'fake.json';config.write_text(json.dumps(dict(events=str(tmp_path/'events'),**options)))
    async def scenario():
        client=QwenAsrClient(generation_mode="legacy", local_model_dir=str(config),local_instance_count=1,execution_mode='process',retry_count=0,
            runtime_factory=lambda *args:LocalQwenProcessRuntime(*args,_runtime_factory=__name__+':CpuRuntime',
                                                           _startup_timeout_s=5,_close_timeout_s=.5))
        await client.discover_model()
        with pytest.raises(RuntimeError):await client.aclose()
        state=client._runtimes[0].snapshot()['model_worker']
        assert state['resource_reaped'] and not state['normal_exit_ok']
        assert state['cleanup_status']=='failed' and not state['worker_alive'] and state['close_error']
    asyncio.run(scenario())


def test_prior_forward_error_does_not_fail_normal_ack_and_exit_zero_close(tmp_path):
    runtime,_=candidate(tmp_path)
    with pytest.raises(RuntimeError,match='controlled forward failure'):
        runtime.transcribe_batch([b'FAIL'])
    runtime.close();runtime.close()
    state=runtime.snapshot()['model_worker']
    assert state['resource_reaped'] and state['normal_exit_ok'] and state['worker_exitcode']==0
    assert state['cleanup_status']=='passed' and state['close_error'] is None


@pytest.mark.parametrize('whole_forward_error',[False,True,'custom'])
def test_real_client_model_and_language_errors_preserve_existing_codes_and_clean_close(tmp_path,whole_forward_error):
    from core.qwen_asr_client import QwenAsrClient,QwenAsrCallError,QwenAsrUnsupportedLanguageError
    config=tmp_path/'fake.json'
    config.write_text(json.dumps({'events':str(tmp_path/'events'),'forward_error':whole_forward_error}))
    async def scenario():
        client=QwenAsrClient(generation_mode="legacy", local_model_dir=str(config),local_instance_count=1,execution_mode='process',retry_count=0,max_inference_batch_size=3,
            runtime_factory=lambda *args:LocalQwenProcessRuntime(*args,_runtime_factory=__name__+':CpuRuntime',
                                                           _startup_timeout_s=5,_close_timeout_s=.5))
        try:
            if whole_forward_error:
                with pytest.raises(QwenAsrCallError) as error:await client.transcribe_bytes(wav_bytes())
                assert error.value.code==4012
            else:
                values=await asyncio.gather(*(client.transcribe_bytes(wav_bytes(),language=language,request_id=str(i))
                                              for i,language in enumerate([None,'bad','broken'])),return_exceptions=True)
                assert isinstance(values[0],str)
                assert isinstance(values[1],QwenAsrUnsupportedLanguageError) and values[1].code==4009
                assert isinstance(values[2],QwenAsrCallError) and values[2].code==4012
        finally:await client.aclose()
        state=client._runtimes[0].snapshot()['model_worker']
        assert state['resource_reaped'] and state['normal_exit_ok'] and state['cleanup_status']=='passed'
    asyncio.run(scenario())


def test_real_spawn_core_worker_entry_matches_current_module_and_optional_so(tmp_path):
    import core.qwen_asr_process as module
    config=tmp_path/'logical-zero.json';config.write_text(json.dumps({'events':str(tmp_path/'events')}))
    runtime=LocalQwenProcessRuntime(str(config),'npu:0',_runtime_factory=__name__+':CpuRuntime',
                                  _startup_timeout_s=5,_close_timeout_s=.5)
    try:
        runtime.ensure_loaded();state=runtime.snapshot();meta=state['model_worker']
        entry=Path(meta['worker_entry_module_file']).resolve()
        assert entry==Path(module.__file__).resolve()
        if os.environ.get('ASR_EXPECT_PROCESS_ENTRY_SO')=='1':
            assert str(entry).endswith('.so')
            assert entry.is_relative_to(Path(os.environ['ASR_EXPECT_PROCESS_STAGE_ROOT']).resolve())
        assert state['device']=='npu:0' and meta['launch_config']['device']=='npu:0'
        assert not meta['ready_preload_observation']['torch_npu_imported']
        assert not meta['ready_preload_observation']['runtime_api_present']
        assert meta['launch_config']['worker_entry_identity']['module_origin']==str(entry)
    finally:runtime.close()
    assert runtime.snapshot()['model_worker']['normal_exit_ok']


def test_abort_is_forbidden_before_explicit_closing(tmp_path):
    runtime,_=candidate(tmp_path)
    try:
        runtime.ensure_loaded()
        with pytest.raises(RuntimeError,match='explicit owner closing'):runtime.abort_owned()
        assert runtime.snapshot()['model_worker']['worker_alive']
    finally:runtime.close()


@pytest.mark.parametrize('force',[False,True])
def test_process_client_shutdown_grace_waits_or_aborts_only_owned_worker(tmp_path,force):
    from unittest.mock import PropertyMock
    from core.config import Settings
    from core.qwen_asr_client import QwenAsrClient,QwenAsrUnavailableError
    gate=tmp_path/'finish';events=tmp_path/'events'
    config=tmp_path/'fake.json';config.write_text(json.dumps({'events':str(events),'gate':str(gate)}))
    async def scenario():
        client=QwenAsrClient(generation_mode="legacy", local_model_dir=str(config),local_instance_count=1,execution_mode='process',retry_count=0,
            runtime_factory=lambda *args:LocalQwenProcessRuntime(*args,_runtime_factory=__name__+':CpuRuntime',
                                                               _startup_timeout_s=5,_close_timeout_s=.5,_abort_timeout_s=.5))
        request=asyncio.create_task(client.transcribe_bytes(wav_bytes()))
        async def started():
            while not read_events(events):await asyncio.sleep(.005)
        await asyncio.wait_for(started(),5)
        with patch.object(Settings,'qwen_process_shutdown_grace_s',new_callable=PropertyMock,return_value=.05 if force else 1):
            close=asyncio.create_task(client.aclose())
            try:
                await asyncio.sleep(.02)
                assert client._closing and client.snapshot()['busy']==1 and not close.done()
                if not force:gate.write_text('finish')
                if force:
                    with pytest.raises(RuntimeError):await asyncio.wait_for(close,3)
                    with pytest.raises(QwenAsrUnavailableError):await request
                    assert client.snapshot()['shutdown']['forced_cleanup']
                else:
                    await close;assert isinstance(await request,str)
                meta=client._runtimes[0].snapshot()['model_worker']
                assert meta['resource_reaped'] and not meta['worker_alive']
                assert meta['normal_exit_ok'] is (not force)
                assert client.snapshot()['busy']==client.snapshot()['batch_scheduler']['inflight']==0
            finally:
                gate.write_text('finish')
                await asyncio.gather(request,close,return_exceptions=True)
    asyncio.run(scenario())


def paused_spawn_context(entered,release,started_pids):
    import multiprocessing
    real=multiprocessing.get_context('spawn')
    class Context:
        def Pipe(self,*args,**kwargs):return real.Pipe(*args,**kwargs)
        def Process(self,*args,**kwargs):
            process=real.Process(*args,**kwargs)
            original=process.start
            def paused():
                entered.set()
                assert release.wait(5),'CPU birth gate deadline'
                # Do not pickle a local test wrapper as an attribute of the child Process.
                del process.start
                original()
                started_pids.append(process.pid)
            process.start=paused
            return process
    return Context()


def test_abort_success_before_birth_prevents_later_spawn(tmp_path):
    import threading
    runtime,_=candidate(tmp_path)
    entered,release=threading.Event(),threading.Event()
    original=runtime._start
    def before_birth():
        entered.set();assert release.wait(5);return original()
    runtime._start=before_birth
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending=executor.submit(runtime.ensure_loaded)
            assert entered.wait(5)
            runtime.begin_closing()
            outcome=runtime.abort_owned('CPU before-birth shutdown')
            assert outcome['resource_reaped']
            release.set()
            with pytest.raises(QwenWorkerUnavailableError):pending.result(timeout=5)
        state=runtime.snapshot()['model_worker']
        assert not state['started'] and state['worker_pid'] is None
        assert state['resource_reaped'] and not state['normal_exit_ok']
        assert runtime._process is None
    finally:
        release.set()
        with pytest.raises(QwenWorkerUnavailableError):runtime.close()


def test_abort_during_real_spawn_waits_for_birth_and_reaps_before_success(tmp_path):
    import threading
    runtime,_=candidate(tmp_path);runtime._abort_timeout_s=2
    entered,release=threading.Event(),threading.Event();started=[]
    context=paused_spawn_context(entered,release,started)
    try:
        with patch('core.qwen_asr_process.multiprocessing.get_context',return_value=context):
            with ThreadPoolExecutor(max_workers=2) as executor:
                pending=executor.submit(runtime.ensure_loaded)
                assert entered.wait(5)
                runtime.begin_closing()
                abort=executor.submit(runtime.abort_owned,'CPU during-birth shutdown')
                time.sleep(.02)
                state=runtime.snapshot()['model_worker']
                assert state['birth_in_progress'] and state['worker_alive'] is None
                assert not state['resource_reaped'] and not abort.done()
                release.set()
                outcome=abort.result(timeout=5)
                assert outcome['resource_reaped']
                with pytest.raises(QwenWorkerUnavailableError):pending.result(timeout=5)
        pid=runtime.snapshot()['model_worker']['worker_pid']
        assert started==[pid] and not runtime._process.is_alive()
        time.sleep(.02)
        assert started==[pid] and not runtime._process.is_alive()
    finally:
        release.set()
        with pytest.raises(QwenWorkerUnavailableError):runtime.close()


def test_birth_coordination_deadline_cannot_report_pid_none_as_reaped(tmp_path):
    import threading
    runtime,_=candidate(tmp_path);runtime._abort_timeout_s=.05
    entered,release=threading.Event(),threading.Event();started=[]
    context=paused_spawn_context(entered,release,started)
    try:
        with patch('core.qwen_asr_process.multiprocessing.get_context',return_value=context):
            with ThreadPoolExecutor(max_workers=1) as executor:
                pending=executor.submit(runtime.ensure_loaded)
                assert entered.wait(5)
                runtime.begin_closing()
                with pytest.raises(QwenWorkerUnavailableError,match='death not confirmed'):
                    runtime.abort_owned('CPU birth deadline')
                state=runtime.snapshot()['model_worker']
                assert state['birth_in_progress'] and state['worker_alive'] is None
                assert not state['resource_reaped'] and not pending.done()
                release.set()
                with pytest.raises(QwenWorkerUnavailableError):pending.result(timeout=5)
        state=runtime.snapshot()['model_worker']
        assert started==[state['worker_pid']]
        assert state['resource_reaped'] and not runtime._process.is_alive()
    finally:
        release.set()
        with pytest.raises(QwenWorkerUnavailableError):runtime.close()


def test_existing_unavailable_worker_is_not_implicitly_killed_by_new_rpc(tmp_path):
    from core.qwen_asr_runtime import LocalModelUnavailable
    runtime,_=candidate(tmp_path,unavailable=True)
    try:
        with pytest.raises(LocalModelUnavailable):runtime.ensure_loaded()
        pid=runtime.snapshot()['model_worker']['worker_pid']
        with patch('core.qwen_asr_process.os.kill',wraps=os.kill) as kill:
            with pytest.raises(LocalModelUnavailable):runtime.ensure_loaded()
        kill.assert_not_called()
        assert runtime.snapshot()['model_worker']['worker_pid']==pid
        assert runtime.snapshot()['model_worker']['worker_alive']
    finally:runtime.close()
    assert runtime.snapshot()['model_worker']['normal_exit_ok']


@pytest.mark.parametrize('operation',['transcribe','load'])
def test_parent_fd_break_during_owned_work_reaps_worker_before_unavailable(tmp_path,operation):
    from core.qwen_asr_client import QwenAsrClient,QwenAsrUnavailableError
    gate=tmp_path/'finish';events=tmp_path/'events';config=tmp_path/'fake.json'
    option='gate' if operation=='transcribe' else 'load_gate'
    config.write_text(json.dumps({'events':str(events),option:str(gate)}))
    async def scenario():
        client=QwenAsrClient(generation_mode="legacy", local_model_dir=str(config),local_instance_count=1,execution_mode='process',retry_count=1,
            runtime_factory=lambda *args:LocalQwenProcessRuntime(*args,_runtime_factory=__name__+':CpuRuntime',
                                                               _startup_timeout_s=5,_close_timeout_s=.5,_abort_timeout_s=.5))
        request=asyncio.create_task(client.transcribe_bytes(wav_bytes()) if operation=='transcribe' else client.discover_model())
        try:
            async def started():
                while not read_events(events):await asyncio.sleep(.005)
            await asyncio.wait_for(started(),5)
            proxy=client._runtimes[0]
            assert client.snapshot()['busy']==1 and proxy.snapshot()['model_worker']['worker_alive']
            # 故障注入：外部破坏父端FD，不是生产调用路径或普通请求timeout。
            proxy._connection.close()
            with pytest.raises(QwenAsrUnavailableError) as error:await asyncio.wait_for(request,3)
            assert error.value.code==4011 and client._execution_failures==0
            meta=proxy.snapshot()['model_worker']
            assert meta['rpc_unrecoverable'] and meta['resource_reaped'] and not meta['worker_alive']
            assert not meta['normal_exit_ok'] and not proxy._process.is_alive()
            assert client.snapshot()['busy']==0 and client._available.qsize()==1
        finally:
            gate.write_text('finish');await asyncio.gather(request,return_exceptions=True)
            with pytest.raises(RuntimeError):await client.aclose()
    asyncio.run(scenario())


@pytest.mark.parametrize('release_by',['done','shutdown'])
def test_unconfirmed_rpc_fault_keeps_real_lease_until_death_and_shutdown_can_reap(tmp_path,release_by):
    from unittest.mock import PropertyMock
    from core.config import Settings
    from core.qwen_asr_client import QwenAsrClient,QwenAsrUnavailableError
    gate=tmp_path/'finish';events=tmp_path/'events';config=tmp_path/'fake.json'
    config.write_text(json.dumps({'events':str(events),'gate':str(gate)}))
    async def scenario():
        client=QwenAsrClient(generation_mode="legacy", local_model_dir=str(config),local_instance_count=1,execution_mode='process',retry_count=0,
            runtime_factory=lambda *args:LocalQwenProcessRuntime(*args,_runtime_factory=__name__+':CpuRuntime',
                                                               _startup_timeout_s=5,_close_timeout_s=.5,_abort_timeout_s=.5))
        request=asyncio.create_task(client.transcribe_bytes(wav_bytes()))
        close=None
        try:
            async def started():
                while not read_events(events):await asyncio.sleep(.005)
            await asyncio.wait_for(started(),5)
            proxy=client._runtimes[0];original=proxy._reap;refusals=[]
            def first_unconfirmed(*,force,deadline):
                if force and not refusals:
                    refusals.append(True)
                    return {'resource_reaped':False,'normal_exit_ok':False,'worker_alive':True}
                return original(force=force,deadline=deadline)
            with patch.object(proxy,'_reap',side_effect=first_unconfirmed):
                proxy._connection.close()
                async def quarantined():
                    while not proxy.snapshot()['model_worker'].get('rpc_quarantine_waiting_death'):
                        await asyncio.sleep(.005)
                await asyncio.wait_for(quarantined(),2)
                await asyncio.sleep(.03)
                assert not request.done() and client.snapshot()['busy']==1 and client._available.qsize()==0
                assert proxy._process.is_alive() and not proxy.snapshot()['model_worker']['resource_reaped']
                if release_by=='shutdown':
                    with patch.object(Settings,'qwen_process_shutdown_grace_s',new_callable=PropertyMock,return_value=.05):
                        close=asyncio.create_task(client.aclose())
                        with pytest.raises(RuntimeError):await asyncio.wait_for(close,3)
                else:
                    gate.write_text('finish')
                with pytest.raises(QwenAsrUnavailableError):await asyncio.wait_for(request,3)
            meta=proxy.snapshot()['model_worker']
            assert meta['resource_reaped'] and not meta['worker_alive'] and not meta['rpc_quarantined']
            assert not meta['normal_exit_ok'] and client.snapshot()['busy']==0
            if release_by=='done':assert any(row['event']=='read_complete' for row in read_events(events))
        finally:
            gate.write_text('finish');await asyncio.gather(request,return_exceptions=True)
            if close is not None:await asyncio.gather(close,return_exceptions=True)
            with pytest.raises(RuntimeError):await client.aclose()
    asyncio.run(scenario())


def test_result_cardinality_fault_is_unavailable_only_after_owned_reap(tmp_path):
    runtime,_=candidate(tmp_path,short_result=True)
    try:
        with pytest.raises(QwenWorkerUnavailableError,match='cardinality'):
            runtime.transcribe_batch([b'a',b'b'])
        state=runtime.snapshot()['model_worker']
        assert state['rpc_unrecoverable'] and state['resource_reaped'] and not state['worker_alive']
    finally:
        with pytest.raises(QwenWorkerUnavailableError):runtime.close()
