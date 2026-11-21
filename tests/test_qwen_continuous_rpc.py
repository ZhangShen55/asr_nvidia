"""CPU private pipe tests; no model, no NPU initialization."""
import io
import multiprocessing
import threading
import time
import wave
from types import SimpleNamespace
from core.qwen_asr_process import _send, _receive
from core.qwen_continuous_process import worker_loop


def test_worker_dedup_cancel_release_and_close(monkeypatch):
    import core.qwen_continuous_runtime as mod
    calls = []
    class Adapter:
        def __init__(self, runtime, **kwargs): self.active = {}; self.events = []
        def prefill(self, items):
            for item in items:
                key=(item['request_id'], item['segment_id']); self.active[key]=0;calls.append(key)
            return []
        def decode(self, keys):
            time.sleep(.005)
            done=[]
            for key in keys:
                self.active[key]+=1
                if key[1]=='short' or self.active[key]>100:
                    del self.active[key];done.append((key, 'text'))
            return done
        def cancel(self,key):
            if key in self.active:
                del self.active[key];return [(key,RuntimeError('cancelled'))]
            return []
        def snapshot(self): return {'active_sequences':len(self.active)}
    monkeypatch.setattr(mod,'ContinuousHFAdapter', Adapter)
    runtime=SimpleNamespace(api=SimpleNamespace(set_device=lambda x:None),device='cpu',
        max_new_tokens=256,processor=SimpleNamespace(apply_transcription_request=lambda:None),
        snapshot=lambda:{},close=lambda:None)
    parent,child=multiprocessing.Pipe()
    errors=[]
    def run():
        try: worker_loop(child,runtime,'launch',dict(max_active=2,max_batch_size=2,
            max_batch_tokens=80000,max_batch_audio_frames=115200,prefill_batch_size=2,
            online_repetition_enabled=False))
        except BaseException as exc:errors.append(exc)
        finally:child.close()
    thread=threading.Thread(target=run);thread.start()
    stream=io.BytesIO()
    with wave.open(stream,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(b'\0'*16000)
    def command(operation,**kwargs):_send(parent,dict(operation=operation,launch_id='launch',**kwargs))
    statuses=[]
    def result():
        while True:
            assert parent.poll(2)
            response=_receive(parent)[0]
            if response.get('kind')!='status':return response
            statuses.append(response)
    try:
        for segment in ('long','short','short'):
            command('submit',key=('a',segment),audio=stream.getvalue(),language=None)
        first=result();assert first['key']==('a','short')
        # Status can be pushed before the first result; results need not repeat
        # the same expensive snapshot inside the one-second sample window.
        assert any(row.get('continuous',{}).get('active_sequences')==2 for row in statuses)
        command('cancel',key=('a','long'))
        assert result()['key']==('a','long')
        command('release_request',request_id='a')
        command('submit',key=('a','short'),audio=stream.getvalue(),language=None)
        assert result()['key']==('a','short')
        command('close');assert result()['kind']=='closed'
        thread.join(2);assert not thread.is_alive()
        assert calls.count(('a','short'))==2
        assert not errors
    finally:
        parent.close();thread.join(2)


def test_command_backpressure_does_not_hold_result_routing_lock(monkeypatch):
    import core.qwen_continuous_process as mod
    runtime=mod.LocalQwenContinuousProcessRuntime('/tmp/nonexistent-cpu-test-model','npu:7')
    runtime._reader=object()
    runtime._connection=object()
    readers=[]
    def blocked_send(connection,message):
        # Model a full outbound pipe: send cannot advance until a result reader
        # acquires its routing lock and drains the worker's previous response.
        drained=threading.Event()
        def drain():
            with runtime._continuous_lock:
                key=tuple(message['key'])
                runtime._futures[key].set_result('ok')
                drained.set()
        thread=threading.Thread(target=drain)
        readers.append(thread);thread.start()
        assert drained.wait(1), 'submit held result routing lock while pipe blocked'
    monkeypatch.setattr(mod,'_send',blocked_send)
    for i in range(16):
        future=runtime.submit(b'x'*262144,request_id='pressure',segment_id=str(i))
        assert future.result(1)=='ok'
    for thread in readers:thread.join(1)
    assert all(not thread.is_alive() for thread in readers)


def test_physical_limit_is_independent_of_logical_active_capacity():
    from core.qwen_continuous_process import LocalQwenContinuousProcessRuntime
    runtime=LocalQwenContinuousProcessRuntime('/tmp/nonexistent-cpu-test-model','npu:7',
        max_active=16,max_batch_size=128)
    assert runtime._launch_config['continuous_options']['max_active']==16
    assert runtime._launch_config['continuous_options']['max_batch_size']==128
    runtime.close()


def test_reader_keeps_structured_sequence_error_identity():
    from concurrent.futures import Future
    import core.qwen_continuous_process as mod
    runtime=mod.LocalQwenContinuousProcessRuntime('/tmp/nonexistent-cpu-test-model','npu:7')
    key=('request','17');future=Future();runtime._futures[key]=future
    replies=iter([
        dict(kind='result',launch_id=runtime._launch_id,key=key,result=dict(error=dict(
            type='ContinuousGenerationError',message='safe message',reason='invalid_output_prefix',
            request_id='request',segment_id='17',category='runtime'))),
        dict(kind='closed',launch_id=runtime._launch_id,state={})])
    runtime._wait_response=lambda *args:(next(replies),0)
    runtime._accept_state=lambda state:None
    runtime._read_continuous()
    error=future.exception()
    assert error.reason=='invalid_output_prefix'
    assert error.request_id=='request' and error.segment_id=='17'
    assert runtime._close_ack.is_set()


def test_reader_keeps_segment_warning_payload_and_does_not_put_body_in_snapshot():
    from concurrent.futures import Future
    import core.qwen_continuous_process as mod
    runtime=mod.LocalQwenContinuousProcessRuntime('/tmp/nonexistent-cpu-test-model','npu:7')
    key=('request','warn');future=Future();runtime._futures[key]=future
    replies=iter([
        dict(kind='result',launch_id=runtime._launch_id,key=key,result=dict(
            text='language Chinese<asr_text>保留内容。',
            warning=dict(reason='repetition_stop',action='recover_prefix',recovered=True),
            recovered=True,original_raw='private repeated original',tokens=[1,2,3])),
        dict(kind='closed',launch_id=runtime._launch_id,state={})])
    runtime._wait_response=lambda *args:(next(replies),0)
    runtime._accept_state=lambda state:None
    runtime._read_continuous()
    value=future.result()
    assert value['warning']['reason']=='repetition_stop'
    assert value['recovered'] and value['original_raw']=='private repeated original'
    assert 'private repeated original' not in str(runtime.snapshot())
    assert future.generated_tokens==[1,2,3]


def test_unsolicited_status_only_updates_cache_and_does_not_complete_future():
    from concurrent.futures import Future
    import core.qwen_continuous_process as mod
    runtime=mod.LocalQwenContinuousProcessRuntime('/tmp/nonexistent-cpu-test-model','npu:7')
    future=Future();runtime._futures[('request','long')]=future
    replies=[dict(kind='status',launch_id=runtime._launch_id,continuous=dict(active_sequences=1,
        events=[dict(request_id='request',segment_id='long',kind='prefill')])),
        dict(kind='closed',launch_id=runtime._launch_id,state={})]
    def response(*args):
        if len(replies)==1:
            assert not future.done()
            assert runtime.snapshot()['continuous']['active_sequences']==1
        return replies.pop(0),0
    runtime._wait_response=response;runtime._accept_state=lambda state:None
    runtime._read_continuous()
    assert not future.done()
    assert runtime._close_ack.is_set()


def test_busy_worker_pushes_status_without_waiting_for_a_result(monkeypatch):
    import core.qwen_continuous_runtime as mod
    class Adapter:
        def __init__(self,*a,**kw):self.active={};self.completed_tokens={}
        def prefill(self,items):
            output=[]
            for row in items:
                key=(row['request_id'],row['segment_id'])
                if key[1]=='short':output.append((key,'short result'))
                else:self.active[key]=0
            return output
        def decode(self,keys):
            time.sleep(.01)
            for key in keys:self.active[key]+=1
            return []
        def cancel(self,key):
            self.active.pop(key,None);return [(key,RuntimeError('cancelled'))]
        def snapshot(self):return dict(active_sequences=len(self.active))
    monkeypatch.setattr(mod,'ContinuousHFAdapter',Adapter)
    runtime=SimpleNamespace(api=SimpleNamespace(set_device=lambda x:None),device='cpu',max_new_tokens=256,
        processor=SimpleNamespace(apply_transcription_request=lambda:None),snapshot=lambda:{},close=lambda:None)
    parent,child=multiprocessing.Pipe();errors=[]
    def run():
        try:worker_loop(child,runtime,'push',dict(max_active=16,max_batch_size=16,
            max_batch_tokens=80000,max_batch_audio_frames=115200,prefill_batch_size=16,
            online_repetition_enabled=False))
        except BaseException as exc:errors.append(exc)
        finally:child.close()
    owner=threading.Thread(target=run);owner.start()
    stream=io.BytesIO()
    with wave.open(stream,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(b'\0'*16000)
    def send(operation,**kwargs):_send(parent,dict(operation=operation,launch_id='push',**kwargs))
    try:
        send('submit',key=('a','short'),audio=stream.getvalue(),language=None)
        assert parent.poll(2);first=_receive(parent)[0]
        assert first['kind']=='result'
        send('submit',key=('b','long'),audio=stream.getvalue(),language=None)
        assert parent.poll(2)
        pushed=_receive(parent)[0]
        assert pushed['kind']=='status'
        assert pushed['continuous']['active_sequences']==1
        assert pushed['continuous']['status_source']=='owner_safe_step_boundary'
        send('cancel',key=('b','long'));send('close')
        while True:
            assert parent.poll(2)
            if _receive(parent)[0].get('kind')=='closed':break
        owner.join(2);assert not owner.is_alive() and not errors
    finally:
        parent.close();owner.join(2)
