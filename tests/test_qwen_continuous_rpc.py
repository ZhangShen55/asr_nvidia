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
    def result():
        assert parent.poll(2)
        return _receive(parent)[0]
    try:
        for segment in ('long','short','short'):
            command('submit',key=('a',segment),audio=stream.getvalue(),language=None)
        first=result();assert first['key']==('a','short')
        assert first['continuous']['active_sequences']==1
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
