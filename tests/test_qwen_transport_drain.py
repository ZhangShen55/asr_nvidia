"""Real private Pipe pressure; CPU fake forward, never a NPU throughput claim."""
import io
import json
import multiprocessing
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace
import core.qwen_continuous_process as process
from core.qwen_asr_process import _send,_receive


def run_trial(monkeypatch,drain_s,count=48):
    import core.qwen_continuous_runtime as mod
    batches=[]
    class Adapter:
        def __init__(self,*a,**kw):self.active={};self.completed_tokens={}
        def prefill(self,items):
            batches.append(len(items));time.sleep(.003)
            return [((r['request_id'],r['segment_id']),'ok') for r in items]
        def snapshot(self):return dict(active_sequences=0)
    monkeypatch.setattr(mod,'ContinuousHFAdapter',Adapter)
    runtime=SimpleNamespace(api=SimpleNamespace(set_device=lambda x:None),device='cpu',max_new_tokens=256,
        processor=SimpleNamespace(apply_transcription_request=lambda:None),snapshot=lambda:{},close=lambda:None)
    parent,child=multiprocessing.Pipe()
    errors=[];responses=[]
    options=dict(max_active=128,max_batch_size=128,max_batch_tokens=80000,max_batch_audio_frames=115200,
        prefill_batch_size=16,online_repetition_enabled=False,transport_drain_s=drain_s)
    def worker():
        try:process.worker_loop(child,runtime,'trial',options)
        except BaseException as e:errors.append(repr(e))
        finally:child.close()
    def reader():
        try:
            while True:
                row,_=_receive(parent);responses.append(row)
                if row.get('kind')=='closed':return
        except BaseException as e:errors.append(repr(e))
    owner=threading.Thread(target=worker);owner.start()
    consumer=threading.Thread(target=reader);consumer.start()
    stream=io.BytesIO()
    with wave.open(stream,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(b'\0'*256000)
    payload=stream.getvalue()
    started=time.monotonic()
    for index in range(count):
        _send(parent,dict(operation='submit',launch_id='trial',key=('audio',str(index)),audio=payload,language=None))
        # Controlled lock handoff / sender scheduling gap after send unblocks.
        time.sleep(.0005)
    _send(parent,dict(operation='close',launch_id='trial'))
    owner.join(10);consumer.join(10);parent.close()
    assert not owner.is_alive() and not consumer.is_alive()
    assert not errors
    results=[r for r in responses if r.get('kind')=='result']
    assert len(results)==count
    assert len({tuple(r['key']) for r in results})==count
    summary=results[-1]['continuous']['transport']
    return dict(drain_s=drain_s,wall_s=time.monotonic()-started,payload_bytes=len(payload),
        result_count=len(results),prefill_calls=len(batches),mean_prefill=count/len(batches),
        max_prefill=max(batches),summary=summary)


def test_bounded_transport_drain_reduces_prefill_calls_without_losing_members(monkeypatch,tmp_path):
    zero=run_trial(monkeypatch,0)
    drained=run_trial(monkeypatch,.003)
    (tmp_path/'result.json').write_text(json.dumps(dict(zero=zero,drained=drained),indent=2))
    assert drained['prefill_calls']<zero['prefill_calls']
    assert drained['max_prefill']<=16
    assert drained['summary']['received_submits']==48


def test_no_submit_does_not_enable_transport_delay(monkeypatch):
    import core.qwen_continuous_runtime as mod
    class Adapter:
        def __init__(self,*a,**kw):self.active={}
    monkeypatch.setattr(mod,'ContinuousHFAdapter',Adapter)
    parent,child=multiprocessing.Pipe();polls=[]
    class Proxy:
        def poll(self,timeout=0):polls.append(timeout);return child.poll(timeout)
        def recv_bytes(self):return child.recv_bytes()
        def send_bytes(self,value):return child.send_bytes(value)
    runtime=SimpleNamespace(api=None,snapshot=lambda:{},close=lambda:None)
    _send(parent,dict(operation='close',launch_id='trial'))
    process.worker_loop(Proxy(),runtime,'trial',dict(max_active=128,max_batch_size=128,
        prefill_batch_size=16,online_repetition_enabled=False,transport_drain_s=.003))
    assert not any(value is not None and value>0 for value in polls)
    assert _receive(parent)[0]['kind']=='closed'
    child.close();parent.close()


def test_constructor_transport_drain_default_zero_and_range():
    import pytest
    runtime=process.LocalQwenContinuousProcessRuntime('/tmp/no-cpu-model','npu:7')
    assert runtime._launch_config['continuous_options']['transport_drain_s']==0
    runtime.close()
    for value in [True,-1,6,float('nan'),float('inf'),'3']:
        with pytest.raises(ValueError):
            process.LocalQwenContinuousProcessRuntime('/tmp/no-cpu-model','npu:7',continuous_admission_drain_ms=value)
    runtime=process.LocalQwenContinuousProcessRuntime('/tmp/no-cpu-model','npu:7',continuous_admission_drain_ms=3)
    assert runtime._launch_config['continuous_options']['transport_drain_s']==.003
    runtime.close()
