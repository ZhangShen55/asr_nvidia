"""Real same-PCM 128-segment NPU:7 transport comparison; root executes only.

Two serial model lifetimes (0/3ms), each one warm + one steady round. The 32
sender threads reproduce client submit transport contention, not concurrent HF
forward. Result warnings stay separate from fatal failures and normal EOS.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import threading
import time
import wave
import numpy as np


def delta(after, before):
    return {key:after.get(key,0)-before.get(key,0) for key in set(after)|set(before)
            if isinstance(after.get(key,0),(float,int)) and isinstance(before.get(key,0),(float,int))}


def run_round(runtime, audios, request_id, deadline):
    def remaining():
        value=deadline-time.monotonic()
        if value<=0:raise FutureTimeout('total420s gate')
        return value
    before=runtime.snapshot().get('continuous',{})
    started=time.monotonic()
    completed_at={}
    lock=threading.Lock()
    def submit(index):
        future=runtime.submit(audios[index],request_id=request_id,segment_id=str(index))
        def record(_):
            with lock:completed_at[index]=time.monotonic()-started
        future.add_done_callback(record)
        return future
    pool=ThreadPoolExecutor(max_workers=32,thread_name_prefix='transport-sender')
    futures=[]
    try:
        sends=[pool.submit(submit,index) for index in range(len(audios))]
        for send in sends:futures.append(send.result(timeout=remaining()))
        all_submits_done_s=time.monotonic()-started
        outputs=[]
        for index,future in enumerate(futures):
            value=future.result(timeout=remaining())
            if not isinstance(value,dict) or not isinstance(value.get('text'),str):
                raise RuntimeError('continuous result schema mismatch')
            outputs.append(dict(value,index=index,tokens=getattr(future,'generated_tokens',None),
                                completed_s=completed_at.get(index)))
        wall=time.monotonic()-started
        after_state=runtime.snapshot()
        after=after_state.get('continuous',{})
        phases=delta(after.get('phase_wall_s',{}),before.get('phase_wall_s',{}))
        steps=delta(after.get('steps',{}),before.get('steps',{}))
        transport=delta(after.get('transport',{}),before.get('transport',{}))
        hist_after=after.get('transport',{}).get('accepted_per_drain',{})
        hist_before=before.get('transport',{}).get('accepted_per_drain',{})
        prefill_calls=transport.get('prefill_calls',0)
        members=transport.get('prefill_members',0)
        counts=Counter(row['warning']['reason'] for row in outputs if row.get('warning'))
        actions=Counter(row['warning']['action'] for row in outputs if row.get('warning'))
        result=dict(request_id=request_id,wall_s=wall,all_submits_done_s=all_submits_done_s,
            result_count=len(outputs),normal_eos_count=sum(row.get('warning') is None for row in outputs),
            warning_counts=dict(counts),warning_actions=dict(actions),phase_wall_s=phases,steps=steps,
            transport=transport,accepted_per_drain=delta(hist_after,hist_before),
            mean_prefill_members=members/prefill_calls if prefill_calls else None,
            peak_active_mode_lifetime=after.get('peak_active'),peak_kv_bytes_mode_lifetime=after.get('peak_kv_bytes'),
            outputs=outputs,state=after_state)
        runtime.release_request(request_id)
        return result
    except BaseException:
        runtime.begin_closing()
        # Worker death confirmation is needed before any blocked transport sender
        # can relinquish ownership at the global test deadline.
        if time.monotonic()>=deadline:
            runtime.abort_owned('transport benchmark420s gate')
        raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--samples-dir',default='/home/zhangs/qwen-om-smoke-20261120/bench128-20261120')
    parser.add_argument('--gate-s',type=float,default=420)
    args=parser.parse_args()
    from core.qwen_continuous_process import LocalQwenContinuousProcessRuntime
    from utils.incremental_repetition import RepetitionGuardConfig
    source=Path(args.samples_dir)
    manifest=json.loads((source/'manifest.json').read_text())['samples']
    if len(manifest)!=128:raise ValueError('expected128 independent real segments')
    samples=np.load(source/'samples.npz')
    audios=[];hashes=[]
    for row in manifest:
        stream=io.BytesIO()
        with wave.open(stream,'wb') as writer:
            writer.setnchannels(1);writer.setsampwidth(2);writer.setframerate(16000)
            writer.writeframes((np.clip(samples[row['audio_key']],-1,32767/32768)*32768).astype('<i2').tobytes())
        audios.append(stream.getvalue());hashes.append(hashlib.sha256(audios[-1]).hexdigest())
    config=asdict(RepetitionGuardConfig());config['enabled']=True
    report=dict(input_manifest=manifest,wav_sha256=hashes,gate_s=args.gate_s,send_workers=32,
        generation=dict(model='/home/model/Qwen3-ASR-1.7B-hf',device='npu:7',dtype='bfloat16',
                        max_new_tokens=256,max_active=128,max_batch_size=128,prefill_batch_size=16,
                        online_repetition_enabled=True,online_repetition_config=config),
        modes={},owners=[],error=None,
        allocator_scope='mode lifetime including model load and warmup; not a separately reset steady peak')
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    def save():target.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    runtime=None;deadline=time.monotonic()+args.gate_s
    try:
        for milliseconds in (0,3):
            if time.monotonic()>=deadline:raise FutureTimeout('total420s gate before next mode')
            mode=str(milliseconds)+'ms';report['modes'][mode]=[]
            runtime=LocalQwenContinuousProcessRuntime('/home/model/Qwen3-ASR-1.7B-hf','npu:7',
                dtype='bfloat16',max_new_tokens=256,max_active=128,max_batch_size=128,
                prefill_batch_size=16,online_repetition_enabled=True,online_repetition_config=config,
                continuous_admission_drain_ms=milliseconds,max_batch_tokens=80000,max_batch_audio_frames=115200,
                _startup_timeout_s=min(120,max(1,deadline-time.monotonic())),_close_timeout_s=120)
            runtime.ensure_loaded()
            for iteration in range(2):
                row=run_round(runtime,audios,mode+'-'+str(iteration),deadline)
                row['phase']='warm' if iteration==0 else 'steady'
                report['modes'][mode].append(row);save()
            runtime.close();owner=runtime.snapshot();report['owners'].append(owner)
            report['modes'][mode][-1]['allocator_after_close']=owner.get('model_worker',{}).get('worker_allocator')
            runtime=None;save()
        left=report['modes']['0ms'][-1]['outputs'];right=report['modes']['3ms'][-1]['outputs']
        report['differences']=[dict(index=i,sample=manifest[i],left=a,right=b)
            for i,(a,b) in enumerate(zip(left,right))
            if a['text']!=b['text'] or a.get('warning')!=b.get('warning') or a.get('tokens')!=b.get('tokens')]
        report['difference_scope']='mode/batch-shape output differences; no human reference or accuracy claim'
    except BaseException as exc:
        report['error']=type(exc).__name__+': '+str(exc)
        raise
    finally:
        if runtime is not None:
            try:runtime.close()
            except BaseException as exc:report['cleanup_error']=type(exc).__name__+': '+str(exc)
            report['owners'].append(runtime.snapshot())
        save()


if __name__=='__main__':main()
