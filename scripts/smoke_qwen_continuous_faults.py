"""Owned NPU:7 language/cancel/fault smoke. Root alone executes this script.

Only explicit abort_owned targets the freshly registered owned worker identity.
An expected forced fault exit is never labeled normal close. No HTTP service or
shared process is stopped. The illegal-language phase observes the runtime
category; its established client mapping is service code4009, not a raw RPC code.
"""
from __future__ import annotations
import argparse
from concurrent.futures import TimeoutError as FutureTimeout
import io
import json
from pathlib import Path
import time
import threading
import wave
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--samples-dir',default='/home/zhangs/qwen-om-smoke-20261120/bench128-20261120')
    parser.add_argument('--failed-samples-dir',default='/tmp/qc-failed-segments')
    parser.add_argument('--gate-s',type=float,default=420)
    args=parser.parse_args()
    from core.qwen_continuous_process import LocalQwenContinuousProcessRuntime
    from core.qwen_asr_runtime import LocalUnsupportedLanguage
    from core.qwen_asr_process import _identity
    source=Path(args.samples_dir)
    rows=json.loads((source/'manifest.json').read_text())['samples']
    ordered=sorted(rows,key=lambda row:row['duration_s'])
    failed_source=Path(args.failed_samples_dir)
    failed_rows=json.loads((failed_source/'manifest.json').read_text())['samples']
    chosen=dict(long=next(row for row in failed_rows if row['audio_key']=='audio_a2'),
        short=next(row for row in rows if row['audio_key']=='audio_001'),
        normal=next(row for row in rows if row['audio_key']=='audio_075'))
    samples=np.load(source/'samples.npz')
    failed_samples=np.load(failed_source/'samples.npz')
    audios={}
    for name,row in chosen.items():
        stream=io.BytesIO()
        with wave.open(stream,'wb') as writer:
            writer.setnchannels(1);writer.setsampwidth(2);writer.setframerate(16000)
            pcm=(failed_samples if name=='long' else samples)[row['audio_key']]
            writer.writeframes((np.clip(pcm,-1,32767/32768)*32768).astype('<i2').tobytes())
        audios[name]=stream.getvalue()
    report=dict(samples=chosen,phases={},owners=[],passed=False,error=None,gate_s=args.gate_s,
        scope='single owned continuous runtime per phase; NPU7 only; no shared service signals')
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+args.gate_s
    runtime=None;expected_forced=False
    def save():target.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    def remaining(limit=120):
        seconds=deadline-time.monotonic()
        if seconds<=0:raise FutureTimeout('fault smoke total420s gate')
        return min(limit,seconds)
    def await_value(future,limit=120):return future.result(timeout=remaining(limit))
    def result_row(value):
        if isinstance(value,dict):return dict(text=value['text'],warning=value.get('warning'),recovered=value.get('recovered'))
        return dict(text=value,warning=None)
    def owned_gone(worker):
        try:
            current=_identity(worker['worker_pid'])
            return current.get('start_ticks')!=worker.get('worker_start_ticks')
        except (OSError,ProcessLookupError):return True
    def new_runtime():
        result=LocalQwenContinuousProcessRuntime('/home/model/Qwen3-ASR-1.7B-hf','npu:7',
            dtype='bfloat16',max_new_tokens=256,max_active=16,max_batch_size=16,prefill_batch_size=16,
            online_repetition_enabled=False,continuous_admission_drain_ms=3,
            _startup_timeout_s=remaining(120),_close_timeout_s=120)
        try:
            result.ensure_loaded()
            return result
        except BaseException:
            try:result.close()
            except BaseException as exc:report['startup_cleanup_error']=type(exc).__name__+': '+str(exc)
            report['owners'].append(dict(expected_forced=False,startup_failed=True,state=result.snapshot()))
            raise
    def start_active_pair(request_id):
        long=runtime.submit(audios['long'],request_id=request_id,segment_id='long')
        anchor=runtime.submit(audios['short'],request_id=request_id,segment_id='anchor')
        anchor_value=await_value(anchor)
        if isinstance(anchor_value,dict) and anchor_value.get('warning') is not None:
            raise RuntimeError('anchor ended with generation warning; cannot use as healthy forward proof')
        observe_deadline=time.monotonic()+remaining(20)
        while True:
            state=runtime.snapshot()
            continuous=state.get('continuous',{})
            evidence=[event for event in continuous.get('events',[]) if event.get('request_id')==request_id
                and event.get('segment_id')=='long' and event.get('kind')=='prefill']
            if evidence and not long.done() and continuous.get('active_sequences',0)>=1:
                break
            if long.done() or time.monotonic()>=observe_deadline:
                raise RuntimeError('cannot confirm real long forward still active; do not claim cancellation/fault exercised')
            # Read local pushed cache only; never issue a busy worker GET/RPC.
            time.sleep(.02)
        return long,dict(anchor=result_row(anchor_value),state_before_action=state,
                         long_prefill_evidence=evidence)
    try:
        runtime=new_runtime()
        # Invalid hint must retain its precise category while a healthy real clip
        # executes in the same worker. This is not merely a READY assertion.
        bad=runtime.submit(audios['short'],'DefinitelyNotASupportedLanguage',request_id='language',segment_id='bad')
        good=runtime.submit(audios['normal'],request_id='language',segment_id='good')
        try:
            await_value(bad)
            raise AssertionError('invalid language was accepted')
        except LocalUnsupportedLanguage as exc:
            report['phases']['language']=dict(runtime_error_type=type(exc).__name__,
                service_mapping_code=4009,raw_runtime_code=getattr(exc,'code',None),
                note='4009 is established client mapping; direct RPC reports LocalUnsupportedLanguage')
        healthy=await_value(good)
        if isinstance(healthy,dict) and healthy.get('warning') is not None:
            raise RuntimeError('healthy language companion returned a warning')
        report['phases']['language']['healthy_result']=result_row(healthy)
        report['phases']['language']['state']=runtime.snapshot()
        runtime.release_request('language');save()
        long,row=start_active_pair('cancel')
        row['cancel_requested_monotonic']=time.monotonic()
        runtime.cancel_segment('cancel','long')
        try:
            await_value(long,30)
            raise AssertionError('cancelled long sequence unexpectedly succeeded')
        except Exception as exc:
            if getattr(exc,'reason',None)!='cancelled':raise
            row['cancel_confirmed']=dict(type=type(exc).__name__,reason=exc.reason,monotonic=time.monotonic())
        following=runtime.submit(audios['normal'],request_id='cancel',segment_id='following')
        following_value=await_value(following)
        if isinstance(following_value,dict) and following_value.get('warning') is not None:
            raise RuntimeError('post-cancel healthy member returned a warning')
        row['following_result']=result_row(following_value)
        row['state_after']=runtime.snapshot();report['phases']['cancel']=row
        runtime.release_request('cancel')
        runtime.close();owner=runtime.snapshot()
        if not owner['model_worker'].get('normal_exit_ok') or not owned_gone(owner['model_worker']):
            raise AssertionError('normal language/cancel worker did not exit normally')
        report['owners'].append(dict(expected_forced=False,state=owner));runtime=None;save()
        # A distinct fresh model lifetime is the only target for destructive fault.
        runtime=new_runtime()
        long,row=start_active_pair('fault')
        before=runtime.snapshot()['model_worker'];row['owned_before_abort']=before
        terminal_observed=threading.Event()
        def observe_terminal(_):
            row['future_terminal_callback']=dict(monotonic=time.monotonic(),
                owned_gone_at_completion=owned_gone(before))
            terminal_observed.set()
        long.add_done_callback(observe_terminal)
        runtime.begin_closing();expected_forced=True
        row['abort_requested_monotonic']=time.monotonic()
        row['abort_result']=runtime.abort_owned('fault injection')
        row['owned_death_confirmed_before_future_check']=owned_gone(before)
        if not row['owned_death_confirmed_before_future_check']:
            raise AssertionError('owned worker death not confirmed')
        try:
            await_value(long,30)
            raise AssertionError('aborted owned work unexpectedly succeeded')
        except Exception as exc:
            if isinstance(exc,AssertionError):raise
            row['future_failure']=dict(type=type(exc).__name__,reason=getattr(exc,'reason',None),
                monotonic=time.monotonic(),owned_gone_at_observation=owned_gone(before))
        if not terminal_observed.wait(remaining(10)) or not row['future_terminal_callback']['owned_gone_at_completion']:
            raise AssertionError('future completed before owned worker death confirmation')
        try:
            runtime.submit(audios['normal'],request_id='forbidden-reuse',segment_id='0')
            raise AssertionError('aborted runtime was reused or restarted')
        except Exception as exc:
            if isinstance(exc,AssertionError):raise
            row['reuse_rejected_type']=type(exc).__name__
        row['state_after']=runtime.snapshot();report['phases']['fault']=row
        report['passed']=True
    except BaseException as exc:
        report['error']=type(exc).__name__+': '+str(exc)
        raise
    finally:
        if runtime is not None:
            try:runtime.close()
            except BaseException as exc:
                field='expected_fault_close_error' if expected_forced else 'cleanup_error'
                report[field]=type(exc).__name__+': '+str(exc)
            state=runtime.snapshot()
            report['owners'].append(dict(expected_forced=expected_forced,state=state,
                owned_gone=owned_gone(state['model_worker']) if state.get('model_worker',{}).get('worker_pid') else True))
        save()


if __name__=='__main__':main()
