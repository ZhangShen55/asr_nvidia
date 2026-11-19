#!/usr/bin/env python3
"""单Qwen真实小批所有权验证；默认只读计划，--execute由owned子进程执行。

仅物理npu:7、batch8、原runtime与权重；wrapper只插入有界返回gate或一次受控异常。
不启动HTTP端口、不安装依赖。总watchdog包含有界收尾，异常回收只针对本工具子进程组。
"""
from __future__ import annotations
import argparse
import copy
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def error_details(exc):
    return {'type': type(exc).__name__, 'code': getattr(exc, 'code', None), 'message': str(exc)}


class Probe:
    def __init__(self, gate_timeout_s=30):
        self.lock = threading.Lock()
        self.events = []
        self.calls = 0
        self.active = 0
        self.peak = 0
        self.gate_timeout_s = gate_timeout_s
        self.prepare('normal')

    def prepare(self, mode, name=None):
        if self.active:
            raise AssertionError('previous owned execution is still active')
        self.mode = mode
        self.phase = name or mode
        self.phase_calls = 0
        self.current_batch_id = None
        self.entered, self.read_done, self.released = threading.Event(), threading.Event(), threading.Event()

    def event(self, kind, **values):
        with self.lock:
            self.events.append({'event': kind, 'monotonic_s': time.monotonic(), **values})

    def snapshot(self):
        with self.lock:
            return {'active': self.active, 'peak': self.peak, 'calls': self.calls, 'events': list(self.events)}


def observed_runtime(base, probe, *, real_npu_runtime=False):
    class Observed(base):
        def transcribe_batch(self, audios, languages=None):
            with probe.lock:
                probe.calls += 1
                call = probe.calls
                probe.phase_calls += 1
                phase_call = probe.phase_calls
                probe.active += 1
                probe.peak = max(probe.peak, probe.active)
            mode, phase, batch_id = probe.mode, probe.phase, probe.current_batch_id
            event_fields = {'call':call, 'phase_call':phase_call, 'phase':phase, 'batch_id':batch_id}
            probe.event('real_runtime_started', **event_fields, members=len(audios), mode=mode,
                        runtime_implementation=base.__module__+'.'+base.__name__,
                        runtime_kind='real_npu_runtime' if real_npu_runtime else 'cpu_test_double')
            probe.entered.set()
            processor = original_decode = None
            decode_was_instance_field = False
            try:
                if mode == 'malformed_output':
                    self.ensure_loaded()
                    processor = self.processor
                    original_decode = processor.batch_decode
                    decode_was_instance_field = 'batch_decode' in getattr(processor, '__dict__', {})
                    def injected_decode(*args, **kwargs):
                        values = list(original_decode(*args, **kwargs))
                        if not values:
                            raise AssertionError('cannot inject malformed member into empty real decode')
                        values[0] = 'controlled malformed prefix'
                        probe.event('malformed_output_injected', **event_fields, member_index=0)
                        return values
                    processor.batch_decode = injected_decode
                values = super().transcribe_batch(audios, languages)
                timings = copy.deepcopy(getattr(self, 'last_batch_timings', None))
                probe.event('real_runtime_read_complete_after_sync', **event_fields,
                            runtime_timings=timings,
                            actual_generate_members=timings.get('batch_size') if timings and timings.get('generate_completed') else None,
                            hardware_execution_confirmed=bool(real_npu_runtime and timings and timings.get('generate_completed')),
                            outputs=[error_details(value) if isinstance(value, BaseException) else value for value in values])
                probe.read_done.set()
                if mode in ('hold', 'fail_after_real'):
                    if not probe.released.wait(probe.gate_timeout_s):
                        raise TimeoutError('owned observation gate deadline exceeded')
                if mode == 'fail_after_real' or (mode == 'retry_fail_once' and phase_call == 1):
                    raise RuntimeError('controlled exception after real batch completed')
                return values
            finally:
                if processor is not None and original_decode is not None:
                    if decode_was_instance_field:
                        processor.batch_decode = original_decode
                    else:
                        delattr(processor, 'batch_decode')
                    probe.event('decode_function_restored', **event_fields,
                                instance_field_restored=decode_was_instance_field)
                probe.event('owned_thread_finished', **event_fields)
                with probe.lock:
                    probe.active -= 1
        def close(self):
            probe.event('runtime_close_started')
            try:
                return super().close()
            finally:
                probe.event('runtime_close_finished', loaded=self.loaded)
    return Observed


async def run_validation(wav_bytes, *, model_dir, dtype='bfloat16', max_new_tokens=256,
                         request_timeout_s=120, observation_timeout_s=60, gate_timeout_s=30,
                         runtime_base=None, report=None, checkpoint=None, should_stop=None):
    from core.qwen_asr_client import QwenAsrClient
    from core.async_utils import await_completion
    from scripts.validate_qwen_cancellation import wait_until
    real_npu_runtime = runtime_base is None
    if runtime_base is None:
        from core.qwen_asr_runtime import LocalQwenRuntime
        runtime_base = LocalQwenRuntime
    probe = Probe(gate_timeout_s)
    class ObservedClient(QwenAsrClient):
        async def _execute_batch(self, members, batch_id):
            probe.current_batch_id = batch_id
            probe.event('logical_batch_started', batch_id=batch_id, phase=probe.phase,
                        members=[{'request_id':m.request_id, 'segment_index':m.segment_index} for m in members])
            try:
                return await super()._execute_batch(members, batch_id)
            finally:
                # 仅观测：super finally已完成busy归还，不重写生产租约/调度逻辑。
                probe.event('instance_release_observed', batch_id=batch_id,
                            busy=self.snapshot()['busy'], available=self._available.qsize())
    client = ObservedClient(local_model_dir=model_dir, local_device='npu:7', local_dtype=dtype,
                          local_instance_count=1, max_new_tokens=max_new_tokens, retry_count=0,
                          request_timeout_s=request_timeout_s, max_inference_batch_size=8,
                          batch_wait_timeout_ms=20, max_batch_audio_frames=102400,
                          max_batch_tokens=78336, batch_max_duration_ratio=4.0,
                          execution_mode="thread", runtime_factory=observed_runtime(runtime_base, probe, real_npu_runtime=real_npu_runtime))
    report = {} if report is None else report
    checkpoint = checkpoint or (lambda: None)
    should_stop = should_stop or (lambda: False)
    report.update(status='running', phases=[], cleanup={'status': 'pending'},
                  runtime_kind='real_npu_runtime' if real_npu_runtime else 'cpu_test_double')
    tasks = []
    def save():
        report['observer'] = probe.snapshot()
        report['client'] = client.snapshot()
        checkpoint()
    def require(value, reason):
        if not value:
            raise AssertionError(reason)
    def stop_check():
        if should_stop():
            raise InterruptedError('stop requested; owned cleanup follows')
    def batch(name, invalid=False):
        group = [asyncio.create_task(client.transcribe_bytes(wav_bytes,
                    language='__not_a_supported_language__' if invalid and i == 0 else None,
                    request_id=f'{name}-{i}', segment_index=i), name=f'{name}-{i}') for i in range(8)]
        tasks.extend(group)
        return group
    def idle():
        state = client.snapshot(); sched = state['batch_scheduler']
        require(state['busy'] == state['waiting'] == sched['waiting'] == sched['running_members'] == sched['inflight'] == 0,
                'owned client or scheduler counts not zero')
        require(probe.active == 0, 'runtime still active')
        require(client._available.qsize() == 1, 'instance must be returned exactly once')
    def strings(values):
        return [value for value in values if isinstance(value, str) and '<asr_text>' in value]
    async def begin_hold(name, mode='hold'):
        stop_check(); idle(); probe.prepare(mode, name)
        group = batch(name)
        await wait_until(probe.entered.is_set, observation_timeout_s, name+' real execution started')
        return group
    try:
        # 懒加载走原客户端；后续timeout注入不把冷加载混为单次推理。
        await client.discover_model(); idle(); save()
        group = await begin_hold('cancel')
        queued = asyncio.create_task(client.transcribe_bytes(wav_bytes, request_id='cancel-queued', segment_index=99))
        tasks.append(queued)
        await wait_until(lambda: client.snapshot()['batch_scheduler']['waiting'] == 1,
                         observation_timeout_s, 'extra member queued behind active batch')
        queued.cancel()
        with_cancel = await asyncio.gather(queued, return_exceptions=True)
        require(isinstance(with_cancel[0], asyncio.CancelledError), 'queued cancellation was lost')
        group[0].cancel(); await asyncio.sleep(0); group[0].cancel(); await asyncio.sleep(0)
        require(not group[0].done() and client.snapshot()['busy'] == 1,
                'repeated cancellation released active batch ownership')
        phase = {'phase': 'partial_repeated_and_queued_cancel', 'before_release': client.snapshot(),
                 'queued_outcome': 'cancelled', 'cancel_requested_monotonic_s': time.monotonic()}
        report['phases'].append(phase); save(); probe.released.set()
        values = await asyncio.gather(*group, return_exceptions=True)
        require(isinstance(values[0], asyncio.CancelledError) and len(strings(values[1:])) == 7,
                'cancelled member affected other seven results')
        idle(); phase.update(status='passed', returned_monotonic_s=time.monotonic(), after=client.snapshot()); save()

        stop_check(); probe.prepare('normal', 'unsupported_member_isolation')
        values = await asyncio.gather(*batch('invalid', invalid=True), return_exceptions=True)
        require(getattr(values[0], 'code', None) == 4009 and len(strings(values[1:])) == 7,
                'invalid language must affect exactly one member')
        idle(); report['phases'].append({'phase': 'unsupported_member_isolation', 'status': 'passed',
                                        'codes': [getattr(value, 'code', None) for value in values]}); save()

        group = await begin_hold('failure', 'fail_after_real')
        await wait_until(probe.read_done.is_set, observation_timeout_s, 'real batch read before one controlled exception')
        require(client.snapshot()['busy'] == 1 and all(not task.done() for task in group), 'failure wrapper lost instance')
        phase = {'phase': 'controlled_failure_after_real_read', 'before_release': client.snapshot()}
        report['phases'].append(phase); save(); probe.released.set()
        values = await asyncio.gather(*group, return_exceptions=True)
        require(all(getattr(value, 'code', None) == 4012 for value in values), 'controlled failure not mapped to 4012')
        idle(); phase.update(status='passed', returned_monotonic_s=time.monotonic(), after=client.snapshot()); save()

        for cancel_all in (False, True):
            stop_check(); idle()
            name = 'retry_all_cancel' if cancel_all else 'retry_partial_cancel'
            probe.prepare('retry_fail_once', name)
            client.retry_count = 1
            failed_before = client._execution_failures
            phase = {'phase':name, 'retry_count':1, 'production_backoff_s':1,
                     'expected_actual_generate_members':[8] if cancel_all else [8,7]}
            report['phases'].append(phase)
            try:
                group = batch(name)
                await wait_until(lambda: client._execution_failures == failed_before + 1 and probe.active == 0,
                                 observation_timeout_s, name+' first real read failed; production backoff entered')
                phase.update(cancel_requested_monotonic_s=time.monotonic(), before_cancel=client.snapshot())
                require(probe.phase_calls == 1 and client.snapshot()['busy'] == 1,
                        'retry cancellation missed the first-attempt backoff window')
                for task in (group if cancel_all else group[:1]):
                    task.cancel()
                values = await asyncio.gather(*group, return_exceptions=True)
                idle()
                require(isinstance(values[0], asyncio.CancelledError), 'retry first member cancellation was lost')
                require(all(isinstance(value, asyncio.CancelledError) for value in values) if cancel_all
                        else len(strings(values[1:])) == 7, 'retry cancellation affected surviving owners')
                state = client.snapshot()
                batch_record = state['batch_scheduler']['recent_batches'][-1]
                attempts = batch_record.get('dispatch_attempts', [])
                reads = [row for row in probe.snapshot()['events']
                         if row['event']=='real_runtime_read_complete_after_sync' and row.get('phase')==name]
                counts = [row['actual_generate_members'] for row in reads]
                expected = phase['expected_actual_generate_members']
                require(counts == expected, 'actual runtime generate members differ from retry ownership contract')
                require(len(attempts) == len(expected), 'retry attempts lost or invented physical calls')
                for index, (attempt, read, count) in enumerate(zip(attempts, reads, expected)):
                    owners = [{'request_id':f'{name}-{i}', 'segment_index':i} for i in (range(8) if index==0 else range(1,8))]
                    require(attempt['input_members'] == owners and attempt['members'] == owners,
                            'attempt member ownership mismatch')
                    require(attempt['actual_dispatched_size'] == count and
                            attempt['runtime_timings']['valid_member_indices'] == list(range(count)),
                            'attempt timings indexed a different dispatch')
                    require(attempt['runtime_timings'].get('generate_started') and
                            attempt['runtime_timings'].get('generate_completed'), 'attempt lacks actual completed generate evidence')
                require(batch_record['selected_size']==8 and batch_record['size']==expected[-1],
                        'summary lost original selection or last physical dispatch')
                require(batch_record['cancelled_before_retry']==(8 if cancel_all else 1), 'retry cancellation count mismatch')
                require(batch_record['runtime_timings']['valid_member_indices']==list(range(expected[-1])),
                        'summary stale timings are associated with a different member list')
                require(batch_record['retry_skipped_all_cancelled'] is cancel_all, 'empty retry was counted as actual execution')
                require(attempts[0]['status']=='failed', 'controlled first attempt failure not retained')
                if not cancel_all:
                    require(attempts[1]['status']=='completed', 'surviving retry did not complete')
                phase.update(status='passed', returned_monotonic_s=time.monotonic(),
                             actual_generate_members=counts, actual_read_events=reads, batch_record=batch_record,
                             outcomes=[error_details(value) if isinstance(value,BaseException) else value for value in values], after=state)
                save()
            finally:
                client.retry_count = 0

        stop_check(); idle(); probe.prepare('malformed_output', 'malformed_output_member_isolation')
        values = await asyncio.gather(*batch('malformed_output'), return_exceptions=True)
        require(getattr(values[0], 'code', None)==4012 and len(strings(values[1:]))==7,
                'malformed prefix must affect only its owner after real generate')
        idle()
        state = client.snapshot(); batch_record = state['batch_scheduler']['recent_batches'][-1]
        timings = batch_record['runtime_timings']
        require(batch_record['size']==8 and timings['batch_size']==8 and timings['output_error_indices']==[0],
                'malformed output was incorrectly removed from physical dispatch')
        report['phases'].append({'phase':'malformed_output_member_isolation','status':'passed',
                                'codes':[getattr(value,'code',None) for value in values], 'batch_record':batch_record,
                                'after':state});save()

        client.request_timeout_s = 0.01
        group = await begin_hold('timeout')
        await asyncio.sleep(.03)
        require(client.snapshot()['busy'] == 1 and all(not task.done() for task in group), 'timeout released a live owned thread')
        phase = {'phase': 'timeout_waits_for_real_execution', 'timeout_s': .01, 'before_release': client.snapshot()}
        report['phases'].append(phase); save()
        await wait_until(probe.read_done.is_set, observation_timeout_s, 'timeout real runtime read complete')
        probe.released.set()
        values = await asyncio.gather(*group, return_exceptions=True)
        require(all(getattr(value, 'code', None) == 4012 for value in values), 'timeout did not map to 4012')
        idle(); phase.update(status='passed', returned_monotonic_s=time.monotonic(), after=client.snapshot()); save()
        client.request_timeout_s = request_timeout_s

        group = await begin_hold('close')
        closer = asyncio.create_task(client.aclose(), name='close-owner'); tasks.append(closer)
        await wait_until(lambda: client._close_task is not None, observation_timeout_s, 'owned close started')
        closer.cancel(); await asyncio.sleep(0); closer.cancel(); await asyncio.sleep(0)
        require(not closer.done() and client.snapshot()['busy'] == 1, 'cancelled close abandoned active work')
        phase = {'phase': 'repeated_cancel_close_while_active', 'before_release': client.snapshot()}
        report['phases'].append(phase); save(); probe.released.set()
        values = await asyncio.gather(*group, return_exceptions=True)
        require(len(strings(values)) == 8, 'close interrupted active eight-member execution')
        closed = await asyncio.gather(closer, return_exceptions=True)
        require(isinstance(closed[0], asyncio.CancelledError), 'close cancellation not propagated after cleanup')
        idle(); require(not client.snapshot()['loaded'], 'runtime still loaded after close')
        phase.update(status='passed', returned_monotonic_s=time.monotonic(), after=client.snapshot()); report['status']='passed'; save()
    except BaseException as exc:
        report.update(status='failed', error=error_details(exc)); save(); raise
    finally:
        probe.released.set()
        try:
            for task in tasks:
                if not task.done(): task.cancel()
            await await_completion(asyncio.gather(*tasks, return_exceptions=True))
            await await_completion(client.aclose())
            idle()
            require(not client.snapshot()['loaded'], 'owned runtime remains loaded')
            report['cleanup']={'status':'passed','client':client.snapshot(),'actual_active':probe.active,
                               'finished_monotonic_s':time.monotonic()}
        except BaseException as exc:
            report['cleanup']={'status':'failed','error':error_details(exc)}; report['status']='failed'; raise
        finally:
            save()
    return report


def checkpoint(path, report):
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(report,ensure_ascii=True,indent=2)+'\n')
    temporary.replace(path)


def read_plan(args):
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    config=tomllib.loads(args.config.read_text())
    qwen=config.get('qwen_asr',{})
    if qwen.get('local_device','npu:7')!='npu:7': raise ValueError('only physical npu:7 is allowed')
    with wave.open(str(args.wav),'rb') as handle:
        if (handle.getframerate(),handle.getnchannels(),handle.getsampwidth(),handle.getcomptype())!=(16000,1,2,'NONE'):
            raise ValueError('WAV must be 16kHz mono PCM16')
        samples=handle.getnframes()
    if abs(samples-65600)>1: raise ValueError('ownership smoke requires a real 4.1-second WAV')
    return {'status':'planned','configuration':{'config':str(args.config.resolve()),'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
            'wav':str(args.wav.resolve()),'wav_sha256':hashlib.sha256(args.wav.read_bytes()).hexdigest(),'samples':samples,
            'device':'npu:7','instances':1,'physical_batch':8,'retry_count':0,'retry_phase_retry_count':1,'retry_phase_count':2,'production_retry_backoff_s':1,'watchdog_s':args.watchdog_s,
            'model_dir':qwen.get('local_model_dir','/home/model/Qwen3-ASR-1.7B-hf'),'dtype':qwen.get('local_dtype','bfloat16'),
            'max_new_tokens':qwen.get('max_new_tokens',256)},'scope':'owned offline real-runtime instrumented smoke; no HTTP ports or hardware exclusivity claim'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--wav',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--watchdog-s',type=float,default=420);parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if not 5<=args.watchdog_s<=420: parser.error('watchdog must be within 5~420 seconds including cleanup')
    if not args.worker and args.output.exists():parser.error('use a new output path; previous evidence is preserved')
    report=read_plan(args);args.output.parent.mkdir(parents=True,exist_ok=True)
    if not args.execute:
        checkpoint(args.output,report);print(json.dumps({'report':str(args.output),'status':'planned'}));return 0
    if args.worker:
        os.environ['CONFIG_PATH']=str(args.config.resolve())
        report=json.loads(args.output.read_text());stops=[]
        for number in (signal.SIGINT,signal.SIGTERM): signal.signal(number,lambda n,f:stops.append(n))
        try:
            q=report['configuration']
            asyncio.run(run_validation(args.wav.read_bytes(),model_dir=q['model_dir'],dtype=q['dtype'],
                         max_new_tokens=q['max_new_tokens'],report=report,
                         checkpoint=lambda:checkpoint(args.output,report),should_stop=lambda:bool(stops)))
            return int(report['status']!='passed')
        except BaseException as exc:
            report.update(status='failed',fatal_error=error_details(exc));return 1
        finally:
            report['stop_signals']=stops;checkpoint(args.output,report)
    checkpoint(args.output,report)
    command=[sys.executable,str(Path(__file__).resolve()),'--config',str(args.config.resolve()),'--wav',str(args.wav.resolve()),
             '--output',str(args.output.resolve()),'--execute','--worker','--watchdog-s',str(args.watchdog_s)]
    started=time.monotonic();reserve=min(30,args.watchdog_s/2);terminated=False;forced=False
    stops=[];old_handlers={}
    for number in (signal.SIGINT,signal.SIGTERM):
        old_handlers[number]=signal.signal(number,lambda n,f:stops.append(n))
    child=None
    try:
        with args.output.with_suffix(args.output.suffix+'.worker.log').open('w') as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try: start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19]
            except (OSError,IndexError): start_ticks=None
            ownership={'pid':child.pid,'parent_pid':os.getpid(),'started_monotonic_s':started,'proc_start_ticks':start_ticks,'new_session':True}
            checkpoint(args.output.with_suffix(args.output.suffix+'.ownership.json'),ownership)
            while child.poll() is None and not stops and time.monotonic()-started < args.watchdog_s-reserve:
                time.sleep(.05)
            if child.poll() is None:
                terminated=True;os.killpg(child.pid,signal.SIGTERM)
                # 总deadline内给SIGKILL后回收预留时间，不在420秒后再额外等一整段宽限。
                remaining=max(.01,args.watchdog_s-(time.monotonic()-started)-min(5,reserve/2))
                try:child.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    forced=True;os.killpg(child.pid,signal.SIGKILL)
                    child.wait(timeout=max(.01,args.watchdog_s-(time.monotonic()-started)))
    finally:
        if child is not None and child.poll() is None:
            forced=True;os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        for number,handler in old_handlers.items():signal.signal(number,handler)
    report=json.loads(args.output.read_text())
    report['supervisor']={'owned_pid':child.pid,'ownership':ownership,'returncode':child.returncode,'elapsed_s':time.monotonic()-started,
                          'watchdog_s':args.watchdog_s,'stop_signals':stops,'term_sent':terminated,'forced_kill':forced,'child_exited':child.poll() is not None}
    if terminated or child.returncode:report['status']='failed'
    checkpoint(args.output,report)
    print(json.dumps({'report':str(args.output),'status':report['status'],'supervisor':report['supervisor']}))
    return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())
