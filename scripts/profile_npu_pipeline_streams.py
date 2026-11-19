#!/usr/bin/env python3
"""NPU:7进程内流水线stream观测；默认只读计划，--execute才加载真实模型。

复用批曲线的manifest/轮次/增强分支/收尾。仅添加有界事件，不切stream、不改精度、
不删除或新增设备同步。当前仅支持stream观测；--profiler明确拒绝执行，避免把缺少
worker线程标签的trace当作完整证据。所有模型必须在owned子进程内并于finally关闭。
"""
from __future__ import annotations
import argparse
import asyncio
from collections import deque
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


class StreamRecorder:
    def __init__(self, limit=4096):
        self.lock=threading.Lock();self.events=deque(maxlen=limit);self.total=0
        self.phase='initial';self.local=threading.local();self.originals=[];self.patched=set()

    def stamp(self, api, tag, phase, event):
        row={'event':event,'model_instance':tag,'phase':phase,'pid':os.getpid(),
             'thread_id':threading.get_ident(),'native_thread_id':threading.get_native_id(),
             'monotonic_ns':time.monotonic_ns(),'utc_time_ns':time.time_ns()}
        try:
            stream=api.current_stream('npu:7');default=api.default_stream('npu:7')
            row.update(current_stream_id=int(stream.stream_id),current_stream_handle=int(stream.npu_stream),
                       default_stream_id=int(default.stream_id),device_index=int(stream.device_index),
                       is_default=stream.stream_id==default.stream_id)
            if row['device_index']!=7:raise ValueError('observed stream is not on physical card7')
        except Exception as exc:
            row['stream_unavailable']=f'{type(exc).__name__}: {exc}'
        with self.lock:self.total+=1;self.events.append(row)

    def patch(self,obj,name,replacement):
        key=(id(obj),name)
        if key in self.patched:return
        owned=name in getattr(obj,'__dict__',{})
        self.originals.append((obj,name,getattr(obj,name),owned));self.patched.add(key)
        setattr(obj,name,replacement)

    def attach(self, runner, name, tag):
        original=getattr(runner,name);api=runner.api
        def wrapped(*args,**kwargs):
            previous=getattr(self.local,'context',None);phase=self.phase;self.local.context=(tag,phase)
            # 原runtime也会选择7号卡；提前选择只为在worker首个观测点避免查询默认0卡。
            api.set_device('npu:7')
            try:
                with api.device('npu:7'):
                    self.stamp(api,tag,phase,'call_started')
                    try:return original(*args,**kwargs)
                    finally:self.stamp(api,tag,phase,'call_finished_after_original_sync')
            finally:self.local.context=previous
        self.patch(runner,name,wrapped)
        if (id(api),'synchronize') not in self.patched:
            synchronize=api.synchronize
            def observed_sync(*args,**kwargs):
                context=getattr(self.local,'context',None) or ('outside_inference',self.phase)
                self.stamp(api,*context,'device_sync_started')
                try:return synchronize(*args,**kwargs)
                finally:self.stamp(api,*context,'device_sync_finished')
            self.patch(api,'synchronize',observed_sync)

    def snapshot(self):
        with self.lock:return {'events':list(self.events),'total_events':self.total,'dropped_events':max(0,self.total-len(self.events))}

    def restore(self):
        for obj,name,original,owned in reversed(self.originals):
            if owned:setattr(obj,name,original)
            else:delattr(obj,name)
        self.originals.clear();self.patched.clear()


def select_segments(rows):
    grouped={}
    for row in rows:
        if 1<=float(row['duration_s'])<=10:grouped.setdefault(row['request_id'],[]).append(row)
    if len(grouped)!=8 or any(len(items)<2 for items in grouped.values()):
        raise ValueError('need eight requests with at least two 1~10 second segments each')
    return [row for key in sorted(grouped) for row in sorted(grouped[key],key=lambda r:(-r['duration_s'],str(r['id'])))[:2]]


async def observe(client,pools,segments,output,*,enhanced_fn,round_fn,close_pools,gate_factory,
                  recorder=None,checkpoint=None):
    from scripts.benchmark_qwen_batch_curve import close_owned
    recorder=recorder or StreamRecorder();checkpoint=checkpoint or (lambda:None)
    output=Path(output);report={'status':'running','rounds':[],'cleanup':{'status':'pending'},
             'profiler':{'supported':False,'requested':False,'thread_label_validation':'not_available',
                         'reason':'stream-only probe; no profiler trace or worker-label completeness claim'}}
    expected={'qwen[0]'}
    try:
        for i,runtime in enumerate(client._runtimes):recorder.attach(runtime,'transcribe_batch',f'qwen[{i}]')
        for name,pool in pools.items():
            for index,runner in enumerate(pool.models):
                if runner is None:raise ValueError('cannot profile an unhealthy enhanced instance')
                tag=f'{name}[{index}]';expected.add(tag);recorder.attach(runner,'generate',tag)
    except BaseException:
        try:await client.aclose()
        finally:
            try:await close_pools()
            finally:recorder.restore()
        raise
    def save():
        report['stream_observation']=recorder.snapshot()
        persist(output/'stream-observation.json',report)
        checkpoint()
    async def work():
        for name,enhanced in [('warmup',enhanced_fn),('pure',None),('B',enhanced_fn)]:
            recorder.phase=name
            result=await round_fn(client,segments,gate_factory(),output/f'{name}.json',
                         warmup=name=='warmup',watchdog_s=360,no_progress_s=120,
                         sample_interval_s=.05,sampler=None,parser=None,enhanced=enhanced,cleanup_timeout_s=30)
            report['rounds'].append({'phase':name,**result});save()
            if result['status']!='completed':raise RuntimeError(f'{name} round failed: {result["status"]}')
            if name=='warmup':
                actual={r['model_instance'] for r in recorder.snapshot()['events'] if r['phase']=='warmup' and r['event']=='call_started'}
                if not expected<=actual:raise RuntimeError(f'not all configured instances warmed: {sorted(expected-actual)}')
        observed=recorder.snapshot()
        if observed['dropped_events'] or any('stream_unavailable' in row for row in observed['events']):
            raise RuntimeError('stream evidence missing or truncated')
        report['status']='completed'
    task=asyncio.create_task(work())
    async def monitor():
        while not task.done():
            save();await asyncio.sleep(1)
    watcher=asyncio.create_task(monitor())
    try:await task
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc));save();raise
    finally:
        watcher.cancel();await asyncio.gather(watcher,return_exceptions=True)
        recorder.phase='cleanup'
        try:
            from core.async_utils import await_completion
            async def drain_all():
                try:await close_owned(client,task,None)
                finally:await close_pools()
            await await_completion(drain_all())
            state=client.snapshot();sched=state.get('batch_scheduler',{})
            if state['busy'] or state['waiting'] or state['loaded'] or sched.get('inflight',0) or sched.get('running_members',0):
                raise RuntimeError('owned client has active resources after close')
            if any(any(model is not None for model in pool.models) for pool in pools.values()):
                raise RuntimeError('owned enhanced pool still loaded after close')
            report['cleanup']={'status':'passed','client':state,'no_ports_started':True}
        except BaseException as exc:
            report.update(status='failed',cleanup={'status':'failed','error':str(exc)});raise
        finally:recorder.restore();save()
    return report


def persist(path,data):
    from scripts.benchmark_qwen_batch_curve import persist as save
    save(path,data)


async def execute(args,summary):
    os.environ['CONFIG_PATH']=str(args.config.resolve())
    from core.config import settings
    from core.qwen_asr_client import QwenAsrClient
    from core.qwen_asr_queue import RequestConcurrency
    from core.models import load_v3_models_if_needed,get_model_pool,close_v3_model_pools
    from scripts.benchmark_qwen_batch_curve import read_manifest,run_round,enhanced_workload
    if (settings.qwen_local_device!='npu:7' or settings.model_device_speech_campplus_devices!=['npu:7']*8
        or settings.model_device_emotion2vec_devices!=['npu:7']*8):raise ValueError('requires single physical7 Qwen and eight CAM/eight emo on7')
    rows=select_segments(read_manifest(args.manifest))
    summary['sample_identity']=[{k:v for k,v in row.items() if k not in ('wav','pcm')} for row in rows]
    persist(args.output/'summary.json',summary)
    client=QwenAsrClient(local_instance_count=1,retry_count=0,max_inference_batch_size=16,batch_wait_timeout_ms=10)
    pools={}
    async def preparation_and_work():
        await client.discover_model()
        await load_v3_models_if_needed(True,True)
        pools.update({name:get_model_pool(name) for name in ('speech_campplus','emotion2vec')})
        if any(sum(model is not None for model in pool.models)!=8 for pool in pools.values()):
            raise RuntimeError('requires eight healthy CAM and eight healthy emo instances')
        details=await observe(client,pools,rows,args.output,enhanced_fn=enhanced_workload,round_fn=run_round,
                      close_pools=close_v3_model_pools,gate_factory=lambda:RequestConcurrency(16,128))
        summary.update(details)
    try:
        await preparation_and_work()
    finally:
        # observe拥有推理阶段；准备阶段失败仍必须关闭本次已创建的客户端/部分池。
        from core.async_utils import await_completion
        try:
            await await_completion(client.aclose())
        finally:
            try:await await_completion(close_v3_model_pools())
            finally:
                summary['finished_monotonic_s']=time.monotonic();persist(args.output/'summary.json',summary)


def plan(args):
    try:import tomllib
    except ImportError:import tomli as tomllib
    cfg=tomllib.loads(args.config.read_text());q=cfg.get('qwen_asr',{});p=cfg.get('model_device_pool',{})
    if q.get('local_device','npu:7')!='npu:7':raise ValueError('only physical npu:7 is allowed')
    if p.get('speech_campplus_devices')!=['npu:7']*8 or p.get('emotion2vec_devices')!=['npu:7']*8:
        raise ValueError('requires explicit eight CAM and eight emo devices on7')
    manifest=json.loads(args.manifest.read_text());rows=manifest.get('segments',[]) if isinstance(manifest,dict) else manifest
    if len(rows)!=128:raise ValueError('curve manifest must declare 128 fixed segments')
    selected=select_segments([dict(row,duration_s=float(row['end_s'])-float(row['start_s'])) for row in rows])
    return {'status':'planned','started_monotonic_s':time.monotonic(),'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
            'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),'pid':os.getpid(),
            'selection':'two longest 1~10s segments per each of eight requests; same 16 inputs in pure and B',
            'selected_ids':[str(row['id']) for row in selected],'owned_device':'npu:7',
            'watchdog_s':args.watchdog_s,'no_service_ports':True,
            'profiler':{'supported':False,'requested':args.profiler,'thread_label_validation':'not_available',
                        'reason':'stream-only implementation; profiler worker-thread labels not captured/validated'}}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--profiler',action='store_true');parser.add_argument('--watchdog-s',type=float,default=420)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);args=parser.parse_args(argv)
    if not math.isfinite(args.watchdog_s) or not 5<=args.watchdog_s<=420:parser.error('watchdog must be5~420 seconds including cleanup')
    if not args.worker and args.output.exists():parser.error('use a new output directory to preserve evidence')
    summary=plan(args)
    if not args.worker:args.output.mkdir(parents=True,exist_ok=False);persist(args.output/'summary.json',summary)
    if not args.execute:return 0
    if args.profiler:
        summary['status']='unsupported';persist(args.output/'summary.json',summary)
        print('profiler暂未支持跨worker完整标签验证；拒绝静默忽略--profiler',file=sys.stderr);return 2
    if args.worker:
        async def signal_owned_run():
            loop=asyncio.get_running_loop();owner=asyncio.current_task()
            for number in (signal.SIGTERM,signal.SIGINT):loop.add_signal_handler(number,owner.cancel)
            try:await execute(args,summary)
            finally:
                for number in (signal.SIGTERM,signal.SIGINT):loop.remove_signal_handler(number)
        try:asyncio.run(signal_owned_run());return int(summary['status']!='completed')
        except BaseException as exc:
            summary.update(status='failed',error_type=type(exc).__name__,error=str(exc));persist(args.output/'summary.json',summary);return 1
    command=[sys.executable,str(Path(__file__).resolve()),'--manifest',str(args.manifest.resolve()),'--config',str(args.config.resolve()),
             '--output',str(args.output.resolve()),'--execute','--worker','--watchdog-s',str(args.watchdog_s)]
    start=time.monotonic();reserve=min(30,args.watchdog_s/2);stops=[];handlers={};child=None;term=False;kill=False;ownership={}
    for number in (signal.SIGINT,signal.SIGTERM):handlers[number]=signal.signal(number,lambda n,f:stops.append(n))
    try:
        with (args.output/'worker.log').open('w') as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19]
            except (OSError,IndexError):ticks=None
            ownership={'pid':child.pid,'parent_pid':os.getpid(),'proc_start_ticks':ticks,'started_monotonic_s':start,'new_session':True}
            persist(args.output/'ownership.json',ownership)
            while child.poll() is None and not stops and time.monotonic()-start<args.watchdog_s-reserve:time.sleep(.05)
            if child.poll() is None:
                term=True;os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=max(.01,args.watchdog_s-(time.monotonic()-start)-min(5,reserve/2)))
                except subprocess.TimeoutExpired:
                    kill=True;os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=max(.01,args.watchdog_s-(time.monotonic()-start)))
    finally:
        if child is not None and child.poll() is None:kill=True;os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        for number,handler in handlers.items():signal.signal(number,handler)
    summary=json.loads((args.output/'summary.json').read_text())
    summary['supervisor']={'ownership':ownership,'elapsed_s':time.monotonic()-start,'returncode':child.returncode,
                           'term_sent':term,'forced_kill':kill,'child_exited':child.poll() is not None,'stop_signals':stops}
    if term or child.returncode:summary['status']='failed'
    persist(args.output/'summary.json',summary)
    print(json.dumps({'output':str(args.output),'status':summary['status'],'supervisor':summary['supervisor']}))
    return int(summary['status']!='completed')


if __name__=='__main__':raise SystemExit(main())
