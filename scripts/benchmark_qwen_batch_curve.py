#!/usr/bin/env python3
"""固定真实VAD段的单Qwen批曲线；仅CLI实际运行时才导入NPU/音频依赖。"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def persist(path, data):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.writing')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str)+'\n')
    os.replace(temporary,path)


def percentile(values,q):
    if not values:return None
    ordered=sorted(values); pos=(len(ordered)-1)*q; lo=int(pos); hi=min(lo+1,len(ordered)-1)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(pos-lo)


def read_manifest(path, ordering='manifest'):
    """只读取音频；原始路径字节不能被Unicode替换而指向其它文件。"""
    from collections import Counter
    body=json.loads(Path(path).read_text())
    rows=body['segments'] if isinstance(body,dict) else body
    if len(rows)!=128:raise ValueError('manifest必须固定128段，八请求各16段')
    counts=Counter(str(row['request_id']) for row in rows)
    if len(counts)!=8 or set(counts.values())!={16}:raise ValueError('manifest必须八request_id各16段')
    if len({str(row['id']) for row in rows})!=128:raise ValueError('manifest段id必须唯一')
    if len({(str(row['request_id']),int(row['segment_index'])) for row in rows})!=128:
        raise ValueError('manifest请求/段序号必须唯一')
    import soundfile as sf
    from utils.audio_utils import audio_sample_bounds
    prepared=[]
    source_hashes={}
    for row in rows:
        source=os.fsdecode(base64.b64decode(row['path_bytes_b64'],validate=True))
        source_bytes=os.fsencode(source)
        declared=row.get('source_sha256',row.get('sha256'))
        if not isinstance(declared,str) or len(declared)!=64:raise ValueError('manifest每段必须声明完整源source_sha256或sha256')
        if source_bytes not in source_hashes:
            digest=hashlib.sha256()
            with open(source_bytes,'rb') as source_file:
                for chunk in iter(lambda:source_file.read(1024*1024),b''):digest.update(chunk)
            source_hashes[source_bytes]=digest.hexdigest()
        if source_hashes[source_bytes]!=declared.lower():raise ValueError('源音频SHA256与manifest不一致')
        start=float(row['start_s']); end=float(row['end_s'])
        if start<0 or end<=start:raise ValueError('无效段时间区间')
        with sf.SoundFile(source_bytes) as f:
            if f.samplerate!=16000:raise ValueError('曲线清单必须引用16kHz标准化源音频')
            if round(end*16000)>len(f):raise ValueError('manifest结束超出完整源音频')
            left,right=audio_sample_bounds(start,end,16000,len(f))
            f.seek(left); pcm=f.read(right-left,dtype='float32',always_2d=True).mean(axis=1)
        if len(pcm)<4800:raise ValueError('manifest不得包含不足0.3秒段')
        buffer=io.BytesIO();sf.write(buffer,pcm,16000,format="WAV",subtype="PCM_16");wav=buffer.getvalue()
        prepared.append(dict(row,request_id=str(row['request_id']),segment_index=int(row['segment_index']),
                             wav=wav,pcm=pcm,duration_s=len(pcm)/16000,input_wav_sha256=hashlib.sha256(wav).hexdigest()))
    if ordering=='length':prepared.sort(key=lambda row:(row['duration_s'],row['request_id'],row['segment_index']))
    return prepared


def resource_sample(reset_peak=False):
    """只选择物理卡7；设备聚合与当前进程allocator明确分开。"""
    import subprocess
    import psutil
    import torch
    import torch_npu  # noqa: F401 - 实测路径显式注册；CPU测试不调用本函数
    torch.npu.set_device('npu:7')
    if reset_peak:torch.npu.reset_peak_memory_stats('npu:7')
    process=psutil.Process()
    result={'monotonic_s':time.monotonic(),'pid':process.pid,'created_at':process.create_time(),
            'rss_bytes':process.memory_info().rss,'cpu_times':process.cpu_times()._asdict(),
            'allocator_allocated_bytes':torch.npu.memory_allocated('npu:7'),
            'allocator_reserved_bytes':torch.npu.memory_reserved('npu:7'),
            'peak_allocator_allocated_bytes':torch.npu.max_memory_allocated('npu:7'),
            'peak_allocator_reserved_bytes':torch.npu.max_memory_reserved('npu:7'),
            'peak_scope':'current_case_since_reset'}
    for kind in ('usages','proc-mem'):
        try:
            command=subprocess.run(['npu-smi','info','-t',kind,'-i','7'],capture_output=True,text=True,timeout=3)
            result['npu7_'+kind]={'returncode':command.returncode,'stdout':command.stdout,'stderr':command.stderr}
        except Exception as exc:result['npu7_'+kind]={'unavailable':str(exc)}
    return result


class RoundBatchObserver:
    """Benchmark-only same-event-loop read of published completed batch state.

    Production snapshot deep-copies its complete history twice. Never invoke it on
    the real client's per-member completion path. Completed batch entries are
    immutable after publication; detach each new ID exactly once for this round.
    """
    def __init__(self,client):
        self.client=client;self.batches={};self.collection_calls=0;self.rich_batch_copies=0
        self.fallback_snapshot_calls=0;self.history_entries_examined=0
        state=self._state();self.baseline=state.get('batch_count',0)
        recent=state.get('recent_batches',[])
        self.last_tail_id=recent[-1]['batch_id'] if recent else None

    def _state(self):
        scheduler=getattr(self.client,'_batch_scheduler',None)
        state=getattr(scheduler,'_state',None)
        if isinstance(state,dict):
            self.source='published_event_loop_state'
            return state
        self.source='snapshot_fallback';self.fallback_snapshot_calls+=1
        return self.client.snapshot().get('batch_scheduler',{})

    def collect(self):
        from copy import deepcopy
        self.collection_calls+=1;state=self._state();recent=state.get('recent_batches',[])
        tail=recent[-1]['batch_id'] if recent else None
        if tail!=self.last_tail_id:
            new=[]
            for batch in reversed(recent):
                self.history_entries_examined+=1
                if batch['batch_id']==self.last_tail_id:break
                if batch['batch_id']>self.baseline and batch['batch_id'] not in self.batches:new.append(batch)
            for batch in reversed(new):
                self.batches[batch['batch_id']]=deepcopy(batch);self.rich_batch_copies+=1
            self.last_tail_id=tail
        # No original source object is modified; omit rich history from status.
        compact={key:value for key,value in state.items() if key not in ('recent_batches','_updated_at')}
        compact['recent_completed_count']=len(recent);compact['latest_completed_batch_id']=tail
        self.last_snapshot={'batch_scheduler':compact,'collection_source':self.source}
        return self.last_snapshot

    def metrics(self):
        return {'source':self.source,'collection_calls':self.collection_calls,
                'rich_batch_copies':self.rich_batch_copies,'fallback_snapshot_calls':self.fallback_snapshot_calls,
                'history_entries_examined':self.history_entries_examined}


async def run_round(client, segments, gate, output, *, warmup=False, watchdog_s=420,
                    no_progress_s=120, sample_interval_s=1, sampler=None, parser=None,
                    enhanced=None, clock=time.monotonic, cleanup_timeout_s=60):
    """逐成员计客户端调用边界；共享物理批按batch_id仅存一份。"""
    started=clock(); started_utc=datetime.now(timezone.utc).isoformat()
    records={}; resources=[]; tasks=[]
    observer=RoundBatchObserver(client);batches=observer.batches
    owned_finished=0;work_done=None;work_done_utc=None
    semaphores={row['request_id']:gate.new_segment_semaphore() for row in segments}
    baseline=observer.baseline
    latest_progress=started
    resource_task=None
    report={'warmup':warmup,'started_monotonic_s':started,'started_utc':started_utc,
            'status':'running','member_start_boundary':'client invocation after external gate; not pure device start'}

    def collect():
        return observer.collect()

    def checkpoint():
        if enhanced is not None:
            from copy import deepcopy
            from core.models import local_model_status
            report['enhanced_pool_status']=deepcopy(local_model_status)
        report.update(members=list(records.values()),batches=list(batches.values()),resources=resources,
                      last_snapshot=collect(),updated_monotonic_s=clock())
        persist(output,report)

    async def member(row):
        nonlocal latest_progress
        record={'id':str(row['id']),'request_id':row['request_id'],'segment_index':row['segment_index'],
                'queued_monotonic_s':clock(),'status':'waiting','duration_s':row['duration_s']}
        records[str(row['id'])]=record
        acquired=False
        try:
            await gate.acquire(semaphores[row['request_id']]); acquired=True
            record.update(start_monotonic_s=clock(),status='calling')
            raw=await client.transcribe_bytes(row['wav'],row.get('language'),
                    request_id=row['request_id'],segment_index=row['segment_index'])
            parsed=parser(raw) if parser else {'text':raw}
            record.update(status='completed',raw=raw,parsed=vars(parsed) if hasattr(parsed,'__dict__') else parsed)
        except asyncio.CancelledError:
            record['status']='cancelled'; raise
        except Exception as exc:
            record.update(status='failed',error_type=type(exc).__name__,error=str(exc),code=getattr(exc,'code',None))
        finally:
            if acquired:await gate.release(semaphores[row['request_id']])
            record['ended_monotonic_s']=clock()
            record['elapsed_s']=record['ended_monotonic_s']-record['queued_monotonic_s']
            latest_progress=clock();collect()
            with Path(str(output)+'.members.jsonl').open('a') as events:
                events.write(json.dumps(record,ensure_ascii=False,default=str)+'\n')

    async def owned(coro):
        nonlocal owned_finished,work_done,work_done_utc
        try:return await coro
        finally:
            owned_finished+=1
            if owned_finished==len(segments)+(enhanced is not None):
                work_done=clock();work_done_utc=datetime.now(timezone.utc).isoformat()

    tasks=[asyncio.create_task(owned(member(row))) for row in segments]
    if enhanced is not None:tasks.append(asyncio.create_task(owned(enhanced(segments))))
    pending=set(tasks);next_check=started
    try:
        while pending:
            now=clock()
            if now>=next_check:
                in_flight=[row for row in records.values() if row['status'] in ('waiting','calling')]
                completed=[row['elapsed_s'] for row in records.values() if row['status']=='completed']
                reason=None
                if any(now-row['queued_monotonic_s']>watchdog_s for row in in_flight):reason='inflight_wall_threshold'
                if any(value is not None and value>watchdog_s for value in (percentile(completed,.5),percentile(completed,.95))):
                    reason='completed_percentile_threshold'
                if now-latest_progress>no_progress_s:reason='no_progress_threshold'
                if reason:
                    report.update(status='diagnostic_aborted',diagnostic={'reason':reason,'monotonic_s':now,'inflight':in_flight})
                    checkpoint();break
                if sampler:
                    if resource_task is not None and resource_task.done():
                        try:resources.append(resource_task.result())
                        except Exception as exc:resources.append({'unavailable':str(exc)})
                        resource_task=None
                    if resource_task is None:resource_task=asyncio.create_task(asyncio.to_thread(sampler))
                checkpoint();next_check=clock()+sample_interval_s
            # Finish immediately on the last owned task; no fixed polling tail.
            _,pending=await asyncio.wait(pending,timeout=max(0,next_check-clock()),return_when=asyncio.FIRST_COMPLETED)
    except BaseException as exc:
        report.update(status='interrupted',error_type=type(exc).__name__,error=str(exc)); checkpoint()
        raise
    finally:
        if report['status'] in ('diagnostic_aborted','interrupted'):
            for task in tasks:
                if not task.done():task.cancel()
        # 不提前归还已启动线程/NPU资源，外层有界收尾负责异常期限。
        done,pending=await asyncio.wait(tasks,timeout=cleanup_timeout_s)
        if pending:
            report.update(status='cleanup_failed',pending_owned_tasks=len(pending)); checkpoint()
            raise RuntimeError('owned round工作未能在收尾期限内结束')
        # This boundary precedes resource sampler/diagnostic serialization cleanup.
        if work_done is None:
            work_done=clock();work_done_utc=datetime.now(timezone.utc).isoformat()
        report.update(work_done_monotonic_s=work_done,ended_monotonic_s=work_done,
                      work_done_utc=work_done_utc,ended_utc=work_done_utc,elapsed_s=work_done-started,
                      elapsed_boundary='all owned Qwen/enhanced tasks completed; excludes final resource/measurement cleanup')
        if resource_task is not None:
            await asyncio.wait({resource_task},timeout=cleanup_timeout_s)
            if resource_task.done():
                try:resources.append(resource_task.result())
                except Exception as exc:resources.append({'unavailable':str(exc)})
        collect()
        failures=[row for row in records.values() if row['status']!='completed']
        task_errors=[str(task.exception()) for task in tasks if not task.cancelled() and task.exception() is not None]
        enhanced_failed=False
        if enhanced is not None and not tasks[-1].cancelled() and tasks[-1].exception() is None:
            report['enhanced']=tasks[-1].result()
            enhanced_failed=report['enhanced'].get('status')!='completed'
        if report['status']=='running':report['status']='failed' if failures or task_errors or enhanced_failed else 'completed'
        report['task_errors']=task_errors
        complete=[row['elapsed_s'] for row in records.values() if row['status']=='completed']
        report['success_count']=len(complete); report['total_count']=len(segments)
        report['p50_s']=percentile(complete,.5); report['p95_s']=percentile(complete,.95)
        report['successful_subset_only']=bool(failures)
        report['audio_seconds']=sum(row['duration_s'] for row in segments)
        report['audio_seconds_per_wall_second']=report['audio_seconds']/max(report['elapsed_s'],1e-9)
        report['segments_per_second']=len(complete)/max(report['elapsed_s'],1e-9)
        report['expected_batch_records']=observer.last_snapshot.get('batch_scheduler',{}).get('batch_count',baseline)-baseline
        report['captured_batch_records']=len(batches)
        if report['captured_batch_records']!=report['expected_batch_records']:
            report['status']='evidence_incomplete'
        if report['status']=='completed' and any(value is not None and value>watchdog_s for value in (report['p50_s'],report['p95_s'])):
            report.update(status='diagnostic_aborted',diagnostic={'reason':'completed_percentile_threshold','monotonic_s':work_done,'inflight':[]})
        if resource_task is not None and not resource_task.done():report['status']='measurement_cleanup_failed'
        checkpoint()
        report['batch_collection']=observer.metrics()
        observer_ended=clock()
        report.update(observer_ended_monotonic_s=observer_ended,observer_inclusive_elapsed_s=observer_ended-started,
                      measurement_cleanup_s=max(0,observer_ended-work_done),resource_sampler_pending=resource_task is not None and not resource_task.done(),
                      observer_boundary='after final resource wait, evidence collection and first final report write; excludes publishing these accounting fields')
        persist(output,report)
    return report


async def enhanced_workload(segments):
    from core.models import get_model_pool
    from core.local_inference import generate_local
    from core.async_utils import run_in_thread
    from core.segment_pipeline import gather_owned_branches
    sem=asyncio.Semaphore(128)
    records=[]
    async def one(row):
        record={'id':str(row['id']),'request_id':row['request_id'],'segment_index':row['segment_index']}
        records.append(record)
        async with sem:
            async def infer(name):
                pool=get_model_pool(name);queued=time.monotonic()
                timing={'queued_monotonic_s':queued,'status':'waiting'};record[name]=timing
                try:
                    async with pool.lease() as (model,device):
                        started=time.monotonic();timing.update(start_monotonic_s=started,status='running')
                        try:
                            options = ({'granularity':'utterance', 'extract_embedding':False}
                                       if name == 'emotion2vec' else {})
                            await run_in_thread(generate_local,model,device,input=row['pcm'],fs=16000,**options)
                            timing['status']='completed'
                        finally:
                            pool.record_inference((started-queued)*1000,(time.monotonic()-started)*1000)
                except asyncio.CancelledError:
                    timing['status']='cancelled';raise
                except Exception as exc:
                    timing.update(status='failed',error_type=type(exc).__name__,error=str(exc))
                finally:
                    timing['ended_monotonic_s']=time.monotonic()
            jobs=[infer('speech_campplus')]
            if 1<=row['duration_s']<=10:jobs.append(infer('emotion2vec'))
            else:record['emotion2vec']={'status':'skipped','label':'平淡'}
            await gather_owned_branches(jobs)
    outcomes=await gather_owned_branches(one(row) for row in segments)
    failed=any(timing.get('status')=='failed' for row in records
               for name,timing in row.items() if name in ('speech_campplus','emotion2vec'))
    failed=failed or any(isinstance(item,BaseException) for item in outcomes)
    return {'status':'failed' if failed else 'completed','members':records}


async def close_owned(client,work_task,enhanced_closer=None):
    if not work_task.done():work_task.cancel()
    await asyncio.gather(work_task,return_exceptions=True)
    await client.aclose()
    if enhanced_closer is not None:await enhanced_closer()


async def wait_cleanup(task,timeout_s):
    deadline=time.monotonic()+timeout_s;cancel_received=False
    while not task.done() and time.monotonic()<deadline:
        try:await asyncio.wait({task},timeout=max(0,deadline-time.monotonic()))
        except asyncio.CancelledError:cancel_received=True
    return {'finished':task.done(),'cancel_received':cancel_received}


async def run(args):
    # 必须先设置配置路径，后导入模块级settings。
    os.environ['CONFIG_PATH']=str(Path(args.config).resolve())
    from core.config import settings
    from core.qwen_asr_client import QwenAsrClient
    from core.qwen_asr_queue import RequestConcurrency
    from core.async_utils import await_completion
    from utils.qwen_response import parse_qwen_text
    if settings.qwen_local_device!='npu:7':raise ValueError('本工具只允许宿主物理npu:7')
    output=Path(args.output); output.mkdir(parents=True,exist_ok=False)
    segments=read_manifest(args.manifest,args.ordering)
    manifest_sha=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    summary={'manifest_sha256':manifest_sha,'config_sha256':hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
             'pid':os.getpid(),'ordering':args.ordering,'enhanced':args.enhanced_load_workload,'cases':[],
             'sample_identity':[ {k:v for k,v in row.items() if k not in ('wav','pcm')} for row in segments]}
    persist(output/'summary.json',summary)
    for size in args.batch_sizes:
        client=QwenAsrClient(request_timeout_s=settings.qwen_request_timeout_s,retry_count=settings.qwen_retry_count,
                             max_inference_batch_size=size,batch_wait_timeout_ms=args.batch_wait_ms)
        case={'batch_size':size,'rounds':[],'status':'running'}; summary['cases'].append(case)
        case['before_load_resources']=None
        async def work():
            case['before_load_resources']=await asyncio.to_thread(resource_sample,True)
            if args.enhanced_load_workload:
                from core.models import load_v3_models_if_needed,get_model_pool
                if settings.model_device_speech_campplus_devices!=['npu:7']*8 or settings.model_device_emotion2vec_devices!=['npu:7']*8:
                    raise ValueError('增强容量试验必须配置八CAM和八emo，全部npu:7')
                await load_v3_models_if_needed(True,True)
                for name in ('speech_campplus','emotion2vec'):
                    pool=get_model_pool(name)
                    if sum(model is not None for model in pool.models)!=8:raise RuntimeError('增强池并非八健康实例')
            for index in range(args.warmup+args.repeats):
                path=output/f'batch-{size}-round-{index}.json'
                result=await run_round(client,segments,RequestConcurrency(16,128),path,
                    warmup=index<args.warmup,watchdog_s=args.diagnostic_wall_timeout_s,
                    no_progress_s=args.no_progress_timeout_s,sample_interval_s=args.sample_interval_s,
                    sampler=resource_sample,parser=parse_qwen_text,
                    enhanced=enhanced_workload if args.enhanced_load_workload else None,cleanup_timeout_s=args.cleanup_timeout_s)
                case['rounds'].append({'path':path.name,'status':result['status'],'elapsed_s':result['elapsed_s']})
                case['load_snapshot']=client.snapshot(); persist(output/'summary.json',summary)
                if result['status']!='completed':case['status']=result['status']; break
            if case['status']=='running':case['status']='completed'
        work_task=asyncio.create_task(work())
        try:
            # 包含首次加载；轮次watchdog独立检查每成员，外层只守护无响应准备阶段。
            prepare_started=time.monotonic()
            while not work_task.done():
                await asyncio.wait({work_task},timeout=1)
                # 无轮次文件更新说明仍在资源/模型准备，不能无限等待加载。
                round_files=list(output.glob(f'batch-{size}-round-*.json'))
                if not round_files and time.monotonic()-prepare_started>args.diagnostic_wall_timeout_s:
                    case.update(status='diagnostic_aborted',diagnostic='preparation_wall_threshold')
                    work_task.cancel();break
            if work_task.done():work_task.result()
        except BaseException as exc:
            case.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        finally:
            closer=None
            if args.enhanced_load_workload:
                from core.models import close_v3_model_pools
                closer=close_v3_model_pools
            task=asyncio.create_task(close_owned(client,work_task,closer))
            cleanup_state=await wait_cleanup(task,args.cleanup_timeout_s)
            case['cancel_received_during_cleanup']=cleanup_state['cancel_received']
            done={task} if cleanup_state['finished'] else set()
            if not done:
                case.update(status='cleanup_failed',cleanup={'timeout_s':args.cleanup_timeout_s})
                persist(output/'summary.json',summary)
                # 只退出本工具进程，避免不可中断线程拖住asyncio.run；不触碰共享PID。
                os._exit(2)
            try:task.result();case['cleanup']={'closed':True,'no_service_ports_started':True}
            except BaseException as exc:case.update(status='cleanup_failed',cleanup={'error':str(exc)})
            persist(output/'summary.json',summary)
        if case['status']!='completed':break
    summary['status']='completed' if all(case['status']=='completed' for case in summary['cases']) else 'failed'
    persist(output/'summary.json',summary)
    return 0 if summary['status']=='completed' else 1


def arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--batch-sizes',default='1,8,16,32,64,128')
    parser.add_argument('--warmup',type=int,default=1);parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--batch-wait-ms',type=float,default=10)
    parser.add_argument('--ordering',choices=('manifest','length'),default='manifest')
    parser.add_argument('--enhanced-load-workload',action='store_true')
    parser.add_argument('--diagnostic-wall-timeout-s',type=float,default=420)
    parser.add_argument('--no-progress-timeout-s',type=float,default=120)
    parser.add_argument('--sample-interval-s',type=float,default=1)
    parser.add_argument('--cleanup-timeout-s',type=float,default=60)
    args=parser.parse_args(argv)
    args.batch_sizes=[int(value) for value in args.batch_sizes.split(',')]
    if (not args.batch_sizes or any(value<1 or value>128 for value in args.batch_sizes)
        or args.warmup<0 or args.repeats<1 or args.batch_wait_ms<0 or args.diagnostic_wall_timeout_s>420
        or any(not math.isfinite(value) or value<=0 for value in
               (args.diagnostic_wall_timeout_s,args.no_progress_timeout_s,args.sample_interval_s,args.cleanup_timeout_s))
        or not math.isfinite(args.batch_wait_ms)):
        parser.error('批大小须在1..128，重复次数及诊断/采样/关闭期限为正数')
    return args


if __name__=='__main__':
    raise SystemExit(asyncio.run(run(arguments())))
