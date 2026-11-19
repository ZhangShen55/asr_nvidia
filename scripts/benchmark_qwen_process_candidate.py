#!/usr/bin/env python3
"""单HF Qwen线程/模型worker隔离候选配对实验；默认只计划，不启动HTTP服务。

先加载父进程八CAM/八emo，依次thread/process；每种各pure/B各1暖3稳。
pure指NN常驻但idle，不等同于卸载NN的纯Qwen曲线。只观测默认global策略原CPU输出。
"""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def persist(path,value):
    temporary=Path(str(path)+'.writing');temporary.write_text(json.dumps(value,ensure_ascii=True,indent=2,default=str)+'\n')
    temporary.replace(path)


def _selection_sample_count(row):
    """计划和实际裁剪统一使用16k整数采样数，避免同长段浮点差改变排序。"""
    samples = row.get('sample_count')
    if 'sample_count' in row and (type(samples) is not int or samples < 1):
        raise ValueError('sample_count must be a positive integer')
    has_endpoints = 'start_s' in row or 'end_s' in row
    if has_endpoints:
        if 'start_s' not in row or 'end_s' not in row:
            raise ValueError('both segment endpoints are required')
        start, end = float(row['start_s']), float(row['end_s'])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError('invalid segment endpoints')
        left, right = round(start*16000), round(end*16000)
        endpoint_samples = right-left
        if samples is not None and samples != endpoint_samples:
            raise ValueError('sample_count differs from rounded segment endpoints')
        samples = endpoint_samples if samples is None else samples
        if 'start_sample' in row or 'end_sample' in row:
            if (type(row.get('start_sample')) is not int or type(row.get('end_sample')) is not int
                    or (row['start_sample'], row['end_sample']) != (left, right)):
                raise ValueError('sample bounds differ from rounded segment endpoints')
    elif samples is None:
        # 仅兼容不含源端点的CPU toy；真实manifest必须由reader验证源与端点。
        duration = float(row['duration_s'])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('invalid toy segment duration')
        samples = round(duration*16000)
    if samples < 4800:
        raise ValueError('selected manifest contains a segment shorter than0.3seconds')
    if 'pcm' in row and len(row['pcm']) != samples:
        raise ValueError('actual PCM length differs from canonical selection samples')
    return samples


def selected_rows(rows,count):
    if len(rows)!=128:raise ValueError('requires a fixed128 manifest, not complete-audio requests')
    counts=Counter(str(row['request_id']) for row in rows)
    if len(counts)!=8 or set(counts.values())!={16} or len({str(row['id']) for row in rows})!=128:
        raise ValueError('requires128 unique segments, eight request owners each16')
    rows = [dict(row, duration_s=_selection_sample_count(row)/16000) for row in rows]
    if count==128:return rows
    if count!=16:raise ValueError('only16 or128 offline segments supported')
    from scripts.profile_npu_pipeline_streams import select_segments
    return select_segments(rows)


def qwen_quality(reference,result,ids):
    def index(records):
        rows={str(row['id']):row for row in records}
        if len(rows)!=len(records) or set(rows)!=set(ids):raise ValueError('missing/duplicate Qwen member evidence')
        if any(row.get('status')!='completed' or not isinstance(row.get('raw'),str) or 'parsed' not in row for row in rows.values()):
            raise ValueError('failed/incomplete Qwen member evidence')
        return rows
    current=index(result)
    baseline=index(reference) if reference is not None else current
    differences=[{'id':key,'raw_equal':baseline[key]['raw']==current[key]['raw'],
                  'parsed_equal':baseline[key]['parsed']==current[key]['parsed']}
                 for key in ids if baseline[key]['raw']!=current[key]['raw'] or baseline[key]['parsed']!=current[key]['parsed']]
    return {'passed':not differences,'expected_members':len(ids),'differences':differences,
            'scope':'same-input raw/parsed agreement, not transcription accuracy'}


def nn_quality(reference,current,segments):
    from scripts.benchmark_npu_stream_policy import compare_results
    expected=Counter(speech_campplus=len(segments),emotion2vec=sum(1<=row['duration_s']<=10 for row in segments))
    counts=Counter(row.get('model') for row in current)
    if counts!=expected:raise ValueError(f'missing original NN result evidence: actual{dict(counts)} expected{dict(expected)}')
    for row in current:
        size=192 if row['model']=='speech_campplus' else 7
        if len(row.get('vector',[]))!=size or row.get('actual_vector_size')!=size:
            raise ValueError('original CAM192/emotion7 vector shape mismatch')
        if row['model']=='emotion2vec' and len(row.get('labels') or [])!=7:
            raise ValueError('original emotion7 labels unavailable')
    comparison=compare_results(reference if reference is not None else current,current,atol=1e-5,rtol=1e-5)
    comparison.update(result_counts=dict(counts),projection9_available=False,
                      scope='original returned CPU CAM192/emotion7; labels exact, atol=rtol=1e-5')
    return comparison


async def paired(segments,output,report,bindings,*,watchdog_s=420):
    """窄实验编排；bindings只供本CLI真实依赖/CPU测试替身，不连HTTP/config工厂。"""
    from core.async_utils import await_completion
    output=Path(output);clients=[];closed=set();policy=None;qwen_reference=nn_reference=None
    ids=[str(row['id']) for row in segments]
    if 'selected_ids' in report and report['selected_ids'] != ids:
        raise ValueError('actual selected segment IDs differ from frozen plan')
    report.update(status='running',rounds=[],transports=[],cleanup={'status':'pending'})
    try:
        began=time.monotonic();pools=await bindings.prepare_pools()
        report['nn_load_s']=time.monotonic()-began;report['nn_loaded_status']=bindings.pool_status()
        policy=bindings.make_policy(pools)
        for transport in ('thread','process'):
            bindings.fence(pools)
            case={'transport':transport,'status':'running','rounds':[]};report['transports'].append(case)
            client=bindings.make_client(transport);clients.append(client)
            load_start=time.monotonic()
            await client.discover_model()
            case.update(qwen_load_s=time.monotonic()-load_start,loaded_snapshot=client.snapshot())
            for mode in ('pure','B'):
                for index in range(4):
                    phase=f'{transport}-{mode}-{index}';policy.phase=phase
                    bindings.fence(pools)
                    # 测量峰值重置/资源准备不进入work_done计时，子进程峰值由其自身RPC边界报告。
                    preparation_started=time.monotonic()
                    before=await asyncio.to_thread(bindings.sample,True)
                    resource_preparation_s=time.monotonic()-preparation_started
                    round_entered=time.monotonic()
                    result=await bindings.round_fn(client,segments,bindings.gate_factory(),output/f'{phase}.json',
                        warmup=index==0,watchdog_s=watchdog_s,no_progress_s=120,sample_interval_s=1,
                        sampler=bindings.sample,parser=bindings.parser,
                        enhanced=bindings.enhanced if mode=='B' else None,cleanup_timeout_s=30)
                    row={'transport':transport,'mode':mode,'round':index,'warmup':index==0,
                         'path':phase+'.json','status':result.get('status'),
                         'work_done_elapsed_s':result.get('elapsed_s'),
                         'measurement_cleanup_s':result.get('measurement_cleanup_s'),
                         'before_resources':before,'resource_preparation_s':resource_preparation_s,
                         'client_after':client.snapshot(),
                         'actual_batch_sizes':[batch.get('actual_dispatched_size',batch.get('size')) for batch in result.get('batches',[])],
                         'worker_call_evidence':bindings.worker_calls(phase)}
                    report['rounds'].append(row);case['rounds'].append(phase)
                    persist(output/'summary.json',report)
                    if result.get('status')!='completed':raise RuntimeError('round failed: '+str(result.get('status')))
                    row['qwen_quality']=qwen_quality(qwen_reference,result.get('members',[]),ids)
                    if qwen_reference is None:
                        qwen_reference=copy.deepcopy(result['members']);report['qwen_baseline_phase']=phase
                    values=policy.phase_results(phase)
                    if mode=='B':
                        row['nn_quality']=nn_quality(nn_reference,values,segments)
                        row['original_nn_cpu_results']=values
                        if nn_reference is None:
                            nn_reference=copy.deepcopy(values);report['nn_baseline_phase']=phase
                    elif values:
                        raise ValueError('pure resident-NN-idle round unexpectedly ran NN workload')
                    evidence=policy.snapshot()
                    if evidence.get('dropped_events') or evidence.get('dropped_results'):
                        raise ValueError('bounded NN evidence overflow')
                    persist(output/f'{phase}-nn-evidence.json',{'results':values,'policy':'default_global'})
                    persist(output/'summary.json',report)
                    row['paired_observer_inclusive_elapsed_s']=time.monotonic()-round_entered
                    row['paired_observer_cleanup_s']=max(0,row['paired_observer_inclusive_elapsed_s']-result['elapsed_s'])
                    row['timing_scope']='work_done from frozen curve; paired observer tail includes final metadata/quality/persistence, no crossPID clock subtraction'
                    persist(output/'summary.json',report)
                    if not row['qwen_quality']['passed'] or (mode=='B' and not row['nn_quality']['passed']):
                        raise ValueError('quality gate failed; stopping without next round/transport')
            bindings.fence(pools)
            await await_completion(client.aclose())
            after=client.snapshot();case['closed_snapshot']=after
            if after.get('loaded') or after.get('busy') or after.get('waiting'):
                raise RuntimeError('previous Qwen did not close cleanly before next transport')
            closed.add(id(client));case['status']='completed';persist(output/'summary.json',report)
        report['status']='completed'
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        errors=[]
        if policy is not None:
            try:
                persist(output/'nn-policy-final.json',policy.snapshot());policy.restore()
            except BaseException as exc:errors.append(f'policy restore: {exc}')
        async def drain():
            for client in clients:
                if id(client) in closed:continue
                try:
                    await client.aclose()
                    state=client.snapshot()
                    if state.get('loaded') or state.get('busy') or state.get('waiting'):
                        raise RuntimeError('Qwen cleanup state not empty/healthy')
                except BaseException as exc:errors.append(f'Qwen close: {exc}')
            try:await bindings.close_pools()
            except BaseException as exc:errors.append(f'NN close: {exc}')
        owned_cleanup=asyncio.create_task(drain());cleanup_cancelled=False
        try:await await_completion(owned_cleanup)
        except asyncio.CancelledError:
            cleanup_cancelled=True
            if owned_cleanup.cancelled():errors.append('owned cleanup task itself cancelled')
            elif owned_cleanup.done():owned_cleanup.result()
        report['cleanup']={'status':'failed' if errors else 'passed','errors':errors,'ports_started':False,
                           'client_count':len(clients),'cancel_received_during_cleanup':cleanup_cancelled,
                           'nn_loaded_after':bindings.pool_status()}
        if errors:report.update(status='failed')
        persist(output/'summary.json',report)
        if errors:raise RuntimeError('owned cleanup failed: '+str(errors))
        if cleanup_cancelled:raise asyncio.CancelledError
    return report


async def execute(args,report):
    os.environ['CONFIG_PATH']=str(args.config.resolve())
    from types import SimpleNamespace
    from core.config import settings
    from core.qwen_asr_client import QwenAsrClient
    from core.qwen_asr_runtime import LocalQwenRuntime
    from core.qwen_asr_queue import RequestConcurrency
    from core.models import load_v3_models_if_needed,get_model_pool,close_v3_model_pools,local_model_status
    from core.async_utils import await_completion
    from utils.qwen_response import parse_qwen_text
    from core.qwen_asr_process import LocalQwenProcessRuntime
    from scripts.benchmark_qwen_batch_curve import read_manifest,run_round,resource_sample,enhanced_workload
    from scripts.benchmark_npu_stream_policy import StreamPolicy
    runtime_origin=str(Path(sys.modules[LocalQwenRuntime.__module__].__file__).resolve())
    if not runtime_origin.endswith('.py'):
        raise ValueError('paired CLI currently accepts source runtime only; protected so validation is separate')
    report['runtime_module_file']=runtime_origin
    pools={};calls=[];phase=['preparation']
    rows=selected_rows(read_manifest(args.manifest),args.segments)
    report['sample_identity']=[{key:value for key,value in row.items() if key not in ('wav','pcm')} for row in rows]
    async def prepare():
        await load_v3_models_if_needed(True,True)
        pools.update({name:get_model_pool(name) for name in ('speech_campplus','emotion2vec')})
        if any(len(pool.models)!=8 or any(model is None for model in pool.models) for pool in pools.values()):
            raise RuntimeError('requires eight healthy CAM and eight healthy emo before Qwen load')
        if any(runner.device!='npu:7' for pool in pools.values() for runner in pool.models):
            raise ValueError('enhanced runner escaped physical7')
        return pools
    def make_client(transport):
        runtime=LocalQwenRuntime if transport=='thread' else LocalQwenProcessRuntime
        client=QwenAsrClient(local_instance_count=1,retry_count=0,execution_mode=transport,
                             max_inference_batch_size=args.segments,
                             batch_wait_timeout_ms=settings.qwen_batch_wait_timeout_ms,runtime_factory=runtime)
        if transport=='process':
            for proxy in client._runtimes:
                original=proxy.transcribe_batch
                def observed(audios,languages=None,_original=original,_proxy=proxy):
                    started=time.monotonic();tag=phase[0]
                    try:return _original(audios,languages)
                    finally:
                        # Immutable published cache reference, no extra RPC/deepcopy/hashing per member/batch.
                        calls.append({'phase':tag,'parent_started_monotonic_s':started,
                                      'parent_ended_monotonic_s':time.monotonic(),
                                      'metadata':_proxy._cache['model_worker']})
                proxy.transcribe_batch=observed
        return client
    def make_policy(pools):
        runners=[(f'{name}[{i}]',runner,'generate') for name,pool in pools.items() for i,runner in enumerate(pool.models)]
        return StreamPolicy('default_global',runners).install()
    def fence(pools):
        for api in {id(runner.api):runner.api for pool in pools.values() for runner in pool.models}.values():
            api.set_device('npu:7');api.synchronize('npu:7')
    async def round_fn(*arguments,**keywords):
        phase[0]=Path(arguments[3]).stem
        return await run_round(*arguments,**keywords)
    async def close_pools():
        await await_completion(close_v3_model_pools())
        if any(any(model is not None for model in pool.models) for pool in pools.values()):
            raise RuntimeError('enhanced pool models remain loaded')
    bindings=SimpleNamespace(prepare_pools=prepare,close_pools=close_pools,pool_status=lambda:copy.deepcopy(local_model_status),
        make_client=make_client,make_policy=make_policy,fence=fence,round_fn=round_fn,
        gate_factory=lambda:RequestConcurrency(16,128),enhanced=enhanced_workload,sample=resource_sample,
        parser=parse_qwen_text,worker_calls=lambda tag:[row for row in calls if row['phase']==tag])
    # Repeated cancellation cannot skip ownership cleanup; outer supervisor is the final bounded limit.
    await paired(rows,args.output,report,bindings,watchdog_s=args.watchdog_s)


def resolve_core_artifact(root, name="qwen_asr_process"):
    """只读取artifact路径，不为默认plan导入runtime/模型；遵循保护扩展优先。"""
    directory = Path(root) / "core"
    extensions = sorted(directory.glob(name + "*.so"))
    if len(extensions) > 1:
        raise ValueError("ambiguous protected core artifact: " + str(extensions))
    if extensions:
        return extensions[0]
    source = directory / (name + ".py")
    if source.is_file():
        return source
    raise ValueError("missing core artifact: " + str(directory / name))


def plan(args):
    try:import tomllib
    except ImportError:import tomli as tomllib
    cfg=tomllib.loads(args.config.read_text());q=cfg.get('qwen_asr',{});p=cfg.get('model_device_pool',{})
    if q.get('local_device')!='npu:7' or q.get('local_instance_count',1)!=1:
        raise ValueError('requires a single physical7 Qwen')
    if any(p.get(key)!=['npu:7']*8 for key in ('speech_campplus_devices','emotion2vec_devices')):
        raise ValueError('requires explicit parent CAM8/emo8 on physical7')
    body=json.loads(args.manifest.read_text());rows=body.get('segments',[]) if isinstance(body,dict) else body
    selected=selected_rows([dict(row,duration_s=float(row['end_s'])-float(row['start_s'])) for row in rows],args.segments)
    sources={str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for name in ('scripts/benchmark_qwen_batch_curve.py','scripts/benchmark_npu_stream_policy.py'):
        path=ROOT/name
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    core_artifact = resolve_core_artifact(ROOT)
    sources[str(core_artifact)] = hashlib.sha256(core_artifact.read_bytes()).hexdigest()
    return {'status':'planned','config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),'source_sha256':sources,
        'selected_ids':[str(row['id']) for row in selected],'segment_count':len(selected),
        'scope':'offline16/128 VAD segments, not N8 complete-audio/service throughput',
        'order':['thread-pure','thread-B','process-pure','process-B'],'warmup_per_case':1,'steady_per_case':3,
        'pure_scope':'parent CAM8/emo8 resident but idle','nn_policy':'default_global; NN16 only; Qwen unhooked',
        'quality_scope':'same-input raw/parsed exact; original CPU CAM192/emo7 atol=rtol1e-5 and labels exact; no projection9',
        'resource_scope':'parent allocator and child allocator/RSS/wire payload separately; never sum crossPID peaks as simultaneous total',
        'watchdog_s':args.watchdog_s,'device':'npu:7','ports_started':False,'candidate_only':True}


def _owned(process,created,sid):
    try:return process.create_time()==created and os.getsid(process.pid)==sid
    except Exception:return False


def supervise(command,args):
    """登记新SID的真实后代；信号只发送PID/ctime/SID仍吻合的owned进程。"""
    import psutil
    started=time.monotonic();deadline=started+args.watchdog_s;reserve=min(30,args.watchdog_s/2)
    registered={};events=[];stops=[];handlers={};child=None;terminated=forced=False
    for number in (signal.SIGINT,signal.SIGTERM):handlers[number]=signal.signal(number,lambda n,f:stops.append(n))
    def register():
        if child is None:return
        candidates=[]
        try:candidates=[psutil.Process(child.pid)]+psutil.Process(child.pid).children(recursive=True)
        except psutil.Error:pass
        # SID remains anchored after leader exit; include same-session orphans before final cleanup.
        if child.poll() is not None:
            for process in psutil.process_iter(['pid','create_time']):
                try:
                    if os.getsid(process.pid)==child.pid:candidates.append(process)
                except OSError:pass
        for process in candidates:
            try:
                created=process.create_time()
                if os.getsid(process.pid)==child.pid:
                    registered.setdefault((process.pid,created),{'pid':process.pid,'create_time':created,'sid':child.pid,'cmdline':process.cmdline()})
            except (psutil.Error,OSError):pass
    def alive():
        found=[]
        for identity in registered.values():
            try:
                process=psutil.Process(identity['pid'])
                if _owned(process,identity['create_time'],identity['sid']) and process.status()!=psutil.STATUS_ZOMBIE:found.append(process)
            except psutil.Error:pass
        return found
    def send(process,signum):
        identity=next((row for row in registered.values() if row['pid']==process.pid
                       and _owned(process,row['create_time'],row['sid'])),None)
        if identity is not None:
            try:os.kill(process.pid,signum);events.append({'pid':process.pid,'signal':int(signum),'monotonic_s':time.monotonic()})
            except ProcessLookupError:pass
    try:
        with (args.output/'worker.log').open('w') as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            register()
            while time.monotonic()<deadline-reserve and not stops and child.poll() is None:
                register();time.sleep(.05)
            if child.poll() is None:
                terminated=True
                register();send(psutil.Process(child.pid),signal.SIGTERM)
            while time.monotonic()<deadline-2:
                register()
                if child.poll() is not None and not alive():break
                time.sleep(.05)
            register()
            remaining=alive()
            if remaining:
                forced=True
                for process in reversed(remaining):send(process,signal.SIGKILL)
            child.wait(timeout=max(.01,deadline-time.monotonic()))
    finally:
        if child is not None:
            register()
            for process in alive():forced=True;send(process,signal.SIGKILL)
            if child.poll() is None:
                try:child.wait(timeout=max(.01,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:events.append({'error':'owned root did not reap by deadline'})
        for number,handler in handlers.items():signal.signal(number,handler)
    register();residual=[process.pid for process in alive()]
    result={'returncode':child.returncode,'elapsed_s':time.monotonic()-started,'watchdog_s':args.watchdog_s,
            'term_sent':terminated,'forced_kill':forced,'stop_signals':stops,'owned_processes':list(registered.values()),
            'signal_events':events,'residual_owned_pids':residual,'child_exited':child.poll() is not None,'cleanup_status':'passed' if not residual and child.poll() is not None else 'failed'}
    persist(args.output/'ownership.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','manifest','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--segments',type=int,choices=(16,128),default=16)
    parser.add_argument('--watchdog-s',type=float,default=420);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);args=parser.parse_args(argv)
    if not math.isfinite(args.watchdog_s) or not 5<=args.watchdog_s<=420:parser.error('total watchdog must be5..420 seconds')
    expected=plan(args)
    if not args.worker:
        args.output.mkdir(parents=True,exist_ok=False);persist(args.output/'summary.json',expected)
    if not args.execute:return 0
    if args.worker:
        report=json.loads((args.output/'summary.json').read_text())
        for key in ('source_sha256','config_sha256','manifest_sha256','selected_ids'):
            if report[key]!=expected[key]:raise ValueError('frozen experiment identity changed before worker: '+key)
        async def work():
            loop=asyncio.get_running_loop();owner=asyncio.current_task()
            for number in (signal.SIGINT,signal.SIGTERM):loop.add_signal_handler(number,owner.cancel)
            try:await execute(args,report)
            finally:
                for number in (signal.SIGINT,signal.SIGTERM):loop.remove_signal_handler(number)
        try:asyncio.run(work());return int(report.get('status')!='completed' or report.get('cleanup',{}).get('status')!='passed')
        except BaseException as exc:
            report.update(status='failed',error=f'{type(exc).__name__}: {exc}');persist(args.output/'summary.json',report);return 1
    command=[sys.executable,str(Path(__file__).resolve()),'--config',str(args.config.resolve()),'--manifest',str(args.manifest.resolve()),
             '--output',str(args.output.resolve()),'--segments',str(args.segments),'--watchdog-s',str(args.watchdog_s),'--execute','--worker']
    supervisor=supervise(command,args);report=json.loads((args.output/'summary.json').read_text());report['supervisor']=supervisor
    if supervisor['returncode'] or supervisor['term_sent'] or supervisor['forced_kill'] or supervisor['cleanup_status']!='passed':
        report['status']='failed'
    persist(args.output/'summary.json',report)
    print(json.dumps({'output':str(args.output),'status':report['status'],'supervisor_cleanup':supervisor['cleanup_status']}))
    return int(report['status']!='completed' or report.get('cleanup',{}).get('status')!='passed')


if __name__=='__main__':raise SystemExit(main())
