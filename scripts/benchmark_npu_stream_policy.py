"""独立NPU:7八池流策略候选；Qwen用default，CAM/emo配对共享池流。"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, deque
from contextlib import contextmanager, nullcontext
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
MODES=("default_global","pooled_global","pooled_current_stream_sync")


def stream_identity(stream):
    # 当前版本Stream池同时复用id和handle；物理身份以设备+ACL handle确认。
    return (int(stream.device_index),int(stream.npu_stream))


def persist(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+".writing")
    temporary.write_text(json.dumps(value,ensure_ascii=True,indent=2)+"\n");temporary.replace(path)


def pcm_identity(value,sample_rate=16000):
    """只查看已在CPU的输入；绝不为观测额外移动NPU tensor。"""
    import numpy as np
    if hasattr(value,"device"):
        if str(value.device)!="cpu":raise ValueError("PCM identity requires original CPU input")
        if hasattr(value,"detach"):value=value.detach()
        if hasattr(value,"numpy"):value=value.numpy()
    pcm=np.asarray(value,dtype="<f4").reshape(-1)
    return {"pcm_float32_sha256":hashlib.sha256(pcm.tobytes()).hexdigest(),"samples":int(pcm.size),"sample_rate":sample_rate}


def cpu_vector(value):
    if hasattr(value,"device"):
        if str(value.device)!="cpu":raise ValueError("refuse extra NPU result readback")
        value=value.reshape(-1).tolist()
    elif hasattr(value,"reshape"):
        value=value.reshape(-1).tolist()
    def flatten(items):
        for item in items:
            if isinstance(item,(list,tuple)):yield from flatten(item)
            else:yield float(item)
    result=list(flatten(value))
    if not all(math.isfinite(item) for item in result):raise ValueError("non-finite original CPU result")
    return result


class StreamPolicy:
    """仅实验进程monkeypatch；TLS外仍执行原设备同步，退出精确恢复属性。"""
    def __init__(self,mode,runners,*,event_limit=32768,result_limit=8192,guard_limit=65536):
        if mode not in MODES:raise ValueError("unknown stream policy")
        self.mode=mode;self.runners=runners;self.local=threading.local();self.lock=threading.Lock()
        self.events=deque(maxlen=event_limit);self.results=deque(maxlen=result_limit)
        self.guards=deque(maxlen=guard_limit);self.guard_total=0;self.guard_alias_coverage=[]
        self.event_total=self.result_total=0;self.originals=[];self.patched=set();self.streams={};self.phase="preparation"

    def patch(self,obj,name,replacement):
        key=(id(obj),name)
        if key in self.patched:return
        self.originals.append((obj,name,getattr(obj,name),name in getattr(obj,"__dict__",{})))
        self.patched.add(key);setattr(obj,name,replacement)

    def observe_guards(self,factory_module=None,alias_modules=None,*,clock=time.monotonic_ns):
        # 生产调用当前为函数内dynamic import；必须观察中央工厂，不能只新增模块属性。
        if factory_module is None:
            import importlib
            factory_module=importlib.import_module("core.model_loading")
            alias_modules=[importlib.import_module(name) for name in
                           ("core.qwen_asr_runtime","core.local_inference","core.models")]
        original=factory_module.model_loading_guard
        wrappers={}
        def observe(delegate):
            @contextmanager
            def guarded(*args,**kwargs):
                context=getattr(self.local,"context",None)
                if context is None:
                    with delegate(*args,**kwargs) as value:yield value
                    return
                phase=self.phase;tag=context[1];requested=clock()
                entered=left=None;depth=getattr(self.local,"guard_depth",0);body_error=error=None
                frame=sys._getframe(1)
                try:
                    while frame is not None and frame.f_globals.get("__name__") in (__name__,"contextlib"):
                        frame=frame.f_back
                    caller={"module":frame.f_globals.get("__name__"),"function":frame.f_code.co_name} if frame is not None else {}
                finally:del frame
                try:
                    with delegate(*args,**kwargs) as value:
                        entered=clock();self.local.guard_depth=depth+1
                        try:yield value
                        except BaseException as exc:
                            body_error=type(exc).__name__;raise
                        finally:
                            left=clock();self.local.guard_depth=depth
                except BaseException as exc:
                    error=type(exc).__name__;raise
                finally:
                    exited=clock()
                    record={"mode":self.mode,"phase":phase,"model_instance":tag,"thread_id":threading.get_ident(),
                            "callsite":caller,"depth":depth,"requested_ns":requested,"entered_ns":entered,
                            "body_left_ns":left,"exited_ns":exited,"entry_wait_ns":None if entered is None else entered-requested,
                            "body_wall_ns":None if left is None else left-entered,"exit_tail_ns":None if left is None else exited-left,
                            "body_error":body_error,"error":error,"body_exception_suppressed":body_error is not None and error is None}
                    with self.lock:self.guard_total+=1;self.guards.append(record)
            return guarded
        for module in [factory_module]+list(alias_modules or []):
            name=getattr(module,"__name__",type(module).__name__)
            if not hasattr(module,"model_loading_guard"):
                self.guard_alias_coverage.append({"module":name,"binding":"dynamic_import_from_central_factory"});continue
            delegate=module.model_loading_guard
            if id(delegate) not in wrappers:wrappers[id(delegate)]=observe(delegate)
            self.patch(module,"model_loading_guard",wrappers[id(delegate)])
            self.guard_alias_coverage.append({"module":name,"binding":"central_factory" if module is factory_module else "existing_alias",
                                              "same_as_central_original":delegate is original})
        return self

    def guard_snapshot(self):
        # 嵌套body含内层；只对depth0单独累计，禁止把所有hold相加当纯CPU。
        rows=list(self.guards);groups={}
        for row in rows:
            key=(row["phase"],row["model_instance"],row["callsite"].get("module"),row["callsite"].get("function"),row["depth"])
            group=groups.setdefault(key,{"phase":key[0],"model_instance":key[1],"module":key[2],"function":key[3],"depth":key[4],"count":0})
            group["count"]+=1
            for field in ("entry_wait_ns","body_wall_ns","exit_tail_ns"):
                if row[field] is not None:
                    group[field+"_sum"]=group.get(field+"_sum",0)+row[field]
                    group[field+"_max"]=max(group.get(field+"_max",0),row[field])
        return {"alias_coverage":self.guard_alias_coverage,"records":rows,"total":self.guard_total,
                "dropped":max(0,self.guard_total-len(rows)),"groups":list(groups.values()),
                "timing_scope":"entry includes original RLock acquisition and dtype setup; body is protected wall time, not pure CPU/kernel time; nested bodies overlap; exit includes dtype restoration and release; TLS-external calls are unchanged and unrecorded"}

    def stamp(self,api,tag,event,**extra):
        current=api.current_stream("npu:7");default=api.default_stream("npu:7")
        if int(current.device_index)!=7:raise ValueError("stream escaped physical7")
        assigned=self.streams.get(tag)
        expected=assigned if assigned is not None else default
        if stream_identity(current)!=stream_identity(expected):raise ValueError("actual worker stream differs from assigned policy")
        record={"mode":self.mode,"phase":self.phase,"model_instance":tag,"event":event,
                "stream_id":int(current.stream_id),"default_stream_id":int(default.stream_id),
                "npu_stream_handle":int(current.npu_stream),"default_npu_stream_handle":int(default.npu_stream),
                "is_default":stream_identity(current)==stream_identity(default),"assigned_stream_id":int(expected.stream_id),
                "assigned_npu_stream_handle":int(expected.npu_stream),"device_index":int(current.device_index),
                "thread_id":threading.get_ident(),"monotonic_ns":time.monotonic_ns(),**extra}
        with self.lock:self.event_total+=1;self.events.append(record)

    def install(self):
        try:
            if self.mode!="default_global":
                expected={"qwen[0]"}|{f"{name}[{i}]" for name in ("speech_campplus","emotion2vec") for i in range(8)}
                if {tag for tag,_,_ in self.runners}!=expected:raise ValueError("pooled policy requires one Qwen and eight CAM/eight emo")
                api=self.runners[0][1].api
                if any(runner.api is not api for _,runner,_ in self.runners):raise ValueError("pooled policy requires the same device API object")
                api.set_device("npu:7")
                with api.device("npu:7"):
                    default=api.default_stream("npu:7")
                    pooled=[api.Stream(device="npu:7") for _ in range(8)]
                identities={stream_identity(stream) for stream in pooled}
                if len(identities)!=8 or stream_identity(default) in identities:
                    raise ValueError("API did not provide eight distinct nondefault pooled stream handles")
                self.streams["qwen[0]"]=default
                for index,stream in enumerate(pooled):
                    self.streams[f"speech_campplus[{index}]"]=stream
                    self.streams[f"emotion2vec[{index}]"]=stream
            for tag,runner,method in self.runners:
                if runner.device!="npu:7":raise ValueError("only physical7 runners allowed")
                api=runner.api;api.set_device("npu:7")
                self.attach(runner,method,tag,api)
            return self
        except BaseException:
            self.restore();raise

    def attach(self,runner,method,tag,api):
        original=getattr(runner,method)
        def wrapped(*args,**kwargs):
            previous=getattr(self.local,"context",None)
            api.set_device("npu:7")
            context=api.stream(self.streams[tag]) if tag in self.streams else nullcontext()
            try:
                with api.device("npu:7"),context:
                    self.local.context=(api,tag)
                    self.stamp(api,tag,"call_started")
                    try:
                        output=original(*args,**kwargs)
                        self.capture(tag,method,args,kwargs,output)
                        return output
                    finally:self.stamp(api,tag,"call_finished_after_original_sync")
            finally:self.local.context=previous
        self.patch(runner,method,wrapped)
        if (id(api),"synchronize") not in self.patched:
            synchronize=api.synchronize
            def routed(*args,**kwargs):
                context=getattr(self.local,"context",None)
                if context is None or context[0] is not api:return synchronize(*args,**kwargs)
                device=args[0] if args else kwargs.get("device")
                if device is not None and str(device) not in ("7","npu:7"):
                    raise ValueError("refuse routing synchronize outside physical7")
                started=time.monotonic_ns();kind="current_stream" if self.mode=="pooled_current_stream_sync" else "device"
                try:
                    if kind=="current_stream":return api.current_stream("npu:7").synchronize()
                    return synchronize(*args,**kwargs)
                finally:self.stamp(api,context[1],"original_sync_finished",sync_kind=kind,elapsed_ns=time.monotonic_ns()-started)
            self.patch(api,"synchronize",routed)

    def capture(self,tag,method,args,kwargs,output):
        if method=="transcribe_batch":
            from core.qwen_asr_runtime import decode_wav_bytes
            waves=args[0] if args else kwargs["audio_bytes_list"]
            languages=(args[1] if len(args)>1 else kwargs.get("languages")) or [None]*len(waves)
            if len(output)!=len(waves):raise ValueError("Qwen member count mismatch")
            rows=[{"model":"qwen","model_instance":tag,"language_hint":language,"input_wav_sha256":hashlib.sha256(wave).hexdigest(),
                   **pcm_identity(decode_wav_bytes(wave)),"raw":raw if isinstance(raw,str) else None,
                   "error":None if isinstance(raw,str) else str(raw)} for wave,language,raw in zip(waves,languages,output)]
        else:
            payload=output[0];identity=pcm_identity(kwargs["input"],kwargs.get("fs",16000))
            model="speech_campplus" if tag.startswith("speech_campplus") else "emotion2vec"
            vector=cpu_vector(payload["spk_embedding"] if model=="speech_campplus" else payload["scores"])
            if model=="speech_campplus" and len(vector)!=192:raise ValueError("CAM original output is not192")
            if model=="emotion2vec" and (len(vector)!=7 or len(payload.get("labels",[]))!=7):
                raise ValueError("expected locked original emotion7 scores/labels; do not reconstruct9")
            rows=[{"model":model,"model_instance":tag,**identity,"vector":vector,"actual_vector_size":len(vector),
                   "labels":payload.get("labels"),"projection9_available":False if model=="emotion2vec" else None}]
        with self.lock:
            for row in rows:
                self.result_total+=1;self.results.append(dict(row,mode=self.mode,phase=self.phase))

    def phase_results(self,phase):
        with self.lock:return [dict(row) for row in self.results if row["phase"]==phase]

    def snapshot(self):
        affinity=[]
        for tag,runner,_ in self.runners:
            stream=self.streams[tag] if tag in self.streams else runner.api.default_stream("npu:7")
            group="default" if tag.startswith("qwen[") or self.mode=="default_global" else "pool_"+tag.split("[")[1].split("]")[0]
            affinity.append({"model_instance":tag,"share_group":group,"device_index":int(stream.device_index),
                             "stream_id":int(stream.stream_id),"npu_stream_handle":int(stream.npu_stream)})
        groups={}
        for row in affinity:
            groups.setdefault(row["share_group"],[]).append(row["model_instance"])
        with self.lock:return {"mode":self.mode,"instance_affinity":affinity,"share_groups":groups,
            "unique_stream_identity_count":len({(row["device_index"],row["npu_stream_handle"]) for row in affinity}),
            "policy_scope":"all default" if self.mode=="default_global" else "one Qwen default plus eight pooled streams; each CAM[i]/emo[i] pair explicitly shares a stream; not17 exclusive streams",
            "events":list(self.events),"event_total":self.event_total,
            "dropped_events":max(0,self.event_total-len(self.events)),"results":list(self.results),
            "result_total":self.result_total,"dropped_results":max(0,self.result_total-len(self.results)),
            "cpu_guard_observation":self.guard_snapshot()}

    def restore(self):
        for obj,name,original,owned in reversed(self.originals):
            if owned:setattr(obj,name,original)
            else:delattr(obj,name)
        self.originals.clear();self.patched.clear()


def compare_results(reference,candidate,*,atol=0,rtol=0):
    def index(rows):
        result={}
        for row in rows:
            key=(row["model"],row["pcm_float32_sha256"],json.dumps(row.get("language_hint"),sort_keys=True))
            result.setdefault(key,[]).append(row)
        return result
    before=index(reference);after=index(candidate);differences=[]
    for key in sorted(set(before)|set(after)):
        left=before.get(key,[]);right=after.get(key,[])
        if len(left)!=len(right):
            differences.append({"key":key,"kind":"count","before":len(left),"after":len(right)});continue
        for a,b in zip(left,right):
            if key[0]=="qwen":
                if a.get("error") or b.get("error") or a.get("raw")!=b.get("raw"):
                    differences.append({"key":key,"kind":"qwen_raw","before":a.get("raw"),"after":b.get("raw")})
            elif a.get("labels")!=b.get("labels") or len(a["vector"])!=len(b["vector"]):
                differences.append({"key":key,"kind":"vector_shape_or_labels"})
            else:
                delta=max((abs(x-y) for x,y in zip(a["vector"],b["vector"])),default=0)
                equal=all(math.isclose(x,y,abs_tol=atol,rel_tol=rtol) for x,y in zip(a["vector"],b["vector"]))
                if delta or not equal:differences.append({"key":key,"kind":"numeric","max_abs":delta,"within_tolerance":equal})
    failures=[row for row in differences if row.get("within_tolerance") is not True]
    return {"passed":not failures,"atol":atol,"rtol":rtol,"differences":differences,"failed_differences":len(failures),
            "reference_result_count":len(reference),"candidate_result_count":len(candidate)}


def plan(args):
    try:import tomllib
    except ImportError:import tomli as tomllib
    config=tomllib.loads(args.config.read_text());q=config.get("qwen_asr",{});p=config.get("model_device_pool",{})
    if q.get("local_device")!="npu:7" or any(p.get(key)!=["npu:7"]*8 for key in ("speech_campplus_devices","emotion2vec_devices")):
        raise ValueError("requires one physical7 Qwen and eight CAM/eight emo on7")
    body=json.loads(args.manifest.read_text());rows=body.get("segments",[]) if isinstance(body,dict) else body
    if args.segments==16:
        from scripts.profile_npu_pipeline_streams import select_segments
        rows=select_segments([dict(row,duration_s=float(row["end_s"])-float(row["start_s"])) for row in rows])
    elif len(rows)!=128:raise ValueError("128 experiment needs fixed128 manifest")
    return {"status":"planned","modes":list(MODES),"selected_ids":[str(row["id"]) for row in rows],"segment_count":len(rows),
        "config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),"manifest_sha256":hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "experiment_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"python_executable":sys.executable,
        "warmup_rounds_per_mode":1,"steady_rounds_per_mode":3,"physical_device":"npu:7","ports_started":False,
        "stream_policy":{"pooled_nondefault_count":8,"qwen_affinity":"default","cam_emo_affinity":"CAM[i] and emo[i] share pooled[i]",
                         "identity_fields":["device_index","npu_stream"],"not_seventeen_exclusive_streams":True,"priority_not_used_to_expand_pool":True},
        "numeric_scope":"original returned CPU CAM192 and actual emotion7+labels; projection9 is not exposed and not reconstructed",
        "candidate_only":True,"production_optimization_accepted":False}


async def execute(args,report):
    os.environ["CONFIG_PATH"]=str(args.config.resolve())
    from core.qwen_asr_client import QwenAsrClient
    from core.qwen_asr_queue import RequestConcurrency
    from core.models import load_v3_models_if_needed,get_model_pool,close_v3_model_pools,local_model_status
    from core.async_utils import await_completion
    from utils.qwen_response import parse_qwen_text
    from scripts.benchmark_qwen_batch_curve import read_manifest,run_round,enhanced_workload
    rows=read_manifest(args.manifest);selected=set(report["selected_ids"]);rows=[row for row in rows if str(row["id"]) in selected]
    client=QwenAsrClient(local_instance_count=1,retry_count=0,max_inference_batch_size=args.segments)
    policy=None;baseline=None;pools={};report.update(status="running",rounds=[],cleanup={"status":"pending"})
    try:
        await client.discover_model();await load_v3_models_if_needed(True,True)
        pools={name:get_model_pool(name) for name in ("speech_campplus","emotion2vec")}
        if any(len(pool.models)!=8 or any(model is None for model in pool.models) for pool in pools.values()):
            raise RuntimeError("all16 enhanced instances must load successfully")
        runners=[(f"qwen[{i}]",runtime,"transcribe_batch") for i,runtime in enumerate(client._runtimes)]
        runners += [(f"{name}[{i}]",runner,"generate") for name,pool in pools.items() for i,runner in enumerate(pool.models)]
        apis={id(runner.api):runner.api for _,runner,_ in runners}
        report["loaded_model_count"]=len(runners)
        from copy import deepcopy
        report["loaded_state"]={"qwen":client.snapshot(),"enhanced":deepcopy(local_model_status)}
        for mode in MODES:
            # TLS外原设备fence隔离各mode，加载/close始终保留原设备同步。
            for api in apis.values():api.set_device("npu:7");api.synchronize("npu:7")
            policy=StreamPolicy(mode,runners).install()
            try:
                policy.observe_guards()
                for index in range(4):
                    phase=f"round-{index}";policy.phase=phase
                    result=await run_round(client,rows,RequestConcurrency(16,128),args.output/f"{mode}-{phase}.json",
                        warmup=index==0,watchdog_s=args.watchdog_s,no_progress_s=120,sample_interval_s=.1,
                        parser=parse_qwen_text,enhanced=enhanced_workload,cleanup_timeout_s=30)
                    values=policy.phase_results(phase)
                    counts=Counter(row["model"] for row in values)
                    expected=Counter(qwen=len(rows),speech_campplus=len(rows),emotion2vec=sum(1<=row["duration_s"]<=10 for row in rows))
                    if counts!=expected:raise RuntimeError("original CPU result evidence missing: "+str(dict(counts)))
                    if index==1 and baseline is None:baseline=values
                    comparison=compare_results(baseline,values,atol=args.atol,rtol=args.rtol) if baseline is not None and index else None
                    report["rounds"].append({"mode":mode,"round":index,"warmup":index==0,"status":result["status"],
                        "elapsed_s":result["elapsed_s"],"quality_gate":comparison,"result_counts":dict(counts),"original_cpu_results":values})
                    persist(args.output/"summary.json",report)
                    if result["status"]!="completed":raise RuntimeError("round failed: "+str(result["status"]))
                evidence=policy.snapshot();persist(args.output/f"{mode}-stream-evidence.json",evidence)
                if evidence["dropped_events"] or evidence["dropped_results"] or evidence["cpu_guard_observation"]["dropped"]:raise RuntimeError("bounded evidence overflow")
                warmed={row["model_instance"] for row in evidence["events"] if row["phase"]=="round-0" and row["event"]=="call_started"}
                if warmed!={tag for tag,_,_ in runners}:raise RuntimeError("not all17 instances actually warmed")
            finally:
                try:persist(args.output/f"{mode}-stream-evidence-final.json",policy.snapshot())
                finally:policy.restore();policy=None
        report["quality_passed"]=all(row["quality_gate"]["passed"] for row in report["rounds"] if row["quality_gate"] is not None)
        report["status"]="completed" if report["quality_passed"] else "quality_failed"
    except BaseException as exc:
        report.update(status="failed",error=f"{type(exc).__name__}: {exc}");raise
    finally:
        # run_round已排空其owned任务；关闭使用恢复后的原设备级sync。
        if policy is not None:
            persist(args.output/f"{policy.mode}-stream-evidence-failed.json",policy.snapshot());policy.restore()
        async def cleanup():
            try:await client.aclose()
            finally:await close_v3_model_pools()
        try:
            await await_completion(cleanup())
            state=client.snapshot()
            if state.get("busy") or state.get("waiting") or state.get("loaded"):raise RuntimeError("client not drained")
            if any(any(model is not None for model in pool.models) for pool in pools.values()):raise RuntimeError("enhanced instances not closed")
            report["cleanup"]={"status":"passed","wrappers_restored":True,"ports_started":False,"client":state}
        except BaseException as exc:
            report.update(status="failed",cleanup={"status":"failed","error":str(exc),"wrappers_restored":True});raise
        finally:persist(args.output/"summary.json",report)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True);parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--execute",action="store_true")
    parser.add_argument("--segments",type=int,choices=(16,128),default=16)
    parser.add_argument("--watchdog-s",type=float,default=420);parser.add_argument("--atol",type=float,default=0);parser.add_argument("--rtol",type=float,default=0)
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if not math.isfinite(args.watchdog_s) or not 5<=args.watchdog_s<=420 or any(not math.isfinite(x) or x<0 for x in (args.atol,args.rtol)):
        parser.error("finite watchdog5..420 and nonnegative tolerances required")
    report=plan(args)
    if not args.worker:
        args.output.mkdir(parents=True,exist_ok=False);persist(args.output/"summary.json",report)
    if not args.execute:return 0
    if args.worker:
        async def work():
            task=asyncio.current_task();loop=asyncio.get_running_loop()
            for number in (signal.SIGINT,signal.SIGTERM):loop.add_signal_handler(number,task.cancel)
            await execute(args,report)
        try:asyncio.run(work());return 0 if report["status"]=="completed" else 2
        except BaseException as exc:
            report.update(status="failed",error=f"{type(exc).__name__}: {exc}");persist(args.output/"summary.json",report);return 2
    command=[sys.executable,str(Path(__file__).resolve()),"--manifest",str(args.manifest.resolve()),"--config",str(args.config.resolve()),
             "--output",str(args.output.resolve()),"--segments",str(args.segments),"--watchdog-s",str(args.watchdog_s),
             "--atol",str(args.atol),"--rtol",str(args.rtol),"--execute","--worker"]
    import psutil
    child=None;forced=False;terminated=False;stops=[];handlers={};started=time.monotonic()
    for number in (signal.SIGINT,signal.SIGTERM):handlers[number]=signal.signal(number,lambda n,f:stops.append(n))
    try:
        with (args.output/"worker.log").open("w") as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            owned=psutil.Process(child.pid);created=owned.create_time()
            persist(args.output/"ownership.json",{"pid":child.pid,"create_time":created,"parent_pid":os.getpid(),"command":command})
            reserve=min(30,args.watchdog_s/2)
            while child.poll() is None and not stops and time.monotonic()-started<args.watchdog_s-reserve:time.sleep(.05)
            if child.poll() is None:
                if owned.create_time()!=created:raise RuntimeError("owned PID identity changed")
                terminated=True;owned.terminate()
                try:child.wait(timeout=max(.01,args.watchdog_s-(time.monotonic()-started)-5))
                except subprocess.TimeoutExpired:forced=True;owned.kill();child.wait(timeout=5)
    finally:
        if child is not None and child.poll() is None:
            forced=True;child.kill();child.wait(timeout=5)
        for number,handler in handlers.items():signal.signal(number,handler)
    report=json.loads((args.output/"summary.json").read_text())
    report["supervisor"]={"owned_pid":child.pid,"returncode":child.returncode,"term_sent":terminated,"forced_kill":forced,
                          "elapsed_s":time.monotonic()-started,"stop_signals":stops,"child_exited":child.poll() is not None}
    if terminated or (child.returncode and report.get("status")!="quality_failed"):report["status"]="failed"
    persist(args.output/"summary.json",report)
    print(json.dumps({"output":str(args.output),"status":report["status"],"supervisor":report["supervisor"]}))
    return 0 if report["status"]=="completed" else 2


if __name__=="__main__":raise SystemExit(main())
