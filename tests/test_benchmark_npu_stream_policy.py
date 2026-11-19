import builtins
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import threading

import pytest

spec=importlib.util.spec_from_file_location("benchmark_npu_stream_policy",Path(__file__).resolve().parents[1]/"scripts/benchmark_npu_stream_policy.py")
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)


class FakeStream:
    def __init__(self,api,identity,handle=None):
        self.api=api;self.stream_id=identity;self.device_index=7;self.npu_stream=1000+identity if handle is None else handle
    def synchronize(self):
        self.api.calls.append(("stream",self.stream_id,threading.get_ident()))


class FakeAPI:
    def __init__(self):
        self.local=threading.local();self.calls=[];self.next=0;self.default=FakeStream(self,0)
    def set_device(self,device):
        assert device=="npu:7"
    @contextmanager
    def device(self,device):
        assert device=="npu:7";yield
    @contextmanager
    def stream(self,stream):
        previous=getattr(self.local,"stream",self.default)
        self.local.stream=stream
        try:yield
        finally:self.local.stream=previous
    def Stream(self,device,priority=0):
        # 与实机相同：normal/high请求都只循环八个ID和八个raw handles。
        assert device=="npu:7";index=self.next%8;self.next+=1;return FakeStream(self,8+index)
    def current_stream(self,device):
        assert device=="npu:7";return getattr(self.local,"stream",self.default)
    def default_stream(self,device):
        assert device=="npu:7";return self.default
    def synchronize(self,device):
        assert device=="npu:7";self.calls.append(("global",self.current_stream(device).stream_id,threading.get_ident()))


class FakeRunner:
    device="npu:7"
    def __init__(self,api,callback=None):self.api=api;self.callback=callback
    def generate(self,**kwargs):
        if self.callback:self.callback()
        self.api.synchronize("npu:7")
        return [{"spk_embedding":[[float(i) for i in range(192)]]}]


class FakeQwen(FakeRunner):
    def transcribe_batch(self,waves,languages):
        self.api.synchronize("npu:7")
        return ["language Chinese<asr_text>原始输出"]*len(waves)


class FakeEmotion(FakeRunner):
    def generate(self,**kwargs):
        self.api.synchronize("npu:7")
        return [{"labels":[str(i) for i in range(7)],"scores":[i/21 for i in range(7)]}]


def runner_set(api,replacements=None):
    entries=[("qwen[0]",FakeQwen(api),"transcribe_batch")]
    entries += [(f"speech_campplus[{i}]",FakeRunner(api),"generate") for i in range(8)]
    entries += [(f"emotion2vec[{i}]",FakeEmotion(api),"generate") for i in range(8)]
    return [(tag,(replacements or {}).get(tag,runner),method) for tag,runner,method in entries]


@pytest.mark.parametrize("mode,expected_kind,default",[(tool.MODES[0],"global",True),(tool.MODES[1],"global",False),(tool.MODES[2],"stream",False)])
def test_three_modes_route_only_inference_scope_and_restore(mode,expected_kind,default):
    api=FakeAPI();runner=FakeRunner(api)
    original=api.synchronize;generate=runner.generate
    policy=tool.StreamPolicy(mode,runner_set(api,{"speech_campplus[0]":runner})).install()
    try:
        result=runner.generate(input=[0.0]*160,fs=16000)
        assert len(result[0]["spk_embedding"][0])==192
        assert api.calls[-1][0]==expected_kind
        started=next(row for row in policy.events if row["event"]=="call_started")
        assert started["is_default"] is default
        assert started["stream_id"]==started["assigned_stream_id"]
        assert started["npu_stream_handle"]==started["assigned_npu_stream_handle"]
        # 加载、close和主线程的同步不在TLS推理上下文内，仍调用原全设备同步。
        api.synchronize("npu:7")
        assert api.calls[-1][0]=="global"
        assert len(policy.results)==1 and policy.results[0]["actual_vector_size"]==192
    finally:policy.restore()
    assert api.synchronize==original and runner.generate==generate
    assert "synchronize" not in api.__dict__ and "generate" not in runner.__dict__


def test_thread_local_routing_does_not_change_outside_thread():
    started=threading.Event();release=threading.Event();errors=[]
    def callback():started.set();assert release.wait(2)
    api=FakeAPI();runner=FakeRunner(api,callback)
    policy=tool.StreamPolicy(tool.MODES[2],runner_set(api,{"speech_campplus[0]":runner})).install()
    def worker():
        try:runner.generate(input=[0.0]*160)
        except BaseException as exc:errors.append(exc)
    thread=threading.Thread(target=worker);thread.start()
    try:
        assert started.wait(2)
        api.synchronize("npu:7")
        assert api.calls[-1][0]=="global"
    finally:
        release.set();thread.join(2);policy.restore()
    assert not thread.is_alive() and not errors
    assert any(kind=="stream" for kind,_,_ in api.calls)


def test_exception_restores_worker_context_and_patch():
    api=FakeAPI()
    def callback():raise RuntimeError("injected failure")
    runner=FakeRunner(api,callback);original=runner.generate
    policy=tool.StreamPolicy(tool.MODES[2],runner_set(api,{"speech_campplus[0]":runner})).install()
    try:
        with pytest.raises(RuntimeError):runner.generate(input=[0.0])
        assert getattr(policy.local,"context",None) is None
        api.synchronize("npu:7");assert api.calls[-1][0]=="global"
    finally:policy.restore()
    assert runner.generate==original and api.current_stream("npu:7").stream_id==0


def test_original_cpu_only_results_no_extra_npu_read():
    class NpuValue:
        device="npu:7"
        def reshape(self,*args):raise AssertionError("must not read device result")
    with pytest.raises(ValueError):tool.cpu_vector(NpuValue())
    with pytest.raises(ValueError):tool.pcm_identity(NpuValue())
    assert tool.cpu_vector([[1,2],[3,4]])==[1.,2.,3.,4.]


def test_quality_gate_keeps_numeric_and_text_differences():
    base=[{"model":"speech_campplus","pcm_float32_sha256":"a","vector":[1.,2.],"labels":None},
          {"model":"qwen","pcm_float32_sha256":"a","raw":"原始文本","language_hint":None,"error":None}]
    other=[dict(row) for row in base];other[0]["vector"]=[1.,2.000001];other[1]["raw"]="不同文本"
    result=tool.compare_results(base,other)
    assert not result["passed"] and result["failed_differences"]==2
    other[1]["raw"]="原始文本"
    result=tool.compare_results(base,other,atol=.001)
    assert result["passed"] and result["differences"] and result["differences"][0]["within_tolerance"]


def test_qwen_original_raw_is_bound_to_decoded_pcm_and_language():
    import io
    import wave
    buffer=io.BytesIO()
    with wave.open(buffer,"wb") as audio:
        audio.setparams((1,2,16000,0,"NONE",""));audio.writeframes(b"\x00\x00"*4800)
    data=buffer.getvalue();api=FakeAPI()
    class Qwen:
        device="npu:7"
        def __init__(self):self.api=api
        def transcribe_batch(self,waves,languages):
            api.synchronize("npu:7")
            return ["language Chinese<asr_text>原始输出"]*len(waves)
    runner=Qwen();policy=tool.StreamPolicy(tool.MODES[2],runner_set(api,{"qwen[0]":runner})).install()
    try:
        result=runner.transcribe_batch([data],[None])
        captured=policy.results[0]
        assert captured["raw"]==result[0] and captured["samples"]==4800
        assert captured["language_hint"] is None and captured["input_wav_sha256"]
        assert captured["pcm_float32_sha256"]==tool.pcm_identity([0.0]*4800)["pcm_float32_sha256"]
    finally:policy.restore()


def test_emotion_records_original_seven_scores_without_reconstructing_nine():
    api=FakeAPI()
    class Emotion(FakeRunner):
        def generate(self,**kwargs):
            api.synchronize("npu:7")
            return [{"labels":[str(i) for i in range(7)],"scores":[i/21 for i in range(7)]}]
    runner=Emotion(api);policy=tool.StreamPolicy(tool.MODES[2],runner_set(api,{"emotion2vec[0]":runner})).install()
    try:
        result=runner.generate(input=[.1]*160)
        captured=policy.results[0]
        assert captured["vector"]==result[0]["scores"] and captured["actual_vector_size"]==7
        assert captured["labels"]==result[0]["labels"] and not captured["projection9_available"]
    finally:policy.restore()


def test_default_plan_never_imports_torch_or_executes_models(tmp_path,monkeypatch):
    config=tmp_path/"config.toml";manifest=tmp_path/"segments.json";output=tmp_path/"plan"
    config.write_text('[qwen_asr]\nlocal_device="npu:7"\n[model_device_pool]\nspeech_campplus_devices='+json.dumps(["npu:7"]*8)+'\nemotion2vec_devices='+json.dumps(["npu:7"]*8)+'\n')
    rows=[{"id":f"{r}-{i}","request_id":str(r),"segment_index":i,"start_s":0,"end_s":2+i*.1} for r in range(8) for i in range(16)]
    manifest.write_text(json.dumps({"segments":rows}))
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name=="torch" or name.startswith("torch_npu"):raise AssertionError("plan imported accelerator runtime")
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",guarded)
    assert tool.main(["--config",str(config),"--manifest",str(manifest),"--output",str(output)])==0
    report=json.loads((output/"summary.json").read_text())
    assert report["status"]=="planned" and report["segment_count"]==16
    assert not report["production_optimization_accepted"] and report["warmup_rounds_per_mode"]==1


def test_realistic_cyclic_pool_does_not_expand_with_high_priority():
    api=FakeAPI()
    normal=[api.Stream(device="npu:7") for _ in range(40)]
    high=[api.Stream(device="npu:7",priority=-1) for _ in range(8)]
    assert len({tool.stream_identity(stream) for stream in normal})==8
    assert {tool.stream_identity(stream) for stream in normal}=={tool.stream_identity(stream) for stream in high}


def test_fixed_pool_affinity_pairs_cam_emo_and_isolates_qwen():
    api=FakeAPI();entries=runner_set(api)
    policy=tool.StreamPolicy(tool.MODES[2],entries).install()
    try:
        state=policy.snapshot()
        assert state["unique_stream_identity_count"]==9
        assert state["share_groups"]["default"]==["qwen[0]"]
        qwen=policy.streams["qwen[0]"]
        for index in range(8):
            cam=policy.streams[f"speech_campplus[{index}]"];emo=policy.streams[f"emotion2vec[{index}]"]
            assert tool.stream_identity(cam)==tool.stream_identity(emo)
            assert tool.stream_identity(cam)!=tool.stream_identity(qwen)
            assert set(state["share_groups"][f"pool_{index}"])=={f"speech_campplus[{index}]",f"emotion2vec[{index}]"}
        # 即使该worker原先处于池流，Qwen调用也显式切回default，返回再恢复原上下文。
        api.local.stream=policy.streams["speech_campplus[0]"]
        entries[0][1].transcribe_batch([],[])
        observed=[row for row in policy.events if row["model_instance"]=="qwen[0]"]
        assert observed and all(row["is_default"] for row in observed)
    finally:policy.restore()
    assert "synchronize" not in api.__dict__
    assert all(method not in runner.__dict__ for _,runner,method in entries)


def test_stream_identity_uses_handle_even_if_python_ids_repeat():
    api=FakeAPI()
    one=FakeStream(api,8,handle=2000);two=FakeStream(api,8,handle=3000)
    assert one.stream_id==two.stream_id and tool.stream_identity(one)!=tool.stream_identity(two)
    policy=tool.StreamPolicy(tool.MODES[0],[("speech_campplus[0]",FakeRunner(api),"generate")]).install()
    try:
        api.local.stream=FakeStream(api,0,handle=9999)
        with pytest.raises(ValueError,match="actual worker stream"):
            policy.runners[0][1].generate(input=[0.0])
    finally:policy.restore()


def test_guard_observation_delegates_outside_tls_and_restores_aliases():
    from types import SimpleNamespace
    calls=[]
    @contextmanager
    def original(dtype=None):
        calls.append(("enter",dtype))
        try:yield "original-value"
        finally:calls.append(("exit",dtype))
    central=SimpleNamespace(__name__="core.model_loading",model_loading_guard=original)
    cached=SimpleNamespace(__name__="core.local_inference",model_loading_guard=original)
    dynamic=SimpleNamespace(__name__="core.qwen_asr_runtime")
    policy=tool.StreamPolicy(tool.MODES[0],[]).observe_guards(central,[cached,dynamic])
    try:
        assert cached.model_loading_guard is central.model_loading_guard
        with central.model_loading_guard("fp32") as value:assert value=="original-value"
        assert calls==[("enter","fp32"),("exit","fp32")] and not policy.guards
        coverage={row["module"]:row["binding"] for row in policy.guard_alias_coverage}
        assert coverage["core.qwen_asr_runtime"]=="dynamic_import_from_central_factory"
        assert not hasattr(dynamic,"model_loading_guard")
    finally:policy.restore()
    assert central.model_loading_guard is original and cached.model_loading_guard is original
    assert not hasattr(dynamic,"model_loading_guard")


def test_guard_nested_original_lock_dtype_exception_and_exact_timing():
    from types import SimpleNamespace
    lock=threading.RLock();state={"dtype":"initial"};ticks=iter(range(1,50))
    @contextmanager
    def original(dtype=None):
        with lock:
            previous=state["dtype"];state["dtype"]=dtype
            try:yield
            finally:state["dtype"]=previous
    central=SimpleNamespace(__name__="core.model_loading",model_loading_guard=original)
    policy=tool.StreamPolicy(tool.MODES[0],[]).observe_guards(central,[],clock=lambda:next(ticks))
    policy.local.context=(object(),"qwen[0]")
    try:
        with pytest.raises(RuntimeError,match="body failure"):
            with central.model_loading_guard("outer"):
                assert state["dtype"]=="outer"
                with central.model_loading_guard("inner"):
                    assert state["dtype"]=="inner"
                assert state["dtype"]=="outer"
                raise RuntimeError("body failure")
        assert state["dtype"]=="initial" and policy.local.guard_depth==0
        rows=sorted(policy.guards,key=lambda row:row["depth"])
        assert [row["depth"] for row in rows]==[0,1]
        assert [rows[0][name] for name in ("requested_ns","entered_ns","body_left_ns","exited_ns")]==[1,2,7,8]
        assert [rows[1][name] for name in ("requested_ns","entered_ns","body_left_ns","exited_ns")]==[3,4,5,6]
        assert rows[0]["entry_wait_ns"]==1 and rows[0]["body_wall_ns"]==5 and rows[0]["exit_tail_ns"]==1
        assert rows[0]["body_error"]==rows[0]["error"]=="RuntimeError"
        assert not rows[0]["body_exception_suppressed"]
        assert {group["depth"] for group in policy.guard_snapshot()["groups"]}=={0,1}
    finally:policy.restore()
    assert central.model_loading_guard is original
    with original("after"):assert state["dtype"]=="after"
    assert state["dtype"]=="initial"


def test_guard_preserves_original_suppression_and_enter_failure_with_bounded_evidence():
    from types import SimpleNamespace
    @contextmanager
    def original(kind):
        if kind=="enter-error":raise LookupError("original enter failure")
        try:yield
        except ValueError:pass
    central=SimpleNamespace(__name__="core.model_loading",model_loading_guard=original)
    policy=tool.StreamPolicy(tool.MODES[0],[],guard_limit=1).observe_guards(central,[])
    policy.local.context=(object(),"emotion2vec[0]")
    try:
        with central.model_loading_guard("suppress"):raise ValueError("original suppression")
        assert policy.guards[0]["body_exception_suppressed"]
        with pytest.raises(LookupError,match="original enter failure"):
            with central.model_loading_guard("enter-error"):raise AssertionError("not entered")
        assert policy.guards[0]["entered_ns"] is None and policy.guards[0]["error"]=="LookupError"
        assert policy.guard_snapshot()["dropped"]==1 and policy.guard_total==2
    finally:policy.restore()
