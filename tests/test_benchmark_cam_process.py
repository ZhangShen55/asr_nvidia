"""CPU-only formal component CLI plan, capture, pairing and cleanup contracts."""
import asyncio
from contextlib import redirect_stdout
import copy
from collections import Counter
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from scripts import benchmark_cam_process as tool
from tests.test_qwen_process_benchmark import rows, tied_selection_rows


def config_text(mode):
    return ('[qwen_asr]\nexecution_mode="process"\nlocal_device="npu:7"\nlocal_instance_count=1\nlocal_dtype="bfloat16"\nmax_new_tokens=256\n'
            '[model_device_pool]\nspeech_campplus_execution_mode="'+mode+'"\n'
            'speech_campplus_devices=['+','.join(['"npu:7"']*8)+']\n'
            'emotion2vec_devices=['+','.join(['"npu:7"']*8)+']\n')


def fake_root(tmp_path, *, protected=False):
    root=tmp_path/'fake-root';(root/'core').mkdir(parents=True);(root/'scripts').mkdir()
    for name in tool.HELPERS:(root/'scripts'/name).write_text('# CPU metadata only, never imported')
    for name in ('cam_process','config','models','local_inference','qwen_asr_process'):
        path=root/'core'/(name+('.cpu.so'if protected else'.py'))
        path.write_bytes(b'CPU invalid artifact: planning must never import this')
    return root


def args_for(tmp_path,mode='thread',segments=128):
    config=tmp_path/(mode+'.toml');config.write_text(config_text(mode))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'scope':{'source':'CPU metadata'},'segments':rows()}))
    return SimpleNamespace(config=config,manifest=manifest,output=tmp_path/'never-created',reference=None,segments=segments,watchdog_s=420)


def members(selected):
    return [{'id':row['id'],'status':'completed','raw':'language Chinese<asr_text>'+row['id'],
             'parsed':{'text':row['id'],'language':'Chinese'}}for row in selected]


def nn_values(selected):
    return [{'model':name,'model_instance':f'{name}[{i%8}]','actual_vector_size':size,'vector':[.1]*size,
             'pcm_float32_sha256':str(row['id']),'labels':['L'+str(j)for j in range(7)]if name=='emotion2vec'else None}
            for i,row in enumerate(selected)for name,size in [('speech_campplus',192),('emotion2vec',7)]]


def complete_reference(report,selected):
    result=copy.deepcopy(report)
    result.update(status='completed',source_changed_during_job=False,cleanup={'status':'passed'},
                  supervisor={'returncode':0,'child_exited':True,'cleanup_status':'passed','term_sent':False,'forced_kill':False,'residual_owned_pids':[]},
                  input_identity=[{'id':str(row['id']),'pcm_float32_sha256':str(row['id'])}for row in selected],
                  quality_reference={'qwen':members(selected),'nn':nn_values(selected)},
                  cold_vs_steady={'qwen':{'passed':True},'nn':{'passed':True}},
                  rounds=[{'round':i,'warmup':i==0,'status':'completed','quality':{'qwen':{'passed':True},'nn':{'passed':True}}}for i in range(4)])
    return result


@pytest.mark.parametrize('count',[16,128])
def test_plan_thread_then_process_only_mode_config_diff_same_root_and_sources(tmp_path,count):
    root=fake_root(tmp_path);before=args_for(tmp_path,'thread',count)
    with patch.object(tool,'ROOT',root):baseline=tool.plan(before)
    selected=tool.selected_rows(rows(),count);reference=complete_reference(baseline,selected)
    ref=tmp_path/'thread-summary.json';ref.write_text(json.dumps(reference));after=args_for(tmp_path,'process',count);after.reference=ref
    with patch.object(tool,'ROOT',root):candidate=tool.plan(after)
    assert candidate['execution_mode']=='process'and candidate['allowed_config_difference']=='.'.join(tool.MODE_PATH)
    assert candidate['source_sha256']==baseline['source_sha256']and candidate['selected_ids']==baseline['selected_ids']
    assert not before.output.exists()and 'not complete-audio'in candidate['scope']


@pytest.mark.parametrize('change',['config','root','source','manifest','members','normal','rounds','quality','actual_pcm'])
def test_pairing_rejects_unrelated_or_incomplete_reference_before_models(tmp_path,change):
    root=fake_root(tmp_path);a=args_for(tmp_path)
    with patch.object(tool,'ROOT',root):before=tool.plan(a)
    selected=tool.selected_rows(rows(),128);baseline=complete_reference(before,selected)
    b=args_for(tmp_path,'process');config=tool.read_config(b.config)
    with patch.object(tool,'ROOT',root):current=dict(before,execution_mode='process',config_path=str(b.config),config_sha256=tool.digest(b.config))
    identity=copy.deepcopy(baseline['input_identity'])
    if change=='config':config['qwen_asr']['max_new_tokens']=257
    elif change=='root':baseline['benchmark_root']='other-root'
    elif change=='source':baseline['source_sha256']={}
    elif change=='manifest':baseline['manifest_sha256']='other'
    elif change=='members':baseline['input_identity']=baseline['input_identity'][:-1]
    elif change=='normal':baseline['supervisor']['forced_kill']=True
    elif change=='rounds':baseline['rounds']=baseline['rounds'][:-1]
    elif change=='quality':baseline['quality_reference']['nn']=[]
    elif change=='actual_pcm':identity[0]['pcm_float32_sha256']='different'
    with pytest.raises(ValueError):tool.validate_reference(baseline,current,config,identity=identity)


def test_default_plan_no_output_no_runtime_import_and_protected_artifact_hashed(tmp_path):
    root=fake_root(tmp_path,protected=True);args=args_for(tmp_path,'thread',16);buffer=StringIO()
    with patch.object(tool,'ROOT',root),patch.object(tool,'execute',side_effect=AssertionError('plan executed')),redirect_stdout(buffer):
        assert tool.main(['--config',str(args.config),'--manifest',str(args.manifest),'--output',str(args.output),'--segments','16'])==0
    plan=json.loads(buffer.getvalue());assert not args.output.exists()
    origin=root/'core/cam_process.cpu.so';assert plan['source_sha256'][str(origin)]==tool.digest(origin)
    assert all(not path.endswith('.py')for path in plan['source_sha256']if str(root/'core')in path)
    args=args_for(tmp_path,'process')
    with patch.object(tool,'ROOT',root),pytest.raises(ValueError,match='reference'):tool.plan(args)


def test_canonical_selection_actual_plan_guard_and_input_shape_count():
    source=tied_selection_rows();expected=tool.selected_rows(source,16)
    actual=[dict(row,pcm=range(row['sample_count']))for row in source]
    assert [row['id']for row in tool.verified_rows(actual,16,[r['id']for r in expected])]==[r['id']for r in expected]
    with pytest.raises(ValueError,match='before loading'):tool.verified_rows(actual,16,list(reversed([r['id']for r in expected])))
    with pytest.raises(ValueError):tool.verified_rows(actual[:-1],128,[])


def batch_result(selected):
    dispatched=[{'request_id':row['request_id'],'segment_index':row['segment_index']}for row in selected]
    return {'status':'completed','batches':[{'status':'completed','actual_dispatched_size':len(selected),'dispatched_members':dispatched,
                'dispatch_attempts':[{'status':'completed','actual_dispatched_size':len(selected)}]}]}


@pytest.mark.parametrize('count',[16,128])
def test_actual_one_batch_members_and_attempt_count_are_strict(count):
    selected=tool.selected_rows(rows(),count);result=batch_result(selected);tool.validate_batches(result,selected)
    for change in ('count','member','attempt','second_batch'):
        broken=copy.deepcopy(result)
        if change=='count':broken['batches'][0]['actual_dispatched_size']-=1
        elif change=='member':broken['batches'][0]['dispatched_members'][0]['segment_index']=-1
        elif change=='attempt':broken['batches'][0]['dispatch_attempts']*=2
        else:broken['batches']*=2
        with pytest.raises(ValueError):tool.validate_batches(broken,selected)


class CPUModel:
    device='npu:7'
    def __init__(self,name,index,mode):self.name=name;self.index=index;self.execution_mode=mode if name=='speech_campplus'else'thread';self.calls=0;self.state={}
    @property
    def api(self):raise AssertionError('observer queried NPU API')
    def snapshot(self):return copy.deepcopy(self.state)


def capture_fixture(selected,mode='thread',**kwargs):
    models={name:SimpleNamespace(models=[CPUModel(name,i,mode)for i in range(8)])for name in ('speech_campplus','emotion2vec')}
    def generate(model,device,**options):
        import torch
        from scripts.benchmark_npu_stream_policy import pcm_identity
        model.calls+=1
        if model.name=='emotion2vec':return [{'scores':[.1]*7,'labels':['L'+str(i)for i in range(7)]}]
        identity=pcm_identity(options['input']);pid=10000+model.index
        model.state={'worker_identity':{'pid':pid},'last_done':{'ok':True,'execution_drained':True,'pcm_sha256':identity['pcm_float32_sha256'],
                     'samples':identity['samples'],'actor':{'pid':pid,'explicit_CPU_test_double':True}},'allocator':{'scope':'CPU toy unavailable'}}
        return [{'spk_embedding':torch.arange(192,dtype=torch.float32).reshape(1,192)}]
    module=SimpleNamespace(generate_local=generate,__file__='/protected/fake_local_inference.cpu.so')
    return module,models,tool.CPUOutputCapture(module,models,selected,mode,**kwargs)


def call_all(module,pools,selected):
    counts={'speech_campplus':0,'emotion2vec':0}
    for i,row in enumerate(selected):
        pcm=np.full(round(row['duration_s']*16000),i/1000,dtype=np.float32)
        for name in ('speech_campplus','emotion2vec'):
            if name=='emotion2vec'and not 1<=row['duration_s']<=10:continue
            model=pools[name].models[counts[name]%8];counts[name]+=1
            module.generate_local(model,'npu:7',input=pcm,fs=16000)


@pytest.mark.parametrize('mode,count',[('thread',16),('process',128)])
def test_cpu_capture_py_so_export_exact_original_tensor_emo_and_all8_without_npu_hooks(mode,count):
    selected=tool.selected_rows(rows(),count)
    if count==128:
        for row in selected[:20]:row['duration_s']=.5
    module,pools,capture=capture_fixture(selected,mode);original=module.generate_local
    capture.install();capture.phase='round-0'
    try:
        call_all(module,pools,selected);values=capture.phase_results('round-0');snap=capture.snapshot()
        assert Counter(row['model']for row in values)=={'speech_campplus':count,'emotion2vec':count-(20 if count==128 else 0)}
        assert all(row['actual_vector_size']==(192 if row['model']=='speech_campplus'else 7)for row in values)
        assert not any('stream_id'in event for event in snap['events'])
        assert all('input'not in row and 'pcm_float32_bytes'not in row for row in values)
        cam=[row for row in values if row['model']=='speech_campplus']
        assert all(('cam_worker'in row)==(mode=='process')for row in cam)
        if mode=='process':assert all(row['cam_worker']['last_done']['actor']['explicit_CPU_test_double']for row in cam)
    finally:capture.restore()
    assert module.generate_local is original and not capture.originals


def test_capture_failure_and_bounded_evidence_do_not_look_complete_and_restore():
    selected=tool.selected_rows(rows(),16);module,pools,capture=capture_fixture(selected,event_limit=1,result_limit=1)
    original=module.generate_local;capture.install();capture.phase='round-0'
    try:
        call_all(module,pools,selected)
        assert capture.snapshot()['dropped_events']and capture.snapshot()['dropped_results']
        with pytest.raises(ValueError):capture.phase_results('round-0')
    finally:capture.restore()
    assert module.generate_local is original


def test_capture_rejects_numpy_or_wrong_shape_instead_of_implicitly_fixing_original_result():
    selected=tool.selected_rows(rows(),16);module,pools,capture=capture_fixture(selected)
    module.generate_local=lambda *a,**kw:[{'spk_embedding':np.arange(192,dtype=np.float32)}]
    capture.original=module.generate_local;capture.install()
    try:
        with pytest.raises(ValueError,match='Tensor'):module.generate_local(pools['speech_campplus'].models[0],'npu:7',input=np.zeros(4800,dtype=np.float32))
    finally:capture.restore()


def test_owned_client_publishes_nn_cleanup_before_waiting_for_qwen_and_delegates_snapshot():
    async def scenario():
        events=[];release=asyncio.Event()
        class Client:
            def snapshot(self):return {'CPU_fake':True}
            async def aclose(self):events.append('Qwen_close');await release.wait()
        client=tool.OwnedClient(Client(),lambda:events.append('NN_begin_close'))
        task=asyncio.create_task(client.aclose());await asyncio.sleep(0)
        assert events==['NN_begin_close','Qwen_close']and not task.done()and client.snapshot()['CPU_fake']
        release.set();await task
    asyncio.run(scenario())


def test_normal_worker_close_requires_actual_zero_exit_reaped_no_force():
    value={'resource_reaped':True,'normal_exit_ok':True,'forced_cleanup':False,'worker_exitcode':0,'worker_alive':False}
    tool.verify_normal_workers([value]*8,8,'CAM')
    for key,bad in [('resource_reaped',False),('normal_exit_ok',False),('forced_cleanup',True),('worker_exitcode',7),('worker_alive',True)]:
        with pytest.raises(RuntimeError):tool.verify_normal_workers([dict(value,**{key:bad})]*8,8,'CAM')


@pytest.mark.parametrize('fail',[False,True])
def test_observer_runs_four_owned_rounds_with_existing_run_policy_and_restores_on_failure(tmp_path,fail):
    from scripts.benchmark_qwen_process_nn_streams import run_policy
    async def scenario():
        selected=tool.selected_rows(rows(),16);module,pools,capture=capture_fixture(selected);original=module.generate_local;events=[]
        class Client:
            loaded=False
            async def discover_model(self):self.loaded=True
            def snapshot(self):return {'loaded':self.loaded,'busy':0,'waiting':0}
            async def aclose(self):events.append('Qwen_close');self.loaded=False
        client=Client()
        async def prepare():return pools
        async def close():events.append('NN_close')
        async def run_round(*args,**kwargs):
            call_all(module,pools,selected[:-1]if fail else selected)
            result=batch_result(selected);tool.validate_batches(result,selected)
            result.update(elapsed_s=1,members=members(selected),enhanced={'status':'completed'})
            return result
        bindings=SimpleNamespace(prepare_pools=prepare,make_policy=lambda pools:capture.install(),make_client=lambda:client,
             fence=lambda pools:None,cache_shapes=lambda pools:{},close_pools=close,pool_status=lambda:{'CPU_fake':True},
             round_fn=run_round,gate_factory=object,sample=lambda reset=False:{'CPU_fake':True},parser=None,enhanced=object())
        report={'watchdog_s':420,'selected_ids':[row['id']for row in selected]}
        if fail:
            with pytest.raises(ValueError):await run_policy(selected,tmp_path,report,bindings)
            assert report['status']=='failed'and not report['rounds']
        else:
            await run_policy(selected,tmp_path,report,bindings)
            assert report['status']=='completed'and len(report['rounds'])==4
            assert all(x['quality']['nn']['passed']and x['quality']['qwen']['passed']for x in report['rounds'][1:])
        assert report['cleanup']['status']=='passed'
        assert events==['Qwen_close','NN_close']and module.generate_local is original
    asyncio.run(scenario())


def test_fresh_process_plan_cannot_import_core_or_pollute_settings(tmp_path):
    import os,subprocess,sys
    root=fake_root(tmp_path);args=args_for(tmp_path,'thread',16)
    code="""import importlib.abc,sys
from pathlib import Path
class NoCore(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='core'or fullname.startswith('core.'):raise AssertionError('plan imported runtime '+fullname)
sys.meta_path.insert(0,NoCore())
from scripts import benchmark_cam_process as tool
tool.ROOT=Path(sys.argv[1])
tool.main(['--config',sys.argv[2],'--manifest',sys.argv[3],'--output',sys.argv[4],'--segments','16'])
assert not any(x=='core'or x.startswith('core.')for x in sys.modules)
"""
    result=subprocess.run([sys.executable,'-c',code,str(root),str(args.config),str(args.manifest),str(args.output)],capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert not args.output.exists()and json.loads(result.stdout)['status']=='planned'


def prefetch_args(tmp_path,depth,*,reference=None,count=16):
    args=args_for(tmp_path,'process',count);args.comparison='prefetch';args.reference=reference
    args.config=tmp_path/f'process-depth{depth}.toml'
    args.config.write_text(config_text('process')+'speech_campplus_prefetch_queue_depth='+str(depth)+'\nemotion2vec_prefetch_queue_depth='+str(depth)+'\n'
                           'speech_campplus_batch_size=1\nemotion2vec_batch_size=1\n')
    return args


def process0_reference(root,tmp_path,count=16):
    args=prefetch_args(tmp_path,0,count=count)
    with patch.object(tool,'ROOT',root):before=tool.plan(args)
    selected=tool.selected_rows(rows(),count);ref=complete_reference(before,selected)
    ref['comparison']='cam-mode'  # A same-version process0 may already have a thread comparison.
    ref.pop('cold_vs_steady')  # External cam-mode references verify the warm round directly.
    for r in ref['rounds']:
        r['members']=members(selected);r['original_nn_cpu_results']=nn_values(selected)
        for record in r['members']:record['raw']='actual process0 '+record['id']
    # This intentionally differs: inherited thread baseline must NOT be returned.
    ref['quality_reference']={'qwen':[{'raw':'inherited thread'}],'nn':[{'vector':['wrong baseline']}]}
    path=tmp_path/'process0-summary.json';path.write_text(json.dumps(ref))
    return args,ref,path


@pytest.mark.parametrize('count',[16,128])
@pytest.mark.parametrize('depth',[1,5])
def test_prefetch0_to_paired1_or5_uses_actual_process0_steady_outputs(tmp_path,count,depth):
    root=fake_root(tmp_path);_,baseline,reference=process0_reference(root,tmp_path,count)
    args=prefetch_args(tmp_path,depth,reference=reference,count=count)
    with patch.object(tool,'ROOT',root):candidate=tool.plan(args)
    assert candidate['comparison']=='prefetch'and candidate['execution_mode']=='process'
    assert candidate['allowed_config_difference']==['model_device_pool.emotion2vec_prefetch_queue_depth','model_device_pool.speech_campplus_prefetch_queue_depth']
    result=tool.validate_reference(baseline,candidate,tool.read_config(args.config),identity=baseline['input_identity'])
    assert result=={'qwen':baseline['rounds'][1]['members'],'nn':baseline['rounds'][1]['original_nn_cpu_results']}
    assert result!=baseline['quality_reference']and candidate['reference_actual_output_phase']=='process-depth0/round-1'


def test_prefetch0_can_form_fresh_process_baseline_only_with_explicit_comparison(tmp_path):
    root=fake_root(tmp_path);args=prefetch_args(tmp_path,0)
    with patch.object(tool,'ROOT',root):
        assert tool.plan(args)['reference_path']is None
        args.comparison='cam-mode'
        with pytest.raises(ValueError,match='thread --reference'):tool.plan(args)
    args.comparison='prefetch';capture=StringIO()
    with patch.object(tool,'ROOT',root),patch.object(tool,'execute',side_effect=AssertionError('plan executed')),redirect_stdout(capture):
        assert tool.main(['--config',str(args.config),'--manifest',str(args.manifest),'--output',str(args.output),'--segments','16','--comparison','prefetch'])==0
    assert not args.output.exists()and json.loads(capture.getvalue())['comparison']=='prefetch'


@pytest.mark.parametrize('bad',['mixed','two','float','bool','thread','compute_batch','extra_config','reference_depth1','reference_thread','same0_reference'])
def test_prefetch_rejects_every_change_outside_exact_process0_paired_depth_transition(tmp_path,bad):
    root=fake_root(tmp_path);_,baseline,reference=process0_reference(root,tmp_path)
    args=prefetch_args(tmp_path,5,reference=reference)
    text=args.config.read_text()
    if bad=='mixed':text=text.replace('emotion2vec_prefetch_queue_depth=5','emotion2vec_prefetch_queue_depth=1')
    elif bad=='two':text=text.replace('queue_depth=5','queue_depth=2')
    elif bad=='float':text=text.replace('queue_depth=5','queue_depth=5.0')
    elif bad=='bool':text=text.replace('queue_depth=5','queue_depth=true')
    elif bad=='thread':text=text.replace('speech_campplus_execution_mode="process"','speech_campplus_execution_mode="thread"')
    elif bad=='compute_batch':text=text.replace('speech_campplus_batch_size=1','speech_campplus_batch_size=2')
    elif bad=='extra_config':text=text.replace('max_new_tokens=256','max_new_tokens=257')
    elif bad in('reference_depth1','reference_thread'):
        path=Path(baseline['config_path']);old=path.read_text()
        path.write_text(old.replace('queue_depth=0','queue_depth=1')if bad=='reference_depth1'else old.replace('speech_campplus_execution_mode="process"','speech_campplus_execution_mode="thread"'))
        baseline['config_sha256']=tool.digest(path)
        if bad=='reference_thread':baseline['execution_mode']='thread'
        reference.write_text(json.dumps(baseline))
    else:text=text.replace('queue_depth=5','queue_depth=0')
    args.config.write_text(text)
    with patch.object(tool,'ROOT',root),pytest.raises(ValueError):tool.plan(args)


def test_prefetch_source_and_actual_pcm_identity_guards_are_not_relaxed(tmp_path):
    root=fake_root(tmp_path);_,baseline,reference=process0_reference(root,tmp_path)
    args=prefetch_args(tmp_path,1,reference=reference)
    with patch.object(tool,'ROOT',root):current=tool.plan(args)
    for field,value in [('source_sha256',{}),('benchmark_root','other-root')]:
        broken=copy.deepcopy(baseline);broken[field]=value
        with pytest.raises(ValueError):tool.validate_reference(broken,current,tool.read_config(args.config))
    identity=copy.deepcopy(baseline['input_identity']);identity[0]['pcm_float32_sha256']='other'
    with pytest.raises(ValueError,match='PCM/WAV'):tool.validate_reference(baseline,current,tool.read_config(args.config),identity=identity)
