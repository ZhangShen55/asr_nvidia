"""配对CLI纯CPU边界替身，不加载模型或NPU。"""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.benchmark_qwen_process_candidate import main,nn_quality,paired,qwen_quality,selected_rows,_owned


def rows():
    return [dict(id=f'{r}-{i}',request_id=str(r),segment_index=i,duration_s=2,
                 start_s=i*2,end_s=i*2+2) for r in range(8) for i in range(16)]


class Policy:
    def __init__(self):self.phase='initial';self.values={};self.restored=False
    def phase_results(self,phase):return self.values.get(phase,[])
    def snapshot(self):return {'dropped_events':0,'dropped_results':0}
    def restore(self):self.restored=True


class Client:
    def __init__(self,name,events,close_failed=False):self.name=name;self.events=events;self.loaded=False;self.closed=0;self.close_failed=close_failed
    async def discover_model(self):self.events.append('load-'+self.name);self.loaded=True
    def snapshot(self):return {'loaded':self.loaded,'busy':0,'waiting':0,'failed_instances':0}
    async def aclose(self):
        self.events.append('close-'+self.name);self.loaded=False;self.closed+=1
        if self.close_failed:raise RuntimeError('controlled close failure')


def binding_fixture(segments,*,missing_nn=False,missing_raw=False,close_failed=False):
    policy=Policy();events=[];clients=[];closed_pools=[];loaded=[False]
    async def prepare():events.append('load-NN16');loaded[0]=True;return {'NN':'fake16'}
    async def close_pools():events.append('close-NN16');closed_pools.append(True);loaded[0]=False
    def make_client(name):
        client=Client(name,events,close_failed=close_failed);clients.append(client);return client
    async def round_fn(client,rows,gate,path,**kwargs):
        events.append(Path(path).stem)
        nn=[]
        if kwargs['enhanced'] is not None:
            for row in rows:
                for model,size in [('speech_campplus',192),('emotion2vec',7)]:
                    nn.append({'model':model,'actual_vector_size':size,'vector':[.1]*size,
                               'pcm_float32_sha256':row['id'],'labels':['L'+str(i) for i in range(7)] if model=='emotion2vec' else None})
            if missing_nn:nn=nn[:-1]
        policy.values[policy.phase]=nn
        members=[{'id':row['id'],'status':'completed','raw':'language Chinese<asr_text>'+row['id'],
                  'parsed':{'text':row['id'],'language':'Chinese'}}for row in rows]
        if missing_raw:members=members[:-1]
        return {'status':'completed','elapsed_s':1,'measurement_cleanup_s':.1,'batches':[{'size':len(rows)}],'members':members}
    bindings=SimpleNamespace(prepare_pools=prepare,close_pools=close_pools,make_client=make_client,
        pool_status=lambda:{'loaded':loaded[0]},make_policy=lambda pools:policy,fence=lambda pools:None,
        round_fn=round_fn,gate_factory=object,enhanced=object(),sample=lambda reset=False:{'parent_only':True},
        parser=None,worker_calls=lambda phase:[])
    return bindings,events,clients,closed_pools,policy


@pytest.mark.parametrize('count',[16,128])
def test_fixed_case_order_selected_segment_scope_and_all_cleanup(tmp_path,count):
    selected=selected_rows(rows(),count);bindings,events,clients,closed_pools,policy=binding_fixture(selected)
    report={};asyncio.run(paired(selected,tmp_path,report,bindings))
    assert report['status']=='completed' and report['cleanup']['status']=='passed'
    expected=[f'{transport}-{kind}-{i}' for transport in ('thread','process') for kind in ('pure','B')for i in range(4)]
    assert [row['path'][:-5] for row in report['rounds']]==expected
    assert all(row['qwen_quality']['expected_members']==count for row in report['rounds'])
    assert events.index('load-NN16')<events.index('load-thread')
    assert events.index('close-thread')<events.index('load-process')<events.index('close-NN16')
    assert len(clients)==2 and all(client.closed==1 for client in clients)
    assert len(closed_pools)==1 and policy.restored
    assert all(row.get('nn_quality',{}).get('passed',True) for row in report['rounds'])


@pytest.mark.parametrize('missing_nn,missing_raw,expected_rounds',[(True,False,5),(False,True,1)])
def test_missing_evidence_fails_closed_and_does_not_start_process_transport(tmp_path,missing_nn,missing_raw,expected_rounds):
    selected=selected_rows(rows(),16)
    bindings,events,clients,closed_pools,policy=binding_fixture(selected,missing_nn=missing_nn,missing_raw=missing_raw)
    report={}
    with pytest.raises(ValueError):asyncio.run(paired(selected,tmp_path,report,bindings))
    assert report['status']=='failed' and report['cleanup']['status']=='passed'
    assert len(report['rounds'])==expected_rounds and len(clients)==1
    assert clients[0].closed==1 and len(closed_pools)==1 and policy.restored
    assert 'load-process' not in events


def test_cleanup_failure_never_passes_or_skips_pool_cleanup(tmp_path):
    selected=selected_rows(rows(),16)
    bindings,events,clients,closed_pools,policy=binding_fixture(selected,close_failed=True)
    report={}
    with pytest.raises(RuntimeError):asyncio.run(paired(selected,tmp_path,report,bindings))
    assert report['status']=='failed' and report['cleanup']['status']=='failed'
    assert len(closed_pools)==1 and policy.restored and 'load-process' not in events


def test_empty_quality_evidence_cannot_pass_or_be_misnamed_complete_audio():
    with pytest.raises(ValueError):qwen_quality(None,[],['expected'])
    with pytest.raises(ValueError):nn_quality(None,[],selected_rows(rows(),16))
    with pytest.raises(ValueError):selected_rows(rows()[:8],16)
    with pytest.raises(ValueError):selected_rows(rows(),8)


def test_default_cli_plans_without_models_and_preserves_segment_scope(tmp_path):
    config=tmp_path/'config.toml';manifest=tmp_path/'manifest.json';output=tmp_path/'plan'
    config.write_text('[qwen_asr]\nlocal_device="npu:7"\nlocal_instance_count=1\n[model_device_pool]\n'
        'speech_campplus_devices=['+','.join(['"npu:7"']*8)+']\n'
        'emotion2vec_devices=['+','.join(['"npu:7"']*8)+']\n')
    manifest.write_text(json.dumps({'segments':rows()}))
    with patch('scripts.benchmark_qwen_process_candidate.execute',side_effect=AssertionError('must not execute')):
        assert main(['--config',str(config),'--manifest',str(manifest),'--output',str(output)])==0
    report=json.loads((output/'summary.json').read_text())
    assert report['status']=='planned' and report['segment_count']==16
    assert 'not N8 complete-audio' in report['scope'] and not report['ports_started']
    with pytest.raises(ValueError):
        config.write_text(config.read_text().replace('local_device="npu:7"','local_device="npu:0"'))
        main(['--config',str(config),'--manifest',str(manifest),'--output',str(tmp_path/'bad')])


def test_owned_signal_predicate_requires_creationtime_and_session():
    process=SimpleNamespace(pid=123,create_time=lambda:10)
    with patch('scripts.benchmark_qwen_process_candidate.os.getsid',return_value=123):
        assert _owned(process,10,123)
        assert not _owned(process,11,123)
        assert not _owned(process,10,999)


def test_cancel_stops_new_rounds_and_repeated_cancel_still_drains_owned_cleanup(tmp_path):
    async def scenario():
        selected=selected_rows(rows(),16)
        bindings,events,clients,closed_pools,policy=binding_fixture(selected)
        started=asyncio.Event()
        async def blocked(*args,**kwargs):
            started.set();await asyncio.Event().wait()
        bindings.round_fn=blocked
        report={};task=asyncio.create_task(paired(selected,tmp_path,report,bindings))
        await started.wait();task.cancel();await asyncio.sleep(0);task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        assert report['status']=='failed' and report['cleanup']['status']=='passed'
        assert len(clients)==1 and clients[0].closed==1 and len(closed_pools)==1 and policy.restored
        assert 'load-process' not in events
    asyncio.run(scenario())


def test_supervisor_real_cpu_owned_child_normal_exit(tmp_path):
    import sys
    from scripts.benchmark_qwen_process_candidate import supervise
    import scripts.benchmark_qwen_process_candidate as module
    release = tmp_path / "cpu-child-release"
    code = ("from pathlib import Path\nimport sys,time\n"
            "while not Path(sys.argv[1]).exists(): time.sleep(.01)\n")
    original_sleep = module.time.sleep
    def release_after_registration(delay):
        # The parent calls sleep after register(); the child cannot disappear
        # before its exact birth identity has been sampled under CPU contention.
        release.touch()
        original_sleep(min(delay, .01))
    with patch.object(module.time, "sleep", release_after_registration):
        report=supervise([sys.executable, '-c', code, str(release)],
                         SimpleNamespace(output=tmp_path,watchdog_s=5))
    assert report['returncode']==0 and report['cleanup_status']=='passed' and report['child_exited']
    assert not report['forced_kill'] and not report['term_sent'] and not report['residual_owned_pids']
    assert report['owned_processes'] and all(row['pid']>0 and row['create_time']>0 for row in report['owned_processes'])


def test_supervisor_real_cpu_reaps_same_session_orphan_and_never_calls_it_passed_experiment(tmp_path):
    import sys
    from scripts.benchmark_qwen_process_candidate import supervise
    code='import subprocess,sys,time; subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]);time.sleep(.2)'
    report=supervise([sys.executable,'-c',code],SimpleNamespace(output=tmp_path,watchdog_s=5))
    assert report['returncode']==0 and report['forced_kill'] and report['child_exited']
    assert not report['residual_owned_pids']
    assert len(report['owned_processes'])>=2
    identities={(row['pid'],row['sid']) for row in report['owned_processes']}
    assert report['signal_events'] and all(any(pid==event['pid'] for pid,sid in identities) for event in report['signal_events'])


def test_plan_prefers_unique_so_without_importing_it_and_rejects_ambiguous_or_missing(tmp_path):
    import hashlib
    from scripts.benchmark_qwen_process_candidate import resolve_core_artifact
    directory=tmp_path/'core';directory.mkdir()
    source=directory/'qwen_asr_process.py';source.write_text('raise AssertionError("must not import source")')
    assert resolve_core_artifact(tmp_path)==source
    extension=directory/'qwen_asr_process.cpython-310-aarch64-linux-gnu.so'
    extension.write_bytes(b'CPU dummy artifact, never import')
    assert resolve_core_artifact(tmp_path)==extension
    import scripts.benchmark_qwen_process_candidate as tool
    script_dir=tmp_path/'scripts';script_dir.mkdir()
    for name in ('benchmark_qwen_batch_curve.py','benchmark_npu_stream_policy.py'):
        (script_dir/name).write_text('# CPU sourcehash fixture only')
    config=tmp_path/'config.toml';manifest=tmp_path/'manifest.json'
    config.write_text('[qwen_asr]\nlocal_device="npu:7"\n[model_device_pool]\n'
        'speech_campplus_devices=['+','.join(['"npu:7"']*8)+']\n'
        'emotion2vec_devices=['+','.join(['"npu:7"']*8)+']\n')
    manifest.write_text(json.dumps({'segments':rows()}))
    with patch.object(tool,'ROOT',tmp_path):
        report=tool.plan(SimpleNamespace(config=config,manifest=manifest,segments=16,watchdog_s=420))
    assert report['source_sha256'][str(extension)]==hashlib.sha256(extension.read_bytes()).hexdigest()
    assert str(source) not in report['source_sha256']
    (directory/'qwen_asr_process.stale.so').write_bytes(b'stale')
    with pytest.raises(ValueError,match='ambiguous'):resolve_core_artifact(tmp_path)
    for path in directory.iterdir():path.unlink()
    with pytest.raises(ValueError,match='missing'):resolve_core_artifact(tmp_path)


def test_protected_stage_plan_resolves_actual_core_artifact_without_import(tmp_path):
    import os
    import scripts.benchmark_qwen_process_candidate as tool
    stage=os.environ.get('ASR_EXPECT_PROCESS_STAGE_ROOT')
    if not stage:pytest.skip('requires actual protected stage')
    stage=Path(stage).resolve()
    config=tmp_path/'config.toml';manifest=tmp_path/'manifest.json';output=tmp_path/'plan'
    config.write_text('[qwen_asr]\nlocal_device="npu:7"\n[model_device_pool]\n'
        'speech_campplus_devices=['+','.join(['"npu:7"']*8)+']\n'
        'emotion2vec_devices=['+','.join(['"npu:7"']*8)+']\n')
    manifest.write_text(json.dumps({'segments':rows()}))
    with patch.object(tool,'ROOT',stage), \
         patch.object(tool,'execute',side_effect=AssertionError('plan must not execute model')):
        assert tool.main(['--config',str(config),'--manifest',str(manifest),'--output',str(output)])==0
    report=json.loads((output/'summary.json').read_text())
    core_paths=[Path(path) for path in report['source_sha256'] if str(stage/'core/qwen_asr_process') in path]
    assert len(core_paths)==1 and str(core_paths[0]).endswith('.so')
    assert core_paths[0].is_relative_to(stage)


def tied_selection_rows():
    """三个真实同长8.01秒段在不同时间轴上，float差曾把ID0挤出前二。"""
    result = rows()
    for row in result:
        index = row['segment_index']
        start = (0.0, 80.0, 100.0)[index] if index < 3 else index*2.0
        end = start + (8.01 if index < 3 else 2.0)
        left, right = round(start*16000), round(end*16000)
        row.update(start_s=start, end_s=end, duration_s=end-start,
                   start_sample=left, end_sample=right, sample_count=right-left)
    return result


@pytest.mark.parametrize('metadata', [True, False])
def test_same_actual_length_ties_use_canonical_samples_not_float_subtraction(metadata):
    from scripts.profile_npu_pipeline_streams import select_segments
    records = tied_selection_rows()
    expected = [f'{owner}-{index}' for owner in range(8) for index in (0,1)]
    assert [row['id'] for row in select_segments(records)] != expected
    if not metadata:
        records = [{key:value for key,value in row.items()
                    if key not in ('sample_count','start_sample','end_sample')} for row in records]
    original = [dict(row) for row in records]
    selected = selected_rows(records,16)
    assert [row['id'] for row in selected] == expected
    assert all(row['duration_s']==128160/16000 for row in selected)
    assert records==original  # 不在调用方清单上原地改写duration或PCM。
    prepared = [dict(row,pcm=range(round(row['end_s']*16000)-round(row['start_s']*16000))) for row in records]
    assert [row['id'] for row in selected_rows(prepared,16)] == expected
    assert [row['id'] for row in selected_rows(records,128)] == [row['id'] for row in records]


@pytest.mark.parametrize('field,value', [('sample_count',128159),('start_sample',1),('end_sample',128159)])
def test_selection_rejects_metadata_that_disagrees_with_actual_rounded_bounds(field,value):
    records=tied_selection_rows();records[0][field]=value
    with pytest.raises(ValueError,match='sample|bounds'):selected_rows(records,16)


def test_selection_rejects_actual_pcm_length_mismatch_and_supports_duration_only_cpu_toy():
    records=tied_selection_rows();records[0]['pcm']=range(128159)
    with pytest.raises(ValueError,match='actual PCM'):selected_rows(records,16)
    toys=[{key:value for key,value in row.items() if key not in ('start_s','end_s')} for row in rows()]
    assert len(selected_rows(toys,16))==16


def test_real_wav_reader_and_metadata_plan_select_same_tied_ids(tmp_path):
    import base64,hashlib
    import numpy as np
    import soundfile as sf
    from scripts.benchmark_qwen_batch_curve import read_manifest
    records=tied_selection_rows();source=tmp_path/'shared.wav'
    sf.write(source,np.zeros(110*16000,dtype='float32'),16000,subtype='PCM_16')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    for row in records:
        row.update(path_bytes_b64=base64.b64encode(bytes(source)).decode(),source_sha256=digest)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'segments':records}))
    actual=read_manifest(manifest)
    planned=selected_rows(records,16);executed=selected_rows(actual,16)
    assert [row['id'] for row in planned]==[row['id'] for row in executed]
    assert {row['id'] for row in executed}=={f'{owner}-{index}'for owner in range(8)for index in (0,1)}
    assert all(len(row['pcm'])==128160 and row['duration_s']==8.01 for row in executed)


def test_paired_actual_selection_mismatch_fails_before_loading_models(tmp_path):
    selected=selected_rows(rows(),16)
    bindings,events,clients,closed,policy=binding_fixture(selected)
    report={'selected_ids':['stale-plan-id']*16}
    with pytest.raises(ValueError,match='actual selected segment IDs'):
        asyncio.run(paired(selected,tmp_path,report,bindings))
    assert not events and not clients and not closed
