"""CPU fakes only; independent policy install/quality/lifecycle and plan contracts."""
import asyncio
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import benchmark_qwen_process_nn_streams as tool
from tests.test_benchmark_npu_stream_policy import FakeAPI, FakeRunner, FakeEmotion
from tests.test_qwen_process_benchmark import rows, binding_fixture


def runners(api):
    return [(f'speech_campplus[{i}]', FakeRunner(api), 'generate') for i in range(8)] + [
            (f'emotion2vec[{i}]', FakeEmotion(api), 'generate') for i in range(8)]


@pytest.mark.parametrize('mode,kind,count', [('default_global','global',1), ('pooled_global','global',8), ('pooled_current_stream_sync','stream',8)])
def test_nn_only_policy_routes_exactly_sixteen_and_restores_original_global(mode,kind,count):
    api = FakeAPI(); entries = runners(api); original = api.synchronize
    policy = tool.ProcessNNPolicy(mode, entries).install()
    try:
        for tag, runner, _ in entries: runner.generate(input=[0.0]*4800, fs=16000)
        assert len(policy.results) == 16 and all(row['actual_vector_size'] in (192,7) for row in policy.results)
        assert all(row[0] == kind for row in api.calls)
        snapshot = policy.snapshot()
        assert not snapshot['qwen_in_parent'] and snapshot['unique_stream_identity_count'] == count
        assert len(snapshot['instance_affinity']) == 16
        assert all('native_thread_id' in e and 'thread_cpu_ns' in e for e in snapshot['events'])
        if mode != 'default_global':
            affinity = {x['model_instance']:x['npu_stream_handle'] for x in snapshot['instance_affinity']}
            assert all(affinity[f'speech_campplus[{i}]'] == affinity[f'emotion2vec[{i}]'] for i in range(8))
        api.synchronize('npu:7'); assert api.calls[-1][0] == 'global'
    finally: policy.restore()
    assert api.synchronize == original and 'synchronize' not in api.__dict__
    with pytest.raises(ValueError): tool.ProcessNNPolicy(mode, entries[:-1]).install()


def bindings_for(segments, **kwargs):
    bindings, events, clients, closed, policy = binding_fixture(segments, **kwargs)
    make = bindings.make_client
    bindings.make_client = lambda: make('process')
    bindings.cache_shapes = lambda pools: {'cpu_fake': {}}
    policy.observe_guards = lambda: policy
    policy.originals = []
    snapshot = policy.snapshot
    def observed():
        result = snapshot()
        result['events'] = [{'phase':policy.phase, 'event':'call_started', 'model_instance':f'{name}[{i}]'}
                            for name in ('speech_campplus','emotion2vec')for i in range(8)]
        return result
    policy.snapshot = observed
    return bindings, events, clients, closed, policy


def test_one_process_four_b_rounds_quality_and_normal_cleanup(tmp_path):
    selected = rows()[:16]; bindings, events, clients, closed, policy = bindings_for(selected)
    report = {'watchdog_s':420, 'policy':'default_global'}
    asyncio.run(tool.run_policy(selected,tmp_path,report,bindings))
    assert report['status'] == 'completed' and report['cleanup']['status'] == 'passed'
    assert len(report['rounds']) == 4 and [r['warmup']for r in report['rounds']] == [True,False,False,False]
    assert len(clients) == 1 and clients[0].name == 'process' and clients[0].closed == 1
    assert len(closed) == 1 and policy.restored and all(q['passed']for q in report['cold_vs_steady'].values())
    assert all(q['passed']for row in report['rounds'][1:]for q in row['quality'].values())


def test_missing_nn_evidence_stops_at_warm_and_closes_everything(tmp_path):
    selected = rows()[:16]; bindings, events, clients, closed, policy = bindings_for(selected,missing_nn=True)
    report = {'watchdog_s':420}
    with pytest.raises(ValueError): asyncio.run(tool.run_policy(selected,tmp_path,report,bindings))
    assert len(report['rounds']) == 1 and report['status'] == 'failed'
    assert clients[0].closed == 1 and len(closed) == 1 and policy.restored


def test_repeat_cancel_waits_cleanup_and_does_not_start_another_round(tmp_path):
    async def scenario():
        selected=rows()[:16]; bindings, events, clients, closed, policy=bindings_for(selected)
        began, closing, release=asyncio.Event(),asyncio.Event(),asyncio.Event()
        async def blocked(*_args,**_kwargs):began.set();await asyncio.Event().wait()
        original_close=bindings.close_pools
        async def close():closing.set();await release.wait();await original_close()
        bindings.round_fn=blocked;bindings.close_pools=close
        report={'watchdog_s':420};task=asyncio.create_task(tool.run_policy(selected,tmp_path,report,bindings))
        await began.wait();task.cancel();await closing.wait();task.cancel();await asyncio.sleep(0)
        assert not task.done();release.set()
        with pytest.raises(asyncio.CancelledError):await task
        assert report['cleanup']['status']=='passed' and len(clients)==1 and clients[0].closed==1 and len(closed)==1
    asyncio.run(scenario())


def test_reference_rejects_changed_conditions_or_nondefault_or_missing_input():
    reference={'policy':'default_global','status':'completed','cleanup':{'status':'passed'},'input_identity':[1],
               'selected_ids':['x'],'supervisor':{'returncode':0,'child_exited':True,'cleanup_status':'passed'},'quality_reference':{'qwen':[1],'nn':[2]},'config_sha256':'a'}
    assert tool.reference_for(reference,[1],['x'],{'config_sha256':'a'})
    with pytest.raises(ValueError):tool.reference_for(reference,[2],['x'])
    with pytest.raises(ValueError):tool.reference_for(reference,[1],['x'],{'config_sha256':'b'})
    with pytest.raises(ValueError):tool.reference_for(dict(reference,policy='pooled_global'),[1],['x'])


def test_plan_is_read_only_process_only_and_resolves_so_without_import(tmp_path):
    config=tmp_path/'config.toml';manifest=tmp_path/'manifest.json';output=tmp_path/'no-output'
    config.write_text('[qwen_asr]\nexecution_mode="process"\nlocal_device="npu:7"\nlocal_instance_count=1\n[model_device_pool]\n'
        'speech_campplus_devices=['+','.join(['"npu:7"']*8)+']\nemotion2vec_devices=['+','.join(['"npu:7"']*8)+']\n')
    manifest.write_text(json.dumps({'scope':{'selection_scope':'fulltime representative'},'segments':rows()}))
    (tmp_path/'scripts').mkdir();(tmp_path/'core').mkdir()
    for name in ('benchmark_qwen_process_candidate','benchmark_npu_stream_policy','benchmark_qwen_batch_curve'):
        (tmp_path/'scripts'/f'{name}.py').write_text('# never imported fixture')
    for name in ('qwen_asr_process','qwen_asr_client','qwen_asr_runtime','qwen_batch_scheduler','models','local_inference','model_loading','async_utils','config'):
        (tmp_path/'core'/f'{name}.cpu.so').write_bytes(b'never import fake .so')
    argv=['--config',str(config),'--manifest',str(manifest),'--output',str(output),'--policy','default_global']
    with patch.object(tool,'ROOT',tmp_path),patch.object(tool,'execute',side_effect=AssertionError('plan executed')),redirect_stdout(StringIO()):
        assert tool.main(argv)==0
    assert not output.exists()
    with patch.object(tool,'ROOT',tmp_path):
        args=SimpleNamespace(config=config,manifest=manifest,output=output,policy='pooled_global',reference=None,segments=128,watchdog_s=420)
        with pytest.raises(ValueError,match='reference'):tool.plan(args)
        with pytest.raises(ValueError,match='source-only'):tool.require_source_execution()


def test_actual_selection_mismatch_is_rejected_before_loading_any_model(tmp_path):
    selected=rows()[:16];bindings,events,clients,closed,policy=bindings_for(selected)
    report={'watchdog_s':420,'policy':'default_global','selected_ids':['wrong-plan-id']*16}
    with pytest.raises(ValueError,match='actual selected segment IDs'):
        asyncio.run(tool.run_policy(selected,tmp_path,report,bindings))
    assert not events and not clients and not closed


def test_stream_and_shared_candidate_selection_use_identical_canonical_function():
    from scripts import benchmark_qwen_process_candidate as paired_tool
    from tests.test_qwen_process_benchmark import tied_selection_rows
    assert tool.selected_rows is paired_tool.selected_rows
    records=tied_selection_rows()
    assert [row['id']for row in tool.selected_rows(records,16)]==[f'{owner}-{index}'for owner in range(8)for index in (0,1)]
