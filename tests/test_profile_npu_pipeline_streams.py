"""Stream观测与收尾CPU桩；不导入NPU、不加载权重、不监听端口。"""
import asyncio
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.profile_npu_pipeline_streams import StreamRecorder, observe, select_segments, main


class Api:
    def __init__(self):self.device_calls=[];self.sync_calls=0
    def set_device(self,value):self.device_calls.append(value)
    def device(self,value):return nullcontext()
    def current_stream(self,value):return SimpleNamespace(stream_id=3,npu_stream=91,device_index=7)
    def default_stream(self,value):return SimpleNamespace(stream_id=3,npu_stream=91,device_index=7)
    def synchronize(self,*args):self.sync_calls+=1


class Runner:
    def __init__(self,api):self.api=api;self.loaded=True;self.calls=0
    def generate(self,*args,**kwargs):self.calls+=1;self.api.synchronize('npu:7');return 'result'
    transcribe_batch=generate


class RecorderTests(unittest.TestCase):
    def test_worker_native_thread_and_stream_and_original_sync_are_recorded(self):
        api=Api();runner=Runner(api);rec=StreamRecorder();rec.phase='B';rec.attach(runner,'generate','cam[0]')
        worker_id=[]
        def execute():worker_id.append(threading.get_native_id());self.assertEqual(runner.generate(),'result')
        thread=threading.Thread(target=execute);thread.start();thread.join()
        self.assertEqual(api.sync_calls,1)
        rows=rec.snapshot()['events'];self.assertEqual(len(rows),4)
        self.assertTrue(all(row['native_thread_id']==worker_id[0] for row in rows))
        self.assertTrue(all(row['current_stream_id']==row['default_stream_id']==3 for row in rows))
        self.assertTrue(all(row['device_index']==7 and row['phase']=='B' for row in rows))
        rec.restore();self.assertNotIn('generate',vars(runner));self.assertNotIn('synchronize',vars(api))
        runner.generate();self.assertEqual(api.sync_calls,2);self.assertEqual(len(rec.snapshot()['events']),4)

    def test_original_exception_and_sync_are_preserved_and_stream_errors_are_explicit(self):
        api=Api();runner=Runner(api);rec=StreamRecorder()
        def fail():api.synchronize('npu:7');raise RuntimeError('original')
        runner.generate=fail
        with patch.object(api,'current_stream',side_effect=RuntimeError('unavailable')):
            rec.attach(runner,'generate','cam[0]')
            with self.assertRaisesRegex(RuntimeError,'original'):runner.generate()
        rec.restore();self.assertIs(runner.generate,fail)
        self.assertEqual(api.sync_calls,1)
        self.assertTrue(all('stream_unavailable' in row for row in rec.snapshot()['events']))


class ObserveTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self,fail=False):
        api=Api();qwen=Runner(api)
        class Client:
            _runtimes=[qwen]
            async def aclose(self):qwen.loaded=False
            def snapshot(self):return {'busy':0,'waiting':0,'loaded':qwen.loaded,'batch_scheduler':{'inflight':0,'running_members':0}}
        client=Client();pools={name:SimpleNamespace(models=[Runner(api) for _ in range(8)]) for name in ['cam','emo']}
        async def enhanced(_):
            await asyncio.gather(*[asyncio.to_thread(runner.generate) for pool in pools.values() for runner in pool.models])
        async def close():
            for pool in pools.values():pool.models.clear()
        async def fake_round(client,segments,gate,path,enhanced=None,**kwargs):
            await asyncio.to_thread(client._runtimes[0].transcribe_batch)
            if enhanced is not None:await enhanced(segments)
            if fail:raise RuntimeError('controlled round failure')
            return {'status':'completed','elapsed_s':0.01}
        with tempfile.TemporaryDirectory() as directory:
            recorder=StreamRecorder()
            if fail:
                with self.assertRaisesRegex(RuntimeError,'controlled round failure'):
                    await observe(client,pools,[],directory,enhanced_fn=enhanced,round_fn=fake_round,
                                  close_pools=close,gate_factory=lambda:object(),recorder=recorder)
            else:
                result=await observe(client,pools,[],directory,enhanced_fn=enhanced,round_fn=fake_round,
                                     close_pools=close,gate_factory=lambda:object(),recorder=recorder)
                self.assertEqual(result['status'],'completed');self.assertEqual(result['cleanup']['status'],'passed')
                self.assertEqual([row['phase'] for row in result['rounds']],['warmup','pure','B'])
                self.assertFalse(result['profiler']['supported'])
                self.assertEqual({row['model_instance'] for row in recorder.snapshot()['events'] if row['phase']=='pure'},{'qwen[0]'})
            self.assertFalse(qwen.loaded);self.assertTrue(all(not pool.models for pool in pools.values()))
            self.assertFalse(recorder.originals);self.assertNotIn('transcribe_batch',vars(qwen))
            self.assertTrue((Path(directory)/'stream-observation.json').exists())
    async def test_rounds_and_all_owned_cleanup(self):await self.exercise()
    async def test_failure_still_restores_hooks_and_closes_all_owned(self):await self.exercise(True)


class PlanTests(unittest.TestCase):
    def test_plan_reads_no_model_and_profiler_execute_explicitly_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=root/'config.toml';manifest=root/'manifest.json'
            cfg.write_text('[qwen_asr]\nlocal_device="npu:7"\n[model_device_pool]\nspeech_campplus_devices='+json.dumps(['npu:7']*8)+'\nemotion2vec_devices='+json.dumps(['npu:7']*8)+'\n')
            rows=[{'id':f'{r}-{s}','request_id':str(r),'start_s':s*2,'end_s':s*2+1.5,'segment_index':s} for r in range(8) for s in range(16)]
            manifest.write_text(json.dumps({'segments':rows}))
            with patch('scripts.profile_npu_pipeline_streams.execute',side_effect=AssertionError('no NPU')):
                out=root/'plan'
                self.assertEqual(main(['--config',str(cfg),'--manifest',str(manifest),'--output',str(out)]),0)
                summary=json.loads((out/'summary.json').read_text());self.assertEqual(summary['status'],'planned')
                self.assertEqual(len(summary['selected_ids']),16)
                reject=root/'unsupported'
                self.assertEqual(main(['--config',str(cfg),'--manifest',str(manifest),'--output',str(reject),'--execute','--profiler']),2)
                self.assertEqual(json.loads((reject/'summary.json').read_text())['status'],'unsupported')
    def test_selection_requires_two_emo_eligible_segments_each_request(self):
        with self.assertRaisesRegex(ValueError,'eight requests'):select_segments([{'request_id':'a','duration_s':0.3}])
