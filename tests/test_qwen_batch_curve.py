"""曲线工具的CPU契约：不加载权重或初始化NPU。"""
import asyncio
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('batch_curve',Path(__file__).parents[1]/'scripts/benchmark_qwen_batch_curve.py')
curve=importlib.util.module_from_spec(spec);spec.loader.exec_module(curve)


class Gate:
    def __init__(self):self.global_sem=asyncio.Semaphore(128);self.active=0
    def new_segment_semaphore(self):return asyncio.Semaphore(16)
    async def acquire(self,sem):
        await sem.acquire();await self.global_sem.acquire();self.active+=1
    async def release(self,sem):self.active-=1;sem.release();self.global_sem.release()


class Client:
    def __init__(self,blocked=False):self.batches=[];self.blocked=blocked;self.calls=[]
    def snapshot(self):return {'batch_scheduler':{'batch_count':len(self.batches),'recent_batches':list(self.batches)}}
    async def transcribe_bytes(self,data,lang,**meta):
        self.calls.append(meta)
        if self.blocked:await asyncio.Future()
        if not self.batches:
            self.batches.append({'batch_id':1,'size':4,'execution_wall_ms':10,'runtime_timings':{'processor_shapes':{'input_ids':[4,20]}}})
        await asyncio.sleep(.002)
        return 'language English<asr_text>'+str(meta['segment_index'])


def segments(n=4):
    return [{'id':str(i),'request_id':str(i%2),'segment_index':i,'wav':b'fake-pcm','duration_s':1,'language':None} for i in range(n)]


class CurveTests(unittest.IsolatedAsyncioTestCase):
    async def test_members_keep_identity_and_batch_envelope_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as folder:
            client=Client();gate=Gate();path=Path(folder)/'round.json'
            report=await curve.run_round(client,segments(),gate,path,sample_interval_s=.001,sampler=lambda:{'cpu':'fake'})
            saved=json.loads(path.read_text())
            self.assertEqual(report['status'],'completed');self.assertEqual(saved['success_count'],4)
            self.assertEqual(len(saved['batches']),1)
            self.assertEqual(saved['batches'][0]['execution_wall_ms'],10)
            self.assertEqual({r['id'] for r in saved['members']},{str(i) for i in range(4)})
            self.assertTrue(all(r['queued_monotonic_s']<=r['start_monotonic_s']<=r['ended_monotonic_s'] for r in saved['members']))
            self.assertEqual(gate.active,0)
            self.assertEqual(len(Path(str(path)+'.members.jsonl').read_text().splitlines()),4)

    async def test_inflight_watchdog_does_not_need_completed_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            client=Client(blocked=True);gate=Gate();path=Path(folder)/'aborted.json'
            report=await curve.run_round(client,segments(),gate,path,watchdog_s=.005,no_progress_s=1,sample_interval_s=.001)
            self.assertEqual(report['status'],'diagnostic_aborted')
            self.assertEqual(report['diagnostic']['reason'],'inflight_wall_threshold')
            self.assertEqual(report['success_count'],0)
            self.assertTrue(all(row['status']=='cancelled' for row in report['members']))
            self.assertEqual(gate.active,0)

    async def test_no_progress_stops_early(self):
        with tempfile.TemporaryDirectory() as folder:
            report=await curve.run_round(Client(blocked=True),segments(),Gate(),Path(folder)/'aborted.json',
                                         watchdog_s=1,no_progress_s=.005,sample_interval_s=.001)
            self.assertEqual(report['diagnostic']['reason'],'no_progress_threshold')

    def test_import_has_no_optional_hardware_import(self):
        script=Path(__file__).parents[1]/'scripts/benchmark_qwen_batch_curve.py'
        code="import runpy,sys;runpy.run_path(sys.argv[1],run_name='import_only');assert 'torch' not in sys.modules;assert 'torch_npu' not in sys.modules"
        subprocess.run([sys.executable,'-c',code,str(script)],check=True)

    def test_invalid_manifest_rejected_before_audio_import(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'manifest.json';path.write_text(json.dumps({'segments':[]}))
            with self.assertRaisesRegex(ValueError,'128'):curve.read_manifest(path)

    def test_percentiles_use_linear_interpolation(self):
        self.assertEqual(curve.percentile([1,2,3,4],.5),2.5)
        self.assertAlmostEqual(curve.percentile([1,2,3,4],.95),3.85)


class CleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_cleanup_waits_owned_work_before_client_and_enhanced_close(self):
        events=[]
        async def work():
            try:await asyncio.Future()
            finally:events.append('owned_work_finished')
        class OwnedClient:
            async def aclose(self):events.append('client_closed')
        async def enhanced_close():events.append('enhanced_closed')
        task=asyncio.create_task(work());await asyncio.sleep(0)
        await curve.close_owned(OwnedClient(),task,enhanced_close)
        self.assertEqual(events,['owned_work_finished','client_closed','enhanced_closed'])

    async def test_repeated_cancel_cannot_interrupt_bounded_cleanup(self):
        release=asyncio.Event()
        task=asyncio.create_task(release.wait())
        waiter=asyncio.create_task(curve.wait_cleanup(task,1))
        await asyncio.sleep(0);waiter.cancel();await asyncio.sleep(0);waiter.cancel()
        await asyncio.sleep(0);self.assertFalse(waiter.done())
        release.set();result=await waiter
        self.assertTrue(result['finished']);self.assertTrue(result['cancel_received'])


class BytesPathManifestTests(unittest.TestCase):
    def test_non_utf8_source_path_and_declared_source_hash(self):
        import os,base64,hashlib,wave
        with tempfile.TemporaryDirectory() as folder:
            source=os.fsencode(folder)+b'/audio_\xff.wav'
            with wave.open(open(source,'wb'),'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000);wav.writeframes(b'\0\0'*4800)
            with open(source,'rb') as stream:digest=hashlib.sha256(stream.read()).hexdigest()
            rows=[{'id':str(i),'request_id':str(i//16),'segment_index':i%16,'start_s':0,'end_s':.3,
                   'language':None,'source_sha256':digest,'path_bytes_b64':base64.b64encode(source).decode()} for i in range(128)]
            path=Path(folder)/'manifest.json';path.write_text(json.dumps({'segments':rows}))
            prepared=curve.read_manifest(path,'length')
            self.assertEqual(len(prepared),128)
            self.assertTrue(all(row['duration_s']==.3 for row in prepared))
            self.assertEqual({(row['request_id'],row['segment_index']) for row in prepared},
                             {(str(i//16),i%16) for i in range(128)})
            self.assertEqual(prepared[0]['input_wav_sha256'],hashlib.sha256(prepared[0]['wav']).hexdigest())
            rows[0]['source_sha256']='0'*64;path.write_text(json.dumps({'segments':rows}))
            with self.assertRaisesRegex(ValueError,'SHA256'):curve.read_manifest(path)


class BatchObserverTests(unittest.TestCase):
    def test_published_state_copies_only_new_completed_ids_once(self):
        from types import SimpleNamespace
        copied=[]
        class RichBatch(dict):
            def __deepcopy__(self,memo):
                copied.append(self['batch_id'])
                return dict(self)
        old=[RichBatch(batch_id=i,size=128,large_history='x'*10000) for i in range(1,65)]
        state={'batch_count':64,'recent_batches':old,'waiting':0}
        class PublishedClient:
            _batch_scheduler=SimpleNamespace(_state=state)
            def snapshot(self):raise AssertionError('rich client snapshot on hot path')
        observer=curve.RoundBatchObserver(PublishedClient())
        for _ in range(128):observer.collect()
        self.assertEqual(copied,[]);self.assertEqual(observer.history_entries_examined,0)
        # Completion order can differ from batch-ID order with multiple workers.
        state['batch_count']=66
        state['recent_batches']=(old+[RichBatch(batch_id=66,size=4)])[-64:]
        observer.collect()
        state['recent_batches']=(state['recent_batches']+[RichBatch(batch_id=65,size=4)])[-64:]
        observer.collect()
        for _ in range(128):observer.collect()
        self.assertEqual(copied,[66,65]);self.assertEqual(observer.rich_batch_copies,2)
        self.assertEqual(observer.history_entries_examined,4)
        self.assertEqual(set(observer.batches),{65,66})
        self.assertNotIn('recent_batches',observer.last_snapshot['batch_scheduler'])
        self.assertEqual(len(old),64);self.assertNotIn('recent_batch_ids',state)

    def test_fake_client_fallback_remains_supported(self):
        client=Client();observer=curve.RoundBatchObserver(client)
        client.batches.append({'batch_id':1,'size':4})
        observer.collect();observer.collect()
        self.assertEqual(observer.rich_batch_copies,1)
        self.assertEqual(observer.fallback_snapshot_calls,3)
        self.assertEqual(observer.source,'snapshot_fallback')


class MeasurementBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_owned_completion_wakes_without_polling_sleep_or_real_snapshot(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        class PublishedClient(Client):
            def __init__(self):
                super().__init__();self._batch_scheduler=SimpleNamespace(_state={'batch_count':0,'recent_batches':[]})
            def snapshot(self):raise AssertionError('must use published state')
            async def transcribe_bytes(self,*args,**kwargs):
                self._batch_scheduler._state={'batch_count':1,'recent_batches':[{'batch_id':1,'size':4}]}
                return 'text'
        original_wait=asyncio.wait;original_sleep=asyncio.sleep;wait_kinds=[]
        async def wait(*args,**kwargs):
            wait_kinds.append(kwargs.get('return_when'));return await original_wait(*args,**kwargs)
        async def sleep(delay,*args,**kwargs):
            self.assertNotEqual(delay,3600,'polling tail sleep must not delay completed work')
            return await original_sleep(delay,*args,**kwargs)
        with tempfile.TemporaryDirectory() as folder,patch.object(curve.asyncio,'wait',wait),patch.object(curve.asyncio,'sleep',sleep):
            report=await curve.run_round(PublishedClient(),segments(),Gate(),Path(folder)/'round.json',sample_interval_s=3600)
        self.assertIn(asyncio.FIRST_COMPLETED,wait_kinds)
        self.assertEqual(report['status'],'completed')
        self.assertEqual(report['batch_collection']['rich_batch_copies'],1)
        self.assertEqual(report['batch_collection']['fallback_snapshot_calls'],0)
        self.assertEqual(report['ended_monotonic_s'],report['work_done_monotonic_s'])

    async def test_sampler_cleanup_is_not_work_time_without_speed_assertions(self):
        import threading
        entered=threading.Event();release=threading.Event();finished=asyncio.Event()
        now=[10.]
        class ClockClient(Client):
            async def transcribe_bytes(self,*args,**kwargs):
                self.batches=[{'batch_id':1,'size':1}]
                now[0]=20.;finished.set();return 'text'
        def sampler():
            entered.set()
            if not release.wait(2):raise RuntimeError('test sampler was not released')
            return {'sample':'owned'}
        with tempfile.TemporaryDirectory() as folder:
            task=asyncio.create_task(curve.run_round(ClockClient(),segments(1),Gate(),Path(folder)/'round.json',
                sampler=sampler,clock=lambda:now[0],sample_interval_s=3600))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait,2))
                await finished.wait();await asyncio.sleep(0);await asyncio.sleep(0)
                self.assertFalse(task.done(),'sampler still owned and awaiting final cleanup')
                now[0]=50.;release.set();report=await task
            finally:release.set()
        self.assertEqual(report['elapsed_s'],10.)
        self.assertEqual(report['observer_inclusive_elapsed_s'],40.)
        self.assertEqual(report['measurement_cleanup_s'],30.)
        self.assertFalse(report['resource_sampler_pending'])
        self.assertEqual(report['resources'],[{'sample':'owned'}])
