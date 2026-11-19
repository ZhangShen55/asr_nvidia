"""CPU processor/model桩验证多成员输入与解码，不初始化NPU。"""
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
import torch
from core.qwen_asr_runtime import LocalQwenRuntime, move_inputs
from tests.test_qwen_batch_scheduler import wav


def _prepare_language_inputs(language, size):
    """测试processor的纯语言校验桩；生产从真实processor函数globals读取。"""
    if language not in (None, 'en'):
        raise ValueError('Unsupported language: '+str(language))
    return [language]*size


class RuntimeBatchTests(unittest.TestCase):
    def test_variable_audio_languages_masks_and_per_member_decode(self):
        runtime=LocalQwenRuntime('/unused','npu:7')
        synced=[]; captured=[]
        runtime.api=SimpleNamespace(set_device=lambda _:None,device=lambda _:nullcontext(),synchronize=lambda d:synced.append(d))
        def prepare(**kwargs):
            index=len(captured);captured.append(kwargs)
            return {'input_ids':torch.tensor([[0,1,2]] if index==0 else [[3,4,5]]),
                    'attention_mask':torch.tensor([[False,True,True]] if index==0 else [[True,True,True]]),
                    'input_features':torch.ones(1,80,100 if index==0 else 200),
                    'input_features_mask':torch.ones(1,100 if index==0 else 200,dtype=torch.bool)}
        def generate(**kwargs):
            self.assertEqual(kwargs['input_ids'].dtype,torch.int64)
            self.assertEqual(kwargs['attention_mask'].dtype,torch.bool)
            self.assertEqual(kwargs['input_features'].dtype,torch.bfloat16)
            return torch.cat([kwargs['input_ids'],torch.tensor([[10,11],[12,13]])],dim=1)
        def decode(ids,**kwargs):
            self.assertEqual(ids.tolist(),[[10,11],[12,13]])
            return ['language English<asr_text>hello<|im_end|>','language Chinese<asr_text>你好<|endoftext|>']
        runtime.processor=SimpleNamespace(apply_transcription_request=prepare,batch_decode=decode,tokenizer=SimpleNamespace(pad_token_id=0))
        runtime.model=SimpleNamespace(dtype=torch.bfloat16,generate=generate)
        with patch('core.qwen_asr_runtime.move_inputs',side_effect=lambda x,**kw:move_inputs(x,device='cpu',dtype=kw['dtype'])):
            out=runtime.transcribe_batch([wav(.3),wav(1)],['en',None])
        self.assertEqual(out,['language English<asr_text>hello','language Chinese<asr_text>你好'])
        self.assertEqual([item['language'] for item in captured],['en',None])
        self.assertEqual([len(item['audio']) for item in captured],[4800,16000])
        self.assertEqual(synced,['npu:7'])
        self.assertEqual(runtime.snapshot()['last_batch_timings']['processor_shapes']['input_ids'],[2,3])

    def test_empty_or_mismatched_batch_rejected_before_load(self):
        runtime=LocalQwenRuntime('/missing','npu:7')
        with self.assertRaises(ValueError): runtime.transcribe_batch([])
        with self.assertRaises(ValueError): runtime.transcribe_batch([wav()],[None,None])
        self.assertFalse(runtime.loaded)


class ClientDispatchCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_selected_batch_cancelled_before_runtime_lease_is_not_executed(self):
        import asyncio
        from core.qwen_asr_client import QwenAsrClient
        from tests.test_local_qwen_runtime import FakeRuntime,until
        client=QwenAsrClient(retry_count=0,execution_mode="thread", runtime_factory=FakeRuntime,local_instance_count=2)
        client._runtimes[0].failure='unavailable'
        runtime=client._runtimes[1]; runtime.block=True
        first=asyncio.create_task(client.transcribe_bytes(wav(),request_id='first'))
        await until(runtime.started.is_set)
        second=asyncio.create_task(client.transcribe_bytes(wav(),language='en',request_id='second'))
        await until(lambda:client._waiting==1)
        second.cancel(); await asyncio.sleep(.01)
        runtime.release.set(); await first
        with self.assertRaises(asyncio.CancelledError):await second
        self.assertEqual(runtime.calls,[None])
        await client.aclose()


class ClientLanguageIsolationTests(unittest.IsolatedAsyncioTestCase):
    def runtime(self):
        runtime=LocalQwenRuntime('/unused','npu:7')
        calls=[]
        runtime.api=SimpleNamespace(set_device=lambda _:None,device=lambda _:nullcontext(),
                                   synchronize=lambda _:None,empty_cache=lambda:None)
        def prepare(**kwargs):
            calls.append(kwargs['language'])
            return {'input_ids':torch.ones(1,2,dtype=torch.long),
                    'attention_mask':torch.ones(1,2,dtype=torch.long),
                    'input_features':torch.ones(1,128,100),
                    'input_features_mask':torch.ones(1,100,dtype=torch.long)}
        runtime.processor=SimpleNamespace(apply_transcription_request=prepare,tokenizer=SimpleNamespace(pad_token_id=0),
            batch_decode=lambda ids,**kw:['language English<asr_text>hello']*ids.shape[0])
        runtime.model=SimpleNamespace(dtype=torch.bfloat16,
            generate=lambda **kw:torch.ones(kw['input_ids'].shape[0],3,dtype=torch.long))
        return runtime,calls

    async def test_mixed_auto_en_invalid_only_bad_member_4009(self):
        import asyncio
        from core.qwen_asr_client import QwenAsrClient,QwenAsrUnsupportedLanguageError
        runtime,calls=self.runtime()
        client=QwenAsrClient(retry_count=0,execution_mode="thread", runtime_factory=lambda *_:runtime,max_inference_batch_size=3)
        with patch('core.qwen_asr_runtime.move_inputs',side_effect=lambda x,**kw:x):
            results=await asyncio.gather(*(client.transcribe_bytes(wav(),lang,request_id=str(i))
                                          for i,lang in enumerate(['auto','en','unsupported-language'])),return_exceptions=True)
        self.assertEqual(results[:2],['language English<asr_text>hello']*2)
        self.assertIsInstance(results[2],QwenAsrUnsupportedLanguageError)
        self.assertEqual(results[2].code,4009)
        self.assertEqual(calls,[None,'en'])
        batch=client.snapshot()['batch_scheduler']['recent_batches'][0]
        self.assertEqual(batch['size'],2)
        self.assertEqual(batch['invalid_before_dispatch'],1)
        self.assertEqual(batch['runtime_timings']['processor_shapes']['input_ids'],[2,2])
        self.assertEqual([m['request_id'] for m in batch['dispatched_members']],['0','1'])
        await client.aclose()

    async def test_all_invalid_produces_no_physical_batch(self):
        import asyncio
        from core.qwen_asr_client import QwenAsrClient,QwenAsrUnsupportedLanguageError
        runtime,calls=self.runtime()
        client=QwenAsrClient(retry_count=0,execution_mode="thread", runtime_factory=lambda *_:runtime,max_inference_batch_size=2)
        results=await asyncio.gather(*(client.transcribe_bytes(wav(),'unsupported-language',request_id=str(i))
                                      for i in range(2)),return_exceptions=True)
        self.assertTrue(all(isinstance(x,QwenAsrUnsupportedLanguageError) for x in results))
        self.assertEqual(calls,[])
        batch=client.snapshot()['batch_scheduler']['recent_batches'][0]
        self.assertEqual(batch['size'],0)
        self.assertEqual(batch['dispatched_members'],[])
        await client.aclose()

    async def test_malformed_output_isolated_from_valid_and_invalid_language_members(self):
        import asyncio
        from core.qwen_asr_client import QwenAsrClient,QwenAsrCallError,QwenAsrUnsupportedLanguageError
        runtime,calls=self.runtime()
        runtime.processor.batch_decode=lambda ids,**kw:[
            'language English<asr_text>valid-member', 'malformed-prefix']
        client=QwenAsrClient(retry_count=0,execution_mode="thread", runtime_factory=lambda *_:runtime,max_inference_batch_size=3)
        try:
            with patch('core.qwen_asr_runtime.move_inputs',side_effect=lambda x,**kw:x):
                results=await asyncio.gather(*(client.transcribe_bytes(wav(),lang,request_id=str(i),segment_index=i)
                    for i,lang in enumerate(['auto','en','unsupported-language'])),return_exceptions=True)
            self.assertEqual(results[0],'language English<asr_text>valid-member')
            self.assertIsInstance(results[1],QwenAsrCallError)
            self.assertEqual(results[1].code,4012)
            self.assertIsInstance(results[2],QwenAsrUnsupportedLanguageError)
            self.assertEqual(results[2].code,4009)
            self.assertEqual(calls,[None,'en'])
            batch=client.snapshot()['batch_scheduler']['recent_batches'][0]
            self.assertEqual(batch['size'],2)
            self.assertEqual(batch['actual_dispatched_size'],2)
            self.assertEqual(batch['invalid_before_dispatch'],1)
            self.assertEqual([m['request_id'] for m in batch['dispatched_members']],['0','1'])
            self.assertEqual(batch['runtime_timings']['valid_member_indices'],[0,1])
            self.assertEqual(batch['runtime_timings']['output_error_indices'],[1])
            self.assertEqual(batch['runtime_timings']['output_error_count'],1)
        finally:
            await client.aclose()
