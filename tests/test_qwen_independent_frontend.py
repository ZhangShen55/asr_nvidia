"""单段前端语义不能因批内其它PCM长度变化；不导入NPU或权重。"""
import unittest
from types import SimpleNamespace
import torch
from core.qwen_asr_runtime import collate_independent_inputs,prepare_independent_inputs


def single(ids,length,valid,*,mask_dtype=torch.long):
    return {'input_ids':torch.tensor([ids],dtype=torch.long),
            'attention_mask':torch.ones(1,len(ids),dtype=mask_dtype),
            'input_features':torch.arange(2*length,dtype=torch.float32).reshape(1,2,length),
            'input_features_mask':torch.tensor([[1]*valid+[0]*(length-valid)],dtype=mask_dtype)}


class IndependentFrontendTests(unittest.TestCase):
    def test_single_returns_original_object(self):
        source=single([1,2],100,30)
        self.assertIs(collate_independent_inputs([source]),source)

    def test_complete_single_features_and_masks_preserved_and_only_extra_padding_zero(self):
        a=single([11,12],100,30,mask_dtype=torch.bool)
        b=single([21,22,23,24],200,101,mask_dtype=torch.bool)
        originals=[{key:value.clone() for key,value in row.items()} for row in [a,b]]
        result=collate_independent_inputs([a,b],pad_token_id=99)
        self.assertEqual(result['input_ids'].tolist(),[[99,99,11,12],[21,22,23,24]])
        self.assertEqual(result['attention_mask'].tolist(),[[False,False,True,True],[True]*4])
        self.assertTrue(torch.equal(result['input_features'][0,:,:100],a['input_features'][0]))
        self.assertTrue(torch.equal(result['input_features_mask'][0,:100],a['input_features_mask'][0]))
        self.assertEqual(int(result['input_features'][0,:,100:].count_nonzero()),0)
        self.assertEqual(int(result['input_features_mask'][0,100:].count_nonzero()),0)
        self.assertEqual(result['input_features'].dtype,torch.float32)
        self.assertEqual(result['input_features_mask'].dtype,torch.bool)
        for row,original in zip([a,b],originals):
            for key in row:self.assertTrue(torch.equal(row[key],original[key]))

    def test_prepare_never_combines_waveforms_or_languages_and_factory_dtype_is_fp32(self):
        calls=[];source=[[.1]*17,[.2]*31]
        def prepare(*,audio,language):
            calls.append((audio,language,torch.get_default_dtype()))
            return single([len(audio)],100 if len(audio)==17 else 200,10)
        processor=SimpleNamespace(apply_transcription_request=prepare,tokenizer=SimpleNamespace(pad_token_id=0))
        before=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            result=prepare_independent_inputs(processor,source,['en',None])
            self.assertEqual(torch.get_default_dtype(),torch.bfloat16)
        finally:torch.set_default_dtype(before)
        self.assertEqual([language for _,language,_ in calls],['en',None])
        self.assertTrue(all(dtype==torch.float32 for _,_,dtype in calls))
        self.assertIs(calls[0][0],source[0]);self.assertIs(calls[1][0],source[1])
        self.assertEqual(result['input_ids'].tolist(),[[17],[31]])

    def test_unknown_keys_bad_shape_and_dtype_are_explicit_failures(self):
        good=single([1],100,30)
        variants=[]
        unknown=single([2],100,30);unknown['unknown']=torch.ones(1);variants.append(unknown)
        bad_mask=single([2],100,30);bad_mask['input_features_mask']=torch.ones(1,99,dtype=torch.long);variants.append(bad_mask)
        bad_dtype=single([2],100,30);bad_dtype['input_features']=bad_dtype['input_features'].double();variants.append(bad_dtype)
        nonbinary=single([2],100,30);nonbinary['attention_mask'].fill_(2);variants.append(nonbinary)
        for variant in variants:
            with self.subTest(keys=list(variant)):
                with self.assertRaises(ValueError):collate_independent_inputs([good,variant],pad_token_id=0)

    def test_original_text_padding_is_replaced_only_on_left(self):
        a=single([99,11,12],100,30);a['attention_mask'][0,0]=0
        b=single([21,22,23,24],100,30)
        result=collate_independent_inputs([a,b],pad_token_id=7)
        self.assertEqual(result['input_ids'].tolist(),[[7,7,11,12],[21,22,23,24]])
        self.assertEqual(result['attention_mask'].sum(1).tolist(),[2,4])
