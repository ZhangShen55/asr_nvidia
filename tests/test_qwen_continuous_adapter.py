"""CPU-only small real text model tests for packed continuous KV semantics."""
from types import SimpleNamespace
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from core.qwen_continuous_runtime import ContinuousHFAdapter, Sequence


def make_model():
    torch.manual_seed(123)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    model = Qwen3ForCausalLM(cfg).eval()
    model.generation_config.eos_token_id = 63
    return model


def test_packed_decode_matches_independent_mixed_cache_lengths():
    model = make_model()
    adapter = ContinuousHFAdapter(SimpleNamespace(model=model, device='cpu', max_new_tokens=100))
    references = {}
    with torch.inference_mode():
        for key, prompt in [(('a','0'), [1,2,3]), (('b','0'), [7,8,9,10,11])]:
            ids = torch.tensor([prompt])
            out = model(ids, use_cache=True)
            token = out.logits[:, -1].argmax(-1)
            adapter.active[key] = Sequence(key=key, tokens=[int(token)], pending=token,
                prompt_length=len(prompt), kv=[(l.keys.clone(),l.values.clone()) for l in out.past_key_values.layers])
            references[key] = (out.past_key_values, token, len(prompt))
        # Force no EOS until the numerical comparison is finished.
        adapter._finish = lambda: []
        for _ in range(4):
            adapter.decode()
            for key, (cache, token, length) in list(references.items()):
                out = model(token[:,None], past_key_values=cache,
                    attention_mask=torch.ones(1,length+1,dtype=torch.long),
                    position_ids=torch.tensor([[length]]), use_cache=True)
                expected = out.logits[:, -1].argmax(-1)
                assert adapter.active[key].tokens[-1] == int(expected)
                for pair, layer in zip(adapter.active[key].kv, out.past_key_values.layers):
                    torch.testing.assert_close(pair[0], layer.keys, atol=1e-6,rtol=1e-5)
                    torch.testing.assert_close(pair[1], layer.values, atol=1e-6,rtol=1e-5)
                references[key] = (out.past_key_values, expected, length+1)
        assert adapter.cancel(('a','0'))[0][0] == ('a','0')
        assert ('a','0') not in adapter.active
        assert ('b','0') in adapter.active
        assert adapter.snapshot()['kv_bytes'] > 0


def test_view_storage_generation_released_after_remaining_member_steps():
    import gc
    import weakref
    model = make_model()
    adapter = ContinuousHFAdapter(SimpleNamespace(model=model, device='cpu', max_new_tokens=100))
    with torch.inference_mode():
        for name,prompt in [('a',[1,2,3]),('b',[7,8,9,10,11])]:
            output=model(torch.tensor([prompt]),use_cache=True)
            token=output.logits[:,-1].argmax(-1)
            adapter.active[(name,'0')]=Sequence(key=(name,'0'),tokens=[int(token)],pending=token,
                kv=[(l.keys,l.values) for l in output.past_key_values.layers])
        del output, token
        adapter._finish=lambda:[]
        adapter.decode()
        a=adapter.active[('a','0')].kv[0][0]
        b=adapter.active[('b','0')].kv[0][0]
        assert a.untyped_storage().data_ptr()==b.untyped_storage().data_ptr()
        storage_type = type(a.untyped_storage())
        old_storage_base = a.untyped_storage()._weak_ref()
        del a,b
        adapter.cancel(('a','0'))
        gc.collect()
        assert not storage_type._expired(old_storage_base)  # remaining member still reads it
        assert adapter.snapshot()['kv_storage_bytes'] > adapter.snapshot()['kv_bytes']
        adapter.decode()
        gc.collect()
        assert storage_type._expired(old_storage_base)  # no history pins the previous generation
        adapter.cancel(('b','0'))
        assert adapter.snapshot()['kv_storage_bytes']==0
        storage_type._free_weak_ref(old_storage_base)


def test_bundle_reuse_repack_new_member_reorder_cancel_and_positions():
    model=make_model()
    adapter=ContinuousHFAdapter(SimpleNamespace(model=model,device='cpu',max_new_tokens=100))
    refs={}
    def add(name,prompt):
        ids=torch.tensor([prompt])
        out=model(ids,use_cache=True)
        token=out.logits[:,-1].argmax(-1)
        # Reference cache must not alias the cache mutated by bundle reuse.
        from transformers import DynamicCache
        reference=DynamicCache(ddp_cache_data=[(l.keys.clone(),l.values.clone()) for l in out.past_key_values.layers])
        group=dict(cache=out.past_key_values,mask=torch.ones(1,len(prompt),dtype=torch.long),keys=((name,'0'),))
        adapter.active[(name,'0')]=Sequence(key=(name,'0'),tokens=[int(token)],pending=token,
            kv=[(l.keys,l.values) for l in out.past_key_values.layers],group=group,row=0)
        refs[(name,'0')]=(reference,token,len(prompt))
    def step(names):
        keys=[(name,'0') for name in names]
        adapter.decode(keys)
        for key in keys:
            cache,token,length=refs[key]
            out=model(token[:,None],past_key_values=cache,
                attention_mask=torch.ones(1,length+1,dtype=torch.long),
                position_ids=torch.tensor([[length]]),use_cache=True)
            token=out.logits[:,-1].argmax(-1)
            assert adapter.active[key].tokens[-1]==int(token)
            for pair,layer in zip(adapter.active[key].kv,out.past_key_values.layers):
                torch.testing.assert_close(pair[0],layer.keys,atol=1e-6,rtol=1e-5)
                torch.testing.assert_close(pair[1],layer.values,atol=1e-6,rtol=1e-5)
            refs[key]=(out.past_key_values,token,length+1)
    with torch.inference_mode():
        adapter._finish=lambda:[]
        add('a',[1,2,3]);add('b',[6,7,8,9,10])
        step(['a','b']);step(['a','b'])
        assert adapter.cache_reuses==1
        add('c',[20,21])
        step(['c','a','b']);step(['c','a','b'])
        adapter.cancel(('b','0'))
        step(['a','c']);step(['a','c'])
        assert adapter.repacks==3
        assert adapter.cache_reuses==3
        assert adapter.active[('a','0')].kv[0][0].shape[-2]==9
        assert adapter.active[('c','0')].kv[0][0].shape[-2]==6


def test_invalid_prefix_eos_is_warning_not_success_eos_or_full_audio_error():
    model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[63]))
    tokenizer=SimpleNamespace(decode=lambda *a,**k:'private output without a valid prefix')
    adapter=ContinuousHFAdapter(SimpleNamespace(model=model,device='cpu',max_new_tokens=256,
        processor=SimpleNamespace(tokenizer=tokenizer)))
    key=('request','12')
    adapter.active[key]=Sequence(key=key,tokens=[63],duration_s=2.0)
    result=adapter._finish()[0][1]
    assert isinstance(result,dict)
    assert result['warning']['reason']=='invalid_output_prefix'
    assert result['warning']['action']=='discard_unrecoverable'
    assert result['text']=='language None<asr_text>'
    assert not result['recovered']
    snapshot=adapter.snapshot()
    assert snapshot['end_reasons']=={'invalid_output_prefix':1}
    assert snapshot['warning_counts']=={'invalid_output_prefix':1}
    assert snapshot['recent_warnings'][0]['segment_id']=='12'
    assert snapshot['error_counts']=={}
    assert 'private output' not in str(snapshot)
    assert 'eos' not in snapshot['end_reasons']


def test_forward_failure_has_structured_bounded_reason_not_cancelled():
    from core.qwen_continuous_runtime import ContinuousGenerationError
    adapter=ContinuousHFAdapter(SimpleNamespace(device='cpu'))
    for i in range(40):
        key=('request',str(i))
        adapter.active[key]=Sequence(key=key,tokens=[1])
        result=adapter.fail(key,RuntimeError('private prompt text'),phase='prefill')
        assert isinstance(result[0][1],ContinuousGenerationError)
        assert result[0][1].reason=='prefill_error'
    snapshot=adapter.snapshot()
    assert snapshot['end_reasons']=={'prefill_error':40}
    assert snapshot['error_counts']=={'prefill:prefill_error:RuntimeError':40}
    assert len(snapshot['recent_errors'])==32
    assert 'private prompt text' not in str(snapshot)
    assert snapshot['active_sequences']==0

def test_bundle_compacts_physical_width_during_many_sequence_generations():
    model=make_model()
    adapter=ContinuousHFAdapter(SimpleNamespace(model=model,device='cpu',max_new_tokens=100))
    refs={}
    from transformers import DynamicCache
    with torch.inference_mode():
        adapter._finish=lambda:[]
        for generation in range(30):
            key=('turnover',str(generation))
            prompt=[1,2,3,4,5][:generation%5+1]
            out=model(torch.tensor([prompt]),use_cache=True)
            pending=out.logits[:,-1].argmax(-1)
            reference=DynamicCache(ddp_cache_data=[(l.keys.clone(),l.values.clone()) for l in out.past_key_values.layers])
            group=dict(cache=out.past_key_values,mask=torch.ones(1,len(prompt),dtype=torch.long),keys=(key,))
            adapter.active[key]=Sequence(key=key,tokens=[int(pending)],pending=pending,group=group,row=0,
                kv=[(l.keys,l.values) for l in out.past_key_values.layers])
            refs[key]=(reference,pending,len(prompt))
            if len(adapter.active)>3:
                oldest=next(iter(adapter.active));adapter.cancel(oldest);refs.pop(oldest)
            adapter.decode()
            assert adapter._bundle['mask'].shape[-1]==max(seq.kv[0][0].shape[-2] for seq in adapter.active.values())
            assert adapter._bundle['mask'].shape[-1]<=8
            for active_key,(cache,token,length) in list(refs.items()):
                reference=model(token[:,None],past_key_values=cache,
                    attention_mask=torch.ones(1,length+1,dtype=torch.long),
                    position_ids=torch.tensor([[length]]),use_cache=True)
                pending=reference.logits[:,-1].argmax(-1)
                assert adapter.active[active_key].tokens[-1]==int(pending)
                for pair,layer in zip(adapter.active[active_key].kv,reference.past_key_values.layers):
                    torch.testing.assert_close(pair[0],layer.keys,atol=1e-6,rtol=1e-5)
                    torch.testing.assert_close(pair[1],layer.values,atol=1e-6,rtol=1e-5)
                refs[active_key]=(reference.past_key_values,pending,length+1)


def test_normal_eos_returns_structured_text_without_warning():
    model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[63]))
    raw='language Chinese<asr_text>正常内容。<|im_end|>'
    adapter=ContinuousHFAdapter(SimpleNamespace(model=model,device='cpu',max_new_tokens=256,
        processor=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *a,**k:raw))))
    key=('request','normal')
    adapter.active[key]=Sequence(key=key,tokens=[63],duration_s=2.0)
    result=adapter._finish()[0][1]
    assert result==dict(text='language Chinese<asr_text>正常内容。',warning=None,recovered=False,original_raw=raw)
    assert adapter.snapshot()['end_reasons']=={'eos':1}


def test_length_exhaustion_is_segment_warning_and_other_member_stays_active():
    raw='language Chinese<asr_text>有效内容。'
    model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[63]))
    adapter=ContinuousHFAdapter(SimpleNamespace(model=model,device='cpu',max_new_tokens=4,
        processor=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *a,**k:raw))))
    adapter.active[('a','0')]=Sequence(key=('a','0'),tokens=[1,2,3,4],duration_s=2.0)
    adapter.active[('b','0')]=Sequence(key=('b','0'),tokens=[1],duration_s=2.0)
    result=adapter._finish()
    assert len(result)==1 and result[0][0]==('a','0')
    assert result[0][1]['warning']['reason']=='length_limit'
    assert ('b','0') in adapter.active


def test_real_online_guard_recovers_short_loop_and_releases_only_finished_member():
    from utils.incremental_repetition import IncrementalRepetitionGuard
    raw = ["language Chinese<asr_text>今天天气" + "很好" * 80]
    tokenizer = SimpleNamespace(batch_decode=lambda *a, **k: [raw[0]], decode=lambda *a, **k: raw[0])
    adapter = ContinuousHFAdapter(SimpleNamespace(device="cpu", max_new_tokens=256,
        model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[63])),
        processor=SimpleNamespace(tokenizer=tokenizer)), online_repetition_enabled=True)
    key = ("audio", "141")
    seq = Sequence(key=key, tokens=[1] * 64, duration_s=2.02)
    seq.guard = IncrementalRepetitionGuard(2.02)
    adapter.active[key] = seq
    other = ("audio", "142")
    adapter.active[other] = Sequence(key=other, tokens=[1], duration_s=2)
    assert adapter._finish() == []  # 首次观测未满足连续确认
    seq.tokens.extend([1] * 4)
    raw[0] += "很好"
    finished = adapter._finish()
    assert finished[0][0] == key
    result = finished[0][1]
    assert result["text"] == "language Chinese<asr_text>今天天气很好"
    assert result["warning"] == dict(reason="repetition_stop", action="repetition_collapsed", recovered=True)
    assert key not in adapter.active and other in adapter.active
    assert adapter.snapshot()["end_reasons"] == {"repetition_stop": 1}
    assert adapter.snapshot()["warning_actions"] == {"repetition_collapsed": 1}
    assert raw[0] not in str(adapter.snapshot())
