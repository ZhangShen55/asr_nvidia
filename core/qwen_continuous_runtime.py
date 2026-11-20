"""Single-owner HF forward adapter. KV remains on the configured device.

The pending token is the last sampled token, not yet represented in KV. Decode
packs left padded independent KV, uses each sequence's logical position, then
keeps read-only valid KV views until their next decode replaces the generation.
No historical generation is retained by completed-token records.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import re
import time


class ContinuousCapacityRejected(RuntimeError):
    accepted = False
    reason = "capacity_rejected"
    code = 4012


class ContinuousGenerationError(RuntimeError):
    code = 4012
    retryable = False

    def __init__(self, reason, *, request_id=None, segment_id=None):
        self.reason = reason
        self.request_id = request_id
        self.segment_id = segment_id
        super().__init__(f'{reason} [request_id={request_id}, segment_id={segment_id}]')



@dataclass
class Sequence:
    key: tuple
    tokens: list = field(default_factory=list)
    kv: list = field(default_factory=list)
    prompt_length: int = 0
    pending: object = None
    guard: object = None
    events: dict = field(default_factory=dict)
    group: object = None
    row: int = 0


class ContinuousHFAdapter:
    def __init__(self, runtime, *, online_repetition_enabled=False, online_repetition_config=None, event_limit=256):
        self.runtime = runtime
        self.active = {}
        self.events = []
        self.event_limit = event_limit
        self.online_repetition_enabled = online_repetition_enabled
        self.online_repetition_config = online_repetition_config or {}
        self.completed = 0
        self.completed_tokens = {}
        self.last_step = None
        self._bundle = None
        self.repacks = 0
        self.cache_reuses = 0
        self.end_reasons = {}
        self.error_counts = {}
        self.recent_errors = []
        self.last_phase = None
        self.phase_wall_s = {'prefill': 0.0, 'decode': 0.0}
        self.steps = {'prefill': 0, 'decode': 0}
        self.peak_active = 0
        self.peak_kv_bytes = 0
        self._memory_sample = {'kv_bytes':0, 'kv_storage_bytes':0}


    def event(self, key, kind, **values):
        self.events.append(dict(request_id=key[0], segment_id=key[1], kind=kind,
                                monotonic=time.monotonic(), **values))
        del self.events[:-self.event_limit]

    def prefill(self, requests):
        import torch
        from core.qwen_asr_runtime import decode_wav_bytes, prepare_independent_inputs, move_inputs
        from utils.incremental_repetition import IncrementalRepetitionGuard, RepetitionGuardConfig
        rt = self.runtime
        self.last_phase = 'prefill'
        audios = [decode_wav_bytes(item['audio']) for item in requests]
        start = time.monotonic()
        inputs = prepare_independent_inputs(rt.processor, audios, [r.get('language') for r in requests])
        inputs = move_inputs(inputs, device=rt.device, dtype=rt.model.dtype)
        lengths = inputs['attention_mask'].sum(-1).tolist()
        positions = inputs['attention_mask'].long().cumsum(-1) - 1
        positions.masked_fill_(inputs['attention_mask'] == 0, 1)
        with torch.inference_mode():
            output = rt.model(**inputs, position_ids=positions, use_cache=True, logits_to_keep=1)
            sampled = output.logits[:, -1].argmax(-1)
            # Readback is a safe boundary for the forward producing these tokens.
            ids = sampled.tolist()
            group = dict(cache=output.past_key_values, mask=inputs['attention_mask'],
                         keys=tuple((item['request_id'],item['segment_id']) for item in requests))
            for i, item in enumerate(requests):
                key = (item['request_id'], item['segment_id'])
                length = int(lengths[i])
                seq = Sequence(key=key, prompt_length=length, pending=sampled[i:i+1], group=group, row=i,
                    kv=[(layer.keys[i:i+1, :, -length:], layer.values[i:i+1, :, -length:])
                        for layer in output.past_key_values.layers])
                if self.online_repetition_enabled:
                    seq.guard = IncrementalRepetitionGuard(len(audios[i])/16000,
                        config=RepetitionGuardConfig(**self.online_repetition_config))
                self.active[key] = seq
                self.event(key, 'prefill', prompt_length=length, wall_s=time.monotonic()-start)
                seq.tokens.append(ids[i])
                self.event(key, 'first_token')
        if getattr(rt, 'api', None) is not None:
            rt.api.synchronize(rt.device)
        wall = time.monotonic()-start
        phase = 'prefill'
        self.phase_wall_s[phase] += wall
        self.steps[phase] += 1
        self._refresh_memory()
        completed = self._finish()
        return completed

    def decode(self, keys=None):
        import torch
        import torch.nn.functional as F
        from transformers import DynamicCache
        rt = self.runtime
        self.last_phase = 'decode'
        sequences = [self.active[key] for key in (keys or list(self.active))]
        if not sequences:
            return []
        start = time.monotonic()
        lengths = [seq.kv[0][0].shape[-2] for seq in sequences]
        width = max(lengths)
        keys = tuple(seq.key for seq in sequences)
        if self._bundle is not None and self._bundle['keys'] == keys:
            # HF DynamicCache extends its layer tensors with cat, never edits old
            # stored KV rows in place. Stable members can reuse the whole cache.
            cache = self._bundle['cache']
            old_mask = self._bundle['mask']
            width = old_mask.shape[-1]
            mask = torch.cat((old_mask, old_mask.new_ones((len(sequences),1))), dim=-1)
            self.cache_reuses += 1
        else:
            # Gather whole cache groups, not per-sequence pad/copy kernels.
            groups = []
            lookup = {}
            for seq in sequences:
                identity = id(seq.group) if seq.group is not None else id(seq)
                if identity not in lookup:
                    lookup[identity] = len(groups)
                    groups.append((seq.group, []))
                groups[lookup[identity]][1].append(seq)
            # Repack is also compaction: physical historical padding must not
            # grow with service-wide decode steps across sequence generations.
            width = max(lengths)
            chunks = []
            flattened = []
            masks = []
            for group,members in groups:
                flattened.extend(seq.key for seq in members)
                if group is None:
                    chunks.append([(k,v) for k,v in members[0].kv])
                    masks.append(torch.ones((1,members[0].kv[0][0].shape[-2]),dtype=torch.long,device=rt.device))
                else:
                    rows = torch.tensor([seq.row for seq in members],dtype=torch.long,device=rt.device)
                    cache_group = group['cache']
                    group_width = max(seq.kv[0][0].shape[-2] for seq in members)
                    chunks.append([(layer.keys.index_select(0,rows)[:, :, -group_width:],
                                    layer.values.index_select(0,rows)[:, :, -group_width:])
                                   for layer in cache_group.layers])
                    masks.append(group['mask'].index_select(0,rows)[:, -group_width:])
            order = [flattened.index(key) for key in keys]
            permutation = None if order == list(range(len(keys))) else torch.tensor(order,dtype=torch.long,device=rt.device)
            packed=[]
            for layer in range(len(sequences[0].kv)):
                pair=[]
                for side in (0,1):
                    parts=[]
                    for chunk in chunks:
                        value=chunk[layer][side]
                        extra=width-value.shape[-2]
                        parts.append(F.pad(value,(0,0,extra,0)) if extra else value)
                    value=torch.cat(parts,dim=0) if len(parts)>1 else parts[0]
                    pair.append(value.index_select(0,permutation) if permutation is not None else value)
                packed.append(tuple(pair))
            old_mask=torch.cat([F.pad(value,(width-value.shape[-1],0)) if value.shape[-1]!=width else value
                                for value in masks],dim=0)
            if permutation is not None:old_mask=old_mask.index_select(0,permutation)
            mask=torch.cat((old_mask,old_mask.new_ones((len(sequences),1))),dim=-1)
            cache=DynamicCache(ddp_cache_data=packed)
            self.repacks += 1
        positions = torch.tensor(lengths, dtype=torch.long, device=rt.device).unsqueeze(1)
        ids = torch.stack([seq.pending[0] for seq in sequences]).unsqueeze(1)
        with torch.inference_mode():
            output = rt.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                              past_key_values=cache, use_cache=True, logits_to_keep=1)
            sampled = output.logits[:, -1].argmax(-1)
            values = sampled.tolist()
            group=dict(cache=output.past_key_values, mask=mask, keys=keys)
            self._bundle=group
            for i, (seq, length) in enumerate(zip(sequences, lengths)):
                seq.kv = [(layer.keys[i:i+1, :, -(length+1):],
                           layer.values[i:i+1, :, -(length+1):])
                          for layer in output.past_key_values.layers]
                seq.group = group
                seq.row = i
                seq.pending = sampled[i:i+1]
                seq.tokens.append(values[i])
        if getattr(rt, 'api', None) is not None:
            rt.api.synchronize(rt.device)
        wall = time.monotonic()-start
        self.last_step = dict(batch_size=len(sequences), wall_s=wall, max_cache_length=width)
        self.phase_wall_s['decode'] += wall
        self.steps['decode'] += 1
        self._refresh_memory()
        return self._finish()

    def _finish(self):
        rt = self.runtime
        eos = rt.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (tuple, list)) else [eos])
        completed = []
        # Batch detokenization every four generated tokens, after the guard's
        # minimum observation count. This bounds detector work without affecting
        # the greedy decode path or normal EOS processing.
        guard_sequences = [seq for seq in self.active.values() if seq.guard is not None
            and len(seq.tokens) >= seq.guard.config.min_generated_tokens
            and len(seq.tokens) % 4 == 0 and seq.tokens[-1] not in eos
            and len(seq.tokens) < rt.max_new_tokens]
        guard_texts = {}
        if guard_sequences:
            values = rt.processor.tokenizer.batch_decode([s.tokens for s in guard_sequences],
                skip_special_tokens=False, clean_up_tokenization_spaces=False)
            guard_texts = {seq.key: raw for seq, raw in zip(guard_sequences, values)}
        for key, seq in list(self.active.items()):
            reason = None
            if seq.tokens[-1] in eos:
                reason = 'eos'
            elif len(seq.tokens) >= rt.max_new_tokens:
                reason = 'length_limit'
            elif key in guard_texts:
                if seq.guard.update(guard_texts[key], generated_tokens=len(seq.tokens)) is not None:
                    reason = 'repetition_stop'
            if reason:
                raw = rt.processor.tokenizer.decode(seq.tokens, skip_special_tokens=False,
                                                    clean_up_tokenization_spaces=False)
                text = re.sub(r'<\|(?:im_end|endoftext|eot_id)\|>', '', raw).strip()
                error = None
                if reason != 'eos':
                    error = ContinuousGenerationError(reason, request_id=key[0], segment_id=key[1])
                elif text and not re.search(r'language\s+[^<]*<asr_text>', text, re.I):
                    reason = 'invalid_output_prefix'
                    error = ContinuousGenerationError(reason, request_id=key[0], segment_id=key[1])
                if error is not None:
                    self.record_error(key, error, self.last_phase, reason)
                self.end_reasons[reason] = self.end_reasons.get(reason, 0)+1
                self.event(key, reason, generated_tokens=len(seq.tokens))
                del self.active[key]
                self.completed += 1
                self.completed_tokens[key] = list(seq.tokens)
                self.event(key, 'resource_reusable')
                completed.append((key, error if error else text))
        return completed

    def record_error(self, key, exc, phase, reason):
        # Store structure only; exception messages may contain model output.
        category = f'{phase}:{reason}:{type(exc).__name__}'
        self.error_counts[category] = self.error_counts.get(category, 0)+1
        self.recent_errors.append(dict(request_id=key[0],segment_id=key[1],
            phase=phase,reason=reason,error_type=type(exc).__name__,monotonic=time.monotonic()))
        del self.recent_errors[:-32]

    def fail(self, key, exc, *, phase):
        if getattr(self.runtime, 'api', None) is not None:
            # A failed forward can still own queued device accesses. If this
            # synchronization fails, let the worker die; reader confirms death.
            self.runtime.api.synchronize(self.runtime.device)
        seq = self.active.pop(key, None)
        reason = phase+'_error'
        self.record_error(key, exc, phase, reason)
        self.end_reasons[reason] = self.end_reasons.get(reason, 0)+1
        self.completed += 1
        if seq is not None:
            self.completed_tokens[key] = list(seq.tokens)
        self._refresh_memory()
        self.event(key, reason, error_type=type(exc).__name__)
        self.event(key, 'resource_reusable')
        error = ContinuousGenerationError(reason,request_id=key[0],segment_id=key[1])
        return [(key,error)]

    def cancel(self, key):
        # Called only at owner-loop boundaries; previous physical step is read back.
        if key in self.active and getattr(self.runtime, 'api', None) is not None:
            self.runtime.api.synchronize(self.runtime.device)
        if self.active.pop(key, None) is not None:
            self._refresh_memory()
            self.end_reasons['cancelled'] = self.end_reasons.get('cancelled', 0)+1
            self.event(key, 'cancelled')
            return [(key, ContinuousGenerationError('cancelled',request_id=key[0],segment_id=key[1]))]
        return []

    def _refresh_memory(self):
        cache_bytes = sum(t.numel()*t.element_size() for seq in self.active.values()
                          for pair in seq.kv for t in pair)
        stores = {}
        for seq in self.active.values():
            for pair in seq.kv:
                for tensor in pair:
                    storage = tensor.untyped_storage()
                    stores[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
        self._memory_sample = dict(kv_bytes=cache_bytes, kv_storage_bytes=sum(stores.values()),
                                   sampled_monotonic=time.monotonic())
        self.peak_active = max(self.peak_active, len(self.active))
        self.peak_kv_bytes = max(self.peak_kv_bytes, cache_bytes)

    def snapshot(self):
        if not self.active:
            self._bundle = None
            self._memory_sample = dict(kv_bytes=0, kv_storage_bytes=0,
                                       sampled_monotonic=time.monotonic())
        return dict(active_sequences=len(self.active), **self._memory_sample,
                    completed=self.completed, last_step=self.last_step, events=list(self.events),
                    error_counts=dict(self.error_counts),recent_errors=list(self.recent_errors),
                    end_reasons=dict(self.end_reasons), phase_wall_s=dict(self.phase_wall_s),
                    steps=dict(self.steps),peak_active=self.peak_active, peak_kv_bytes=self.peak_kv_bytes,
                    repacks=self.repacks, cache_reuses=self.cache_reuses)
