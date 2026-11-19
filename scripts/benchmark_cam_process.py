#!/usr/bin/env python3
"""Formal Settings-selected CAM thread/process component validation; default plan only.

NN8+8 and single original HF/BF16 Qwen worker; 16/128 canonical real segments,
one warm + three steady rounds. Does not patch model factories, math, device streams
or synchronization. CPU outputs and parent call envelopes only; supports .py/.so core.
"""
from __future__ import annotations
import argparse
import asyncio
from collections import deque
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.benchmark_qwen_process_candidate import persist, selected_rows, qwen_quality, nn_quality, supervise

MODE_PATH = ('model_device_pool', 'speech_campplus_execution_mode')
PREFETCH_KEYS = ('speech_campplus_prefetch_queue_depth', 'emotion2vec_prefetch_queue_depth')
PREFETCH_PATHS = sorted(('model_device_pool', key) for key in PREFETCH_KEYS)
HELPERS = ('benchmark_qwen_process_candidate.py', 'benchmark_qwen_process_nn_streams.py',
           'benchmark_qwen_batch_curve.py', 'benchmark_npu_stream_policy.py', 'profile_npu_pipeline_streams.py')


def read_config(path):
    try: import tomllib
    except ImportError: import tomli as tomllib
    return tomllib.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity(root=ROOT):
    """Hash all runtime core artifacts, including the actual CAM py/so, without importing them."""
    root = Path(root).resolve()
    paths = {Path(__file__).resolve()}
    for directory in ('core', 'utils', 'entity'):
        paths.update(p.resolve() for p in (root/directory).rglob('*') if p.is_file() and p.suffix in ('.py', '.so'))
    for name in HELPERS:
        paths.add((root/'scripts'/name).resolve())
    core = [p for p in paths if p.parent == root/'core' and p.name.startswith('cam_process')]
    if not core: raise ValueError('formal CAM core artifact is missing')
    return {str(path): digest(path) for path in sorted(paths)}


def config_differences(left, right, prefix=()):
    if isinstance(left, dict) and isinstance(right, dict):
        return [item for key in sorted(set(left)|set(right))
                for item in config_differences(left.get(key), right.get(key), prefix+(key,))]
    return [] if type(left) is type(right) and left == right else [prefix]


def prefetch_depth(config):
    pool = config.get('model_device_pool', {})
    values = tuple(pool.get(key, 0) for key in PREFETCH_KEYS)
    if any(type(v) is not int or v not in (0,1,5) for v in values) or values[0] != values[1]:
        raise ValueError('prefetch comparison requires paired integer depth0/1/5')
    for key in ('speech_campplus_batch_size','emotion2vec_batch_size'):
        if type(pool.get(key,1)) is not int or pool.get(key,1) != 1:
            raise ValueError('prefetch comparison requires actual CAM/emo compute batch1')
    return values[0]


def validate_reference(report, current, config, *, identity=None):
    comparison = current.get('comparison','cam-mode')
    expected_mode = 'thread' if comparison == 'cam-mode' else 'process'
    if (report.get('benchmark') != 'formal_cam' or report.get('execution_mode') != expected_mode
            or report.get('status') != 'completed' or report.get('cleanup', {}).get('status') != 'passed'
            or report.get('source_changed_during_job') is not False):
        raise ValueError(comparison+' requires a normally closed formal '+expected_mode+' reference')
    owner = report.get('supervisor', {})
    if (owner.get('returncode') != 0 or not owner.get('child_exited') or owner.get('cleanup_status') != 'passed'
            or owner.get('term_sent') or owner.get('forced_kill') or owner.get('residual_owned_pids')):
        raise ValueError('thread reference owned cleanup is not normal')
    for key in ('benchmark_root', 'source_sha256', 'manifest_sha256', 'selected_ids', 'segments'):
        if report.get(key) != current.get(key): raise ValueError('thread reference differs: '+key)
    baseline_path = Path(report['config_path'])
    if digest(baseline_path) != report['config_sha256']: raise ValueError('thread reference config file changed')
    before = read_config(baseline_path)
    if comparison == 'cam-mode':
        if before.get('model_device_pool', {}).get(MODE_PATH[1], 'thread') != 'thread':
            raise ValueError('reference config is not thread')
        if config_differences(before, config) != [MODE_PATH]:
            raise ValueError('configs must differ only in model_device_pool.speech_campplus_execution_mode')
    else:
        if (before.get('model_device_pool', {}).get(MODE_PATH[1], 'thread') != 'process'
                or config.get('model_device_pool', {}).get(MODE_PATH[1], 'thread') != 'process'
                or prefetch_depth(before) != 0 or prefetch_depth(config) not in (1,5)):
            raise ValueError('prefetch requires same-process depth0 reference -> paired depth1 or5')
        if config_differences(before, config) != PREFETCH_PATHS:
            raise ValueError('prefetch configs must differ in exactly the two paired waiting-depth keys')
    actual_ids = [str(row['id']) for row in report.get('input_identity', [])]
    if actual_ids != current['selected_ids']: raise ValueError('reference plan and actual members differ')
    if identity is not None and report.get('input_identity') != identity:
        raise ValueError('reference actual PCM/WAV identity differs')
    rounds = report.get('rounds', [])
    if len(rounds) != 4 or [r.get('warmup') for r in rounds] != [True,False,False,False]:
        raise ValueError('thread reference must contain one warm and three steady rounds')
    if any(r.get('status') != 'completed' for r in rounds): raise ValueError('thread reference round incomplete')
    warm_passed = all(report.get('cold_vs_steady', {}).get(k, {}).get('passed') for k in ('qwen','nn'))
    if comparison == 'prefetch':
        warm_passed = warm_passed or all(rounds[0].get('quality', {}).get(k, {}).get('passed') for k in ('qwen','nn'))
    if not warm_passed: raise ValueError('reference warm/steady quality missing')
    if any(not all(r.get('quality', {}).get(k, {}).get('passed') for k in ('qwen','nn')) for r in rounds[1:]):
        raise ValueError('thread reference steady quality missing')
    # A cam-mode process0 report may retain the thread quality_reference. Prefetch
    # must compare its actual process0 steady outputs, never that inherited baseline.
    reference = ({'qwen':rounds[1].get('members'), 'nn':rounds[1].get('original_nn_cpu_results')}
                 if comparison == 'prefetch' else report.get('quality_reference'))
    if not isinstance(reference, dict) or not reference.get('qwen') or not reference.get('nn'):
        raise ValueError('thread reference original quality evidence missing')
    qwen_quality(None, reference['qwen'], current['selected_ids'])
    nn_quality(None, reference['nn'], [{'duration_s':d} for d in current['selected_durations_s']])
    return copy.deepcopy(reference)


def plan(args):
    config = read_config(args.config)
    q, pool = config.get('qwen_asr', {}), config.get('model_device_pool', {})
    mode = pool.get(MODE_PATH[1], 'thread')
    comparison = getattr(args,'comparison','cam-mode')
    if comparison not in ('cam-mode','prefetch'): raise ValueError('unknown comparison')
    if mode not in ('thread', 'process'): raise ValueError('CAM mode must be thread/process')
    if (q.get('execution_mode', 'thread') != 'process' or q.get('local_device') != 'npu:7'
            or q.get('local_instance_count', 1) != 1 or q.get('local_dtype', 'bfloat16') != 'bfloat16'):
        raise ValueError('component validation requires single original BF16 process Qwen on NPU7')
    if any(pool.get(key) != ['npu:7']*8 for key in ('speech_campplus_devices', 'emotion2vec_devices')):
        raise ValueError('component validation requires actual CAM8/emo8 on NPU7')
    if comparison == 'cam-mode':
        if mode == 'thread' and args.reference is not None: raise ValueError('fresh thread baseline must not reference another job')
        if mode == 'process' and args.reference is None: raise ValueError('process requires same-ROOT formal thread --reference')
    else:
        if mode != 'process': raise ValueError('prefetch comparison requires process CAM')
        depth = prefetch_depth(config)
        if depth == 0 and args.reference is not None: raise ValueError('fresh process depth0 baseline must not reference another job')
        if depth in (1,5) and args.reference is None: raise ValueError('prefetch depth1/5 requires same-source process depth0 --reference')
    body = json.loads(args.manifest.read_text())
    rows = selected_rows(body.get('segments', []) if isinstance(body, dict) else body, args.segments)
    if sum(1 <= row['duration_s'] <= 10 for row in rows) < 8:
        raise ValueError('selected segments cannot exercise all eight emo instances')
    result = {'benchmark': 'formal_cam', 'benchmark_root': str(ROOT.resolve()), 'status': 'planned',
              'execution_mode': mode, 'comparison':comparison, 'prefetch_depths':{key:pool.get(key,0)for key in PREFETCH_KEYS}, 'policy': 'original_math_and_sync_unchanged', 'segments': args.segments,
              'selected_ids': [str(row['id']) for row in rows], 'selected_durations_s':[row['duration_s'] for row in rows], 'config_path': str(args.config.resolve()),
              'config_sha256': digest(args.config), 'manifest_sha256': digest(args.manifest),
              'source_sha256': source_identity(ROOT), 'warmup_rounds': 1, 'steady_rounds': 3,
              'watchdog_s': args.watchdog_s, 'ports_started': False,
              'manifest_declared_scope': body.get('scope') if isinstance(body, dict) else None,
              'manifest_coverage': body.get('coverage_per_source') if isinstance(body, dict) else None,
              'scope': 'fixed 16/128 actual segments, not complete-audio or all corpus dynamic-shape acceptance',
              'observation': 'original completed CPU Tensor[1,192] / emo7+labels; parent thread envelopes, no stream/sync patch',
              'resource_scope': 'parent allocator excludes CAM/Qwen children; child allocator excludes CANN; per-PID smi separately; IPC extra RSS',
              'benchmark_overrides': {'qwen_retry_count': 0, 'qwen_batch_size': args.segments, 'per_audio_window': 16, 'global_window': 128},
              'reference_path': str(args.reference.resolve()) if args.reference else None,
              'reference_sha256': digest(args.reference) if args.reference else None}
    if args.reference:
        validate_reference(json.loads(args.reference.read_text()), result, config)
        result['allowed_config_difference'] = ('.'.join(MODE_PATH) if comparison == 'cam-mode' else ['.'.join(path)for path in PREFETCH_PATHS])
        if comparison == 'prefetch': result['reference_actual_output_phase'] = 'process-depth0/round-1'
    return result


class CPUOutputCapture:
    """Wrap the exported generate_local attribute, including Cython modules; no factory/NPU hooks."""
    def __init__(self, module, pools, rows, mode, *, event_limit=8192, result_limit=4096):
        self.module, self.pools, self.rows, self.mode = module, pools, rows, mode
        self.models = {id(model): (name, index) for name, pool in pools.items() for index, model in enumerate(pool.models)}
        self.original = module.generate_local; self.originals = []
        self.phase = 'preparation'; self.lock = threading.Lock(); self.sequence = 0
        self.events = deque(maxlen=event_limit); self.results = deque(maxlen=result_limit)
        self.event_total = self.result_total = 0

    def install(self):
        def observe(model, device, **kwargs):
            tag = self.models.get(id(model))
            if tag is None: return self.original(model, device, **kwargs)
            name, index = tag; instance = f'{name}[{index}]'; phase = self.phase
            with self.lock: self.sequence += 1; sequence = self.sequence
            start, cpu = time.monotonic_ns(), time.thread_time_ns()
            common = {'phase': phase, 'model_instance': instance, 'sequence': sequence,
                      'native_thread_id': threading.get_native_id(), 'observation_kind': 'parent_CPU_call_envelope_not_NPU_stream'}
            self.event(dict(common, event='call_started', monotonic_ns=start, thread_cpu_ns=cpu))
            error = None
            try:
                output = self.original(model, device, **kwargs)
            except BaseException as exc:
                error = type(exc).__name__; raise
            finally:
                end, end_cpu = time.monotonic_ns(), time.thread_time_ns()
                self.event(dict(common, event='call_finished_after_original_sync', monotonic_ns=end, thread_cpu_ns=end_cpu,
                                wall_s=(end-start)/1e9, calling_thread_cpu_s=(end_cpu-cpu)/1e9, error_type=error))
            capture_start = time.monotonic_ns()
            self.capture(model, name, instance, phase, sequence, kwargs, output)
            self.event(dict(common, event='CPU_capture_completed', monotonic_ns=time.monotonic_ns(),
                            capture_wall_s=(time.monotonic_ns()-capture_start)/1e9))
            return output
        self.wrapper = observe
        self.module.generate_local = observe
        self.originals = [(self.module, 'generate_local', self.original)]
        return self

    def event(self, value):
        with self.lock: self.event_total += 1; self.events.append(value)

    def capture(self, model, name, instance, phase, sequence, kwargs, output):
        from scripts.benchmark_npu_stream_policy import pcm_identity, cpu_vector
        import torch
        if not isinstance(output, list) or len(output) != 1: raise ValueError('original NN result envelope differs')
        payload = output[0]
        if name == 'speech_campplus':
            tensor = payload['spk_embedding']
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu'
                    or tensor.dtype != torch.float32 or list(tensor.shape) != [1, 192]):
                raise ValueError('original CAM result must be CPU float32 Tensor[1,192]')
            vector = cpu_vector(tensor)
        else:
            vector = cpu_vector(payload['scores'])
            if len(vector) != 7 or len(payload.get('labels', [])) != 7: raise ValueError('original emotion7 output missing')
        record = {'model': name, 'model_instance': instance, 'mode': self.mode, 'phase': phase, 'sequence': sequence,
                  **pcm_identity(kwargs['input'], kwargs.get('fs', 16000)), 'vector': vector, 'actual_vector_size': len(vector),
                  'labels': payload.get('labels'), 'projection9_available': False if name == 'emotion2vec' else None}
        if name == 'speech_campplus' and getattr(model, 'execution_mode', 'thread') == 'process':
            state = model.snapshot(); done = state.get('last_done') or {}
            if (not done.get('ok') or not done.get('execution_drained') or done.get('pcm_sha256') != record['pcm_float32_sha256']
                    or done.get('samples') != record['samples'] or done.get('actor', {}).get('pid') != (state.get('worker_identity') or {}).get('pid')):
                raise ValueError('formal CAM DONE/PCM/actor evidence missing')
            record['cam_worker'] = {key: copy.deepcopy(state.get(key)) for key in
                                    ('worker_identity', 'last_done', 'allocator', 'last_parent_rpc_wall_s', 'last_request_wire_bytes', 'last_response_wire_bytes', 'last_audio_bytes')}
        with self.lock: self.result_total += 1; self.results.append(record)

    def observe_guards(self):
        return self  # intentionally no Python/Cython guard-factory rewriting

    def phase_results(self, phase):
        with self.lock: values = copy.deepcopy([row for row in self.results if row['phase'] == phase])
        nn_quality(None, values, self.rows)
        for name in ('speech_campplus', 'emotion2vec'):
            tags = {row['model_instance'] for row in values if row['model'] == name}
            if tags != {f'{name}[{index}]' for index in range(8)}: raise ValueError('not all eight actual NN instances executed: '+name)
        return values

    def snapshot(self):
        with self.lock:
            return {'mode': self.mode, 'events': copy.deepcopy(list(self.events)), 'results': copy.deepcopy(list(self.results)),
                    'event_total': self.event_total, 'result_total': self.result_total,
                    'dropped_events': max(0, self.event_total-len(self.events)), 'dropped_results': max(0, self.result_total-len(self.results)),
                    'cpu_guard_observation': {'dropped': 0, 'supported': False, 'reason': 'no source-only instrumentation'},
                    'scope': 'parent CPU original-call envelopes; actual CAM child actor/allocator only in per-result cam_worker.last_done'}

    def restore(self):
        if self.originals:
            if self.module.generate_local is not self.wrapper: raise RuntimeError('generate_local changed outside owned observer')
            self.module.generate_local = self.original
            self.originals.clear()


def verified_rows(rows, count, planned_ids):
    result = selected_rows(rows, count)
    if [str(row['id']) for row in result] != list(planned_ids):
        raise ValueError('actual selected members differ from plan before loading')
    return result


class OwnedClient:
    """Begin NN ownership cleanup before awaiting the real Qwen close."""
    def __init__(self, delegate, begin_nn_close):
        self.delegate, self.begin_nn_close = delegate, begin_nn_close
    def __getattr__(self, name): return getattr(self.delegate, name)
    async def aclose(self):
        self.begin_nn_close()
        await self.delegate.aclose()


def validate_batches(result, rows):
    if result.get('status') != 'completed': return
    batches = result.get('batches', [])
    expected = {(str(row['request_id']), int(row['segment_index'])) for row in rows}
    if len(batches) != 1: raise ValueError('expected one actual Qwen batch')
    batch = batches[0]
    members = batch.get('dispatched_members', [])
    actual = {(str(row['request_id']), int(row['segment_index'])) for row in members}
    attempts = batch.get('dispatch_attempts', [])
    if (batch.get('actual_dispatched_size') != len(rows) or len(members) != len(rows) or actual != expected
            or batch.get('status') != 'completed' or len(attempts) != 1
            or attempts[0].get('actual_dispatched_size') != len(rows) or attempts[0].get('status') != 'completed'):
        raise ValueError('actual Qwen batch/count/attempt ownership differs')


def check_cam_ready(model, report):
    state = model.snapshot(); ready = state.get('ready') or {}; actor = ready.get('actor') or {}
    if (not state.get('loaded') or state.get('worker_alive') is not True or not state.get('healthy')
            or ready.get('test_double') is not False or ready.get('device') != 'npu:7'
            or ready.get('precision') != 'fp32' or ready.get('math_mode') != 'fp32_ops_hf32_disabled'
            or actor.get('pid') != (state.get('worker_identity') or {}).get('pid') or actor.get('device_index') != 7):
        raise ValueError('actual CAM child READY/FP32/NPU7 missing')
    storage = ready.get('storage_identity') or {}
    if (not all(storage.get(k) for k in ('available','all_floating_fp32','all_parameters_on_device','all_floating_buffers_on_device'))
            or not storage.get('parameter_tensor_count') or not storage.get('floating_tensor_count')
            or (ready.get('operator_adaptations') or {}).get('campplus_segment_pooling') is not True):
        raise ValueError('actual CAM FP32/device/adaptation storage evidence missing')
    for key in ('worker_entry_identity', 'factory_identity'):
        item = ready.get(key) or {}; path = Path(item.get('module_origin', '/missing')).resolve()
        if not path.is_relative_to(ROOT.resolve()) or report['source_sha256'].get(str(path)) != item.get('module_sha256'):
            raise ValueError('actual CAM child artifact differs from frozen ROOT: '+key)
    return state


def verify_normal_workers(states, count, prefix):
    if len(states) != count: raise RuntimeError(prefix+' worker count differs')
    for state in states:
        if (not state.get('resource_reaped') or state.get('normal_exit_ok') is not True or state.get('forced_cleanup')
                or state.get('worker_exitcode') != 0 or state.get('worker_alive')):
            raise RuntimeError(prefix+' worker did not exit normally')


async def execute(args, report):
    os.environ['CONFIG_PATH'] = str(args.config.resolve())
    from core.config import settings
    from core import models, local_inference
    from core.qwen_asr_client import QwenAsrClient
    from core.qwen_asr_process import LocalQwenProcessRuntime
    from core.qwen_asr_queue import RequestConcurrency
    from core.async_utils import await_completion
    from scripts.benchmark_qwen_batch_curve import read_manifest, run_round, resource_sample, enhanced_workload
    from scripts.benchmark_qwen_process_nn_streams import input_identity, run_policy
    from utils.qwen_response import parse_qwen_text
    if settings.model_device_speech_campplus_execution_mode != report['execution_mode']:
        raise ValueError('actual Settings CAM mode differs from frozen plan')
    if report.get('comparison') == 'prefetch':
        if any(getattr(settings,key) != report['prefetch_depths'][key] for key in PREFETCH_KEYS):
            raise ValueError('actual Settings prefetch depths differ from frozen plan')
    report['runtime_artifacts'] = {name: str(Path(importlib.import_module('core.'+name).__file__).resolve()) for name in
                                   ('cam_process', 'models', 'local_inference', 'config', 'qwen_asr_process', 'qwen_asr_client', 'qwen_asr_runtime', 'qwen_batch_scheduler')}
    if any(report['source_sha256'].get(path) != digest(path) for path in report['runtime_artifacts'].values()):
        raise ValueError('loaded runtime origin differs from frozen artifact')
    rows = verified_rows(read_manifest(args.manifest), args.segments, report['selected_ids'])
    report['input_identity'] = input_identity(rows)
    reference = validate_reference(json.loads(args.reference.read_text()), report, read_config(args.config), identity=report['input_identity']) if args.reference else None
    pools = {}; cam_models = []; holder = {}; nn_close = None
    async def prepare():
        load_started = time.monotonic()
        await models.load_v3_models_if_needed(True, True)
        report['nn_load_elapsed_s'] = time.monotonic()-load_started
        pools.update({name: models.get_model_pool(name) for name in ('speech_campplus', 'emotion2vec')})
        for name, pool in pools.items():
            pool.refresh_status()
            if len(pool.models) != 8 or models.local_model_status[name]['instances'] != 8:
                raise ValueError('requires eight actual healthy instances: '+name)
            if any(m is None or m.device != 'npu:7' for m in pool.models): raise ValueError('NN device mismatch')
        cam_models.extend(pools['speech_campplus'].models)
        if any(getattr(m, 'execution_mode', 'thread') != report['execution_mode'] for m in cam_models):
            raise ValueError('actual ModelPool runner mode differs from Settings')
        if report['execution_mode'] == 'process': report['cam_loaded_workers'] = [check_cam_ready(m, report) for m in cam_models]
        return pools
    def make_policy(loaded):
        capture = CPUOutputCapture(local_inference, loaded, rows, report['execution_mode']).install()
        holder['capture'] = capture
        return capture
    def begin_nn_close():
        nonlocal nn_close
        if nn_close is None: nn_close = asyncio.create_task(models.close_v3_model_pools())
    def make_client():
        return OwnedClient(QwenAsrClient(local_instance_count=1, execution_mode='process', runtime_factory=LocalQwenProcessRuntime,
                                        retry_count=0, max_inference_batch_size=args.segments), begin_nn_close)
    async def close_pools():
        begin_nn_close(); await await_completion(nn_close)
        if any(any(m is not None for m in pool.models) for pool in pools.values()): raise RuntimeError('NN resources still loaded')
    async def measured_round(client, values, gate, output, **kwargs):
        result = await run_round(client, values, gate, output, **kwargs)
        validate_batches(result, values)
        return result
    def caches(loaded):
        return {f'emotion2vec[{i}]': {str(k): {'shape': list(v.shape), 'dtype': str(v.dtype), 'device': str(v.device)}
                                    for k, v in getattr(m.model, 'alibi_biases', {}).items()}
                for i, m in enumerate(loaded['emotion2vec'].models)}
    bindings = SimpleNamespace(prepare_pools=prepare, make_policy=make_policy, make_client=make_client,
        fence=lambda loaded: None, cache_shapes=caches, close_pools=close_pools,
        pool_status=lambda: copy.deepcopy(models.local_model_status), round_fn=measured_round,
        gate_factory=lambda: RequestConcurrency(16,128), sample=resource_sample, parser=parse_qwen_text, enhanced=enhanced_workload)
    try:
        await run_policy(rows, args.output, report, bindings, reference)
    finally:
        report['cam_final_workers'] = copy.deepcopy(models.local_model_status['speech_campplus'].get('model_workers', []))
        if 'capture' in holder: persist(args.output/'nn-observation-final.json', holder['capture'].snapshot())
        persist(args.output/'summary.json', report)
    if report['execution_mode'] == 'process': verify_normal_workers(report['cam_final_workers'], 8, 'CAM')
    verify_normal_workers(report['closed_qwen'].get('model_workers', []), 1, 'Qwen')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','manifest','output'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--comparison', choices=('cam-mode','prefetch'), default='cam-mode')
    parser.add_argument('--segments', type=int, choices=(16,128), default=128)
    parser.add_argument('--watchdog-s', type=float, default=420)
    parser.add_argument('--execute', action='store_true'); parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not math.isfinite(args.watchdog_s) or not 5 <= args.watchdog_s <= 420: parser.error('watchdog must be5..420')
    for name in ('config','manifest','output','reference'):
        if getattr(args, name) is not None: setattr(args, name, getattr(args, name).resolve())
    expected = plan(args)
    if not args.execute: print(json.dumps(expected, ensure_ascii=True, indent=2)); return 0
    keys = ('source_sha256','config_sha256','manifest_sha256','selected_ids','reference_sha256','comparison')
    if not args.worker:
        args.output.mkdir(parents=True, exist_ok=False); persist(args.output/'summary.json', expected)
    if args.worker:
        report = json.loads((args.output/'summary.json').read_text())
        if any(report[key] != expected[key] for key in keys): raise ValueError('frozen identity changed before worker')
        async def work():
            import asyncio
            loop = asyncio.get_running_loop(); task = asyncio.current_task()
            for number in (signal.SIGINT,signal.SIGTERM): loop.add_signal_handler(number, task.cancel)
            try: await execute(args, report)
            finally:
                for number in (signal.SIGINT,signal.SIGTERM): loop.remove_signal_handler(number)
        try:
            import asyncio
            asyncio.run(work())
        except BaseException as exc: report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        finally:
            try:
                after = plan(args)
                report['source_changed_during_job'] = any(report[key] != after[key] for key in keys)
            except BaseException as exc:
                report['source_changed_during_job'] = True
                report['identity_recheck_error'] = f'{type(exc).__name__}: {exc}'
            if report['source_changed_during_job']: report['status'] = 'source_changed'
            persist(args.output/'summary.json', report)
        return int(report.get('status') != 'completed' or report.get('cleanup',{}).get('status') != 'passed')
    command = [sys.executable, str(Path(__file__).resolve()), '--config', str(args.config), '--manifest', str(args.manifest),
               '--output', str(args.output), '--comparison', args.comparison, '--segments', str(args.segments), '--watchdog-s', str(args.watchdog_s), '--execute', '--worker']
    if args.reference: command += ['--reference', str(args.reference)]
    owner = supervise(command, args); report = json.loads((args.output/'summary.json').read_text()); report['supervisor'] = owner
    if owner['returncode'] or owner['term_sent'] or owner['forced_kill'] or owner['cleanup_status'] != 'passed': report['status'] = 'failed'
    persist(args.output/'summary.json', report)
    return int(report.get('status') != 'completed' or report.get('cleanup',{}).get('status') != 'passed')


if __name__ == '__main__': raise SystemExit(main())
