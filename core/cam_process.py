"""Owned CAM-only spawn runner; original FP32 model and CPU Tensor embedding contract.

No model, process or NPU is touched by construction/status. A private owned Pipe carries
PCM and float32[192] bytes, never Qwen text. Ordinary model errors leave original health
semantics intact; broken ownership/transport is quarantined until actual child death.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import re
import sys
import threading
import time
import uuid

from core.local_inference import InferenceRunner, InstanceFailure

_DEFAULT_FACTORY = 'core.local_inference:TorchNpuRunner'
_MATH_MODE = 'fp32_ops_hf32_disabled'


class _BirthCancelled(InstanceFailure):
    """Explicit closing withdrew a never-created child while its grace is valid."""


_BOOT_TIME = None
_BOOT_TIME_LOCK = threading.Lock()


def _boot_time():
    # 同一宿主本次启动的btime不变；首次真实身份观察时读取，导入不触碰procfs。
    global _BOOT_TIME
    with _BOOT_TIME_LOCK:
        if _BOOT_TIME is None:
            _BOOT_TIME = next(int(x.split()[1]) for x in Path('/proc/stat').read_text().splitlines()
                              if x.startswith('btime '))
        return _BOOT_TIME


def _identity(pid):
    # PID/start_ticks/SID每次重新验证，缓存btime不缓存worker身份。
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    boot = _boot_time()
    ticks = fields[19]
    return {'pid': pid, 'start_ticks': ticks, 'create_time': boot + int(ticks) / os.sysconf('SC_CLK_TCK'), 'sid': os.getsid(pid)}


def _rss_mb(pid):
    try:
        return int(Path(f'/proc/{pid}/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 1048576
    except (OSError, ValueError, IndexError):
        return None


def _artifact(module_name, qualname):
    module = sys.modules.get(module_name)
    path = getattr(module, '__file__', None)
    if path is None:
        spec = importlib.util.find_spec(module_name)
        path = getattr(spec, 'origin', None)
    origin = Path(path).resolve() if path else None
    if origin is None or not origin.is_file():
        raise ValueError(f'CAM worker artifact not found: {module_name}')
    return {'module': module_name, 'qualname': qualname, 'module_origin': str(origin),
            'module_sha256': hashlib.sha256(origin.read_bytes()).hexdigest()}


def _model_identity(directory):
    root = Path(directory).resolve()
    names = ('config.yaml', 'campplus_cn_en_common.pt')
    if any(not (root / name).is_file() for name in names):
        raise ValueError('CAM requires original config.yaml and campplus_cn_en_common.pt')
    return {'model_dir': str(root), 'sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names},
            'hash_scope': 'full original CAM config and weight bytes'}


def _send(pipe, value):
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    pipe.send_bytes(payload)
    return len(payload)


def _receive(pipe):
    payload = pipe.recv_bytes()
    return pickle.loads(payload), len(payload)  # only our private owned Pipe, never network input


def _storage_identity(runner, device):
    network = getattr(runner, 'model', None)
    if network is None or not hasattr(network, 'named_parameters'):
        return {'available': False, 'reason': 'explicit CPU test double has no native NPU model storage'}
    counts = {}; floating_ok = parameters_ok = buffers_ok = True
    floating_count = parameter_count = 0
    for category, values in (('parameter', network.named_parameters()), ('buffer', network.named_buffers())):
        group = counts.setdefault(category, {})
        for _name, tensor in values:
            dtype, actual_device = str(tensor.dtype), str(tensor.device)
            key = dtype + '@' + actual_device
            group[key] = group.get(key, 0) + 1
            floating = tensor.is_floating_point() or tensor.is_complex()
            if floating:
                floating_count += 1
                floating_ok = floating_ok and dtype == 'torch.float32'
            if category == 'parameter':
                parameter_count += 1
                parameters_ok = parameters_ok and actual_device == device
            elif floating:
                buffers_ok = buffers_ok and actual_device == device
    return {'available': True, 'dtype_device_counts': counts, 'floating_tensor_count': floating_count,
            'parameter_tensor_count': parameter_count, 'all_floating_fp32': floating_ok,
            'all_parameters_on_device': parameters_ok, 'all_floating_buffers_on_device': buffers_ok,
            'scope': 'storage metadata only; no weights read and no new tensor computation'}


def _actor(runner, device, test_double):
    api = getattr(runner, 'api', None)
    if api is None:
        return {'available': False, 'test_double': test_double, 'configured_device': device,
                'scope': 'explicit CPU toy, no claimed NPU execution'}
    api.set_device(device)
    stream = api.current_stream(device)
    return {'available': True, 'pid': os.getpid(), 'native_thread_id': threading.get_native_id(),
            'device_index': int(stream.device_index), 'stream_id': int(stream.stream_id),
            'npu_stream_handle': int(stream.npu_stream),
            'test_double': test_double,
            'scope': 'child process/device namespace; bare handles are not comparable across PIDs'}


def _allocator(runner, device, *, reset=False):
    api = getattr(runner, 'api', None)
    if reset:
        try:
            api.set_device(device)
            api.reset_peak_memory_stats(device)
            runner._cam_peak_scope = 'last_generate_only'
        except Exception:
            runner._cam_peak_scope = 'not_reset_unknown_or_includes_load'
    result = {'worker_pid': os.getpid(), 'worker_rss_mb': _rss_mb(os.getpid()),
              'source': 'worker_torch_npu_allocator' if api is not None else 'unavailable_CPU_test_double',
              'scope': 'this CAM child allocator; excludes CANN/runtime/non-PyTorch allocations',
              'peak_scope': getattr(runner, '_cam_peak_scope', 'not_reset_unknown_or_includes_load')}
    for field in ('memory_allocated', 'memory_reserved', 'max_memory_allocated', 'max_memory_reserved'):
        try:
            result[field + '_bytes'] = int(getattr(api, field)(device))
        except Exception:
            result[field + '_bytes'] = None
    return result


def _error(exc):
    return {'type': type(exc).__name__, 'message': str(exc),
            'category': 'instance' if isinstance(exc, InstanceFailure) else 'model'}


def _restore_error(value):
    types = {'InstanceFailure': InstanceFailure, 'RuntimeError': RuntimeError, 'ValueError': ValueError,
             'TypeError': TypeError, 'TimeoutError': TimeoutError, 'AssertionError': AssertionError}
    cls = InstanceFailure if value.get('category') == 'instance' else types.get(value.get('type'), RuntimeError)
    return cls(value['message'] if value.get('type') in types else value.get('type', 'RemoteError') + ': ' + value['message'])


def _worker(pipe, launch, directory, device, factory_spec):
    """Top-level importable entry also used by protected .so spawn; no parent model exists."""
    runner = None; close_attempted = False
    try:
        import numpy as np
        module_name, qualname = factory_spec.split(':', 1)
        factory = importlib.import_module(module_name)
        for part in qualname.split('.'):
            factory = getattr(factory, part)
        runner = factory('speech_campplus', directory, device)
        test_double = factory_spec != _DEFAULT_FACTORY
        if test_double and not getattr(runner, '_cam_cpu_test_double', False):
            raise ValueError('private CAM factory must explicitly declare CPU test double')
        if getattr(runner, 'actual_precision', None) != 'fp32' or getattr(runner, 'math_mode', None) != _MATH_MODE:
            raise ValueError('CAM worker FP32/math mode mismatch')
        if str(getattr(runner, 'device', device)) != device:
            raise ValueError('CAM runner configured device mismatch')
        storage = _storage_identity(runner, device)
        actor = _actor(runner, device, test_double)
        adaptations = copy.deepcopy(getattr(runner, 'operator_adaptations', None))
        if not test_double and (not isinstance(adaptations, dict) or adaptations.get('campplus_segment_pooling') is not True):
            raise ValueError('native CAM pooling adaptation was not reported active')
        if not test_double and (not storage['available'] or not storage.get('floating_tensor_count')
                or not storage.get('parameter_tensor_count') or not all(storage.get(k) for k in
                ('all_floating_fp32', 'all_parameters_on_device', 'all_floating_buffers_on_device'))
                or not actor['available'] or actor['device_index'] != int(device.split(':')[1])):
            raise ValueError('native CAM parameter/buffer FP32/device/actor audit failed')
        born = _identity(os.getpid())
        _send(pipe, {'kind': 'READY', 'launch': launch, 'identity': born, 'device': device,
                     'factory_identity': _artifact(module_name, qualname), 'worker_entry_identity': _artifact(__name__, '_worker'),
                     'model_identity': _model_identity(directory), 'precision': runner.actual_precision,
                     'math_mode': runner.math_mode, 'storage_identity': storage, 'actor': actor, 'operator_adaptations': adaptations,
                     'test_double': test_double, 'allocator': _allocator(runner, device)})
        while True:
            request, request_bytes = _receive(pipe)
            operation = request['operation']
            if request['launch'] != launch or request['device'] != device:
                raise ValueError('foreign CAM launch/device request')
            response = {'kind': 'CLOSE_ACK' if operation == 'CLOSE' else 'DONE', 'launch': launch,
                        'rpc': request['rpc'], 'identity': born, 'device': device, 'ok': False}
            try:
                if operation == 'CLOSE':
                    close_attempted = True
                    runner.close()
                    response['ok'] = True
                elif operation == 'GENERATE':
                    payload = request['pcm_float32_bytes']
                    if request['fs'] != 16000 or len(payload) != request['samples'] * 4:
                        raise ValueError('invalid CAM PCM envelope')
                    array = np.frombuffer(payload, dtype='<f4').copy()
                    if not len(array) or not np.isfinite(array).all():
                        raise ValueError('invalid CAM PCM values')
                    response.update(pcm_sha256=hashlib.sha256(payload).hexdigest(), samples=len(array), sample_rate=16000)
                    _allocator(runner, device, reset=True)
                    started, cpu = time.monotonic(), time.thread_time()
                    try:
                        import torch
                        value = runner.generate(input=array, fs=16000)[0]['spk_embedding']
                        if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.dtype != torch.float32 or list(value.shape) != [1, 192]:
                            raise ValueError('original CAM result must be CPU torch.float32 Tensor[1,192]')
                        vector = value.detach().numpy()
                        if not np.isfinite(vector).all():
                            raise ValueError('CAM embedding contains nonfinite values')
                        response.update(ok=True, execution_drained=True, vector_float32_bytes=vector.astype('<f4', copy=False).tobytes(),
                                        result_shape=[1, 192], result_dtype='float32')
                    finally:
                        response.update(call_wall_s=time.monotonic() - started, calling_thread_cpu_s=time.thread_time() - cpu,
                                        actor=_actor(runner, device, test_double), allocator=_allocator(runner, device))
                    request = payload = array = value = vector = None
                else:
                    raise ValueError('unknown CAM operation')
            except Exception as exc:
                response['ok'] = False
                for field in ('vector_float32_bytes', 'result_shape', 'result_dtype'):
                    response.pop(field, None)
                # Native generate already fences on success. Only failures need an explicit
                # confirmation: a finally/fence exception must not look like drained model work.
                drained = True
                if operation == 'GENERATE':
                    api = getattr(runner, 'api', None)
                    fence = {'attempted': api is not None, 'status': 'CPU_test_double_no_device_work'}
                    if api is not None:
                        try:
                            from core.local_inference import synchronize_npu
                            synchronize_npu(api, device)
                            fence['status'] = 'exception_path_fence_completed'
                        except Exception as fence_error:
                            drained = False
                            fence.update(status='exception_path_fence_failed', error=_error(fence_error))
                            exc = InstanceFailure(f'CAM device work not confirmed drained: {fence_error}')
                    if isinstance(exc, InstanceFailure) or not getattr(runner, 'healthy', True):
                        runner.healthy = False; runner.failure = str(exc)
                        if not isinstance(exc, InstanceFailure): exc = InstanceFailure(str(exc))
                    response.update(execution_drained=drained, error_fence=fence)
                response['error'] = _error(exc)
            response.update(healthy=bool(getattr(runner, 'healthy', True)), failure=getattr(runner, 'failure', None),
                            request_wire_bytes=request_bytes)
            _send(pipe, response)
            if operation == 'CLOSE':
                break
            request = response = payload = array = value = vector = None
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as exc:
        try:
            _send(pipe, {'kind': 'STARTUP_ERROR', 'launch': launch, 'error': _error(exc)})
        except (OSError, EOFError):
            pass
    finally:
        if runner is not None and not close_attempted:
            try:
                runner.close()
            except Exception:
                pass
        pipe.close()


class CamProcessRunner(InferenceRunner):
    effective_backend = 'torch_npu'
    execution_mode = 'process'
    math_mode = _MATH_MODE

    def __init__(self, model_dir, device, *, startup_timeout_s=60, shutdown_grace_s=30, _factory_spec=None):
        if not isinstance(device, str) or re.fullmatch(r'npu:[0-9]+', device.strip().lower()) is None:
            raise ValueError('CAM model process requires configured npu:N')
        device = 'npu:' + str(int(device.strip().lower().split(':')[1]))
        for key, value in (('startup_timeout_s', startup_timeout_s), ('shutdown_grace_s', shutdown_grace_s)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(key + ' must be a positive finite number')
        if _factory_spec is not None and (not isinstance(_factory_spec, str) or ':' not in _factory_spec or '<' in _factory_spec):
            raise ValueError('CAM test factory must be an importable module:qualname string')
        self.model_dir, self.device = str(model_dir), device
        self.startup_timeout_s, self.shutdown_grace_s = float(startup_timeout_s), float(shutdown_grace_s)
        self._factory_spec = _factory_spec or _DEFAULT_FACTORY
        self._launch = uuid.uuid4().hex
        self._rpc_lock = threading.Lock(); self._birth_lock = threading.RLock(); self._reap_lock = threading.RLock()
        self._close_lock = threading.Lock(); self._state_lock = threading.Lock()
        self._process = self._pipe = None; self._born = None; self._rpc = 0
        self._closing = self._closed = self._abnormal = False; self._close_deadline = None
        self.healthy = True; self.failure = None; self.actual_precision = None
        self._cache = {'execution_mode': 'process', 'runtime_kind': 'original_torch_npu_fp32_cam' if self._factory_spec == _DEFAULT_FACTORY else 'cpu_test_double',
                       'worker_identity': None, 'worker_alive': False, 'birth_in_progress': False,
                       'resource_reaped': True, 'normal_exit_ok': None, 'forced_cleanup': False, 'active': False,
                       'loaded': False, 'closing': False, 'closed': False, 'healthy': True, 'failure': None,
                       'close_deadline_monotonic_s': None, 'abnormal_reasons': [], 'ready': None,
                       'allocator': None, 'last_done': None, 'last_close_ack': None, 'operator_adaptations': None,
                       'last_request_wire_bytes': None, 'last_response_wire_bytes': None, 'last_audio_bytes': None,
                       'IPC_scope': 'PCM input copy and 768-byte embedding; child/IPC RSS is additional to pipeline PCM budget',
                       'allocation_scope': 'parent allocator excludes CAM child; unavailable is not zero'}

    def _update(self, **fields):
        with self._state_lock:
            self._cache = dict(self._cache, **fields)

    def cached_health(self):
        """仅租约热路径使用；不做OS观察，也不能作为恢复已知死亡的证据。"""
        with self._state_lock:
            return {key: self._cache[key] for key in
                    ('loaded', 'closing', 'closed', 'healthy', 'resource_reaped')}

    def snapshot(self):
        with self._state_lock:
            value = copy.deepcopy(self._cache)
        born = value.get('worker_identity')
        if value['resource_reaped']:
            value['worker_alive'] = False
        elif born is None:
            value['worker_alive'] = None
        else:
            try:
                actual = _identity(born['pid'])
                state = Path(f"/proc/{born['pid']}/stat").read_text().rsplit(')', 1)[1].split()[0]
                value['worker_alive'] = actual == born and state not in ('Z', 'X')
            except (FileNotFoundError, ProcessLookupError):
                value['worker_alive'] = False
            except (OSError, ValueError, IndexError):
                value['worker_alive'] = None
        if value['loaded'] and value['worker_alive'] is False and not value['resource_reaped']:
            value['healthy'] = False
            value['failure'] = value.get('failure') or 'registered CAM child is no longer alive; not yet reaped'
        value['parent_rss_mb'] = _rss_mb(os.getpid())
        return value

    @property
    def operator_adaptations(self):
        with self._state_lock:
            return copy.deepcopy(self._cache['operator_adaptations'])

    @property
    def memory(self):
        return self.snapshot()

    @property
    def memory_info(self):
        return self.snapshot()

    def request_close(self):
        """Nonblocking closing publication. It never spawns/RPCs or waits for birth/RPC locks."""
        with self._state_lock:
            self._closing = True
            if self._close_deadline is None:
                self._close_deadline = time.monotonic() + self.shutdown_grace_s
            self._cache = dict(self._cache, closing=True, close_deadline_monotonic_s=self._close_deadline)

    def _remaining_grace(self):
        with self._state_lock:
            deadline = self._close_deadline
        return None if deadline is None else max(0., deadline - time.monotonic())

    def _wait(self, process, pipe, deadline=None):
        while True:
            if pipe.poll(.02):
                return _receive(pipe)
            if not process.is_alive():
                raise EOFError(f'CAM child exited: {process.exitcode}')
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('CAM startup deadline expired')
            remaining = self._remaining_grace()
            if remaining is not None and remaining <= 0:
                raise TimeoutError('CAM explicit shutdown grace expired')

    def _mark_abnormal(self, reason, *, unhealthy=True):
        with self._state_lock:
            self._abnormal = True
            reasons = self._cache['abnormal_reasons']
            if str(reason) not in reasons:
                reasons = (reasons + [str(reason)])[-16:]
            if unhealthy:
                self.healthy = False
                self.failure = self.failure or str(reason)
            self._cache = dict(self._cache, abnormal_reasons=reasons, normal_exit_ok=False,
                               healthy=self.healthy, failure=self.failure)

    def _reap(self, force):
        """Never waits for RPC lock. Unresolved birth/death keeps resource_reaped false."""
        with self._reap_lock:
            if not self._birth_lock.acquire(timeout=.05):
                return False
            try:
                process = self._process
                if process is None:
                    return self._cache['resource_reaped']
                if process.pid is None:
                    return False
                if force:
                    for action in ('terminate', 'kill'):
                        if not process.is_alive():
                            break
                        try:
                            current = _identity(process.pid)
                        except (OSError, ValueError, IndexError):
                            process.join(0)
                            if not process.is_alive():
                                break
                            return False
                        if self._born is None:
                            # This Process owns this child, but birth registration may have raised.
                            try:
                                stat = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()
                                parent_pid = int(stat[1])
                            except (OSError, ValueError, IndexError):
                                process.join(0)
                                if not process.is_alive(): break
                                return False
                            if parent_pid != os.getpid() or current['sid'] != os.getsid(0):
                                return False
                            self._born = current
                            self._update(worker_identity=current)
                        if current != self._born:
                            return False
                        try:
                            getattr(process, action)()
                        except (OSError, ProcessLookupError):
                            process.join(0)
                            if not process.is_alive(): break
                            return False
                        self._update(forced_cleanup=True)
                        process.join(.5)
                else:
                    remaining = self._remaining_grace()
                    process.join(min(.05, remaining) if remaining is not None else .05)
                if process.is_alive():
                    return False
                process.join(0)
                exitcode = process.exitcode
                process.close(); self._process = None
                if self._pipe is not None:
                    self._pipe.close(); self._pipe = None
                self._update(resource_reaped=True, worker_alive=False, worker_exitcode=exitcode, loaded=False)
                return True
            finally:
                self._birth_lock.release()

    def _confirmed_death(self, force):
        while not self._reap(force):
            if not force and self._remaining_grace() == 0:
                force = True
                self._mark_abnormal('CAM explicit shutdown grace expired')
            time.sleep(.01)

    def _poison(self, reason):
        self._mark_abnormal(reason)
        self.request_close()
        self._update(quarantined=True)
        self._confirmed_death(True)
        with self._state_lock:
            self._closed = True
            self._cache = dict(self._cache, quarantined=False, closed=True, loaded=False, active=False)
        raise InstanceFailure('CAM ownership/transport failed after confirmed child death: ' + str(reason))

    def _start_locked(self):
        with self._state_lock:
            if self._closing or self._closed or not self.healthy:
                raise InstanceFailure('CAM runner closing/unhealthy: ' + str(self.failure))
        if self._process is not None:
            return
        child = None
        try:
            expected_model = _model_identity(self.model_dir)
            module, qualname = self._factory_spec.split(':', 1)
            expected_factory = _artifact(module, qualname)
            expected_entry = _artifact(__name__, '_worker')
            with self._birth_lock:
                if self._closing or self._closed:
                    if self._process is None and self._cache['resource_reaped'] and self._remaining_grace() != 0:
                        raise _BirthCancelled('CAM birth cancelled before process creation')
                    raise TimeoutError('CAM closing grace expired before birth')
                context = multiprocessing.get_context('spawn')
                parent, child = context.Pipe(duplex=True)
                process = context.Process(target=_worker, args=(child, self._launch, self.model_dir, self.device, self._factory_spec),
                                          daemon=False, name='cam-fp32-model-worker')
                self._process, self._pipe = process, parent
                self._update(resource_reaped=False, birth_in_progress=True)
                try:
                    process.start(); child.close(); child = None
                    self._born = _identity(process.pid)
                    self._update(worker_identity=self._born, worker_alive=True)
                finally:
                    self._update(birth_in_progress=False)
            ready, wire_bytes = self._wait(process, parent, time.monotonic() + self.startup_timeout_s)
            if (ready.get('kind') != 'READY' or ready.get('launch') != self._launch or ready.get('identity') != self._born
                    or ready.get('device') != self.device or ready.get('model_identity') != expected_model
                    or ready.get('factory_identity') != expected_factory or ready.get('worker_entry_identity') != expected_entry
                    or ready.get('precision') != 'fp32' or ready.get('math_mode') != _MATH_MODE
                    or ready.get('test_double') != (self._factory_spec != _DEFAULT_FACTORY)):
                raise ValueError('CAM READY identity/artifact/model/FP32 mismatch: ' + str(ready.get('error')))
            actor, storage = ready['actor'], ready['storage_identity']
            if self._factory_spec == _DEFAULT_FACTORY and (not actor.get('available') or actor.get('pid') != self._born['pid']
                    or actor.get('device_index') != int(self.device.split(':')[1]) or not storage.get('available')
                    or not storage.get('parameter_tensor_count') or not storage.get('floating_tensor_count')
                    or not all(storage.get(k) for k in ('all_floating_fp32', 'all_parameters_on_device', 'all_floating_buffers_on_device'))):
                raise ValueError('CAM READY actual FP32/device storage mismatch')
            if self._factory_spec == _DEFAULT_FACTORY and (not isinstance(ready.get('operator_adaptations'), dict)
                    or ready['operator_adaptations'].get('campplus_segment_pooling') is not True):
                raise ValueError('CAM READY pooling adaptation mismatch')
            with self._state_lock:
                if self._closing or self._closed:
                    raise InstanceFailure('CAM startup interrupted by explicit closing')
                self.actual_precision = 'fp32'
                self._cache = dict(self._cache, ready=ready, allocator=ready['allocator'], loaded=True,
                                   last_response_wire_bytes=wire_bytes, healthy=self.healthy, operator_adaptations=ready.get('operator_adaptations'))
        except _BirthCancelled:
            # This exact guard precedes Pipe/Process creation. No uncertain birth or
            # real startup failure is allowed through this expected-close branch.
            raise InstanceFailure('CAM birth cancelled before process creation') from None
        except BaseException as exc:
            if child is not None:
                child.close()
            # A failed Process.start with no PID created no child; do not invent resource death otherwise.
            with self._birth_lock:
                if self._process is not None and self._process.pid is None:
                    self._process.close(); self._process = None
                    if self._pipe is not None:
                        self._pipe.close(); self._pipe = None
                    self._update(resource_reaped=True)
            self._poison(f'startup: {type(exc).__name__}: {exc}')

    def ensure_loaded(self):
        with self._rpc_lock:
            self._start_locked()

    def _validate_response(self, response, rpc, kind):
        if (not isinstance(response, dict) or response.get('kind') != kind or response.get('launch') != self._launch
                or response.get('rpc') != rpc or response.get('identity') != self._born or response.get('device') != self.device):
            raise ValueError('CAM response identity/RPC/device mismatch')
        if type(response.get('ok')) is not bool or type(response.get('healthy')) is not bool:
            raise ValueError('CAM response health/result envelope malformed')
        if response['ok'] and 'error' in response:
            raise ValueError('CAM success response also contains an error')
        if not response['ok'] and any(k in response for k in ('vector_float32_bytes', 'result_shape', 'result_dtype')):
            raise ValueError('CAM failed response retains successful embedding payload')
        if not response['ok'] and (not isinstance(response.get('error'), dict)
                or not all(isinstance(response['error'].get(k), str) for k in ('type', 'message', 'category'))
                or response['error'].get('category') not in ('model', 'instance')):
            raise ValueError('CAM response error envelope malformed')

    def generate(self, *, input, fs=16000, **kwargs):
        import numpy as np
        if not isinstance(input, np.ndarray) or input.dtype != np.float32 or input.ndim != 1 or fs != 16000 or not len(input) or not np.isfinite(input).all():
            raise ValueError('CAM requires original finite float32 16k mono PCM')
        value = np.ascontiguousarray(input, dtype='<f4')
        payload = value.tobytes(); digest = hashlib.sha256(payload).hexdigest()
        with self._rpc_lock:
            self._start_locked()
            process, pipe = self._process, self._pipe
            self._rpc += 1; rpc = self._rpc
            self._update(active=True, last_rpc=rpc, last_audio_bytes=len(payload))
            started = time.monotonic()
            try:
                wire_bytes = _send(pipe, {'operation': 'GENERATE', 'launch': self._launch, 'rpc': rpc, 'device': self.device,
                                          'fs': fs, 'samples': len(value), 'pcm_float32_bytes': payload})
                self._update(last_request_wire_bytes=wire_bytes)
                response, wire_bytes = self._wait(process, pipe)
                self._validate_response(response, rpc, 'DONE')
                if (response.get('pcm_sha256') != digest or response.get('samples') != len(value) or response.get('sample_rate') != 16000):
                    raise ValueError('CAM DONE PCM identity mismatch')
                if type(response.get('execution_drained')) is not bool:
                    raise ValueError('CAM DONE missing device-drain confirmation')
                if not response['execution_drained']:
                    self._update(last_done=response, allocator=response.get('allocator'), last_response_wire_bytes=wire_bytes)
                    raise InstanceFailure('CAM worker could not confirm device work drained')
                if response['ok']:
                    if (response.get('result_shape') != [1, 192] or response.get('result_dtype') != 'float32'
                            or not isinstance(response.get('vector_float32_bytes'), bytes) or len(response['vector_float32_bytes']) != 768):
                        raise ValueError('CAM DONE embedding shape/dtype/length mismatch')
                    vector = np.frombuffer(response.pop('vector_float32_bytes'), dtype='<f4').copy().reshape(1, 192)
                    if not np.isfinite(vector).all():
                        raise ValueError('CAM DONE embedding is nonfinite')
                actor = response.get('actor', {})
                if self._factory_spec == _DEFAULT_FACTORY and (response['ok'] or actor) and (actor.get('pid') != self._born['pid'] or actor.get('device_index') != int(self.device.split(':')[1])):
                    raise ValueError('CAM DONE actual actor mismatch')
                self._update(last_done=response, allocator=response.get('allocator'), last_response_wire_bytes=wire_bytes,
                             last_parent_rpc_wall_s=time.monotonic() - started)
                if not response['healthy'] or response.get('error', {}).get('category') == 'instance':
                    self.healthy = False; self.failure = response.get('failure') or response.get('error', {}).get('message')
                    self._update(healthy=False, failure=self.failure)
            except BaseException as exc:
                self._poison(f'RPC: {type(exc).__name__}: {exc}')
            finally:
                self._update(active=False)
            if not response['ok']:
                raise _restore_error(response['error'])
            import torch
            return [{'spk_embedding': torch.from_numpy(vector)}]

    def close(self):
        self.request_close()
        with self._close_lock:
            with self._state_lock:
                if self._closed and self._cache['resource_reaped']:
                    if self._abnormal:
                        raise InstanceFailure('CAM previous abnormal shutdown remains abnormal: ' + str(self.failure))
                    return
            remaining = self._remaining_grace()
            acquired = self._rpc_lock.acquire(timeout=remaining)
            ack_ok = False
            try:
                with self._state_lock:
                    already_reaped = self._closed and self._cache['resource_reaped']
                if already_reaped:
                    if self._abnormal:
                        raise InstanceFailure('CAM abnormal shutdown remains abnormal')
                    return
                if acquired and self._process is None:
                    ack_ok = True  # never born; closing forbids a late birth
                elif acquired:
                    self._rpc += 1
                    _send(self._pipe, {'operation': 'CLOSE', 'launch': self._launch, 'device': self.device, 'rpc': self._rpc})
                    response, wire_bytes = self._wait(self._process, self._pipe)
                    self._validate_response(response, self._rpc, 'CLOSE_ACK')
                    self._update(last_close_ack=response, last_response_wire_bytes=wire_bytes)
                    ack_ok = response['ok']
                    if not ack_ok:
                        self._mark_abnormal('CAM close ACK reported error: ' + response['error']['message'])
                else:
                    self._mark_abnormal('CAM explicit close grace expired while RPC/birth remained active')
                # Valid negative ACK still gets a chance to exit naturally; no false forced cleanup.
                valid_ack = acquired and (self._process is None or self._cache.get('last_close_ack') is not None)
                self._confirmed_death(not valid_ack)
                code = self._cache.get('worker_exitcode', 0)
                if code != 0:
                    self._mark_abnormal(f'CAM worker exited with code {code}')
                if not ack_ok:
                    self._mark_abnormal('CAM did not complete a successful explicit close ACK')
            except BaseException as exc:
                self._mark_abnormal(f'close: {type(exc).__name__}: {exc}')
                if not self._cache['resource_reaped']:
                    self._confirmed_death(True)
            finally:
                if acquired:
                    self._rpc_lock.release()
                with self._state_lock:
                    self._closed = bool(self._cache['resource_reaped'])
                    self._cache = dict(self._cache, closed=self._closed, loaded=False, active=False,
                                       normal_exit_ok=not self._abnormal if self._closed else False)
            if self._abnormal:
                raise InstanceFailure('CAM close failed after confirmed child death: ' + str(self.failure))
