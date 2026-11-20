"""单Qwen模型spawn worker，同HF权重/精度/原同步，不创建HTTP服务。

默认factory由client的execution_mode选择；设备沿用已配置逻辑npu:N及CANN可见映射。
snapshot只读缓存；普通推理/取消等待真实设备读回，只有显式closing grace允许owned abort。
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import multiprocessing
import os
import pickle
import re
from pathlib import Path
import signal
import sys
import threading
import time
import uuid


from core.qwen_asr_runtime import LocalModelUnavailable


class QwenWorkerUnavailableError(LocalModelUnavailable):
    pass


class RemoteRuntimeError(RuntimeError):
    pass


def _identity(pid):
    """Linux进程出生身份；不以kill(pid,0)替代PID复用检查。"""
    fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
    ticks = fields[19]
    boot = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime '))
    return {'pid':pid, 'start_ticks':ticks, 'create_time':boot + int(ticks) / os.sysconf('SC_CLK_TCK')}


def _send(connection, value):
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    connection.send_bytes(payload)
    return len(payload)


def _receive(connection):
    payload = connection.recv_bytes()
    return pickle.loads(payload), len(payload)  # private trusted Pipe, never network input


def _rss_mb(pid):
    try:
        return int(Path(f'/proc/{pid}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')/1048576
    except (OSError, ValueError, IndexError):
        return None


def _model_identity(model_dir):
    root = Path(model_dir).resolve()
    return {'model_dir':str(root), 'metadata_sha256':{
        name:hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in ('config.json','model.safetensors.index.json') if (root/name).is_file()},
        'weight_hash_scope':'metadata_only_not_full_weights'}


def _factory_identity(path):
    module_name, qualname = path.split(':',1)
    spec = importlib.util.find_spec(module_name)
    origin = Path(spec.origin) if spec and spec.origin else None
    return {'factory':path, 'module_origin':str(origin) if origin else None,
            'module_sha256':hashlib.sha256(origin.read_bytes()).hexdigest() if origin and origin.is_file() else None}


def _allocator(runtime, *, reset=False):
    api = getattr(runtime, 'api', None)
    metadata = {'source':'worker_torch_npu_allocator' if api is not None else 'unavailable',
                'worker_pid':os.getpid(), 'worker_rss_mb':_rss_mb(os.getpid())}
    if reset:
        try:
            api.set_device(runtime.device)
            api.reset_peak_memory_stats(runtime.device)
            runtime._qwen_worker_peak_scope = 'last_transcribe_batch'
        except Exception:
            runtime._qwen_worker_peak_scope = 'not_reset_unknown_or_includes_load'
    metadata['peak_scope'] = getattr(runtime, '_qwen_worker_peak_scope', 'not_reset_unknown_or_includes_load')
    for name in ('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved'):
        try:
            metadata[name+'_mb'] = getattr(api,name)(runtime.device)/1048576
        except Exception:
            metadata[name+'_mb'] = None
    return metadata


def _error(exc):
    # 仅轻量值跨进程，绝不pickle异常/traceback/cause或线程frame/PCM引用。
    from core.qwen_asr_runtime import LocalUnsupportedLanguage
    category = ('unavailable' if isinstance(exc,LocalModelUnavailable) else
                'language' if isinstance(exc,LocalUnsupportedLanguage) else 'runtime')
    return {'type':type(exc).__name__, 'message':str(exc), 'code':getattr(exc, 'code', None), 'category':category}


def _restore_error(value):
    from core.qwen_asr_runtime import LocalModelUnavailable, LocalUnsupportedLanguage
    kinds = {'LocalModelUnavailable':LocalModelUnavailable, 'LocalUnsupportedLanguage':LocalUnsupportedLanguage,
             'RuntimeError':RuntimeError, 'ValueError':ValueError, 'TimeoutError':TimeoutError,
             'QwenWorkerUnavailableError':QwenWorkerUnavailableError}
    cls = kinds.get(value['type'], RemoteRuntimeError)
    if value.get('category')=='unavailable':cls=LocalModelUnavailable
    elif value.get('category')=='language':cls=LocalUnsupportedLanguage
    error = cls(value['type']+': '+value['message'] if cls is RemoteRuntimeError else value['message'])
    if value.get('code') is not None:
        error.code = value['code']
    return error


def _state(runtime):
    snapshot = runtime.snapshot()
    return {'snapshot':snapshot, 'last_batch_timings':copy.deepcopy(getattr(runtime, 'last_batch_timings', None)),
            'worker_allocator':_allocator(runtime)}


def _worker(connection, runtime_args, factory_path, launch_id, launch_config):
    runtime, close_attempted = None, False
    try:
        module_name, qualname = factory_path.split(':',1)
        factory = importlib.import_module(module_name)
        for attribute in qualname.split('.'):
            factory = getattr(factory, attribute)
        runtime = factory(*runtime_args)
        if launch_config.get('continuous_options') is not None:
            runtime.ensure_loaded()
        ready_config = dict(launch_config, factory_identity=_factory_identity(factory_path),
                            model_identity=_model_identity(runtime_args[0]),
                            model_dir=str(getattr(runtime, 'model_dir', runtime_args[0])),
                            device=str(getattr(runtime, 'device', runtime_args[1])),
                            dtype=str(getattr(runtime, 'dtype', runtime_args[2])),
                            max_new_tokens=int(getattr(runtime, 'max_new_tokens', runtime_args[3])))
        _send(connection, {'kind':'ready', 'launch_id':launch_id, 'worker_identity':_identity(os.getpid()),
                           'launch_config':dict(ready_config,worker_entry_identity=_factory_identity(__name__+':_worker')),
                           'worker_entry_module_file':__file__, 'state':_state(runtime),
                           'preload_observation':{
                               'runtime_module_file':getattr(sys.modules.get(type(runtime).__module__), '__file__', None),
                               'torch_imported':'torch' in sys.modules,
                               'torch_npu_imported':'torch_npu' in sys.modules,
                               'runtime_api_present':getattr(runtime, 'api', None) is not None,
                               'loaded':bool(getattr(runtime, 'loaded', False)),
                               'npu_initialization_observation':
                                   'no_torch_no_torch_npu_no_runtime_api' if 'torch' not in sys.modules
                                   and 'torch_npu' not in sys.modules and getattr(runtime,'api',None) is None else 'not_queried'}})
        if launch_config.get('continuous_options') is not None:
            from core.qwen_continuous_process import worker_loop
            close_attempted = True
            worker_loop(connection, runtime, launch_id, launch_config['continuous_options'])
            return
        while True:
            request, _ = _receive(connection)
            rpc_id, operation = request['rpc_id'], request['operation']
            if request.get('launch_id') != launch_id:
                raise QwenWorkerUnavailableError('worker request belongs to another launch')
            member_ids = request.get('member_ids')
            try:
                if operation == 'ensure_loaded':
                    runtime.ensure_loaded()
                    result = None
                elif operation == 'transcribe_batch':
                    _allocator(runtime, reset=True)
                    values = runtime.transcribe_batch(request['audios'], request['languages'])
                    # 结果索引与输入一一对应；单成员异常不升级为整批失败。
                    result = [{'error':_error(value)} if isinstance(value, Exception) else {'text':value} for value in values]
                    if any('text' in value and not isinstance(value['text'], str) for value in result):
                        raise TypeError('worker runtime returned a non-text/non-error member')
                elif operation == 'close':
                    close_attempted = True
                    runtime.close()
                    result = None
                else:
                    raise ValueError('unsupported private Qwen worker operation')
                _send(connection, {'rpc_id':rpc_id, 'launch_id':launch_id, 'member_ids':member_ids,
                                   'ok':True, 'result':result, 'state':_state(runtime)})
            except Exception as exc:
                _send(connection, {'rpc_id':rpc_id, 'launch_id':launch_id, 'member_ids':member_ids,
                                   'ok':False, 'error':_error(exc), 'state':_state(runtime)})
            if operation == 'close':
                break
            request = values = result = None  # do not retain the previous PCM batch while idle
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            _send(connection, {'kind':'startup_error', 'launch_id':launch_id, 'error':_error(exc)})
        except (OSError, EOFError):
            pass
    finally:
        if runtime is not None and not close_attempted:
            try:
                runtime.close()
            except Exception:
                pass
        connection.close()


class LocalQwenProcessRuntime:
    def __init__(self, model_dir, device, dtype='bfloat16', max_new_tokens=256, *,
                 _runtime_factory=None, _startup_timeout_s=30, _close_timeout_s=30, _abort_timeout_s=5):
        if not isinstance(device,str) or re.fullmatch(r'npu:[0-9]+',device.strip().lower()) is None:
            raise ValueError('Qwen model worker requires configured npu:N')
        device='npu:'+str(int(device.strip().lower().split(':')[1]))
        if _startup_timeout_s <= 0 or _close_timeout_s <= 0:
            raise ValueError('worker lifecycle deadlines must be positive')
        self.model_dir, self.device, self.dtype = str(model_dir), str(device), str(dtype)
        self.max_new_tokens = int(max_new_tokens)
        self.last_batch_timings = None
        if _runtime_factory is not None and (not isinstance(_runtime_factory,str) or ':' not in _runtime_factory
                                             or '<' in _runtime_factory):
            raise ValueError('private test factory must be an importable module:qualname string')
        self._factory = _runtime_factory or 'core.qwen_asr_runtime:LocalQwenRuntime'
        self._launch_id = uuid.uuid4().hex
        self._launch_config = {'model_dir':self.model_dir,'device':self.device,'dtype':self.dtype,
                               'max_new_tokens':self.max_new_tokens,
                               'factory_identity':_factory_identity(self._factory),
                               'model_identity':_model_identity(self.model_dir),
                               'worker_entry_identity':_factory_identity(__name__+':_worker')}
        self._startup_timeout_s, self._close_timeout_s = _startup_timeout_s, _close_timeout_s
        self._rpc_lock, self._state_lock = threading.Lock(), threading.Lock()
        self._reap_lock = threading.RLock()
        self._birth_lock = threading.Lock()
        self._birth_in_progress = threading.Event()
        self._explicit_closing = False
        self._abort_timeout_s = _abort_timeout_s
        self._process = self._connection = None
        self._rpc_id, self._closed = 0, False
        self._close_error = None
        self._cache = {'device':self.device, 'dtype':self.dtype, 'loaded':False, 'loading':False,
                       'failure':None, 'memory_allocated_mb':0, 'memory_reserved_mb':0,
                       'model_worker':{'started':False, 'worker_pid':None, 'worker_create_time':None,
                           'worker_alive':False, 'closed':False, 'closing':False, 'birth_in_progress':False,
                           'cleanup_status':'not_started',
                           'resource_reaped':True, 'normal_exit_ok':None, 'forced_cleanup':False,
                           'runtime_kind':'original_hf_runtime' if _runtime_factory is None else 'cpu_test_double',
                           'start_method':'spawn', 'cann_environment_inherited':True, 'launch_id':self._launch_id,
                           'device_scope':'configured logical device; physical mapping follows deployment',
                           'launch_config':self._launch_config,
                           'ipc_memory_scope':'wire_payload_bytes_and_process_RSS_are_additional_to_main_PCM_budget'}}

    @property
    def failure(self):
        return self._cache['failure']

    @property
    def loaded(self):
        return self._cache['loaded']

    @property
    def loading(self):
        return self._cache['loading']

    def _update(self, **values):
        with self._state_lock:
            self._cache = dict(self._cache, **values)

    def _metadata(self, **values):
        with self._state_lock:
            self._cache = dict(self._cache, model_worker=dict(self._cache['model_worker'], **values))

    def _failed(self, reason, *, dead=False):
        self._update(failure=self.failure or str(reason), loading=False,
                     **({'loaded':False} if dead else {}))

    def _accept_state(self, state):
        incoming = dict(state['snapshot'])
        incoming.pop('model_worker', None)
        incoming['failure'] = self.failure or incoming.get('failure')
        self.last_batch_timings = copy.deepcopy(state['last_batch_timings'])
        self._update(**incoming)
        self._metadata(worker_allocator=copy.deepcopy(state['worker_allocator']), parent_rss_mb=_rss_mb(os.getpid()))

    def snapshot(self):
        # 仅本地缓存及OS存活检查；无Pipe操作，亦不会懒启动。
        process = self._process
        if (process is not None and process.pid is not None
                and not self._birth_in_progress.is_set()
                and not process.is_alive() and not self._closed):
            self._failed(f'Qwen owned worker exited unexpectedly: exitcode={process.exitcode}', dead=True)
            self._metadata(worker_alive=False)
        with self._state_lock:
            return copy.deepcopy(self._cache)

    def _wait_response(self, process, connection, deadline=None):
        while True:
            if connection.poll(.02):
                return _receive(connection)
            if not process.is_alive():
                raise QwenWorkerUnavailableError(f'Qwen owned worker died: exitcode={process.exitcode}')
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('Qwen owned worker lifecycle deadline exceeded')

    def _start(self):
        child = None
        cleanup_needed = False
        try:
            # birth锁覆盖关闭复核、spawn及出生身份登记；不覆盖RPC读取/模型推理。
            # abort先发布_closed，再有界等待此锁，因此pid尚未出版不能被当作已死亡。
            with self._birth_lock:
                if self._closed:
                    raise QwenWorkerUnavailableError("Qwen model worker already closed")
                if self.failure:
                    raise QwenWorkerUnavailableError(self.failure)
                if self._process is not None:
                    return
                cleanup_needed = True
                self._birth_in_progress.set()
                self._metadata(birth_in_progress=True, worker_alive=None, resource_reaped=False)
                try:
                    context = multiprocessing.get_context("spawn")
                    parent, child = context.Pipe(duplex=True)
                    process = context.Process(
                        target=_worker,
                        args=(child, (self.model_dir, self.device, self.dtype, self.max_new_tokens),
                              self._factory, self._launch_id, self._launch_config),
                        name="owned-qwen-" + self.device.replace(":", "-"),
                        daemon=False,
                    )
                    self._process, self._connection = process, parent
                    process.start()
                    child.close()
                    identity = _identity(process.pid)
                    self._owned_identity = identity
                    self._metadata(
                        started=True, worker_pid=process.pid,
                        worker_create_time=identity["create_time"],
                        worker_start_ticks=identity["start_ticks"], worker_alive=True,
                        cleanup_status="pending", resource_reaped=False, normal_exit_ok=None,
                    )
                finally:
                    self._metadata(birth_in_progress=False)
                    self._birth_in_progress.clear()
            # 关闭可能发生于Process.start期间。必须先登记PID再回收，不能发布一个晚生worker。
            if self._closed:
                raise QwenWorkerUnavailableError("Qwen worker birth interrupted by explicit closing")
            response, ready_bytes = self._wait_response(
                process, parent, time.monotonic() + self._startup_timeout_s
            )
            if (response.get("kind") != "ready"
                    or response.get("launch_id") != self._launch_id
                    or response.get("worker_identity") != identity
                    or response.get("launch_config") != self._launch_config):
                raise QwenWorkerUnavailableError(str(response.get("error", "worker did not acknowledge ready")))
            self._metadata(
                ready_wire_payload_bytes=ready_bytes, ready_verified=True,
                worker_ready_identity=response["worker_identity"],
                worker_entry_module_file=response["worker_entry_module_file"],
                ready_preload_observation=response["preload_observation"],
            )
            if self._closed:
                raise QwenWorkerUnavailableError("Qwen worker startup interrupted by explicit closing")
            self._accept_state(response["state"])
        except BaseException as exc:
            if not cleanup_needed:
                # 旧worker的failure/closed拒绝不等于新birth失败，不得在普通调用中偷偷kill。
                raise
            if child is not None:
                child.close()
            self._failed(f"Qwen worker startup failed: {type(exc).__name__}: {exc}")
            self._reap(force=True, deadline=time.monotonic() + self._close_timeout_s)
            raise QwenWorkerUnavailableError(self.failure) from None

    def _unrecoverable_rpc(self, process, reason):
        # 这不是普通推理timeout：通道或归属已失信，无法再取得可信DONE。
        # 先poison禁止复用，仅回收本实例；死亡未确认则当前线程/实际租约继续隔离持有。
        self._closed = True
        self._close_error = "Qwen unrecoverable RPC fault: " + str(reason)
        self._failed(self._close_error)
        self._metadata(rpc_unrecoverable=True, rpc_quarantined=True, rpc_fault=str(reason))
        outcome = self._reap(force=True, deadline=time.monotonic() + self._abort_timeout_s)
        if not outcome["resource_reaped"]:
            self._metadata(rpc_quarantine_waiting_death=True)
            while True:
                if not process.is_alive():
                    outcome = self._reap(force=False, deadline=time.monotonic() + self._abort_timeout_s)
                    if outcome["resource_reaped"]:
                        break
                # 不等RPC锁的显式shutdown abort仍可执行；不能因自己的收尾deadline假归还名额。
                time.sleep(.02)
        self._metadata(
            rpc_quarantined=False, rpc_quarantine_waiting_death=False,
            resource_reaped=True, normal_exit_ok=False, cleanup_status="failed",
            forced_cleanup=True, closed=True, close_error=self._close_error,
        )
        raise QwenWorkerUnavailableError(self.failure) from None

    def _rpc(self, operation, *, audios=None, languages=None, deadline=None):
        self._start()
        process, connection = self._process, self._connection
        self._rpc_id += 1
        request = {"rpc_id": self._rpc_id, "launch_id": self._launch_id, "operation": operation}
        if operation == "transcribe_batch":
            request.update(audios=audios, languages=languages, member_ids=list(range(len(audios))))
        try:
            started = time.monotonic()
            request_bytes = _send(connection, request)
            response, response_bytes = self._wait_response(process, connection, deadline)
            self._metadata(
                last_request_wire_payload_bytes=request_bytes, last_response_wire_payload_bytes=response_bytes,
                last_audio_bytes=sum(len(audio) for audio in audios) if audios is not None else 0,
                last_parent_rpc_wall_ms=(time.monotonic() - started) * 1000,
            )
            if (not isinstance(response, dict)
                    or response.get("rpc_id") != request["rpc_id"]
                    or response.get("launch_id") != self._launch_id
                    or response.get("member_ids") != request.get("member_ids")):
                raise QwenWorkerUnavailableError("Qwen worker response ownership mismatch")
            if (type(response.get("ok")) is not bool
                    or not isinstance(response.get("state"), dict)
                    or not {"snapshot", "last_batch_timings", "worker_allocator"} <= response["state"].keys()
                    or not isinstance(response["state"]["snapshot"], dict)
                    or not isinstance(response["state"]["worker_allocator"], dict)):
                raise QwenWorkerUnavailableError("Qwen worker response state is incomplete")
            if response["ok"] and operation == "transcribe_batch":
                values = response.get("result")
                if not isinstance(values, list) or len(values) != len(audios):
                    raise QwenWorkerUnavailableError("Qwen worker result cardinality mismatch")
                for value in values:
                    if not isinstance(value, dict) or ("text" in value) == ("error" in value):
                        raise QwenWorkerUnavailableError("Qwen worker member payload is incomplete")
                    if "text" in value and not isinstance(value["text"], str):
                        raise QwenWorkerUnavailableError("Qwen worker member text is invalid")
                    if "error" in value and (not isinstance(value["error"], dict)
                            or not {"type", "message"} <= value["error"].keys()):
                        raise QwenWorkerUnavailableError("Qwen worker member error is incomplete")
            if not response["ok"] and (not isinstance(response.get("error"), dict)
                    or not {"type", "message"} <= response["error"].keys()):
                raise QwenWorkerUnavailableError("Qwen worker error payload is incomplete")
        except (OSError, EOFError, QwenWorkerUnavailableError, TimeoutError) as exc:
            self._unrecoverable_rpc(process, f"{type(exc).__name__}: {exc}")
        # 仅接受归属、基数均可信的完整响应。正常模型error payload仍按原分类抛出/重试。
        self._accept_state(response["state"])
        if not response["ok"]:
            raise _restore_error(response["error"]) from None
        return response["result"]

    def ensure_loaded(self):
        with self._rpc_lock:
            self._update(loading=True)
            try:
                self._rpc('ensure_loaded')
            finally:
                self._update(loading=False)

    def transcribe_batch(self, audios, languages=None):
        # 不设推理RPC超时，不在取消/超时时提前归还活跃worker。
        with self._rpc_lock:
            self.last_batch_timings = None
            self._update(loading=not self.loaded)
            try:
                values = self._rpc('transcribe_batch', audios=list(audios),
                                   languages=list(languages) if languages is not None else None)
                return [_restore_error(value['error']) if 'error' in value else value['text'] for value in values]
            finally:
                self._update(loading=False)

    def transcribe(self, audio, language=None):
        value = self.transcribe_batch([audio], [language])[0]
        if isinstance(value, Exception):
            raise value
        return value

    def _reap(self, *, force, deadline):
        def unconfirmed(reason):
            self._failed(reason)
            outcome = {
                "worker_alive": self._cache["model_worker"].get("worker_alive"),
                "resource_reaped": False, "normal_exit_ok": False,
                "forced_cleanup": force, "cleanup_status": "failed",
                "birth_in_progress": self._birth_in_progress.is_set(),
                "coordination_error": reason,
            }
            self._metadata(**outcome)
            return outcome

        # 独立于RPC读锁；若birth/reap协调超时，只能报告未确认死亡，不能提前归还名额。
        acquired_reap = self._reap_lock.acquire(timeout=max(0, deadline - time.monotonic()))
        if not acquired_reap:
            return unconfirmed("Qwen reap coordination deadline exceeded")
        try:
            acquired_birth = self._birth_lock.acquire(timeout=max(0, deadline - time.monotonic()))
            if not acquired_birth:
                return unconfirmed("Qwen birth coordination deadline exceeded")
            try:
                return self._reap_locked(force=force, deadline=deadline)
            finally:
                self._birth_lock.release()
        finally:
            self._reap_lock.release()

    def _reap_locked(self, *, force, deadline):
        process = self._process
        if process is None or process.pid is None:
            if self._connection is not None:
                self._connection.close()
            self._connection = None
            abnormal = force or self._close_error is not None
            outcome={'worker_alive':False, 'resource_reaped':True, 'normal_exit_ok':not abnormal,
                     'forced_cleanup':abnormal, 'worker_exitcode':None,
                     'cleanup_status':'failed' if abnormal else 'passed'}
            self._metadata(**outcome)
            return outcome
        # 仅对本代理最初登记且出生身份仍吻合的worker发信号。
        for signum in ((signal.SIGTERM, signal.SIGKILL) if force else ()):
            if not process.is_alive():
                break
            try:
                if _identity(process.pid) != self._owned_identity:
                    raise QwenWorkerUnavailableError('owned worker PID/creationtime identity mismatch; refusing signal')
                os.kill(process.pid, signum)
                self._metadata(last_cleanup_signal=int(signum))
            except ProcessLookupError:
                break
            except Exception as exc:
                self._failed(f'Qwen owned cleanup failed: {exc}')
                outcome={'worker_alive':True, 'resource_reaped':False, 'normal_exit_ok':False,
                         'forced_cleanup':force, 'worker_exitcode':process.exitcode, 'cleanup_status':'failed'}
                self._metadata(**outcome)
                return outcome
            process.join(timeout=max(0, min((deadline-time.monotonic())/2, 1)))
        process.join(timeout=max(0, deadline-time.monotonic()))
        alive = process.is_alive()
        outcome={'worker_alive':alive, 'resource_reaped':not alive,
                 'normal_exit_ok':not alive and process.exitcode==0 and not force,
                 'forced_cleanup':force, 'worker_exitcode':process.exitcode}
        outcome['cleanup_status']='passed' if outcome['normal_exit_ok'] else 'failed'
        self._metadata(**outcome)
        if alive:
            self._failed('Qwen owned worker survived bounded cleanup')
        else:
            self._update(loaded=False, loading=False)
            if self._connection is not None:
                self._connection.close()
            self._connection = None
        return outcome

    def begin_closing(self):
        # 只发布显式关闭状态；不得等待RPC读锁，也不会启动worker。
        self._explicit_closing = True
        self._metadata(closing=True)

    def abort_owned(self, reason="explicit closing grace exceeded"):
        if not self._explicit_closing:
            raise RuntimeError("worker abort requires explicit owner closing")
        metadata = self._cache["model_worker"]
        if self._closed and self._close_error is None and metadata.get("normal_exit_ok"):
            return {"resource_reaped": True, "aborted": False, "reason": "already normally closed"}
        # 先禁止后续birth，再有界协调正在出版的PID。只有join确认死亡才能报告回收成功。
        self._closed = True
        self._close_error = "forced Qwen worker cleanup: " + str(reason)
        self._failed(self._close_error)
        outcome = self._reap(force=True, deadline=time.monotonic() + self._abort_timeout_s)
        self._metadata(
            closed=True, close_error=self._close_error, forced_cleanup=True,
            normal_exit_ok=False, cleanup_status="failed",
        )
        if not outcome["resource_reaped"]:
            raise QwenWorkerUnavailableError("owned worker death not confirmed; ownership remains active")
        return dict(outcome, aborted=True, normal_exit_ok=False, forced_cleanup=True, cleanup_status="failed")

    def close(self):
        self.begin_closing()
        deadline = time.monotonic()+self._close_timeout_s
        acquired = self._rpc_lock.acquire(timeout=self._close_timeout_s)
        if not acquired:
            self._closed = True
            self._close_error = 'Qwen close deadline while owned RPC active; forced cleanup requested'
            self._failed(self._close_error)
            self._reap(force=True, deadline=deadline+.5)
            self._metadata(closed=True, close_error=self._close_error)
            raise QwenWorkerUnavailableError(self._close_error)
        try:
            if self._closed:
                if self._close_error:
                    raise QwenWorkerUnavailableError(self._close_error)
                return
            self._closed = True
            if self._process is None or self._process.pid is None:
                self._reap(force=False, deadline=deadline)
                self._metadata(closed=True)
                return
            try:
                if self._process.is_alive():
                    self._rpc_id += 1
                    _send(self._connection, {'rpc_id':self._rpc_id, 'launch_id':self._launch_id, 'operation':'close'})
                    response, _ = self._wait_response(self._process, self._connection, deadline)
                    if (response.get('rpc_id') != self._rpc_id or response.get('launch_id') != self._launch_id
                            or not response.get('ok')):
                        raise QwenWorkerUnavailableError('owned worker close ACK failed')
                    self._accept_state(response['state'])
                outcome=self._reap(force=False, deadline=deadline)
                if not outcome['normal_exit_ok']:
                    raise QwenWorkerUnavailableError('owned worker abnormal exit: '+str(outcome))
            except Exception as exc:
                self._close_error=f'Qwen close failed: {type(exc).__name__}: {exc}'
                self._failed(self._close_error)
                # A nonzero already-exited worker stays recorded as nonzero; do not relabel it forced.
                self._reap(force=self._process.is_alive(), deadline=deadline+.5)
                self._metadata(normal_exit_ok=False, cleanup_status='failed')
            self._metadata(closed=True, close_error=self._close_error)
            if self._close_error:
                raise QwenWorkerUnavailableError(self._close_error)
        finally:
            self._rpc_lock.release()
