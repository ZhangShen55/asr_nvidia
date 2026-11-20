"""Multiplexed private RPC for the single-owner continuous HF worker."""
from concurrent.futures import Future
import threading
import time
from core.qwen_asr_process import (LocalQwenProcessRuntime, _send, _receive, _state,
    _error, _restore_error, QwenWorkerUnavailableError)
from core.qwen_continuous_runtime import ContinuousCapacityRejected, ContinuousGenerationError


def worker_loop(connection, runtime, launch_id, options):
    from collections import deque
    from core.qwen_continuous_runtime import ContinuousHFAdapter
    adapter = ContinuousHFAdapter(runtime, online_repetition_enabled=options['online_repetition_enabled'],
                                  online_repetition_config=options.get('online_repetition_config'))
    waiting = deque()
    known = set()
    reservations = {}
    closing = False
    decode_cursor = 0
    last_snapshot = 0.0
    last_resources = 0.0

    def emit(key, value):
        nonlocal last_snapshot, last_resources
        reservations.pop(key, None)
        payload = {'error':_error(value)} if isinstance(value, Exception) else {'text':value}
        if isinstance(value, Exception):
            payload['error'].update(reason=getattr(value,'reason',None), request_id=key[0],segment_id=key[1])
        payload['tokens'] = getattr(adapter, 'completed_tokens', {}).pop(key, None)
        now = time.monotonic()
        response = dict(kind='result', launch_id=launch_id, key=key, result=payload)
        if now-last_snapshot >= 1.0 or (not waiting and not adapter.active):
            response['continuous'] = adapter.snapshot()
            last_snapshot = now
        if now-last_resources >= 1.0:
            response['state'] = _state(runtime)
            last_resources = now
        _send(connection, response)

    try:
        while True:
            if not waiting and not adapter.active and not closing:
                connection.poll(None)
            # Drain bounded IPC commands so a stream of SUBMIT cannot starve decode.
            for _ in range(options['max_active']+8):
                if not connection.poll():
                    break
                msg, _ = _receive(connection)
                if msg.get('launch_id') != launch_id:
                    raise RuntimeError('continuous launch ownership mismatch')
                operation = msg['operation']
                if operation == 'submit':
                    key = tuple(msg['key'])
                    if key in known:
                        continue
                    from core.qwen_asr_runtime import decode_wav_bytes, LocalUnsupportedLanguage
                    method = runtime.processor.apply_transcription_request
                    function = getattr(method, '__func__', method)
                    validator = getattr(function, '__globals__', {}).get('_prepare_language_inputs')
                    if validator is not None:
                        try:
                            validator(msg['language'], 1)
                        except ValueError as exc:
                            emit(key, LocalUnsupportedLanguage(str(exc)))
                            continue
                    frames = (len(decode_wav_bytes(msg['audio'])) + 159)//160 + 100
                    msg['estimated_frames'] = frames
                    estimate = (frames+7)//8 + runtime.max_new_tokens + 256
                    if closing or len(waiting)+len(adapter.active) >= options['max_active'] or sum(reservations.values())+estimate > options['max_batch_tokens'] or frames > options['max_batch_audio_frames']:
                        emit(key, ContinuousCapacityRejected('continuous capacity rejected before acceptance'))
                    else:
                        known.add(key)
                        reservations[key] = estimate
                        waiting.append(dict(request_id=key[0], segment_id=key[1], audio=msg['audio'], language=msg['language'], estimated_frames=frames))
                elif operation == 'cancel':
                    key = tuple(msg['key'])
                    for item in list(waiting):
                        if (item['request_id'], item['segment_id']) == key:
                            waiting.remove(item)
                            emit(key, ContinuousGenerationError('cancelled'))
                            break
                    for pair in adapter.cancel(key):
                        emit(*pair)
                elif operation == 'release_request':
                    request_id = msg['request_id']
                    if any(k[0] == request_id for k in adapter.active) or any(r['request_id'] == request_id for r in waiting):
                        raise RuntimeError('cannot release request while work is active')
                    known.difference_update(k for k in list(known) if k[0] == request_id)
                elif operation == 'close':
                    closing = True
                else:
                    raise RuntimeError('unsupported continuous command')
            if waiting:
                # Bounded prefill chunk, followed by decode, rather than unbounded new arrivals.
                batch = []
                max_frames = 0
                while waiting and len(batch) < options['prefill_batch_size']:
                    next_frames = max(max_frames, waiting[0]['estimated_frames'])
                    if batch and next_frames*(len(batch)+1) > options['max_batch_audio_frames']:
                        break
                    max_frames = next_frames
                    batch.append(waiting.popleft())
                try:
                    runtime.api.set_device(runtime.device)
                    for key, value in adapter.prefill(batch):
                        emit(key, value)
                except Exception as exc:
                    for item in batch:
                        key = (item['request_id'], item['segment_id'])
                        for pair in adapter.fail(key, exc, phase='prefill'):
                            emit(*pair)
            if adapter.active:
                try:
                    all_keys = list(adapter.active)
                    decode_cursor %= len(all_keys)
                    rotated = all_keys[decode_cursor:]+all_keys[:decode_cursor]
                    keys = rotated[:options['max_batch_size']]
                    decode_cursor += len(keys)
                    for key, value in adapter.decode(keys):
                        emit(key, value)
                except Exception as exc:
                    # Device errors invalidate the affected active set, never retry its work here.
                    for key in list(adapter.active):
                        for pair in adapter.fail(key, exc, phase='decode'):
                            emit(*pair)
            if closing and not waiting and not adapter.active:
                runtime.close()
                _send(connection, dict(kind='closed', launch_id=launch_id, state=_state(runtime)))
                return
    finally:
        runtime.close()


class LocalQwenContinuousProcessRuntime(LocalQwenProcessRuntime):
    def __init__(self, *args, max_active=128, max_batch_size=128,
                 online_repetition_enabled=False, online_repetition_config=None, prefill_batch_size=16,
                 max_batch_tokens=80000, max_batch_audio_frames=115200, **kwargs):
        kwargs.setdefault('_startup_timeout_s', 120)
        super().__init__(*args, **kwargs)
        if not (1 <= max_batch_size <= 384 and 1 <= max_active <= 384):
            raise ValueError('continuous batch and active capacity must independently be within 1..384')
        self._launch_config['continuous_options'] = dict(max_active=max_active,
            max_batch_size=max_batch_size, online_repetition_enabled=online_repetition_enabled,
            online_repetition_config=online_repetition_config,
            prefill_batch_size=min(prefill_batch_size, max_batch_size),
            max_batch_tokens=max_batch_tokens, max_batch_audio_frames=max_batch_audio_frames)
        self._futures = {}
        self._send_lock = threading.Lock()
        self._reader = None
        self._close_ack = threading.Event()
        self._continuous_lock = threading.RLock()

    def ensure_loaded(self):
        # The legacy load RPC runs before switching the pipe to its sole reader.
        # Continuous worker requires loading as part of construction handshake below.
        with self._continuous_lock:
            if self._reader is not None:
                if self.failure:
                    raise QwenWorkerUnavailableError(self.failure)
                return
            with self._rpc_lock:
                self._start()
            self._reader = threading.Thread(target=self._read_continuous, daemon=True,
                                            name='qwen-continuous-reader')
            self._reader.start()

    def _read_continuous(self):
        try:
            while True:
                response, _ = self._wait_response(self._process, self._connection)
                if response.get('launch_id') != self._launch_id:
                    raise RuntimeError('continuous response launch mismatch')
                if response['kind'] == 'closed':
                    self._accept_state(response['state'])
                    self._close_ack.set()
                    return
                if response['kind'] != 'result':
                    raise RuntimeError('continuous response kind mismatch')
                key = tuple(response['key'])
                with self._continuous_lock:
                    future = self._futures.get(key)
                if future is None or future.done():
                    raise RuntimeError('continuous result ownership mismatch')
                if 'state' in response:
                    self._accept_state(response['state'])
                if 'continuous' in response:
                    self._update(continuous=response['continuous'])
                payload = response['result']
                future.generated_tokens = payload.get('tokens')
                if 'error' in payload:
                    value = payload['error']
                    if value['type'] == 'ContinuousCapacityRejected':
                        error = ContinuousCapacityRejected(value['message'])
                        # A rejected ID has not been accepted and may be retried.
                        with self._continuous_lock:
                            self._futures.pop(key, None)
                    elif value['type'] == 'ContinuousGenerationError':
                        error = ContinuousGenerationError(value.get('reason') or 'generation_error',
                            request_id=value.get('request_id'),segment_id=value.get('segment_id'))
                    else:
                        error = _restore_error(value)
                    error.reason = value.get('reason') or getattr(error,'reason','generation_error')
                    error.request_id = value.get('request_id')
                    error.segment_id = value.get('segment_id')
                    future.set_exception(error)
                else:
                    future.set_result(payload['text'])
        except BaseException as exc:
            self._failed('continuous reader failed: '+str(exc))
            # Confirm owned death before resolving outstanding ownership futures.
            try:
                self._unrecoverable_rpc(self._process, str(exc))
            except BaseException:
                pass
            with self._continuous_lock:
                futures = list(self._futures.values())
            for future in futures:
                if not future.done():
                    future.set_exception(QwenWorkerUnavailableError(self.failure))
            self._close_ack.set()

    def submit(self, audio, language=None, *, request_id, segment_id, attempt_id=0):
        with self._continuous_lock:
            if self._reader is None:
                raise QwenWorkerUnavailableError('continuous runtime must be preloaded before submit')
            if self._explicit_closing or self.failure:
                raise QwenWorkerUnavailableError(self.failure or 'continuous runtime closing')
            key = (str(request_id), str(segment_id))
            if key in self._futures:
                return self._futures[key]
            future = Future()
            self._futures[key] = future
        # Never hold the result-routing lock across potentially blocking Pipe IO.
        # The reader must drain worker results even when the command pipe is full.
        with self._send_lock:
            if self._explicit_closing:
                future.set_exception(QwenWorkerUnavailableError('continuous runtime closing'))
                return future
            try:
                _send(self._connection, dict(operation='submit', launch_id=self._launch_id,
                    key=key, attempt_id=attempt_id, audio=audio, language=language))
            except BaseException as exc:
                self._failed(str(exc))
                try:
                    self._unrecoverable_rpc(self._process, 'continuous send failed: '+str(exc))
                finally:
                    if not future.done():
                        future.set_exception(QwenWorkerUnavailableError(self.failure))
                raise
        return future

    def release_request(self, request_id):
        with self._continuous_lock:
            keys = [k for k in self._futures if k[0] == str(request_id)]
            if any(not self._futures[k].done() for k in keys):
                raise RuntimeError('cannot release active request')
        # Parent request is terminal; callers cannot submit more IDs for it.
        with self._send_lock:
            _send(self._connection, dict(operation='release_request', launch_id=self._launch_id,
                                        request_id=str(request_id)))
        with self._continuous_lock:
            for key in keys:
                self._futures.pop(key, None)

    def cancel_segment(self, request_id, segment_id):
        with self._send_lock:
            _send(self._connection, dict(operation='cancel', launch_id=self._launch_id,
                                        key=(str(request_id), str(segment_id))))

    cancel = cancel_segment

    def close(self):
        if self._closed:
            if self._close_error:
                raise QwenWorkerUnavailableError(self._close_error)
            return
        self.begin_closing()
        if self._reader is None:
            return super().close()
        try:
            with self._send_lock:
                _send(self._connection, dict(operation='close', launch_id=self._launch_id))
        except (OSError, EOFError) as exc:
            self.abort_owned('continuous close send failed: '+str(exc))
            raise QwenWorkerUnavailableError('continuous close send failed') from None
        if not self._close_ack.wait(self._close_timeout_s):
            self.abort_owned('continuous close deadline')
            raise QwenWorkerUnavailableError('continuous close deadline')
        self._reader.join(timeout=1)
        outcome = self._reap(force=False, deadline=time.monotonic()+self._close_timeout_s)
        self._closed = True
        self._metadata(closed=True)
        if not outcome['normal_exit_ok']:
            raise QwenWorkerUnavailableError('continuous worker abnormal close')
