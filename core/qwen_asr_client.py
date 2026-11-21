"""Qwen3-ASR 本地异步适配器，保留既有客户端与错误码契约。"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import time
from pathlib import Path
from typing import Optional

from core.async_utils import await_completion
from core.config import settings
from core.qwen_batch_scheduler import BatchScheduler
from core.qwen_asr_runtime import LocalQwenRuntime, LocalModelUnavailable, LocalUnsupportedLanguage

logger = logging.getLogger(__name__)


class QwenAsrError(Exception):
    code = 4012
    message = "Qwen 服务调用失败，请稍后再试"

    def __init__(self, message: str | None = None):
        super().__init__(message or self.message)
        self.message = message or self.message


class QwenAsrUnavailableError(QwenAsrError):
    code = 4011
    message = "Qwen 服务不可用或未发现可用模型"


class QwenAsrCallError(QwenAsrError):
    code = 4012
    message = "Qwen 服务调用失败，请稍后再试"


class QwenAsrUnsupportedLanguageError(QwenAsrError):
    code = 4009
    message = "不支持的语言"


def normalize_language_param(language: Optional[str]) -> Optional[str]:
    """auto/空值由模型自动识别，具体语种交给锁定版本的processor验证。"""
    value = (language or "").strip()
    return None if not value or value.lower() == "auto" else value


class QwenAsrClient:
    supports_batch_metadata = True
    def __init__(self, base_url: str = "", model: str = "", request_timeout_s: float = 120.0,
                 connect_timeout_s: float = 10.0, retry_count: int = 1, *,
                 local_model_dir=None, local_device=None, local_dtype=None,
                 local_instance_count=None, max_new_tokens=None, runtime_factory=None,
                 max_inference_batch_size=None, batch_wait_timeout_ms=None,
                 max_batch_audio_frames=None, max_batch_tokens=None, batch_max_duration_ratio=None,
                 execution_mode=None, generation_mode=None):
        # 旧参数仍可被现有调用者传入，但不再创建HTTP连接池或发模型发现请求。
        self.base_url = str(base_url).rstrip("/")
        self.model = (model or "").strip()
        self.request_timeout_s = float(request_timeout_s)
        self.connect_timeout_s = float(connect_timeout_s)
        self.retry_count = max(0, int(retry_count))
        self.local_model_dir = str(local_model_dir if local_model_dir is not None else getattr(
            settings, "qwen_local_model_dir", "/home/model/Qwen3-ASR-1.7B-hf"))
        self.local_device = str(local_device if local_device is not None else getattr(settings, "qwen_local_device", "npu:7"))
        self.local_dtype = str(local_dtype if local_dtype is not None else getattr(settings, "qwen_local_dtype", "bfloat16"))
        count = int(local_instance_count if local_instance_count is not None else getattr(settings, "qwen_local_instance_count", 1))
        tokens = int(max_new_tokens if max_new_tokens is not None else getattr(settings, "qwen_max_new_tokens", 256))
        if count < 1 or tokens < 1 or self.request_timeout_s <= 0:
            raise ValueError("Qwen 实例数、生成长度与请求超时必须为正数")
        self.execution_mode = settings.qwen_execution_mode if execution_mode is None else execution_mode
        if not isinstance(self.execution_mode, str) or self.execution_mode not in {"thread", "process"}:
            raise ValueError("execution_mode must be thread or process")
        self.generation_mode = (getattr(settings, 'qwen_generation_mode', 'legacy')
                                if generation_mode is None else generation_mode)
        if self.generation_mode not in {'legacy', 'continuous'}:
            raise ValueError('generation_mode must be legacy or continuous')
        if self.generation_mode == 'continuous' and (self.execution_mode != 'process' or count != 1):
            raise ValueError('continuous generation requires one process model owner')
        if runtime_factory is not None:
            factory = runtime_factory
        elif self.generation_mode == 'continuous':
            from core.qwen_continuous_process import LocalQwenContinuousProcessRuntime
            factory = lambda *args: LocalQwenContinuousProcessRuntime(*args,
                max_active=int(getattr(settings, 'qwen_request_concurrency', 128)),
                max_batch_size=int(max_inference_batch_size if max_inference_batch_size is not None else
                                   getattr(settings, 'qwen_max_inference_batch_size', 128)),
                prefill_batch_size=int(getattr(settings, 'qwen_continuous_prefill_batch_size', 16)),
                continuous_admission_drain_ms=int(getattr(settings, 'qwen_continuous_admission_drain_ms', 0)),
                max_batch_tokens=int(max_batch_tokens if max_batch_tokens is not None else
                                     getattr(settings, 'qwen_max_batch_tokens', 80000)),
                max_batch_audio_frames=int(max_batch_audio_frames if max_batch_audio_frames is not None else
                                           getattr(settings, 'qwen_max_batch_audio_frames', 115200)),
                online_repetition_enabled=bool(getattr(settings, 'qwen_online_repetition_enabled', False)),
                online_repetition_config=(__import__('dataclasses').asdict(settings.qwen_online_repetition_config)
                    if hasattr(settings, 'qwen_online_repetition_config') else None))
        elif self.execution_mode == "process":
            from core.qwen_asr_process import LocalQwenProcessRuntime
            factory = LocalQwenProcessRuntime
        else:
            factory = LocalQwenRuntime
        self._shutdown_state = {"phase": "open", "forced_cleanup": False}
        self._drain_task = None
        self._runtimes = [factory(self.local_model_dir, self.local_device, self.local_dtype, tokens) for _ in range(count)]
        self._available = asyncio.Queue(maxsize=count)
        for index in range(count):
            self._available.put_nowait(index)
        self._changed = asyncio.Event()
        self._busy = [False] * count
        self._waiting = 0
        self._closing = False
        self._close_task = None
        self._discovered_model = None
        self._discovery_lock = asyncio.Lock()
        self._discovering_all = False
        self._execution_failures = 0
        self._batch_scheduler = BatchScheduler(
            self._execute_batch, workers=count,
            batch_size=int(max_inference_batch_size if max_inference_batch_size is not None else
                           (getattr(settings, "qwen_max_inference_batch_size", 1) if
                            all(hasattr(r, "transcribe_batch") for r in self._runtimes) else 1)),
            wait_ms=float(batch_wait_timeout_ms if batch_wait_timeout_ms is not None else
                          getattr(settings, "qwen_batch_wait_timeout_ms", 10)),
            audio_frame_budget=int(max_batch_audio_frames if max_batch_audio_frames is not None else
                                   getattr(settings, "qwen_max_batch_audio_frames", 102400)),
            token_budget=int(max_batch_tokens if max_batch_tokens is not None else
                             getattr(settings, "qwen_max_batch_tokens", 78336)),
            duration_ratio=(batch_max_duration_ratio if batch_max_duration_ratio is not None else
                            getattr(settings, "qwen_batch_max_duration_ratio", 4.0)),
            max_new_tokens=tokens, capacity=int(getattr(settings, "qwen_pending_segment_limit", 256)),
        )
        if self.generation_mode == 'continuous':
            from core.qwen_continuous_scheduler import ContinuousScheduler
            self._batch_scheduler = ContinuousScheduler(self._execute_sequence,
                capacity=int(getattr(settings, 'qwen_request_concurrency', 128)),
                audio_frame_budget=self._batch_scheduler.audio_frame_budget,
                token_budget=self._batch_scheduler.token_budget, max_new_tokens=tokens)
        if self.base_url:
            logger.info("[配置迁移] qwen_asr.base_url/connect_timeout_s 仅兼容读取，本地推理不发起HTTP调用")

    async def _run_owned(self, function, *args):
        worker = asyncio.create_task(asyncio.to_thread(function, *args))
        try:
            # Python3.10的wait_for在完成/取消竞态中可能吞掉取消；wait不取消底层线程任务。
            done, _ = await asyncio.wait({worker}, timeout=self.request_timeout_s)
            if worker in done:
                return worker.result()
            # 超时不能停止Python线程或NPU kernel；先完成原工作再允许重试/复用实例。
            try:
                await await_completion(worker)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("Qwen 超时工作收尾时失败", exc_info=True)
            raise TimeoutError("Qwen 本地推理超时（设备工作已收尾）")
        except asyncio.CancelledError as cancelled:
            try:
                await await_completion(worker)
            except BaseException:
                # 请求已取消，原线程的错误不能掩盖取消；重复取消同样等待收尾。
                pass
            raise cancelled

    async def _lease(self):
        self._waiting += 1
        try:
            while True:
                # 扫描与清除通知之间没有await，避免归还时丢失唤醒；失败实例不再发放。
                self._changed.clear()
                if self._closing or all(runtime.failure for runtime in self._runtimes):
                    raise QwenAsrUnavailableError()
                if self._discovering_all:
                    await self._changed.wait()
                    continue
                for _ in range(self._available.qsize()):
                    index = self._available.get_nowait()
                    if not self._runtimes[index].failure:
                        return index
                    self._available.put_nowait(index)
                await self._changed.wait()
        finally:
            self._waiting -= 1

    async def preload(self):
        # 启动预加载复用全体租约/取消排空流程，显示model名不能跳过真实加载。
        name = await self._discover_all_process_runtimes()
        if not all(runtime.loaded and not runtime.failure for runtime in self._runtimes):
            raise QwenAsrUnavailableError()
        return name

    async def discover_model(self):
        if self.execution_mode == "process" and len(self._runtimes) > 1:
            return await self._discover_all_process_runtimes()
        if self.model:
            return self.model
        if self._discovered_model:
            return self._discovered_model
        index = await self._lease()
        self._busy[index] = True
        try:
            if self._closing:
                raise QwenAsrUnavailableError()
            await self._run_owned(self._runtimes[index].ensure_loaded)
            self._discovered_model = Path(self.local_model_dir).name
            return self._discovered_model
        except LocalModelUnavailable as exc:
            logger.error("[Qwen] 本地加载失败: %s", exc)
            raise QwenAsrUnavailableError() from exc
        except QwenAsrError:
            raise
        except Exception as exc:
            raise QwenAsrUnavailableError() from exc
        finally:
            self._busy[index] = False
            self._available.put_nowait(index)
            self._changed.set()

    async def _discover_all_process_runtimes(self):
        # 一位发现者取得全部租约；其它发现者先等锁，不能各持一份后互等。
        async with self._discovery_lock:
            if self._closing:
                raise QwenAsrUnavailableError()
            if self._discovered_model:
                # 冷启动已全体验证；之后允许健康已加载副本继续服务，不重启失败者。
                if any(runtime.loaded and not runtime.failure for runtime in self._runtimes):
                    return self._discovered_model
                raise QwenAsrUnavailableError()
            if any(runtime.failure for runtime in self._runtimes):
                raise QwenAsrUnavailableError()
            if all(runtime.loaded for runtime in self._runtimes):
                self._discovered_model = self.model or Path(self.local_model_dir).name
                return self._discovered_model
            acquired = []
            self._discovering_all = True
            self._waiting += 1
            try:
                # 不用只发健康实例的_lease：部分失败也必须取回并归还全部登记索引。
                for _ in self._runtimes:
                    index = await self._available.get()
                    acquired.append(index)
                    self._busy[index] = True
                    if self._closing or self._runtimes[index].failure:
                        raise QwenAsrUnavailableError()
                outcomes = await await_completion(asyncio.gather(
                    *(self._run_owned(self._runtimes[index].ensure_loaded) for index in acquired),
                    return_exceptions=True,
                ))
                if (self._closing or any(isinstance(value, BaseException) for value in outcomes)
                        or not all(runtime.loaded and not runtime.failure for runtime in self._runtimes)):
                    raise QwenAsrUnavailableError()
                self._discovered_model = self.model or Path(self.local_model_dir).name
                return self._discovered_model
            except (QwenAsrError, asyncio.CancelledError):
                raise
            except Exception as exc:
                raise QwenAsrUnavailableError() from exc
            finally:
                for index in acquired:
                    self._busy[index] = False
                    self._available.put_nowait(index)
                self._waiting -= 1
                self._discovering_all = False
                self._changed.set()

    def _runtime_attempt_identity(self, index, runtime):
        # 仅复制本实例已有快照，不加载、不发RPC；诊断失败不能覆盖业务结果。
        identity = {"instance_index": index, "execution_mode": self.execution_mode}
        snapshot = getattr(runtime, "snapshot", None)
        if not callable(snapshot):
            identity["metadata_available"] = False
            return identity
        try:
            state = snapshot()
            worker = state.get("model_worker", {})
            identity.update({key: worker.get(key) for key in (
                "worker_pid", "worker_create_time", "worker_start_ticks", "launch_id",
                "runtime_kind", "ready_verified", "worker_entry_module_file",
                "resource_reaped", "normal_exit_ok", "forced_cleanup", "worker_exitcode",
            )})
            identity["last_rpc_id"] = getattr(runtime, "_rpc_id", None)
            identity["metadata_available"] = bool(worker)
        except Exception as exc:
            identity["metadata_available"] = False
            identity["snapshot_error_type"] = type(exc).__name__
        return identity

    async def transcribe_bytes(self, audio_bytes: bytes, language: Optional[str] = None,
                               filename: str = "segment.wav", *, request_id=None,
                               segment_index=None) -> str | dict:
        if self._closing:
            raise QwenAsrUnavailableError()
        try:
            return await self._batch_scheduler.submit(
                audio_bytes, normalize_language_param(language),
                request_id=request_id, segment_index=segment_index)
        except LocalUnsupportedLanguage as exc:
            raise QwenAsrUnsupportedLanguageError() from exc
        except (QwenAsrError, asyncio.CancelledError):
            raise
        except Exception as exc:
            if self.generation_mode == 'continuous':
                reason = getattr(exc, 'reason', 'generation_error')
                error = QwenAsrCallError(f'Qwen sequence failed: reason={reason}, request_id={request_id}, segment_id={segment_index}')
                error.reason, error.request_id, error.segment_id = reason, request_id, segment_index
                error.backend_error_type = type(exc).__name__
                logger.warning('[Qwen] sequence failed request=%s segment=%s reason=%s type=%s',
                    request_id, segment_index, reason, type(exc).__name__)
                raise error from exc
            raise QwenAsrCallError() from exc

    async def _execute_sequence(self, member):
        # Discovery is serialized; after it completes there is no whole-model lease.
        # All member futures are routed by the single worker's device owner.
        runtime = self._runtimes[0]
        try:
            async with self._discovery_lock:
                if not runtime.loaded:
                    await self._run_owned(runtime.ensure_loaded)
        except LocalModelUnavailable as exc:
            raise QwenAsrUnavailableError() from exc
        if self._closing or runtime.failure:
            raise QwenAsrUnavailableError()
        rejected_at = None
        for attempt in range(4):
            if member.cancelled:
                raise asyncio.CancelledError()
            try:
                # Pipe IO can block behind a device step. Keep it off the event
                # loop, but retain ownership if cancellation arrives during send.
                send = asyncio.create_task(asyncio.to_thread(runtime.submit,
                    member.audio, member.language, request_id=member.request_id,
                    segment_id=member.segment_index))
                try:
                    submitted = await asyncio.shield(send)
                except asyncio.CancelledError as cancelled:
                    try:
                        submitted = await await_completion(send)
                    except BaseException:
                        raise cancelled
                    owned = asyncio.wrap_future(submitted)
                    try:
                        try:
                            await await_completion(asyncio.to_thread(runtime.cancel_segment,
                                member.request_id, member.segment_index))
                        finally:
                            await await_completion(owned)
                    except BaseException:
                        pass
                    raise cancelled
                future = asyncio.wrap_future(submitted)
                return await self._await_sequence(runtime, member, future)
            except LocalModelUnavailable as exc:
                raise QwenAsrUnavailableError() from exc
            except Exception as exc:
                if not (getattr(exc, 'accepted', None) is False and
                        getattr(exc, 'reason', None) == 'capacity_rejected'):
                    raise
                now = time.monotonic()
                rejected_at = now if rejected_at is None else rejected_at
                delay = (0.1, 0.5, 1.0)[min(attempt, 2)]
                if attempt == 3 or now + delay - rejected_at > 30:
                    raise QwenAsrCallError('Qwen capacity retry exhausted') from exc
                await asyncio.sleep(delay)

    async def _await_sequence(self, runtime, member, future):
        started = time.monotonic()
        cancel_sent = False
        timeout = False
        try:
            while not future.done():
                timeout = timeout or time.monotonic() - started >= self.request_timeout_s
                if (member.cancelled or timeout) and not cancel_sent:
                    cancel_sent = True
                    await await_completion(asyncio.to_thread(runtime.cancel_segment,
                        member.request_id, member.segment_index))
                await asyncio.wait({future}, timeout=0.05)
            # Retrieve the owned completion before applying the caller's first
            # cause: backend cancellation is an acknowledgement, not a new cause.
            backend_error = None
            try:
                value = future.result()
            except BaseException as exc:
                backend_error = exc
            if member.cancelled:
                raise asyncio.CancelledError() from backend_error
            if timeout:
                error = TimeoutError('Qwen sequence timed out after owned cleanup')
                error.reason = 'request_timeout'
                raise error from backend_error
            if backend_error is not None:
                raise backend_error
            return value
        except asyncio.CancelledError as cancelled:
            try:
                try:
                    if not cancel_sent:
                        await await_completion(asyncio.to_thread(runtime.cancel_segment,
                            member.request_id, member.segment_index))
                finally:
                    await await_completion(future)
            except BaseException:
                pass
            raise cancelled

    async def _execute_batch(self, members, batch_id):
        index = await self._lease()
        self._busy[index] = True
        runtime = self._runtimes[index]
        active = [member for member in members if not member.cancelled]
        self._batch_scheduler.record_dispatch(batch_id, active)
        try:
            if not active:
                return [None] * len(members)
            for attempt in range(self.retry_count + 1):
                # backoff期间可单独取消成员；新物理attempt不得再次执行其音频。
                active = [member for member in members if not member.cancelled]
                if attempt:
                    self._batch_scheduler.record_retry_cancellations(batch_id,
                        [member for member in members if member.cancelled], attempt + 1)
                if not active:
                    return [None] * len(members)
                if attempt:
                    self._batch_scheduler.record_dispatch(batch_id, active)
                # 明确每次物理attempt的timings归属，不能把旧batch的有效索引套到幸存成员。
                if hasattr(runtime, "last_batch_timings"):
                    runtime.last_batch_timings = None
                attempt_status, attempt_error = "failed", None
                try:
                    if hasattr(runtime, "transcribe_batch"):
                        values = await self._run_owned(runtime.transcribe_batch,
                            [m.audio for m in active], [m.language for m in active])
                        if len(values) != len(active):
                            raise RuntimeError("Qwen batch result cardinality mismatch")
                        by_member = dict(zip((id(m) for m in active), values))
                        attempt_status = "completed"
                        return [by_member.get(id(member)) for member in members]
                    if len(active) != 1:
                        raise RuntimeError("当前运行器不支持真实批执行，不能逐段冒充批推理")
                    value = await self._run_owned(runtime.transcribe, active[0].audio, active[0].language)
                    attempt_status = "completed"
                    return [value if member is active[0] else None for member in members]
                except LocalModelUnavailable as exc:
                    attempt_error = type(exc).__name__
                    logger.error("[Qwen] 本地模型不可用: %s", exc)
                    raise QwenAsrUnavailableError() from exc
                except LocalUnsupportedLanguage as exc:
                    attempt_error = type(exc).__name__
                    raise QwenAsrUnsupportedLanguageError() from exc
                except asyncio.CancelledError:
                    attempt_status, attempt_error = "cancelled", "CancelledError"
                    raise
                except Exception as exc:
                    attempt_error = type(exc).__name__
                    self._execution_failures += 1
                    logger.warning("[Qwen] batch=%s size=%s attempt=%d type=%s detail=%s",
                                   batch_id, len(active), attempt+1, type(exc).__name__, exc)
                    if attempt == self.retry_count:
                        raise QwenAsrCallError() from exc
                finally:
                    self._batch_scheduler.record_runtime_timings(batch_id,
                        getattr(runtime, "last_batch_timings", None),
                        status=attempt_status, error_type=attempt_error)
                    self._batch_scheduler.record_runtime_instance(
                        batch_id, self._runtime_attempt_identity(index, runtime))
                # 失败attempt的归属/包络已落盘；等待期间取消不会污染其有效成员索引。
                await asyncio.sleep(2 ** attempt)
        finally:
            self._busy[index] = False
            self._available.put_nowait(index)
            self._changed.set()

    def release_request(self, request_id):
        """Forget deduplication only after the parent's owned segments have drained."""
        if self.generation_mode == 'continuous':
            self._runtimes[0].release_request(str(request_id))

    async def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        data = await asyncio.to_thread(Path(audio_path).read_bytes)
        return await self.transcribe_bytes(data, language=language, filename=Path(audio_path).name)

    def snapshot(self, *, recent_batch_limit=None):
        details = [dict(runtime.snapshot(), index=index, busy=self._busy[index]) for index, runtime in enumerate(self._runtimes)]
        batch_snapshot = (self._batch_scheduler.snapshot() if recent_batch_limit is None else
                          self._batch_scheduler.snapshot(recent_limit=recent_batch_limit))
        if self.generation_mode == 'continuous':
            details[0]['busy'] = bool(batch_snapshot['running_members'])
        return {
            **({"generation_mode": self.generation_mode} if self.generation_mode == "continuous" else {}),
            "execution_mode": self.execution_mode,
            "memory_process_scope": "model_worker" if self.execution_mode == "process" else "service_process",
            "configured_model_workers": len(details) if self.execution_mode == "process" else 0,
            "model_workers": [dict(item["model_worker"], index=item["index"])
                              for item in details if "model_worker" in item],
            "shutdown": dict(self._shutdown_state),
            "backend": "transformers_torch_npu", "model_dir": self.local_model_dir,
            "device": self.local_device, "dtype": self.local_dtype,
            "loaded": any(x["loaded"] for x in details),
            "loading": any(x.get("loading") for x in details),
            "configured_instances": len(details),
            "capacity": sum(x["loaded"] and not x.get("failure") for x in details),
            "busy": sum(bool(item.get("busy")) for item in details), "waiting": self._waiting + batch_snapshot["waiting"],
            "batch_scheduler": batch_snapshot,
            "failed_instances": sum(bool(x.get("failure")) for x in details),
            "execution_failures": self._execution_failures, "closing": self._closing,
            "memory_allocated_mb": sum(x.get("memory_allocated_mb", 0) for x in details),
            "memory_reserved_mb": sum(x.get("memory_reserved_mb", 0) for x in details),
            "memory_source": "pytorch_load_delta_estimate", "instances": details,
        }

    async def _drain(self):
        self._closing = True
        self._changed.set()
        await self._batch_scheduler.close(QwenAsrUnavailableError())
        if self.execution_mode == "process" and len(self._runtimes) > 1:
            # discovery和drain都要持全体索引，不能各持一部分后互等。
            async with self._discovery_lock:
                await self._drain_runtimes()
        else:
            await self._drain_runtimes()

    async def _drain_runtimes(self):
        acquired = []
        try:
            for _ in self._runtimes:
                acquired.append(await self._available.get())
            await await_completion(asyncio.to_thread(self._close_runtimes))
        finally:
            for index in acquired:
                self._available.put_nowait(index)

    async def _close(self):
        if self.execution_mode != "process":
            self._shutdown_state.update(phase="draining")
            try:
                result = await self._drain()
            except BaseException:
                self._shutdown_state.update(phase="failed")
                raise
            self._shutdown_state.update(phase="closed")
            return result

        # 显式aclose才启动独立宽限期；普通推理timeout不会触发worker终止。
        grace = settings.qwen_process_shutdown_grace_s
        self._shutdown_state.update(
            phase="grace", grace_s=grace, started_monotonic_s=time.monotonic()
        )
        self._drain_task = asyncio.create_task(self._drain())

        def observe_drain(task):
            # 关闭已失败时也保留后台拥有者；将来的异常需要被观察，不能取消或假归还名额。
            if not task.cancelled():
                task.exception()

        self._drain_task.add_done_callback(observe_drain)
        done, _ = await asyncio.wait({self._drain_task}, timeout=grace)
        if done:
            try:
                result = self._drain_task.result()
            except BaseException:
                self._shutdown_state.update(phase="failed")
                raise
            self._shutdown_state.update(phase="closed")
            return result

        self._shutdown_state.update(phase="owned_abort", forced_cleanup=True)
        outcomes = await asyncio.gather(
            *(asyncio.to_thread(runtime.abort_owned, "explicit shutdown grace exceeded")
              for runtime in self._runtimes),
            return_exceptions=True,
        )
        errors = [str(outcome) for outcome in outcomes if isinstance(outcome, BaseException)]
        if errors:
            self._shutdown_state.update(phase="failed", abort_errors=errors)
            # 未确认实际死亡时，不取消reader/drain，也不提前归还仍有设备工作的租约。
            raise RuntimeError("Qwen owned shutdown could not confirm worker death")

        done, _ = await asyncio.wait({self._drain_task}, timeout=5)
        self._shutdown_state.update(phase="failed")
        if not done:
            raise RuntimeError("Qwen worker reaped but parent owned drain did not finish")
        # 强制路径必须保留失败，即使资源已回收也不能伪装成正常关闭。
        try:
            self._drain_task.result()
        except BaseException as exc:
            raise RuntimeError("Qwen forced shutdown cleanup failed") from exc
        raise RuntimeError("Qwen worker was forcibly reaped during shutdown")

    def _close_runtimes(self):
        if self.execution_mode == "process" and len(self._runtimes) > 1:
            # 原drain已持有全部归还的租约。并发CLOSE不把正常收尾串成N倍deadline。
            errors = []
            with ThreadPoolExecutor(max_workers=len(self._runtimes)) as executor:
                closes = [executor.submit(runtime.close) for runtime in self._runtimes]
                for index, close in enumerate(closes):
                    try:
                        close.result()
                    except Exception as exc:
                        errors.append((index, exc))
            if errors:
                raise RuntimeError(f"Qwen 多实例关闭失败: {errors}")
            return
        errors = []
        for runtime in self._runtimes:
            try:
                runtime.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(f"Qwen 实例关闭失败: {errors}") from errors[0]

    async def aclose(self):
        if self._close_task is None:
            self._closing = True
            self._changed.set()
            if self.execution_mode == "process":
                for runtime in self._runtimes:
                    begin = getattr(runtime, "begin_closing", None)
                    if begin is not None:
                        begin()
            self._close_task = asyncio.create_task(self._close())
        await await_completion(self._close_task)


_client_instance: Optional[QwenAsrClient] = None
_client_lock = asyncio.Lock()


async def get_qwen_client() -> QwenAsrClient:
    global _client_instance
    if _client_instance is None:
        async with _client_lock:
            if _client_instance is None:
                _client_instance = QwenAsrClient(
                    base_url=settings.qwen_base_url, model=settings.qwen_model,
                    request_timeout_s=settings.qwen_request_timeout_s,
                    connect_timeout_s=settings.qwen_connect_timeout_s,
                    retry_count=settings.qwen_retry_count,
                )
    return _client_instance


async def release_qwen_request(request_id) -> None:
    """Drain the existing client's Pipe send without blocking the event loop."""
    client = _client_instance
    if client is not None:
        await await_completion(asyncio.to_thread(client.release_request, request_id))


async def preload_qwen_client() -> str:
    client = await get_qwen_client()
    return await client.preload()


def get_qwen_status(*, recent_batch_limit=None):
    if _client_instance is not None:
        if recent_batch_limit is None:
            return _client_instance.snapshot()
        return _client_instance.snapshot(recent_batch_limit=recent_batch_limit)
    # 构造运行器无权重加载，但状态查询连运行器都不必创建。
    count = int(getattr(settings, "qwen_local_instance_count", 1))
    device = getattr(settings, "qwen_local_device", "npu:7")
    dtype = getattr(settings, "qwen_local_dtype", "bfloat16")
    mode = settings.qwen_execution_mode
    return {
        "execution_mode": mode,
        "memory_process_scope": "model_worker" if mode == "process" else "service_process",
        "configured_model_workers": count if mode == "process" else 0,
        "model_workers": [], "shutdown": {"phase": "open", "forced_cleanup": False},
        "backend": "transformers_torch_npu", "model_dir": getattr(settings, "qwen_local_model_dir", "/home/model/Qwen3-ASR-1.7B-hf"),
        "device": device, "dtype": dtype, "loaded": False, "loading": False,
        "configured_instances": count, "capacity": 0, "busy": 0, "waiting": 0,
        "failed_instances": 0, "execution_failures": 0, "closing": False,
        "memory_allocated_mb": 0.0, "memory_reserved_mb": 0.0,
        "memory_source": "pytorch_load_delta_estimate",
        "instances": [{"index": i, "device": device, "dtype": dtype, "loaded": False,
                       "loading": False, "busy": False, "failure": None,
                       "memory_allocated_mb": 0.0, "memory_reserved_mb": 0.0,
                       "memory_source": "pytorch_load_delta_estimate"} for i in range(count)],
    }


async def close_qwen_client() -> None:
    global _client_instance
    if _client_instance is not None:
        client = _client_instance
        # 多次取消期间保留全局引用，避免旧设备工作未结束时创建第二个客户端。
        try:
            await client.aclose()
        finally:
            if (client._close_task is not None and client._close_task.done()
                    and not client._close_task.cancelled() and client._close_task.exception() is None
                    and _client_instance is client):
                _client_instance = None


__all__ = ["QwenAsrClient", "QwenAsrError", "QwenAsrUnavailableError", "QwenAsrCallError",
           "QwenAsrUnsupportedLanguageError", "close_qwen_client", "get_qwen_client", "get_qwen_status",
           "normalize_language_param", "preload_qwen_client"]
