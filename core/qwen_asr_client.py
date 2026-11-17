"""Qwen3-ASR 本地异步适配器，保留既有客户端与错误码契约。"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from core.async_utils import await_completion
from core.config import settings
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
    def __init__(self, base_url: str = "", model: str = "", request_timeout_s: float = 120.0,
                 connect_timeout_s: float = 10.0, retry_count: int = 1, *,
                 local_model_dir=None, local_device=None, local_dtype=None,
                 local_instance_count=None, max_new_tokens=None, runtime_factory=None):
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
        factory = runtime_factory or LocalQwenRuntime
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
        self._execution_failures = 0
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
                for _ in range(self._available.qsize()):
                    index = self._available.get_nowait()
                    if not self._runtimes[index].failure:
                        return index
                    self._available.put_nowait(index)
                await self._changed.wait()
        finally:
            self._waiting -= 1

    async def discover_model(self):
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

    async def transcribe_bytes(self, audio_bytes: bytes, language: Optional[str] = None,
                               filename: str = "segment.wav") -> str:
        index = await self._lease()
        self._busy[index] = True
        runtime = self._runtimes[index]
        try:
            if self._closing:
                raise QwenAsrUnavailableError()
            for attempt in range(self.retry_count + 1):
                try:
                    return await self._run_owned(runtime.transcribe, audio_bytes, normalize_language_param(language))
                except LocalModelUnavailable as exc:
                    logger.error("[Qwen] 本地模型不可用: %s", exc)
                    raise QwenAsrUnavailableError() from exc
                except LocalUnsupportedLanguage as exc:
                    raise QwenAsrUnsupportedLanguageError() from exc
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._execution_failures += 1
                    logger.warning("[Qwen] 本地推理失败 attempt=%d type=%s detail=%s", attempt + 1, type(exc).__name__, exc)
                    if attempt == self.retry_count:
                        raise QwenAsrCallError() from exc
                    await asyncio.sleep(2 ** attempt)
        finally:
            self._busy[index] = False
            self._available.put_nowait(index)
            self._changed.set()

    async def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        data = await asyncio.to_thread(Path(audio_path).read_bytes)
        return await self.transcribe_bytes(data, language=language, filename=Path(audio_path).name)

    def snapshot(self):
        details = [dict(runtime.snapshot(), index=index, busy=self._busy[index]) for index, runtime in enumerate(self._runtimes)]
        return {
            "backend": "transformers_torch_npu", "model_dir": self.local_model_dir,
            "device": self.local_device, "dtype": self.local_dtype,
            "loaded": any(x["loaded"] for x in details),
            "loading": any(x.get("loading") for x in details),
            "configured_instances": len(details),
            "capacity": sum(x["loaded"] and not x.get("failure") for x in details),
            "busy": sum(self._busy), "waiting": self._waiting,
            "failed_instances": sum(bool(x.get("failure")) for x in details),
            "execution_failures": self._execution_failures, "closing": self._closing,
            "memory_allocated_mb": sum(x.get("memory_allocated_mb", 0) for x in details),
            "memory_reserved_mb": sum(x.get("memory_reserved_mb", 0) for x in details),
            "memory_source": "pytorch_load_delta_estimate", "instances": details,
        }

    async def _close(self):
        self._closing = True
        self._changed.set()
        acquired = []
        try:
            for _ in self._runtimes:
                acquired.append(await self._available.get())
            await await_completion(asyncio.to_thread(self._close_runtimes))
        finally:
            for index in acquired:
                self._available.put_nowait(index)

    def _close_runtimes(self):
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


def get_qwen_status():
    if _client_instance is not None:
        return _client_instance.snapshot()
    # 构造运行器无权重加载，但状态查询连运行器都不必创建。
    count = int(getattr(settings, "qwen_local_instance_count", 1))
    device = getattr(settings, "qwen_local_device", "npu:7")
    dtype = getattr(settings, "qwen_local_dtype", "bfloat16")
    return {
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
           "normalize_language_param"]
