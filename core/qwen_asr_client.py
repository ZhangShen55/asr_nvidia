"""Qwen3-ASR（OpenAI 兼容接口）异步客户端。"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

import httpx

from core.config import settings

logger = logging.getLogger(__name__)


class QwenAsrError(Exception):
    """Qwen 链路错误基类，携带对外错误码。"""

    code = 4012
    message = "Qwen 服务调用失败，请稍后再试"

    def __init__(self, message: str | None = None):
        super().__init__(message or self.message)
        self.message = message or self.message


class QwenAsrUnavailableError(QwenAsrError):
    """Qwen 服务不可用或未发现可用模型。"""

    code = 4011
    message = "Qwen 服务不可用或未发现可用模型"


class QwenAsrCallError(QwenAsrError):
    """Qwen 服务调用失败（超时、连接错误、5xx 等）。"""

    code = 4012
    message = "Qwen 服务调用失败，请稍后再试"


class QwenAsrUnsupportedLanguageError(QwenAsrError):
    """Qwen 服务不支持请求指定的语种。"""

    code = 4009
    message = "不支持的语言"


def normalize_language_param(language: Optional[str]) -> Optional[str]:
    """按规则决定是否下发 language 字段。

    - ``None`` / 空串 / ``auto``：不下发（Qwen 服务端不接受 auto，会直接 400）。
    - 其它取值：原样透传。
    """
    value = (language or "").strip()
    if not value or value.lower() == "auto":
        return None
    return value


class QwenAsrClient:
    """封装 /v1/models 模型发现与 /v1/audio/transcriptions 调用。"""

    def __init__(
        self,
        base_url: str,
        model: str = "",
        request_timeout_s: float = 120.0,
        connect_timeout_s: float = 10.0,
        retry_count: int = 1,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = (model or "").strip()
        self.request_timeout_s = request_timeout_s
        self.connect_timeout_s = connect_timeout_s
        self.retry_count = max(0, int(retry_count))
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()
        self._discovered_model: Optional[str] = None

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            self.request_timeout_s,
            connect=self.connect_timeout_s,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    limits = httpx.Limits(
                        max_connections=max(settings.qwen_request_concurrency * 2, 16),
                        max_keepalive_connections=max(settings.qwen_request_concurrency, 8),
                    )
                    self._client = httpx.AsyncClient(timeout=self._timeout(), limits=limits)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def discover_model(self) -> str:
        """获取可用模型标识；已配置模型名时直接返回。"""
        if self.model:
            return self.model
        if self._discovered_model:
            return self._discovered_model

        client = await self._get_client()
        last_error: Optional[Exception] = None
        for attempt in range(self.retry_count + 1):
            try:
                response = await client.get(f"{self.base_url}/v1/models")
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"模型发现失败: HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                payload = response.json()
                data = payload.get("data") or []
                if not data:
                    raise QwenAsrUnavailableError("Qwen 服务未返回可用模型")
                model_id = str(data[0].get("id") or "").strip()
                if not model_id:
                    raise QwenAsrUnavailableError("Qwen 服务未返回可用模型")
                self._discovered_model = model_id
                logger.info(f"[Qwen] 模型发现成功: {model_id}")
                return model_id
            except QwenAsrUnavailableError:
                raise
            except Exception as e:
                last_error = e
                if attempt < self.retry_count:
                    await asyncio.sleep(2 ** attempt)

        logger.error(f"[Qwen] 模型发现失败: {last_error}")
        raise QwenAsrUnavailableError() from last_error

    async def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """转写一个音频片段，返回服务返回的原始文本。"""
        model_id = await self.discover_model()
        payload_language = normalize_language_param(language)

        client = await self._get_client()
        last_error: Optional[Exception] = None

        for attempt in range(self.retry_count + 1):
            request_started = time.perf_counter()
            try:
                with open(audio_path, "rb") as audio_file:
                    form: dict = {"model": model_id}
                    if payload_language:
                        form["language"] = payload_language
                    response = await client.post(
                        f"{self.base_url}/v1/audio/transcriptions",
                        data=form,
                        files={"file": (os.path.basename(audio_path), audio_file, "audio/wav")},
                    )

                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"Qwen 转写失败: HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                if response.status_code == 400:
                    detail = response.text
                    if "language" in detail.lower():
                        logger.warning(f"[Qwen] 语种不被支持: {detail}")
                        raise QwenAsrUnsupportedLanguageError()
                    raise QwenAsrCallError(f"Qwen 转写请求被拒绝: HTTP 400 {detail[:200]}")
                if response.status_code >= 400:
                    raise QwenAsrCallError(
                        f"Qwen 转写失败: HTTP {response.status_code} {response.text[:200]}"
                    )

                text = self._extract_text(response)
                logger.debug(
                    f"[Qwen] 转写完成 language={payload_language or 'auto'} "
                    f"耗时={time.perf_counter() - request_started:.3f}s"
                )
                return text
            except QwenAsrError:
                raise
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_error = e
                logger.warning(f"[Qwen] 请求异常（第 {attempt + 1} 次）: {e}")
            except Exception as e:
                last_error = e
                logger.warning(f"[Qwen] 请求失败（第 {attempt + 1} 次）: {e}")

            if attempt < self.retry_count:
                await asyncio.sleep(2 ** attempt)

        logger.error(f"[Qwen] 转写重试耗尽: {last_error}")
        raise QwenAsrCallError() from last_error

    @staticmethod
    def _extract_text(response: httpx.Response) -> str:
        """兼容 JSON 与纯文本两种响应形态。"""
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                payload = response.json()
            except ValueError as e:
                raise QwenAsrCallError("Qwen 返回内容不是合法 JSON") from e
            if isinstance(payload, dict):
                if "text" in payload:
                    return str(payload.get("text") or "")
                error = payload.get("error")
                if error:
                    detail = error.get("message") if isinstance(error, dict) else str(error)
                    if "language" in str(detail).lower():
                        raise QwenAsrUnsupportedLanguageError()
                    raise QwenAsrCallError(f"Qwen 返回错误: {detail}")
            return ""
        return response.text or ""


_client_instance: Optional[QwenAsrClient] = None
_client_lock = asyncio.Lock()


async def get_qwen_client() -> QwenAsrClient:
    """获取进程级共享的 Qwen 客户端。"""
    global _client_instance
    if _client_instance is None:
        async with _client_lock:
            if _client_instance is None:
                _client_instance = QwenAsrClient(
                    base_url=settings.qwen_base_url,
                    model=settings.qwen_model,
                    request_timeout_s=settings.qwen_request_timeout_s,
                    connect_timeout_s=settings.qwen_connect_timeout_s,
                    retry_count=settings.qwen_retry_count,
                )
    return _client_instance


async def close_qwen_client() -> None:
    global _client_instance
    if _client_instance is not None:
        await _client_instance.aclose()
        _client_instance = None


__all__ = [
    "QwenAsrClient",
    "QwenAsrError",
    "QwenAsrUnavailableError",
    "QwenAsrCallError",
    "QwenAsrUnsupportedLanguageError",
    "close_qwen_client",
    "get_qwen_client",
    "normalize_language_param",
]
