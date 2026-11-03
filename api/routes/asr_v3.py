"""v3 转写接口：Qwen3-ASR 链路。"""

import asyncio
import logging
import time

from fastapi import APIRouter, Depends

from api.routes.asr_common import cleanup_temp_files, prepare_asr_context
from api.routes.asr_qwen_common import (
    get_phase_stats, reset_phase_stats, resolve_v3_capabilities, run_qwen_asr,
)
from core.async_utils import await_completion
from core.config import settings
from core.models import ModelUnavailableError, validate_v3_capabilities
from core.qwen_asr_queue import QueueFullError, QueueTimeoutError, get_slot_manager
from core.request_telemetry import new_request_id, record_request
from core.vad_pool import VadPoolError
from entity.data import AsrRequestParams, get_asr_params
from utils.asr_stats import add_processing_task, remove_processing_task, update_fail_task

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/v3/seacraft_asr")
async def api_asr_v3(request: AsrRequestParams = Depends(get_asr_params)):
    """语音转写：本地 VAD 切分后并发调用 Qwen3-ASR。"""
    request_started = time.perf_counter()
    request_id = new_request_id()
    reset_phase_stats()
    capabilities = resolve_v3_capabilities(request)
    response = None
    ctx = None
    lease = None
    processing_registered = False
    cleanup_ms = 0.0
    cancelled = False
    failed = False
    audio_wait_ms = 0.0
    slots = get_slot_manager()
    admission = {"category": capabilities.category, "outcome": "not_attempted",
                 "reason": None, "snapshot": slots.snapshot()}
    try:
        if not settings.qwen_asr_enabled:
            admission.update(outcome="rejected", reason="qwen_disabled")
            response = {"msg": "Qwen 转写能力未开启", "code": 4003}
            return response
        try:
            validate_v3_capabilities(capabilities.need_role, capabilities.need_emotion)
        except ModelUnavailableError as exc:
            admission.update(outcome="rejected", reason=exc.reason)
            update_fail_task()
            response = {"msg": exc.message, "code": exc.code}
            return response
        wait_started = time.perf_counter()
        try:
            try:
                lease = await slots.acquire(capabilities.category)
                admission.update(outcome="admitted", snapshot=lease.snapshot)
                try:
                    # request_id 在准入前生成，覆盖音频准备、VAD、推理和清理的完整处理周期。
                    add_processing_task(request_id)
                    processing_registered = True
                except Exception:
                    # 任务列表是运维统计，写入失败不能阻断已经准入的业务请求。
                    logger.exception("[V3] 写入处理中任务列表失败 request_id=%s", request_id)
            finally:
                audio_wait_ms = (time.perf_counter() - wait_started) * 1000
        except (QueueFullError, QueueTimeoutError) as exc:
            admission.update(outcome="timeout" if isinstance(exc, QueueTimeoutError) else "rejected",
                             reason=exc.reason, snapshot=exc.snapshot)
            logger.warning("[V3] 音频准入失败 code=%s: %s", exc.code, exc.message)
            response = {"msg": exc.message, "code": exc.code}
            return response

        try:
            err, ctx = await prepare_asr_context(request, unique_tmp=True)
            if err:
                response = err
                return response
            try:
                response = await run_qwen_asr(ctx)
                return response
            except (VadPoolError, ModelUnavailableError) as exc:
                logger.warning("[V3] 处理失败 code=%s: %s", exc.code, exc.message)
                update_fail_task()
                response = {"msg": exc.message, "code": exc.code}
                return response
            except Exception:
                logger.exception("[V3] 转写异常")
                response = {"msg": "未知错误", "code": 4005}
                return response
        finally:
            try:
                if ctx is not None:
                    cleanup_started = time.perf_counter()
                    try:
                        cleanup_temp_files(ctx.tmp_paths)
                    except Exception:
                        # 清理错误保留日志，不能覆盖原业务结果或取消，也不能漏还名额。
                        logger.exception("[V3] 请求临时文件清理失败")
                    finally:
                        cleanup_ms = (time.perf_counter() - cleanup_started) * 1000
            finally:
                if lease is not None:
                    try:
                        await await_completion(slots.release(lease))
                    finally:
                        if processing_registered:
                            try:
                                remove_processing_task(request_id)
                            except Exception:
                                # 释放和业务结果不能被统计文件故障影响；下次启动会清理残留。
                                logger.exception("[V3] 移除处理中任务失败 request_id=%s", request_id)
    except asyncio.CancelledError:
        cancelled = True
        if lease is None:
            admission.update(outcome="cancelled", reason="cancelled", snapshot=slots.snapshot())
        raise
    except BaseException:
        failed = True
        raise
    finally:
        payload = response if isinstance(response, dict) else {}
        phases = get_phase_stats()
        phases["audio_wait_ms"] = audio_wait_ms
        phases["cleanup_ms"] = cleanup_ms
        phase_code = payload.get("code")
        status = "cancelled" if cancelled else (
            "failed" if failed or phase_code not in (None, 0) else "completed")
        logger.info("[V3准入] request_id=%s category=%s outcome=%s reason=%s code=%s snapshot=%s",
                    request_id, admission["category"], admission["outcome"], admission["reason"],
                    phase_code, admission["snapshot"])
        record_request(
            request_id=request_id,
            task_id=getattr(ctx, "task_id", None),
            filename=getattr(ctx, "filename", getattr(request.audioFile, "filename", None)),
            started_at=request_started,
            status=status,
            http_status=None if cancelled else 200,
            code=phase_code,
            retry_count=settings.qwen_retry_count,
            valid_segments=len(payload.get("segments") or []),
            phases=phases,
            admission=admission,
            error="请求取消" if cancelled else payload.get("msg") if status == "failed" else None,
        )
