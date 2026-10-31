"""v2 转写接口：Qwen3-ASR 链路。"""

import logging
import time

from fastapi import APIRouter, Depends

from api.routes.asr_common import cleanup_temp_files, prepare_asr_context
from api.routes.asr_qwen_common import get_phase_stats, run_qwen_asr
from core.config import settings
from core.models import ModelUnavailableError
from core.qwen_asr_queue import QueueFullError, QueueTimeoutError, get_slot_manager
from core.vad_pool import VadPoolError
from entity.data import AsrRequestParams, get_asr_params
from utils.asr_stats import update_fail_task
from core.request_telemetry import new_request_id, record_request

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v2/seacraft_asr")
async def api_asr_v2(request: AsrRequestParams = Depends(get_asr_params)):
    """语音转写：本地 VAD 切分后并发调用 Qwen3-ASR，响应结构与 /v1.1.8 同构。"""
    request_started = time.perf_counter()
    request_id = new_request_id()
    response = None
    ctx = None
    cleanup_ms = 0.0
    slots = get_slot_manager()
    try:
        if not settings.qwen_asr_enabled:
            response = {"msg": "Qwen 转写能力未开启", "code": 4003}
            return response
        try:
            await slots.acquire()
        except QueueFullError as e:
            logger.warning(f"[V2] 处理队列已满: {e.message}")
            response = {"msg": e.message, "code": e.code}
            return response
        except QueueTimeoutError as e:
            logger.warning(f"[V2] 排队超时: {e.message}")
            response = {"msg": e.message, "code": e.code}
            return response

        try:
            # 并发请求使用唯一临时文件名，避免同名上传文件互相覆盖
            err, ctx = await prepare_asr_context(request, unique_tmp=True)
            if err:
                response = err
                return response
            try:
                response = await run_qwen_asr(ctx)
                return response
            except VadPoolError as e:
                # VAD 排队超时 / worker 异常：错误码 4014，便于上游与 Qwen 侧错误区分
                logger.error(f"[V2] VAD 阶段异常: {e.message}")
                update_fail_task()
                response = {"msg": e.message, "code": e.code}
                return response
            except ModelUnavailableError as e:
                logger.warning(f"[V2] 本地模型不可用: {e.message}")
                update_fail_task()
                response = {"msg": e.message, "code": e.code}
                return response
            except Exception as e:  # noqa: BLE001 - 兜底，避免整条链路 500
                logger.exception(f"[V2] 转写异常: {e}")
                response = {"msg": "未知错误", "code": 4005}
                return response
            finally:
                cleanup_started = time.perf_counter()
                cleanup_temp_files(ctx.tmp_paths)
                cleanup_ms = (time.perf_counter() - cleanup_started) * 1000
        finally:
            await slots.release()
    finally:
        payload = response if isinstance(response, dict) else {}
        phases = get_phase_stats()
        phases["cleanup_ms"] = cleanup_ms
        phase_code = payload.get("code")
        record_request(
            request_id=request_id,
            task_id=getattr(ctx, "task_id", None),
            filename=getattr(ctx, "filename", getattr(request.audioFile, "filename", None)),
            started_at=request_started,
            status="completed" if phase_code in (None, 0) else "failed",
            http_status=200,
            code=phase_code,
            # 当前 Qwen 客户端按每个段执行重试；这里记录本次配置的重试上限，
            # 便于把异常请求与重试策略关联，实际重试细节仍由 Qwen 日志保留。
            retry_count=settings.qwen_retry_count,
            valid_segments=len(payload.get("segments") or []) if isinstance(payload, dict) else None,
            phases=phases,
            error=payload.get("msg") if phase_code not in (None, 0) else None,
        )
