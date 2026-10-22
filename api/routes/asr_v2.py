"""v2 转写接口：Qwen3-ASR 链路。"""

import logging

from fastapi import APIRouter, Depends

from api.routes.asr_common import cleanup_temp_files, prepare_asr_context
from api.routes.asr_qwen_common import run_qwen_asr
from core.config import settings
from core.qwen_asr_queue import QueueFullError, QueueTimeoutError, get_slot_manager
from entity.data import AsrRequestParams, get_asr_params

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v2/seacraft_asr")
async def api_asr_v2(request: AsrRequestParams = Depends(get_asr_params)):
    """语音转写：本地 VAD 切分后并发调用 Qwen3-ASR，响应结构与 /v1.1.8 同构。"""
    if not settings.qwen_asr_enabled:
        return {"msg": "Qwen 转写能力未开启", "code": 4003}

    slots = get_slot_manager()
    try:
        await slots.acquire()
    except QueueFullError as e:
        logger.warning(f"[V2] 处理队列已满: {e.message}")
        return {"msg": e.message, "code": e.code}
    except QueueTimeoutError as e:
        logger.warning(f"[V2] 排队超时: {e.message}")
        return {"msg": e.message, "code": e.code}

    try:
        # 并发请求使用唯一临时文件名，避免同名上传文件互相覆盖
        err, ctx = await prepare_asr_context(request, unique_tmp=True)
        if err:
            return err
        try:
            return await run_qwen_asr(ctx)
        except Exception as e:  # noqa: BLE001 - 兜底，避免整条链路 500
            logger.exception(f"[V2] 转写异常: {e}")
            return {"msg": "未知错误", "code": 4005}
        finally:
            cleanup_temp_files(ctx.tmp_paths)
    finally:
        await slots.release()
