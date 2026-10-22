from fastapi import APIRouter, Depends

from entity.data import AsrRequestParams, get_asr_params
from core.config import settings
from core.models import get_whisper_model
from api.routes.asr_common import (
    cleanup_temp_files,
    prepare_asr_context,
    run_paraformer_asr,
    run_whisper_asr,
)

router = APIRouter()
PARAFORMER_LANGUAGES = {"auto", "zh", "en"}
WHISPER_LANGUAGES = {"fr"}


@router.post("/v1.1.8/seacraft_asr")
async def api_asr_v18(request: AsrRequestParams = Depends(get_asr_params)):
    """语音转写：中英文走 Paraformer，法语走 Whisper。"""
    language = (request.language or "").strip().lower()
    request.language = language

    whisper_model = None
    if language in PARAFORMER_LANGUAGES:
        pass
    elif language in WHISPER_LANGUAGES:
        if not settings.open_mul_lang:
            return {"msg": "未开启小语种识别或模型未就绪", "code": 4003}
        whisper_model = get_whisper_model()
        if whisper_model is None:
            return {"msg": "未开启小语种识别或模型未就绪", "code": 4003}
    else:
        return {"msg": f"不支持的语言: {language}", "code": 4009}

    err, ctx = await prepare_asr_context(request)
    if err:
        return err

    try:
        if whisper_model is not None:
            return await run_whisper_asr(ctx, whisper_model)
        return await run_paraformer_asr(ctx)
    finally:
        cleanup_temp_files(ctx.tmp_paths)
