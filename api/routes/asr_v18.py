from fastapi import APIRouter, Depends

from entity.data import AsrRequestParams, get_asr_params
from api.routes.asr_common import cleanup_temp_files, prepare_asr_context, run_paraformer_asr

router = APIRouter()


@router.post("/v1.1.8/seacraft_asr")
async def api_asr_v18(request: AsrRequestParams = Depends(get_asr_params)):
    """语音转写（仅 Paraformer，不含小语种 Whisper）+ 身份判定 + 情绪 + 音轨分离"""
    err, ctx = await prepare_asr_context(request)
    if err:
        return err

    try:
        return await run_paraformer_asr(ctx)
    finally:
        cleanup_temp_files(ctx.tmp_paths)
