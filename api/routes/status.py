from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from core.status import build_status_payload
from core.resource_diagnostics import unknown_resource_snapshot
from utils.asr_stats import read_stats, clear_processing_tasks

router = APIRouter()


@router.get("/get_status")
async def get_status(request: Request):
    # 每app只维护一个TTL singleflight；首次快照unknown，GET永不等待重资源采样。
    sampler = getattr(request.app.state, "resource_sampler", None)
    if sampler is None:
        resource = unknown_resource_snapshot("资源采样器尚未由app lifespan初始化")
        resource["sampling"] = {"has_snapshot": False, "age_s": None, "in_progress": False,
                                "last_error": None, "lifecycle": "not_initialized"}
    else:
        resource = sampler.snapshot()
    stats = read_stats()
    run_start_time = request.app.state.run_start_time
    payload = build_status_payload(stats, run_start_time, resource_snapshot=resource,
                                   qwen_recent_batch_limit=8)
    # payload只含JSON内建类型；直接render避免FastAPI再次递归编码完整诊断树。
    return JSONResponse(content=payload)


@router.delete("/clear_tasks_list")
async def clear_tasks_list():
    clear_processing_tasks()
    return {"msg": "正在处理的任务列表已清空", "code": 0}
