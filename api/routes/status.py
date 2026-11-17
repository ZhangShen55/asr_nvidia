import asyncio

from fastapi import APIRouter, Request

from core.status import build_status_payload
from core.resource_diagnostics import status_snapshot
from utils.asr_stats import read_stats, clear_processing_tasks

router = APIRouter()


@router.get("/get_status")
async def get_status(request: Request):
    # npu-smi/procfs/cache扫描可能等待数秒；仅重采样进入线程，
    # 模型池/准入等可变状态仍在主事件循环组装，避免跨线程访问竞态。
    try:
        resource = await asyncio.to_thread(status_snapshot)
    except Exception as exc:
        resource = {"available": False, "reason": f"资源采样失败: {type(exc).__name__}: {exc}"}
    stats = read_stats()
    run_start_time = request.app.state.run_start_time
    return build_status_payload(stats, run_start_time, resource_snapshot=resource)


@router.delete("/clear_tasks_list")
async def clear_tasks_list():
    clear_processing_tasks()
    return {"msg": "正在处理的任务列表已清空", "code": 0}
