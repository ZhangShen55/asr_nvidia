from fastapi import APIRouter, Request

from core.status import build_status_payload
from utils.asr_stats import read_stats, clear_processing_tasks

router = APIRouter()


@router.get("/get_status")
async def get_status(request: Request):
    stats = read_stats()
    run_start_time = request.app.state.run_start_time
    return build_status_payload(stats, run_start_time)


@router.delete("/clear_tasks_list")
async def clear_tasks_list():
    clear_processing_tasks()
    return {"msg": "正在处理的任务列表已清空", "code": 0}
