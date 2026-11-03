"""状态查询的响应组装。

引擎标识与版本不再来自配置（配置项已随接口下线一并删除），
这里返回固定占位值，保证运维平台的字段结构保持不变。
"""

from datetime import timedelta, datetime

from core.models import get_local_model_status
from core.qwen_asr_queue import get_slot_manager
from core.request_telemetry import snapshot as request_telemetry_snapshot
from core.resource_diagnostics import status_snapshot
from core.vad_pool import get_vad_pool
from core.config import settings

# 引擎标识：配置项已删除，返回固定占位值以保持响应字段结构不变
ID_ENGINE_PLACEHOLDER = "1"

# 应用版本：固定值
APP_VERSION = "latest"


def format_timedelta(td: timedelta) -> str:
    total_seconds = int(td.total_seconds())
    days = total_seconds // (3600 * 24)
    hours = (total_seconds // 3600) % 24
    minutes = (total_seconds // 60) % 60
    seconds = total_seconds % 60
    return f"{days}天 {hours}小时 {minutes}分 {seconds}秒"


def build_status_payload(stats: dict, run_start_time: datetime) -> dict:
    """组装 `/get_status` 的响应体。"""
    run_time = datetime.utcnow() - run_start_time
    admission = get_slot_manager().snapshot()
    resource = status_snapshot()
    services = resource.get("services", {})
    resource_diagnostics = {
        "available": True,
        "timestamp": resource.get("timestamp"),
        "host": resource.get("host", {"available": False, "reason": "未采集"}),
        "vmstat": resource.get("vmstat", {"available": False, "reason": "未采集"}),
        "backend": resource.get("backend", {"available": False, "reason": "未采集"}),
        "gpu": resource.get("gpu", {"available": False, "reason": "未采集"}),
        "cache": resource.get("cache", {"available": False, "reason": "未采集"}),
        "processes": {
            name: {
                "count": len(rows),
                "pids": [row.get("pid") for row in rows if row.get("pid") is not None],
            }
            for name, rows in services.items()
        },
        "requestTelemetry": request_telemetry_snapshot(include_events=False),
    }
    return {
        "id_engine": ID_ENGINE_PLACEHOLDER,
        "status": "living",
        "appVersion": APP_VERSION,
        "nowTime": (datetime.utcnow() + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
        "runTime": format_timedelta(run_time),
        "totalHaveDoneProcessTasks": stats["offline"] + stats["online"],
        "totalFailedTasks": stats["total_failed_tasks"],
        "offlineDone": stats["offline"],
        "onlineDone": stats["online"],
        "queuedTaskList": stats["queued_task_ids"],
        "processingTaskList": stats["processing_task_ids"],
        **admission,
        # CAM++ / emotion2vec 模型池的加载状态与显存明细；VAD 状态统一放在 vadPool。
        "localModels": get_local_model_status(),
        # VAD 进程池的实时运行状况：设备、当前 worker 数、在途与排队请求
        "vadPool": get_vad_pool().stats(),
        # 资源诊断为附加字段；采集失败明确标记 available=false，不影响业务字段。
        "resourceDiagnostics": resource_diagnostics,
        "deployment": {
            "profile": settings.deployment_profile,
            "config_path": settings.config_path,
            "cuda_tensorrt": settings.is_cuda_tensorrt_profile,
        },
    }


__all__ = [
    "APP_VERSION",
    "ID_ENGINE_PLACEHOLDER",
    "build_status_payload",
    "format_timedelta",
]
