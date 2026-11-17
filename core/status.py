"""状态查询的响应组装。

引擎标识与版本不再来自配置（配置项已随接口下线一并删除），
这里返回固定占位值。状态中的并发、模型和资源字段来自当前 v3 运行时；旧版
onlineDone、queuedTaskList 不再对外返回，processingTaskList 保留为当前已准入请求的文件名。
"""

from datetime import timedelta, datetime
import os

from core.npu_resource_status import get_live_allocator_status
from core.models import get_local_model_status
from core.qwen_asr_queue import get_slot_manager
from core.qwen_asr_client import get_qwen_status
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


def build_status_payload(stats: dict, run_start_time: datetime, *, resource_snapshot=None) -> dict:
    """组装状态；HTTP预采样重资源，直接同步调用保留原入口兼容。"""
    run_time = datetime.utcnow() - run_start_time
    processing_names = stats.get("processing_task_names", {})
    if not isinstance(processing_names, dict):
        processing_names = {}
    admission = get_slot_manager().snapshot()
    if resource_snapshot is None:
        try:
            resource = status_snapshot()
        except Exception as exc:
            resource = {"available": False, "reason": f"资源采样失败: {type(exc).__name__}: {exc}"}
    else:
        resource = resource_snapshot
    # live allocator是主服务进程/设备聚合，不能冒充逐模型或整卡HBM。
    # helper只在本进程NPU已初始化时采样，首次状态查询不会创建上下文。
    backend = dict(resource.get("backend", {"available": False, "reason": "未采集"}))
    backend["npu_allocator"] = get_live_allocator_status(settings.qwen_local_device)
    services = resource.get("services", {})
    resource_diagnostics = {
        "available": bool(resource.get("available", True)),
        "timestamp": resource.get("timestamp"),
        "host": resource.get("host", {"available": False, "reason": "未采集"}),
        "vmstat": resource.get("vmstat", {"available": False, "reason": "未采集"}),
        "backend": backend,
        "npu": resource.get("npu", {"available": False, "reason": "未采集"}),
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
    if not resource_diagnostics["available"]:
        resource_diagnostics["reason"] = resource.get("reason", "资源采样不可用")
    return {
        "id_engine": ID_ENGINE_PLACEHOLDER,
        "pid": os.getpid(),
        "status": "living",
        "appVersion": APP_VERSION,
        "nowTime": (datetime.utcnow() + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
        "runTime": format_timedelta(run_time),
        "totalHaveDoneProcessTasks": stats.get("offline", 0) + stats.get("online", 0),
        "totalFailedTasks": stats["total_failed_tasks"],
        "offlineDone": stats.get("offline", 0),
        # 只报告已经通过音频准入、正在执行完整处理生命周期的音频文件名。
        # 通过 request_id 映射文件名，保证同名文件并发时仍能独立移除。
        "processingTaskList": [
            processing_names.get(task_id, task_id)
            for task_id in stats.get("processing_task_ids", [])
        ],
        **admission,
        # CAM++ / emotion2vec 模型池的加载状态与明确口径的 HBM 明细；VAD 状态统一放在 vadPool。
        "localModels": get_local_model_status(),
        "qwenModel": get_qwen_status(),
        # VAD 进程池的实时运行状况：设备、当前 worker 数、在途与排队请求
        "vadPool": get_vad_pool().stats(),
        # 资源诊断为附加字段；采集失败明确标记 available=false，不影响业务字段。
        "resourceDiagnostics": resource_diagnostics,
        "deployment": {
            "profile": settings.deployment_profile,
            "config_path": settings.config_path,
            "backend": "torch_npu" if any(device.startswith("npu:") for device in (
                settings.vad_device + settings.model_device_speech_campplus_devices +
                settings.model_device_emotion2vec_devices +
                ([settings.qwen_local_device] if settings.qwen_asr_enabled else [])
            )) else "pytorch_cpu",
            "qwen_device": settings.qwen_local_device if settings.qwen_asr_enabled else None,
            "vad_devices": settings.vad_device,
        },
    }


__all__ = [
    "APP_VERSION",
    "ID_ENGINE_PLACEHOLDER",
    "build_status_payload",
    "format_timedelta",
]
