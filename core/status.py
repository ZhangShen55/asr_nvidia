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
from core.segment_pipeline import pending_snapshot

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



def _cam_model_workers(local_models):
    """主循环取得的模型池缓存，含loading/retired实例；不触发runner加载或RPC。"""
    pool = local_models.get("speech_campplus", {})
    return [row for row in pool.get("model_workers", [])
            if isinstance(row, dict) and (row.get("execution_mode") == "process"
                                         or pool.get("execution_mode") == "process")]


def _worker_identity_matches(expected, actual):
    if actual.get("available") is False:
        return False
    if actual.get("pid") is not None and actual["pid"] != expected.get("pid"):
        return False
    ticks, measured_ticks = expected.get("start_ticks"), actual.get("start_ticks")
    created, measured_created = expected.get("create_time"), actual.get("create_time")
    same_time = (type(created) in (int, float) and type(measured_created) in (int, float)
                 and abs(created - measured_created) < .001)
    if ticks is not None and measured_ticks is not None:
        verified = str(ticks) == str(measured_ticks) and (
            created is None or measured_created is None or same_time)
    else:
        verified = same_time
    sid, measured_sid = expected.get("sid"), actual.get("sid")
    return verified and (sid is None or measured_sid is None or sid == measured_sid)


def _classify_npu_apps(resource, qwen, local_models=None):
    """只匹配缓存OS出生身份与主循环模型快照；GET不扫描进程、不发RPC。"""
    npu = dict(resource.get("npu", {"available": False, "reason": "未采集"}))
    workers = {}
    for worker in qwen.get("model_workers", []):
        if not isinstance(worker, dict) or type(worker.get("worker_pid")) is not int:
            continue
        expected = {"pid": worker["worker_pid"], "start_ticks": worker.get("worker_start_ticks"),
                    "create_time": worker.get("worker_create_time")}
        workers.setdefault(expected["pid"], []).append(("qwen", expected, worker))
    for worker in _cam_model_workers(local_models or {}):
        expected = worker.get("worker_identity")
        if isinstance(expected, dict) and type(expected.get("pid")) is int:
            workers.setdefault(expected["pid"], []).append(("cam", expected, worker))
    cached_service_pids = {row.get("pid") for rows in resource.get("services", {}).values() for row in rows}
    own, other, unverified = [], [], []
    for original in npu.get("compute_apps", []):
        row = dict(original)
        pid = row.get("pid")
        if pid == os.getpid():
            row["service_role"] = "backend"
            own.append(row)
        elif pid in workers:
            candidates = workers[pid]
            matching = [(kind, expected, worker) for kind, expected, worker in candidates
                        if worker.get("resource_reaped") is not True and worker.get("worker_alive") is not False
                        and _worker_identity_matches(expected, row.get("process_identity", {}))]
            if len(matching) == 1:
                kind, expected, worker = matching[0]
                row.update(service_role=f"{kind}_model_worker", ownership_identity_verified=True)
                own.append(row)
            else:
                kinds = sorted({kind for kind, _, _ in candidates})
                row.update(service_role="unverified_" + "_".join(kinds) + "_pid",
                           ownership_identity_verified=False,
                           ownership_reason="birth identity unavailable/mismatched, retired worker, or ambiguous registry")
                unverified.append(row)
        elif pid in cached_service_pids:
            row["service_role"] = "cached_service_process"
            own.append(row)
        else:
            identity = row.get("process_identity", {})
            if identity.get("start_ticks") is None or type(identity.get("create_time")) not in (int, float):
                row.update(service_role="unverified_process_identity", ownership_identity_verified=False)
                unverified.append(row)
            else:
                other.append(row)
    npu.update(service_compute_apps=own, other_compute_apps=other, unverified_compute_apps=unverified,
               backend_compute_apps=[row for row in own if row["service_role"] == "backend"],
               qwen_compute_apps=[row for row in own if row["service_role"] == "qwen_model_worker"],
               cam_compute_apps=[row for row in own if row["service_role"] == "cam_model_worker"],
               ownership_scope="cached resource sample PID/birth identity matched to current backend and registered model workers; not fresh OS sampling")
    return npu


def build_status_payload(stats: dict, run_start_time: datetime, *, resource_snapshot=None,
                         qwen_recent_batch_limit=None) -> dict:
    """组装状态；HTTP预采样重资源，直接同步调用保留原入口兼容。"""
    run_time = datetime.utcnow() - run_start_time
    processing_names = stats.get("processing_task_names", {})
    if not isinstance(processing_names, dict):
        processing_names = {}
    admission = get_slot_manager().snapshot()
    qwen = (get_qwen_status() if qwen_recent_batch_limit is None else
            get_qwen_status(recent_batch_limit=qwen_recent_batch_limit))
    local_models = get_local_model_status()
    cam_workers = _cam_model_workers(local_models)
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
        "npu": _classify_npu_apps(resource, qwen, local_models),
        "qwen_execution": qwen.get("execution_mode", "unknown"),
        "qwen_memory_process_scope": qwen.get("memory_process_scope", "unknown"),
        "model_workers": qwen.get("model_workers", []),
        "cam_execution": getattr(settings, "model_device_speech_campplus_execution_mode", "thread"),
        "cam_model_workers": cam_workers,
        "cam_memory_process_scope": "each CAM worker cached allocator/last RPC and OS-sampled RSS/HBM separate from parent; no cross-time peak sums",
        "sampling": resource.get("sampling", {"mode": "direct_sync_compatibility", "age_s": None, "in_progress": False}),
        "cache": resource.get("cache", {"available": False, "reason": "未采集"}),
        "processes": {
            name: {
                "count": len(rows),
                "pids": [row.get("pid") for row in rows if row.get("pid") is not None],
            }
            for name, rows in services.items()
        },
        "requestTelemetry": request_telemetry_snapshot(include_events=False),
        "segmentPipeline": pending_snapshot(),
    }
    if qwen.get("execution_mode") == "process":
        known = [row for row in qwen.get("model_workers", []) if type(row.get("worker_pid")) is int]
        alive = [row for row in known if row.get("worker_alive") is True]
        resource_diagnostics["processes"]["qwen_model_worker"] = {
            "count": len(alive), "pids": [row["worker_pid"] for row in alive],
            "unknown_liveness_count": sum(row.get("worker_alive") is None for row in known),
            "identities": [{key: row.get(key) for key in ("worker_pid", "worker_create_time", "worker_start_ticks", "worker_alive")}
                           for row in known],
            "source": "cached model-worker liveness/registered identities; resource sample may be older",
        }
    if resource_diagnostics["cam_execution"] == "process" or cam_workers:
        known = [row for row in cam_workers if isinstance(row.get("worker_identity"), dict)
                 and type(row["worker_identity"].get("pid")) is int]
        unreaped = [row for row in cam_workers if row.get("resource_reaped") is not True]
        alive = [row for row in known if row.get("worker_alive") is True and row.get("resource_reaped") is not True]
        unknown = [row for row in unreaped if row.get("worker_alive") is None
                   or not isinstance(row.get("worker_identity"), dict)
                   or type(row["worker_identity"].get("pid")) is not int]
        resource_diagnostics["processes"]["cam_model_worker"] = {
            "count": None if unknown else len(alive),
            "pids": [row["worker_identity"]["pid"] for row in alive],
            "registered_not_reaped_count": len(unreaped),
            "unknown_liveness_count": len(unknown),
            "identities": [{"worker_identity": row.get("worker_identity"),
                            "worker_alive": row.get("worker_alive"),
                            "resource_reaped": row.get("resource_reaped"),
                            "birth_in_progress": row.get("birth_in_progress")}
                           for row in cam_workers],
            "source": "cached model-worker liveness/registered identities; resource sample may be older; unknown is not zero",
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
        "localModels": local_models,
        "qwenModel": qwen,
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
