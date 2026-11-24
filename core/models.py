"""v3 链路的本地模型加载、模型池调度与显存登记。"""

import asyncio
import logging
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, Dict

import torch

from core.config import settings
from core.async_utils import await_completion, run_in_thread
from core.request_telemetry import record_enhancement_phase
from core.local_inference import (
    InstanceFailure, TorchRunner, TorchNpuRunner, backend_for_device, npu_api,
)

logger = logging.getLogger(__name__)

# 延迟导入 FunASR，便于状态接口和配置单测在没有音频后端时启动；测试可以
# patch 这个模块属性替换真实模型构造器。
AutoModel = None
CamProcessRunner = None

_model_spk_v3 = None
_model_emotion_v3 = None
_model_lock = asyncio.Lock()
_LOCAL_MODELS = ("vad", "speech_campplus", "emotion2vec")


class ModelUnavailableError(RuntimeError):
    """请求要求的本地模型池未启用或没有健康副本。"""

    _DETAILS = {
        "speech_campplus": (4015, "CAM++ 模型未启用或没有可用实例（没有可用副本）"),
        "emotion2vec": (4016, "emotion2vec 模型未启用或没有可用实例（没有可用副本）"),
        "spk_emo": (4018, "CAM++ 与 emotion2vec 能力均未启用"),
    }

    def __init__(self, name: str, *, reason: str | None = None):
        code, message = self._DETAILS.get(name, (4015, f"{name} 模型未启用或没有可用实例"))
        self.model_name = name
        self.code = code
        self.message = message
        self.reason = reason
        super().__init__(message)


def validate_v3_capabilities(need_role: bool = False, need_emotion: bool = False) -> None:
    """先检查完整需求，避免一项已加载后才发现另一项被配置关闭。"""
    role_disabled = need_role and not settings.speech_campplus_enabled
    emotion_disabled = need_emotion and not settings.emotion2vec_enabled
    if role_disabled and emotion_disabled:
        raise ModelUnavailableError("spk_emo", reason="both_disabled")
    if role_disabled:
        raise ModelUnavailableError("speech_campplus", reason="role_disabled")
    if emotion_disabled:
        raise ModelUnavailableError("emotion2vec", reason="emotion_disabled")


local_model_status: Dict[str, Dict[str, Any]] = {
    name: {
        "loaded": False, "device": None, "allocated_mb": 0.0, "instances": 0,
        "configured_instances": 0,
        "devices": [],
        "available_instances": 0, "failed_instances": 0, "busy": 0,
        "waiting": 0, "failure": None,
        "wait_total_ms": 0.0, "inference_total_ms": 0.0,
        "calls": 0, "last_wait_ms": 0.0,
        "waiting_summary": {},
    }
    for name in _LOCAL_MODELS
}


def device_allocated_mb(device: str) -> float | None:
    """只采样本进程 PyTorch 分配器；失败为未知，不能假报资源为零。"""
    if device == "cpu":
        return 0.0
    try:
        if backend_for_device(device) != "torch_npu":
            return None
        api = npu_api()
        api.set_device(device)
        return float(api.memory_allocated(device)) / (1024 ** 2)
    except Exception:
        return None


def record_model_memory(name: str, device: str, allocated_mb: float) -> None:
    if name not in local_model_status:
        return
    entry = local_model_status[name]
    entry["loaded"] = True
    entry["device"] = device
    entry["instances"] = max(1, entry["instances"])
    entry["allocated_mb"] = round(max(0.0, allocated_mb), 2)


def record_vad_runtime(
    device,
    instances,
    loaded,
    allocated_mb,
    pool_size=None,
    pool_max=None,
    *,
    configured_instances=None,
    successful_instances=None,
    failed_instances=0,
    devices=None,
    busy=0,
    waiting=0,
    failures=None,
):
    """登记 VAD worker 状态，委托给未编译的动态字典 helper。"""
    from model_status_runtime import update_vad_runtime

    update_vad_runtime(
        local_model_status["vad"], device, instances, loaded, allocated_mb,
        configured_instances, successful_instances, failed_instances,
        devices, busy, waiting, failures,
    )


def get_local_model_status() -> Dict[str, Dict[str, Any]]:
    for pool in _MODEL_POOLS.values():
        pool.refresh_status()
    # VAD 由 vadPool 单独暴露；localModels 只描述按需加载的增强模型池。
    return {
        name: dict(local_model_status[name])
        for name in ("speech_campplus", "emotion2vec")
    }


def _resolve_device(index: int | str) -> str:
    if isinstance(index, str):
        spec = index.strip().lower()
        if spec == "cpu":
            return spec
        if spec.startswith("npu:") and spec[4:].isdigit():
            api = npu_api()
            number = int(spec[4:])
            if not api.is_available() or number >= api.device_count():
                raise RuntimeError(f"NPU 设备不可用: {spec}")
            return spec
    raise RuntimeError(f"本地模型设备 {index!r} 不支持；请将旧 CUDA 配置迁移为 cpu 或 npu:N")


def resolve_vad_device() -> str:
    devices = settings.vad_device
    if not devices:
        raise RuntimeError("vad.vad_device 不能为空，至少配置一个 VAD worker")
    return _resolve_device(devices[0])


class ModelPool:
    """按设备数组加载模型副本；每个句柄同一时刻只服务一个任务。"""

    def __init__(self, name: str, model_dir: str, device_getter=None, count_getter=None,
                 *, devices_getter=None, fp16=False, enabled_getter=None, prefetch_depth_getter=None):
        self.name = name
        self.model_dir = model_dir
        self.device_getter = device_getter
        self.count_getter = count_getter
        self.devices_getter = devices_getter
        self._return_device = devices_getter is not None
        self.fp16 = fp16
        self.enabled_getter = enabled_getter
        self.prefetch_depth_getter = prefetch_depth_getter
        self._admission_waiters = {}
        self._last_released = []
        self.handoff_total_ms = 0.0
        self.handoff_count = 0
        self.last_handoff_ms = 0.0
        self._closing = False
        self.models: list[Any] = []
        self.devices: list[str | None] = []
        self.locks: list[asyncio.Lock] = []
        self.failures: list[str | None] = []
        self._unclosed_instances: dict[int, Any] = {}
        self._load_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        # lazy代理必须在ensure_loaded出生之前登记；关闭和登记共享此线程锁。
        self._ownership_lock = threading.Lock()
        self._process_runners = {}
        self._health_cache = {}
        self._condition = asyncio.Condition()
        self._waiters = deque()
        self._next_instance = 0
        self.waiting = 0
        self.total_allocated_mb = 0.0
        self.memory_complete = True
        self.wait_total_ms = 0.0
        self.inference_total_ms = 0.0
        self.calls = 0
        self.last_wait_ms = 0.0
        self._leased_instance = ContextVar(f"{name}_leased_instance", default=None)
        self._instance_metrics = []
        self._outcomes = dict.fromkeys(("success", "empty", "failed", "cancelled"), 0)
        self.duration_skips = 0

    def _metrics_for(self, index):
        # 固定池最多只有配置数组那么多记录，不保留逐段日志或模型引用。
        while len(self._instance_metrics) <= index:
            self._instance_metrics.append(dict.fromkeys((
                "load_attempts", "load_failures", "load_ms", "load_work_ms",
                "load_dispatch_wait_ms", "leases", "wait_ms", "calls", "inference_ms",
                "success", "empty", "failed", "cancelled",
            ), 0))
        return self._instance_metrics[index]

    def _record_phase(self, key, value):
        prefix = {"speech_campplus": "spk", "emotion2vec": "emo"}.get(self.name)
        if prefix is not None:
            record_enhancement_phase(f"{prefix}_{key}", value)

    def _metrics_status(self):
        metrics = [dict(row, instance_index=index) for index, row in enumerate(self._instance_metrics)]
        return {
            "telemetry_scope": "since_last_close; host elapsed includes dispatch/device wait, not kernel time",
            "instance_metrics": metrics,
            "called_instances": sum(row["calls"] > 0 for row in metrics),
            "unattributed_calls": max(0, self.calls - sum(row["calls"] for row in metrics)),
            "inference_outcomes": dict(self._outcomes),
            "unclassified_calls": max(0, self.calls - sum(self._outcomes.values())),
            "duration_skips": self.duration_skips,
            "load_total_ms": round(sum(row["load_ms"] for row in metrics), 3),
        }

    @property
    def configured_instances(self) -> int:
        return len(self.configured_devices)

    @property
    def configured_devices(self) -> list[str]:
        if self.devices_getter is not None:
            devices = list(self.devices_getter())
        else:
            count = self.count_getter() if self.count_getter is not None else 1
            devices = [self.device_getter()] * count
        return devices

    @property
    def execution_mode(self) -> str:
        if self.name != "speech_campplus":
            return "thread"
        return getattr(settings, "model_device_speech_campplus_execution_mode", "thread")

    @staticmethod
    def _process_snapshot(model):
        try:
            return model.snapshot()
        except Exception as exc:
            return {"loaded": False, "worker_alive": None,
                    "snapshot_error": f"{type(exc).__name__}: {exc}"}

    def _is_healthy(self, model) -> bool:
        if model is None or not getattr(model, "healthy", True):
            return False
        if getattr(model, "execution_mode", "thread") != "process":
            return True
        # Condition谓词只读缓存，避免notify_all把128等待者放大成反复OS扫描。
        return self._health_cache.get(id(model), False)

    @staticmethod
    def _healthy_process_state(model, state):
        return bool(getattr(model, "healthy", True) and state.get("loaded")
                    and state.get("worker_alive") is True
                    and not state.get("closing") and not state.get("closed"))

    def _validate_process_candidate(self, model):
        if getattr(model, "execution_mode", "thread") != "process":
            return self._is_healthy(model)
        state = self._process_snapshot(model)
        self._health_cache[id(model)] = self._healthy_process_state(model, state)
        return self._health_cache[id(model)]

    def _owned_process_entries(self):
        with self._ownership_lock:
            return list(self._process_runners.values())

    def _process_entry(self, model):
        with self._ownership_lock:
            return self._process_runners.get(id(model))

    def _request_process_close(self, entry):
        if entry["close_requested"]:
            return
        entry["close_requested"] = True
        try:
            # 同步发布绝对deadline；不等线程池排队后才启动新的grace。
            entry["runner"].request_close()
        except Exception as exc:
            entry["close_error"] = entry["close_error"] or f"request_close: {type(exc).__name__}: {exc}"

    async def _close_process_entry(self, entry):
        model = entry["runner"]
        self._request_process_close(entry)
        before = self._process_snapshot(model)
        if (before.get("forced_cleanup") or before.get("normal_exit_ok") is False
                or before.get("worker_exitcode") not in (None, 0)
                or (before.get("loaded") and before.get("worker_alive") is False
                    and not before.get("resource_reaped"))):
            entry["close_error"] = entry["close_error"] or "CAM worker prior abnormal cleanup"
        try:
            await run_in_thread(model.close)
        except Exception as exc:
            entry["close_error"] = entry["close_error"] or f"{type(exc).__name__}: {exc}"
        # 正式runner只在DONE或确认死亡后返回；即使坏替身违反契约也不提前归还。
        while not self._process_snapshot(model).get("resource_reaped"):
            await asyncio.sleep(0.01)
        state = self._process_snapshot(model)
        if (state.get("forced_cleanup") or state.get("normal_exit_ok") is False
                or state.get("worker_exitcode") not in (None, 0)):
            entry["close_error"] = entry["close_error"] or "CAM worker abnormal confirmed-death cleanup"
        entry["close_completed"] = True
        entry["last_state"] = state

    def _ensure_process_close_task(self, entry):
        self._request_process_close(entry)
        if entry["close_task"] is None:
            entry["close_task"] = asyncio.create_task(self._close_process_entry(entry))
        return entry["close_task"]

    def _check_enabled(self) -> None:
        if self._closing:
            raise ModelUnavailableError(self.name, reason="closing")
        if self.enabled_getter is not None and not self.enabled_getter():
            raise ModelUnavailableError(self.name)

    async def ensure_loaded(self, *, observe_process: bool = True) -> None:
        self._check_enabled()
        loading_needed = len(self.models) < self.configured_instances
        lock_started = time.perf_counter()
        async with self._load_lock:
            if loading_needed:
                self._record_phase("load_lock_wait_ms", (time.perf_counter() - lock_started) * 1000)
            self._check_enabled()
            if len(self.models) >= self.configured_instances:
                self.refresh_status(observe_process=observe_process)
                return
            # 包住整个登记过程，取消不能遗失刚在线程中创建的运行器。
            await await_completion(self._load_instances())

    def _build_instance(self, configured_device):
        if self.execution_mode == "process":
            # 子进程验证实际可见卡号；父进程不能为CAM分配器探测而初始化NPU。
            device = configured_device.strip().lower() if isinstance(configured_device, str) else configured_device
            if backend_for_device(device) != "torch_npu":
                raise ValueError("CAM process execution requires a configured npu:N device")
            global CamProcessRunner
            if CamProcessRunner is None:
                from core.cam_process import CamProcessRunner as ProcessRunner
                CamProcessRunner = ProcessRunner
            with self._ownership_lock:
                if self._closing:
                    raise ModelUnavailableError(self.name, reason="closing")
                model = CamProcessRunner(
                    self.model_dir, device,
                    startup_timeout_s=getattr(settings, "model_device_speech_campplus_process_startup_timeout_s", 60),
                    shutdown_grace_s=getattr(settings, "model_device_speech_campplus_process_shutdown_grace_s", 30),
                )
                self._process_runners[id(model)] = {
                    "runner": model, "index": len(self.models), "device": device,
                    "close_requested": False, "close_task": None,
                    "close_completed": False, "close_error": None, "last_state": None,
                }
            model.ensure_loaded()
            return model, device, None
        device = _resolve_device(configured_device)
        backend = backend_for_device(device)
        before = device_allocated_mb(device)
        if backend == "torch_npu":
            model = TorchNpuRunner(self.name, self.model_dir, device)
        else:
            global AutoModel
            import torch
            from core.model_loading import model_loading_guard
            with model_loading_guard(dtype=torch.float32):
                if AutoModel is None:
                    from funasr import AutoModel as FunasrAutoModel
                    AutoModel = FunasrAutoModel
                model = TorchRunner(AutoModel(model=self.model_dir, device=device,
                                    disable_update=True, disable_pbar=True), device)
        after = device_allocated_mb(device)
        allocated = max(0.0, after - before) if before is not None and after is not None else None
        return model, device, allocated

    async def _load_instances(self):
        devices = self.configured_devices
        while len(self.models) < len(devices):
            self._check_enabled()
            configured_device = devices[len(self.models)]
            metrics = self._metrics_for(len(self.models))
            metrics["load_attempts"] += 1
            load_started = time.perf_counter()
            work_ms = 0.0

            def timed_build():
                nonlocal work_ms
                work_started = time.perf_counter()
                try:
                    return self._build_instance(configured_device)
                finally:
                    work_ms = (time.perf_counter() - work_started) * 1000

            model, device, allocated, failure = None, configured_device, 0.0, None
            try:
                model, device, allocated = await run_in_thread(timed_build)
            except Exception as exc:  # noqa: BLE001
                failure = str(exc)
                logger.exception("[模型池] %s %s 副本加载失败", self.name, configured_device)
            finally:
                elapsed_ms = (time.perf_counter() - load_started) * 1000
                dispatch_ms = max(0.0, elapsed_ms - work_ms)
                for key, value in (("load_ms", elapsed_ms), ("load_work_ms", work_ms),
                                   ("load_dispatch_wait_ms", dispatch_ms)):
                    metrics[key] += value
                    self._record_phase(key, value)
                if failure is not None:
                    metrics["load_failures"] += 1
                    self._record_phase("load_failures", 1)
            if self._closing and self.execution_mode == "process":
                # loading线程刚返回也不能晚发布READY或继续构建后面的实例。
                for entry in self._owned_process_entries():
                    await await_completion(self._ensure_process_close_task(entry))
                raise ModelUnavailableError(self.name, reason="closing")
            self.models.append(model)
            self.devices.append(device)
            self.locks.append(asyncio.Lock())
            self._last_released.append(None)
            self.failures.append(failure)
            if self.execution_mode == "process":
                allocated = None
            self.memory_complete = self.memory_complete and allocated is not None
            self.total_allocated_mb += allocated or 0.0
            self.refresh_status()
            if self._closing:
                raise ModelUnavailableError(self.name, reason="closing")

    @property
    def prefetch_depth(self) -> int:
        return self.prefetch_depth_getter() if self.prefetch_depth_getter is not None else 5

    @property
    def waiting_capacity(self) -> int:
        return sum(self._is_healthy(model) for model in self.models) * self.prefetch_depth

    def _mark_grant(self, index: int) -> None:
        # 只统计有待执行任务的交接，不把无负载时的空闲时间当调度开销。
        if index < len(self._last_released) and self._last_released[index] is not None:
            self.last_handoff_ms = max(0.0, (time.perf_counter() - self._last_released[index]) * 1000)
            self.handoff_total_ms += self.last_handoff_ms
            self.handoff_count += 1
            self._last_released[index] = None

    def _refresh_hot_status(self) -> None:
        """调度转换只更新轻量计数；完整worker证据由显式status观察刷新。"""
        for model in self.models:
            if model is None or getattr(model, "execution_mode", "thread") != "process":
                continue
            cached_health = getattr(model, "cached_health", None)
            if cached_health is not None:
                state = cached_health()
                if (not state.get("loaded") or state.get("closing") or state.get("closed")
                        or state.get("healthy") is False):
                    self._health_cache[id(model)] = False
            # 不把旧READY/缓存alive当作新证据：OS已判dead的false一直保留，
            # 只有真实selected验证或显式refresh_status才可更新该观察。
        healthy = sum(self._is_healthy(model) for model in self.models)
        busy = sum(lock.locked() for lock in self.locks)
        now = time.perf_counter()
        oldest = min([t for _, t in self._waiters] + list(self._admission_waiters.values()), default=now)
        local_model_status[self.name].update({
            **self._metrics_status(),
            "instances": healthy,
            "available_instances": 0 if self._closing else max(0, healthy - busy),
            "busy": busy,
            "waiting": self.waiting + len(self._admission_waiters),
            "prefetch_waiting": self.waiting,
            "waiting_capacity": self.waiting_capacity,
            "backpressure_waiting": len(self._admission_waiters),
            "oldest_wait_ms": round(max(0.0, (now - oldest) * 1000), 3),
            "handoff_count": self.handoff_count,
            "last_handoff_ms": round(self.last_handoff_ms, 3),
            "handoff_total_ms": round(self.handoff_total_ms, 3),
            "wait_total_ms": round(self.wait_total_ms, 2),
            "inference_total_ms": round(self.inference_total_ms, 2),
            "calls": self.calls,
            "last_wait_ms": round(self.last_wait_ms, 2),
            "waiting_summary": {
                "current": self.waiting + len(self._admission_waiters),
                "total_ms": round(self.wait_total_ms, 2),
                "average_ms": round(self.wait_total_ms / max(1, self.calls), 2),
            },
        })

    def refresh_status(self, *, observe_process: bool = True) -> None:
        if not observe_process and self.execution_mode == "process":
            self._refresh_hot_status()
            return
        entry = local_model_status[self.name]
        owned_entries = self._owned_process_entries()
        states = {id(owned["runner"]): self._process_snapshot(owned["runner"])
                  for owned in owned_entries}
        for model in self.models:
            if model is not None and getattr(model, "execution_mode", "thread") == "process":
                if id(model) not in states:
                    states[id(model)] = self._process_snapshot(model)
                self._health_cache[id(model)] = self._healthy_process_state(model, states[id(model)])
        loaded = [model for model in self.models if self._is_healthy(model)]
        configured_devices = self.configured_devices
        busy = sum(lock.locked() for lock in self.locks)
        details = []
        owned_by_index = {entry["index"]: entry for entry in owned_entries}
        worker_states = []
        for owned in owned_entries:
            state = states[id(owned["runner"])]
            worker_states.append(dict(state, pool_instance_index=owned["index"],
                                      configured_device=owned["device"], close_error=owned["close_error"]))
        for index, device in enumerate(configured_devices):
            model = self.models[index] if index < len(self.models) else None
            owned = owned_by_index.get(index)
            attempted = index < len(self.models) or owned is not None
            recorded_failure = self.failures[index] if index < len(self.failures) else None
            if owned and owned["close_error"]:
                recorded_failure = recorded_failure or owned["close_error"]
            backend = backend_for_device(device)
            observed = model if model is not None else (owned["runner"] if owned else None)
            healthy = self._is_healthy(model)
            process_state = states.get(id(observed), {})
            is_process = getattr(observed, "execution_mode", "thread") == "process"
            if not recorded_failure and is_process:
                if process_state.get("failure"):
                    recorded_failure = process_state["failure"]
                elif (process_state.get("forced_cleanup") or process_state.get("normal_exit_ok") is False
                      or process_state.get("worker_exitcode") not in (None, 0)):
                    recorded_failure = "worker abnormal cleanup"
                elif (process_state.get("healthy") is False or not getattr(observed, "healthy", True)
                      or (process_state.get("loaded") and process_state.get("worker_alive") is False
                          and not process_state.get("resource_reaped"))):
                    recorded_failure = "worker unavailable"
            elif (not recorded_failure and model is not None
                  and not getattr(model, "healthy", True)):
                recorded_failure = "worker unavailable"
            if recorded_failure:
                health = "close_failed" if self._closing else "failed"
            elif is_process:
                if process_state.get("closed") and process_state.get("resource_reaped"):
                    health = "closed"
                elif self._closing or process_state.get("closing"):
                    health = "closing"
                elif index >= len(self.models):
                    health = "loading"
                else:
                    health = "healthy" if healthy else "unknown"
            elif self._closing and attempted:
                health = "closed"
            else:
                health = "healthy" if healthy else ("failed" if attempted else "not_loaded")
            details.append({
                "instance_index": index,
                "telemetry": dict(self._instance_metrics[index]) if index < len(self._instance_metrics) else None,
                "configured_device": device, "configured_backend": backend,
                "busy": index < len(self.locks) and self.locks[index].locked(),
                "execution_mode": getattr(observed, "execution_mode", "thread"),
                "effective_backend": getattr(observed, "effective_backend", None),
                "actual_precision": getattr(observed, "actual_precision", None),
                "math_mode": getattr(observed, "math_mode", None),
                "operator_adaptations": getattr(observed, "operator_adaptations", None),
                "health": health,
                "failure": recorded_failure,
                "memory": (states.get(id(observed)) if getattr(observed, "execution_mode", "thread") == "process"
                           else getattr(observed, "memory", None)),
            })
        failed = [detail["failure"] for detail in details if detail["failure"]]
        now = time.perf_counter()
        oldest = min([t for _, t in self._waiters] + list(self._admission_waiters.values()), default=now)
        entry.update({
            **self._metrics_status(),
            "execution_mode": self.execution_mode,
            "model_workers": worker_states,
            "loaded": (any(model is not None and getattr(model, "execution_mode", "thread") != "process"
                           for model in self.models) or
                       any(state.get("loaded") and state.get("worker_alive") is True
                           for state in states.values())),
            "device": next((device for model, device in zip(self.models, self.devices)
                             if self._is_healthy(model)), None),
            "devices": list(self.devices) if self.models else configured_devices,
            "allocated_mb": round(self.total_allocated_mb, 2) if self.memory_complete else None,
            "torch_allocated_mb": round(self.total_allocated_mb, 2) if self.memory_complete else None,
            "runtime_hbm_mb": None if any(d.startswith("npu:") for d in configured_devices) else 0.0,
            "allocated_mb_scope": "PyTorch allocator delta at load; excludes runtime/non-PyTorch HBM and other processes",
            "instance_details": details,
            "instances": len(loaded),
            "configured_instances": self.configured_instances,
            "available_instances": 0 if self._closing else max(0, len(loaded) - busy),
            "closing": self._closing,
            "pending_close_instances": (sum(owned["close_requested"] and not owned["close_completed"]
                                            for owned in owned_entries)
                                        + len(self._unclosed_instances)
                                        + (sum(model is not None and getattr(model, "execution_mode", "thread") != "process"
                                               for model in self.models) if self._closing else 0)),
            "failed_instances": len(failed),
            "busy": busy,
            "waiting": self.waiting + len(self._admission_waiters),
            "prefetch_waiting": self.waiting,
            "prefetch_queue_depth": self.prefetch_depth,
            "waiting_capacity": self.waiting_capacity,
            "backpressure_waiting": len(self._admission_waiters),
            "oldest_wait_ms": round(max(0.0, (now - oldest) * 1000), 3),
            "actual_batch_size": 1,
            "handoff_count": self.handoff_count,
            "last_handoff_ms": round(self.last_handoff_ms, 3),
            "handoff_total_ms": round(self.handoff_total_ms, 3),
            "failure": failed[len(failed) - 1] if failed else None,
            "wait_total_ms": round(self.wait_total_ms, 2),
            "inference_total_ms": round(self.inference_total_ms, 2),
            "calls": self.calls,
            "last_wait_ms": round(self.last_wait_ms, 2),
            "waiting_summary": {
                "current": self.waiting + len(self._admission_waiters),
                "total_ms": round(self.wait_total_ms, 2),
                "average_ms": round(self.wait_total_ms / max(1, self.calls), 2),
            },
        })

    def _available_index(self) -> int | None:
        """按轮转游标选择一个健康且未占用的副本。"""
        count = len(self.models)
        if not count:
            return None
        for offset in range(count):
            index = (self._next_instance + offset) % count
            if (self._is_healthy(self.models[index]) and not self.locks[index].locked()
                    and self._validate_process_candidate(self.models[index])):
                self._next_instance = (index + 1) % count
                return index
        return None

    async def _dispatch_waiters(self) -> None:
        """在条件锁内按 FIFO 发放租约；取消的等待者会被跳过。"""
        if self._closing or not any(self._is_healthy(model) for model in self.models):
            while self._waiters:
                future, _ = self._waiters.popleft()
                self.waiting = max(0, self.waiting - 1)
                if not future.done():
                    future.set_exception(ModelUnavailableError(self.name))
            return
        while self._waiters:
            future, enqueued_at = self._waiters[0]
            if future.cancelled():
                self._waiters.popleft()
                self.waiting = max(0, self.waiting - 1)
                continue
            index = self._available_index()
            if index is None:
                if not any(self._is_healthy(model) for model in self.models):
                    await self._dispatch_waiters()
                return
            await self.locks[index].acquire()
            self._mark_grant(index)
            self._waiters.popleft()
            self.waiting = max(0, self.waiting - 1)
            wait_ms = (time.perf_counter() - enqueued_at) * 1000
            future.set_result((index, self.models[index], wait_ms))
        if not self._waiters and not self._admission_waiters:
            for index, lock in enumerate(self.locks):
                if not lock.locked():
                    self._last_released[index] = None

    @asynccontextmanager
    async def lease(self):
        self._check_enabled()
        # 每完整音频的显式ensure_loaded仍做OS健康预检；每段只在选中实例时验证。
        await self.ensure_loaded(observe_process=False)
        if not any(self._is_healthy(model) for model in self.models):
            raise ModelUnavailableError(self.name)
        selected = None
        future = None
        token = object()
        enqueued_at = time.perf_counter()
        try:
            async with self._condition:
                # 共享预取队列只存健康实例数×depth个待执行任务，溢出在生产端反压。
                # depth=0只等待空闲实例，绝不解释成无限队列。
                self._admission_waiters[token] = enqueued_at
                self.refresh_status(observe_process=False)
                await self._condition.wait_for(lambda: self._closing
                    or not any(self._is_healthy(model) for model in self.models)
                    or (next(iter(self._admission_waiters), None) is token
                        and ((not self._waiters and self._has_available())
                             or len(self._waiters) < self.waiting_capacity)))
                self._check_enabled()
                if not any(self._is_healthy(model) for model in self.models):
                    raise ModelUnavailableError(self.name)
                self._admission_waiters.pop(token, None)
                if not self._waiters:
                    index = self._available_index()
                    if index is not None:
                        await self.locks[index].acquire()
                        self._mark_grant(index)
                        selected = (index, self.models[index], (time.perf_counter() - enqueued_at) * 1000)
                if selected is None:
                    future = asyncio.get_running_loop().create_future()
                    self._waiters.append((future, enqueued_at))
                    self.waiting += 1
                    await self._dispatch_waiters()
                self.refresh_status(observe_process=False)
                self._condition.notify_all()
            if selected is None:
                # 取消只撤回本等待者；若已授予则由撤回路径归还，不让Future被隐式取消。
                selected = await asyncio.shield(future)
            index, model, wait_ms = selected
            self.wait_total_ms += wait_ms
            self.last_wait_ms = wait_ms
            metrics = self._metrics_for(index)
            metrics["leases"] += 1
            metrics["wait_ms"] += wait_ms
            self.refresh_status(observe_process=False)
            instance_token = self._leased_instance.set(index)
            try:
                if self._return_device:
                    yield model, self.devices[index]
                else:
                    yield model
            except InstanceFailure as exc:
                self.failures[index] = str(exc)
                self.models[index] = None
                entry = self._process_entry(model)
                if entry is not None:
                    await await_completion(self._ensure_process_close_task(entry))
                else:
                    close = getattr(model, "close", None)
                    if close is not None:
                        try:
                            await run_in_thread(close)
                        except Exception as close_error:
                            # Quarantine it from leasing, but retain ownership for
                            # an explicit close retry and visible failure reporting.
                            self._unclosed_instances[index] = model
                            self.failures[index] = f"{exc}; close failed: {close_error}"
                            self.memory_complete = False
                            logger.exception("[模型池] 故障实例清理失败: %s", self.name)
                raise
            finally:
                self._leased_instance.reset(instance_token)
                await await_completion(self._release(index))
        except asyncio.CancelledError:
            await await_completion(self._withdraw_waiter(token, future, selected))
            raise
        except Exception:
            await await_completion(self._withdraw_waiter(token, future, selected))
            raise

    def _has_available(self) -> bool:
        return any(self._is_healthy(model) and not lock.locked()
                   for model, lock in zip(self.models, self.locks))

    async def _withdraw_waiter(self, token, future, selected):
        async with self._condition:
            self._admission_waiters.pop(token, None)
            if future is not None and selected is None:
                item = next((item for item in self._waiters if item[0] is future), None)
                if item is not None:
                    self._waiters.remove(item)
                    self.waiting = max(0, self.waiting - 1)
                    future.cancel()
                elif future.done() and not future.cancelled():
                    error = future.exception()
                    if error is None:
                        index, _, _ = future.result()
                        self.locks[index].release()
                        self._last_released[index] = time.perf_counter() if self._waiters or self._admission_waiters else None
            await self._dispatch_waiters()
            self.refresh_status(observe_process=False)
            self._condition.notify_all()

    async def _release(self, index):
        async with self._condition:
            self.locks[index].release()
            self._last_released[index] = time.perf_counter() if self._waiters or self._admission_waiters else None
            await self._dispatch_waiters()
            # 故障关闭期间的重复取消也必须经由本次真实release发布失败。
            # 正常交接仍只更新轻量计数，不做整池OS观察。
            if self.failures[index] is not None:
                self.refresh_status()
            else:
                self.refresh_status(observe_process=False)
            self._condition.notify_all()

    def record_inference(self, wait_ms: float, inference_ms: float) -> None:
        """实际NN调用包络；不把时长skip或仅租到实例视为调用/成功。"""
        self.calls += 1
        self.inference_total_ms += max(0.0, float(inference_ms))
        self.last_wait_ms = max(0.0, float(wait_ms))
        index = self._leased_instance.get()
        if index is not None:
            metrics = self._metrics_for(index)
            metrics["calls"] += 1
            metrics["inference_ms"] += max(0.0, float(inference_ms))
        self._record_phase("nn_calls", 1)
        self.refresh_status(observe_process=False)

    def record_outcome(self, outcome: str) -> None:
        """由调用方在真实收尾后分类，空embedding和取消不能充当成功。"""
        if outcome not in self._outcomes:
            return
        self._outcomes[outcome] += 1
        index = self._leased_instance.get()
        if index is not None:
            self._metrics_for(index)[outcome] += 1
        self._record_phase(f"nn_{outcome}", 1)
        self.refresh_status(observe_process=False)

    def record_skip(self) -> None:
        """时长外按原平淡规则返回，不加载/租实例、不增加NN调用数。"""
        self.duration_skips += 1
        self._record_phase("duration_skips", 1)
        self.refresh_status(observe_process=False)

    async def close(self) -> None:
        await await_completion(self._close())

    async def _close(self):
        async with self._close_lock:
            await self._close_locked()

    async def _close_locked(self):
        with self._ownership_lock:
            self._closing = True
            process_entries = list(self._process_runners.values())
        for entry in process_entries:
            self._request_process_close(entry)
        process_tasks = [self._ensure_process_close_task(entry) for entry in process_entries]
        # 看护先运行；不得先等loading/lease再轮到进程grace。
        if process_tasks:
            await asyncio.sleep(0)
        async with self._load_lock:
            async with self._condition:
                await self._dispatch_waiters()
                self._condition.notify_all()
                await self._condition.wait_for(lambda: not any(lock.locked() for lock in self.locks))
            if process_tasks:
                await await_completion(asyncio.gather(*process_tasks))
            errors = [(entry["index"], RuntimeError(entry["close_error"]))
                      for entry in process_entries if entry["close_error"]]
            owned = [(index, model) for index, model in enumerate(self.models) if model is not None]
            owned += list(self._unclosed_instances.items())
            for index, model in owned:
                if self._process_entry(model) is not None:
                    entry = self._process_entry(model)
                    if not entry["close_error"]:
                        self.models[index] = None
                    continue
                close = getattr(model, "close", None)
                try:
                    if close is not None:
                        await run_in_thread(close)
                except Exception as exc:
                    self.failures[index] = f"close failed: {type(exc).__name__}: {exc}"
                    errors.append((index, exc))
                    logger.exception("[模型池] 关闭实例失败: %s#%d", self.name, index)
                else:
                    self.models[index] = None
                    self._unclosed_instances.pop(index, None)
                    self.failures[index] = None
            if errors:
                # Keep failed resources and indices owned, reject all future
                # leases, and permit close() to retry only the remaining ones.
                self.memory_complete = False
                self.refresh_status()
                details = "; ".join(f"instance#{index}: {type(exc).__name__}: {exc}" for index, exc in errors)
                raise RuntimeError(f"模型池 {self.name} 关闭失败: {details}") from errors[0][1]
            self.models.clear()
            self.devices.clear()
            self.locks.clear()
            self._last_released.clear()
            self.failures.clear()
            self._unclosed_instances.clear()
            self.total_allocated_mb = 0.0
            self.memory_complete = True
            self.waiting = 0
            self.wait_total_ms = 0.0
            self.inference_total_ms = 0.0
            self.calls = 0
            self.last_wait_ms = 0.0
            self._instance_metrics.clear()
            self._outcomes = dict.fromkeys(self._outcomes, 0)
            self.duration_skips = 0
            self.handoff_total_ms = 0.0
            self.handoff_count = 0
            self.last_handoff_ms = 0.0
            for future, _ in self._waiters:
                if not future.done():
                    future.cancel()
            self._waiters.clear()
            # process池关闭后保持终态；旧CPU/thread池保留显式重载兼容行为。
            self._closing = bool(process_entries) or self.execution_mode == "process"
            self.refresh_status()


_MODEL_POOLS = {
    "speech_campplus": ModelPool(
        "speech_campplus", settings.spk_model_dir,
        devices_getter=lambda: settings.model_device_speech_campplus_devices,
        enabled_getter=lambda: settings.speech_campplus_enabled,
        prefetch_depth_getter=lambda: getattr(settings, "speech_campplus_prefetch_queue_depth", 0),
        fp16=True,
    ),
    "emotion2vec": ModelPool(
        "emotion2vec", settings.emotion_model_dir,
        devices_getter=lambda: settings.model_device_emotion2vec_devices,
        enabled_getter=lambda: settings.emotion2vec_enabled,
        prefetch_depth_getter=lambda: getattr(settings, "emotion2vec_prefetch_queue_depth", 0),
        fp16=True,
    ),
}


async def preload_v3_model_pools() -> None:
    # 就绪前完成增强池加载尝试；健康容量与失败仍沿用各池的降级契约。
    async with _model_lock:
        for pool in _MODEL_POOLS.values():
            if not pool.enabled_getter() or not pool.configured_instances:
                continue
            # ensure_loaded逐实例登记失败，不因一个池全失败而跳过另一池。
            # 取消/意外异常不吞掉，由lifespan finally回收已创建的资源。
            await pool.ensure_loaded()


async def load_v3_models_if_needed(need_role: bool = False, need_emotion: bool = False):
    """按请求校验并复用对应模型池，兼容直接调用时尚未加载的情况。"""
    global _model_spk_v3, _model_emotion_v3
    validate_v3_capabilities(need_role, need_emotion)
    lock_started = time.perf_counter()
    async with _model_lock:
        record_enhancement_phase("model_load_lock_wait_ms", (time.perf_counter() - lock_started) * 1000)
        if need_role:
            await _MODEL_POOLS["speech_campplus"].ensure_loaded()
            if not any(_MODEL_POOLS["speech_campplus"]._is_healthy(model)
                       for model in _MODEL_POOLS["speech_campplus"].models):
                raise ModelUnavailableError("speech_campplus")
            _model_spk_v3 = next((model for model in _MODEL_POOLS["speech_campplus"].models
                                  if _MODEL_POOLS["speech_campplus"]._is_healthy(model)), None)
        if need_emotion:
            await _MODEL_POOLS["emotion2vec"].ensure_loaded()
            if not any(_MODEL_POOLS["emotion2vec"]._is_healthy(model)
                       for model in _MODEL_POOLS["emotion2vec"].models):
                raise ModelUnavailableError("emotion2vec")
            _model_emotion_v3 = next((model for model in _MODEL_POOLS["emotion2vec"].models
                                      if _MODEL_POOLS["emotion2vec"]._is_healthy(model)), None)


def get_model_pool(name: str) -> ModelPool:
    return _MODEL_POOLS[name]


def get_spk_model():
    return _model_spk_v3


def get_emotion_v3_model():
    return _model_emotion_v3


async def close_v3_model_pools() -> None:
    async def close_all():
        results = await asyncio.gather(*[pool.close() for pool in _MODEL_POOLS.values()], return_exceptions=True)
        global _model_spk_v3, _model_emotion_v3
        if not isinstance(results[0], BaseException):
            _model_spk_v3 = None
        if not isinstance(results[1], BaseException):
            _model_emotion_v3 = None
        failures = [(name, result) for name, result in zip(_MODEL_POOLS, results) if isinstance(result, BaseException)]
        if failures:
            raise RuntimeError("本地模型池关闭失败: " + "; ".join(f"{name}: {error}" for name, error in failures)) from failures[0][1]

    await await_completion(close_all())
