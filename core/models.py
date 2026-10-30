"""v2 链路的本地模型加载、模型池调度与显存登记。"""

import asyncio
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Dict

import torch

from core.config import settings

logger = logging.getLogger(__name__)

# 延迟导入 FunASR，便于状态接口和配置单测在没有音频后端时启动；测试可以
# patch 这个模块属性替换真实模型构造器。
AutoModel = None

_model_spk_v2 = None
_model_emotion_v2 = None
_model_lock = asyncio.Lock()
_LOCAL_MODELS = ("vad", "speech_campplus", "emotion2vec")


class ModelUnavailableError(RuntimeError):
    """请求要求的本地模型池未启用或没有健康副本。"""

    _DETAILS = {
        "speech_campplus": (4015, "CAM++ 模型未启用或没有可用实例（没有可用副本）"),
        "emotion2vec": (4016, "emotion2vec 模型未启用或没有可用实例（没有可用副本）"),
    }

    def __init__(self, name: str):
        code, message = self._DETAILS.get(name, (4015, f"{name} 模型未启用或没有可用实例"))
        self.model_name = name
        self.code = code
        self.message = message
        super().__init__(message)
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


def device_allocated_mb(device: str) -> float:
    """取指定设备上 torch 已分配的显存（MB）；不可用时返回 0。"""
    try:
        index = int(str(device).split(":")[1])
    except (AttributeError, IndexError, ValueError):
        return 0.0
    try:
        return torch.cuda.memory_allocated(index) / (1024 ** 2)
    except Exception:  # noqa: BLE001
        return 0.0


def record_model_memory(name: str, device: str, allocated_mb: float) -> None:
    if name not in local_model_status:
        return
    entry = local_model_status[name]
    entry["loaded"] = True
    entry["device"] = device
    entry["instances"] = max(1, entry["instances"])
    entry["allocated_mb"] = round(max(0.0, allocated_mb), 2)


def record_vad_runtime(
    device: str | list[str],
    instances: int,
    loaded: bool,
    allocated_mb: float,
    pool_size: int | None = None,
    pool_max: int | None = None,
    *,
    configured_instances: int | None = None,
    successful_instances: int | None = None,
    failed_instances: int = 0,
    devices: list[str] | None = None,
    busy: int = 0,
    waiting: int = 0,
    failures: list[str] | None = None,
) -> None:
    entry = local_model_status["vad"]
    entry["loaded"] = bool(loaded)
    entry["device"] = device[0] if isinstance(device, list) and device else device
    entry["devices"] = list(devices if devices is not None else (device if isinstance(device, list) else [device]))
    entry["instances"] = max(0, int(instances))
    entry["allocated_mb"] = round(max(0.0, allocated_mb), 2)
    entry["configured_instances"] = int(configured_instances if configured_instances is not None else (instances or 0))
    entry["available_instances"] = max(0, int(successful_instances if successful_instances is not None else instances) - int(busy))
    entry["failed_instances"] = int(failed_instances)
    entry["busy"] = int(busy)
    entry["waiting"] = int(waiting)
    entry["failure"] = (failures or [None])[-1]


def get_local_model_status() -> Dict[str, Dict[str, Any]]:
    for pool in _MODEL_POOLS.values():
        pool.refresh_status()
    # VAD 由 vadPool 单独暴露；localModels 只描述按需加载的增强模型池。
    return {
        name: dict(local_model_status[name])
        for name in ("speech_campplus", "emotion2vec")
    }


def _resolve_device(index: int | str) -> str:
    if isinstance(index, str) and index.strip().lower() == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError("v2 链路要求使用 GPU，但 CUDA 不可用")
    if isinstance(index, str):
        spec = index.strip().lower()
        if spec == "cuda":
            index = 0
        elif spec.startswith("cuda:") and spec.split(":", 1)[1].isdigit():
            index = int(spec.split(":", 1)[1])
        else:
            raise RuntimeError(f"本地模型设备 {index!r} 不支持，必须使用 cuda:N")
    if not isinstance(index, int) or index < 0:
        raise RuntimeError(f"本地模型设备 {index!r} 不支持，必须使用 cuda:N")
    visible = torch.cuda.device_count()
    if index >= visible:
        raise RuntimeError(f"GPU 索引 {index} 越界，可见 CUDA 设备数量为 {visible}")
    return f"cuda:{index}"


def resolve_vad_device() -> str:
    devices = settings.vad_device
    if not devices:
        raise RuntimeError("qwen_asr.vad_device 不能为空，至少配置一个 VAD worker")
    return _resolve_device(devices[0])


class ModelPool:
    """按设备数组加载模型副本；每个句柄同一时刻只服务一个任务。"""

    def __init__(self, name: str, model_dir: str, device_getter=None, count_getter=None,
                 *, devices_getter=None, fp16=False):
        self.name = name
        self.model_dir = model_dir
        self.device_getter = device_getter
        self.count_getter = count_getter
        self.devices_getter = devices_getter
        self._return_device = devices_getter is not None
        self.fp16 = fp16
        self.models: list[Any] = []
        self.devices: list[str | None] = []
        self.locks: list[asyncio.Lock] = []
        self.failures: list[str | None] = []
        self._load_lock = asyncio.Lock()
        self._condition = asyncio.Condition()
        self._waiters = deque()
        self._next_instance = 0
        self.waiting = 0
        self.total_allocated_mb = 0.0
        self.wait_total_ms = 0.0
        self.inference_total_ms = 0.0
        self.calls = 0
        self.last_wait_ms = 0.0

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

    async def ensure_loaded(self) -> None:
        async with self._load_lock:
            target = self.configured_instances
            if len(self.models) >= target:
                self.refresh_status()
                return
            devices = self.configured_devices
            while len(self.models) < target:
                configured_device = devices[len(self.models)]
                model = None
                failure = None
                device = None
                try:
                    global AutoModel
                    if AutoModel is None:
                        from funasr import AutoModel as FunasrAutoModel

                        AutoModel = FunasrAutoModel

                    device = _resolve_device(configured_device)
                    before = device_allocated_mb(device)
                    kwargs = {"model": self.model_dir, "device": device,
                              "disable_update": True, "disable_pbar": True}
                    if self.fp16:
                        kwargs["fp16"] = True
                    model = AutoModel(**kwargs)
                except Exception as exc:  # noqa: BLE001
                    failure = str(exc)
                    logger.exception("[模型池] %s 副本加载失败", self.name)
                self.models.append(model)
                self.devices.append(device or configured_device)
                self.locks.append(asyncio.Lock())
                self.failures.append(failure)
                if model is not None:
                    self.total_allocated_mb += max(0.0, device_allocated_mb(device) - before)
            self.refresh_status()

    def refresh_status(self) -> None:
        entry = local_model_status[self.name]
        loaded = [model for model in self.models if model is not None]
        failed = [failure for failure in self.failures if failure]
        configured_devices = self.configured_devices
        busy = sum(lock.locked() for lock in self.locks)
        entry.update({
            "loaded": bool(loaded),
            "device": next((device for model, device in zip(self.models, self.devices)
                             if model is not None), None),
            "devices": list(self.devices) if self.models else configured_devices,
            "allocated_mb": round(self.total_allocated_mb, 2),
            "instances": len(loaded),
            "configured_instances": self.configured_instances,
            "available_instances": max(0, len(loaded) - busy),
            "failed_instances": len(failed),
            "busy": busy,
            "waiting": self.waiting,
            "failure": failed[-1] if failed else None,
            "wait_total_ms": round(self.wait_total_ms, 2),
            "inference_total_ms": round(self.inference_total_ms, 2),
            "calls": self.calls,
            "last_wait_ms": round(self.last_wait_ms, 2),
            "waiting_summary": {
                "current": self.waiting,
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
            if self.models[index] is not None and not self.locks[index].locked():
                self._next_instance = (index + 1) % count
                return index
        return None

    async def _dispatch_waiters(self) -> None:
        """在条件锁内按 FIFO 发放租约；取消的等待者会被跳过。"""
        while self._waiters:
            future, enqueued_at = self._waiters[0]
            if future.cancelled():
                self._waiters.popleft()
                self.waiting = max(0, self.waiting - 1)
                continue
            index = self._available_index()
            if index is None:
                return
            await self.locks[index].acquire()
            self._waiters.popleft()
            self.waiting = max(0, self.waiting - 1)
            wait_ms = (time.perf_counter() - enqueued_at) * 1000
            future.set_result((index, self.models[index], wait_ms))

    @asynccontextmanager
    async def lease(self):
        await self.ensure_loaded()
        if not any(model is not None for model in self.models):
            raise ModelUnavailableError(self.name)
        selected = None
        future = None
        try:
            async with self._condition:
                # 有排队者时，新请求必须排在队尾，保证 FIFO。
                if not self._waiters:
                    index = self._available_index()
                    if index is not None:
                        await self.locks[index].acquire()
                        selected = (index, self.models[index], 0.0)
                if selected is None:
                    future = asyncio.get_running_loop().create_future()
                    self._waiters.append((future, time.perf_counter()))
                    self.waiting += 1
                    await self._dispatch_waiters()
                    self.refresh_status()
            if selected is None:
                selected = await future
            index, model, wait_ms = selected
            self.wait_total_ms += wait_ms
            self.last_wait_ms = wait_ms
            self.refresh_status()
            try:
                if self._return_device:
                    yield model, self.devices[index]
                else:
                    yield model
            finally:
                self.locks[index].release()
                async with self._condition:
                    await self._dispatch_waiters()
                    self.refresh_status()
                    self._condition.notify_all()
        except asyncio.CancelledError:
            async with self._condition:
                if future is not None and not future.done():
                    try:
                        self._waiters.remove(next(item for item in self._waiters if item[0] is future))
                    except (StopIteration, ValueError):
                        pass
                    else:
                        self.waiting = max(0, self.waiting - 1)
                elif future is not None and future.done() and not future.cancelled() and selected is None:
                    index, _model, _wait_ms = future.result()
                    self.locks[index].release()
                await self._dispatch_waiters()
                self.refresh_status()
                self._condition.notify_all()
            raise
        except Exception:
            self.refresh_status()
            raise

    def record_inference(self, wait_ms: float, inference_ms: float) -> None:
        """记录一次模型池租约等待和实际推理耗时。"""
        self.calls += 1
        self.inference_total_ms += max(0.0, float(inference_ms))
        self.last_wait_ms = max(0.0, float(wait_ms))
        self.refresh_status()

    async def close(self) -> None:
        async with self._load_lock:
            self.models.clear()
            self.devices.clear()
            self.locks.clear()
            self.failures.clear()
            self.total_allocated_mb = 0.0
            self.waiting = 0
            self.wait_total_ms = 0.0
            self.inference_total_ms = 0.0
            self.calls = 0
            self.last_wait_ms = 0.0
            for future, _ in self._waiters:
                if not future.done():
                    future.cancel()
            self._waiters.clear()
            self.refresh_status()


_MODEL_POOLS = {
    "speech_campplus": ModelPool(
        "speech_campplus", settings.spk_model_dir,
        devices_getter=lambda: settings.model_device_speech_campplus_devices,
        fp16=True,
    ),
    "emotion2vec": ModelPool(
        "emotion2vec", settings.emotion_model_dir,
        devices_getter=lambda: settings.model_device_emotion2vec_devices,
        fp16=True,
    ),
}


async def load_v2_models_if_needed(need_role: bool = False, need_emotion: bool = False):
    """按请求开关懒加载对应模型池。"""
    global _model_spk_v2, _model_emotion_v2
    async with _model_lock:
        if settings.open_v2_role and need_role:
            await _MODEL_POOLS["speech_campplus"].ensure_loaded()
            if not any(model is not None for model in _MODEL_POOLS["speech_campplus"].models):
                raise ModelUnavailableError("speech_campplus")
            _model_spk_v2 = next((model for model in _MODEL_POOLS["speech_campplus"].models if model), None)
        if settings.open_v2_emotion and need_emotion:
            await _MODEL_POOLS["emotion2vec"].ensure_loaded()
            if not any(model is not None for model in _MODEL_POOLS["emotion2vec"].models):
                raise ModelUnavailableError("emotion2vec")
            _model_emotion_v2 = next((model for model in _MODEL_POOLS["emotion2vec"].models if model), None)


def get_model_pool(name: str) -> ModelPool:
    return _MODEL_POOLS[name]


def get_spk_model():
    return _model_spk_v2


def get_emotion_v2_model():
    return _model_emotion_v2


async def close_v2_model_pools() -> None:
    await asyncio.gather(*[pool.close() for pool in _MODEL_POOLS.values()])
    global _model_spk_v2, _model_emotion_v2
    _model_spk_v2 = None
    _model_emotion_v2 = None
