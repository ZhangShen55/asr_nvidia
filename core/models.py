"""v2 链路的本地模型加载与显存登记。

`/v1.1.8/seacraft_asr` 下线后，本模块只保留 v2 需要的三个本地模型：
FSMN-VAD（由 `core/vad.py` 的实例池管理）、CAM++ 说话人嵌入、emotion2vec 情绪。

三个模型由同一个服务进程承载，进程名统一为 `asr-offline-model`（见 `main.py`），
因此 `nvidia-smi` 只能看到聚合显存；这里额外登记每个模型的加载状态与显存明细，
供 `/get_status` 暴露给运维。
"""

import asyncio
import logging
from typing import Any, Dict

import torch
from funasr import AutoModel

from core.config import settings

logger = logging.getLogger(__name__)

# 单例缓存
_model_spk_v2 = None
_model_emotion_v2 = None

# 线程锁
_model_lock = asyncio.Lock()

# 本地模型的加载状态与显存明细：模型名 -> {loaded, device, allocated_mb, instances}
_LOCAL_MODELS = ("vad", "speech_campplus", "emotion2vec")
local_model_status: Dict[str, Dict[str, Any]] = {
    name: {"loaded": False, "device": None, "allocated_mb": 0.0, "instances": 0}
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
    except Exception:  # noqa: BLE001 - 显存统计失败不应影响推理
        return 0.0


def record_model_memory(
    name: str,
    device: str,
    allocated_mb: float,
    instance_added: bool = False,
) -> None:
    """登记本地模型的加载状态与显存占用。

    instance_added=True 用于按实例累加的模型（VAD 实例池）。
    """
    if name not in local_model_status:
        return
    entry = local_model_status[name]
    entry["loaded"] = True
    entry["device"] = device
    if instance_added:
        entry["instances"] += 1
        entry["allocated_mb"] = round(entry["allocated_mb"] + max(0.0, allocated_mb), 2)
    else:
        entry["instances"] = max(1, entry["instances"])
        entry["allocated_mb"] = round(max(0.0, allocated_mb), 2)


def get_local_model_status() -> Dict[str, Dict[str, Any]]:
    """返回三个本地模型的加载状态与显存明细（供状态接口使用）。"""
    return {name: dict(entry) for name, entry in local_model_status.items()}


def _resolve_device(index: int) -> str:
    """把 GPU 索引解析为 torch 设备串。"""
    if not torch.cuda.is_available():
        raise RuntimeError("v2 链路要求使用 GPU，但 CUDA 不可用")
    visible = torch.cuda.device_count()
    if index < 0 or index >= visible:
        raise RuntimeError(f"GPU 索引 {index} 越界，可见 CUDA 设备数量为 {visible}")
    return f"cuda:{index}"


async def load_v2_models_if_needed(need_role: bool = False, need_emotion: bool = False):
    """懒加载 v2 链路使用的独立模型实例（CAM++ / emotion2vec）。

    - CAM++ 由 `open_v2_role` 单独控制，emotion2vec 由 `open_v2_emotion` 单独控制，
      两者互不依赖，任一关闭都不影响另一个；
    - 两者各自落在 [model_device] 指定的 GPU 上；
    - 请求未触发（need_role / need_emotion 为 false）时不加载权重；
    - VAD 由 core/vad.py 的实例池单独管理。
    """
    global _model_spk_v2, _model_emotion_v2

    async with _model_lock:
        if settings.open_v2_role and need_role and _model_spk_v2 is None:
            device = _resolve_device(settings.model_device_speech_campplus)
            before = device_allocated_mb(device)
            _model_spk_v2 = AutoModel(
                model=settings.spk_model_dir,
                device=device,
                disable_update=True,
                disable_pbar=True,
            )
            allocated = device_allocated_mb(device) - before
            record_model_memory("speech_campplus", device, allocated)
            logger.info(
                f"[模型] CAM++ 已加载到 {device}，显存占用约 {allocated:.1f} MB"
            )

        if settings.open_v2_emotion and need_emotion and _model_emotion_v2 is None:
            device = _resolve_device(settings.model_device_emotion2vec)
            before = device_allocated_mb(device)
            # 精度对半即可，业务上有结果就行；权重占用从约 1.9GB 降到约 1GB。
            _model_emotion_v2 = AutoModel(
                model=settings.emotion_model_dir,
                device=device,
                fp16=True,
                disable_update=True,
                disable_pbar=True,
            )
            allocated = device_allocated_mb(device) - before
            record_model_memory("emotion2vec", device, allocated)
            logger.info(
                f"[模型] emotion2vec 已加载到 {device}，显存占用约 {allocated:.1f} MB"
            )


def get_spk_model():
    return _model_spk_v2


def get_emotion_v2_model():
    return _model_emotion_v2
