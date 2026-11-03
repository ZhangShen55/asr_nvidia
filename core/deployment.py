"""CUDA Docker profile 的启动前环境与 TensorRT 资产校验。"""

from __future__ import annotations

import importlib
from pathlib import Path

from core.model_assets import validate_manifest


class DeploymentPreflightError(RuntimeError):
    """部署 profile 的必需运行条件不满足。"""


def validate_cuda_tensorrt_runtime(settings) -> dict:
    """在真实 CUDA 设备上校验 Docker profile 的 fp16 engine。"""
    if not settings.is_cuda_tensorrt_profile:
        return {"profile": settings.deployment_profile, "checked": False}

    try:
        import torch
    except Exception as exc:  # pragma: no cover - only reached in broken runtime images
        raise DeploymentPreflightError(f"无法导入 PyTorch CUDA 运行时: {exc}") from exc
    if not torch.cuda.is_available():
        raise DeploymentPreflightError("CUDA TensorRT profile 要求可用 NVIDIA GPU")
    try:
        trt = importlib.import_module("tensorrt")
    except Exception as exc:  # pragma: no cover - depends on deployment image
        raise DeploymentPreflightError(f"无法导入 TensorRT: {exc}") from exc

    results = []
    models = (
        ("speech_campplus", settings.spk_model_dir, settings.model_device_speech_campplus_devices),
        ("emotion2vec", settings.emotion_model_dir, settings.model_device_emotion2vec_devices),
    )
    for name, model_dir, devices in models:
        if not devices:
            raise DeploymentPreflightError(f"{name} 设备数组为空")
        for device in devices:
            if not device.startswith("cuda:"):
                raise DeploymentPreflightError(f"{name} 设备 {device} 不是 CUDA 设备")
            try:
                device_index = int(device.split(":", 1)[1])
                if device_index >= torch.cuda.device_count():
                    raise DeploymentPreflightError(
                        f"{name} 设备 {device} 不可见，可见设备数为 {torch.cuda.device_count()}"
                    )
                with torch.cuda.device(device):
                    environment = {
                        "tensorrt": trt.__version__,
                        "cuda": torch.version.cuda,
                        "compute_capability": list(torch.cuda.get_device_capability(device)),
                    }
                    engine_path, manifest = validate_manifest(
                        Path(model_dir), name, "fp16", environment=environment
                    )
            except DeploymentPreflightError:
                raise
            except Exception as exc:
                raise DeploymentPreflightError(
                    f"{name} {device} 的 fp16 TensorRT 资产校验失败: {exc}"
                ) from exc
            results.append({
                "model": name,
                "device": device,
                "engine": str(engine_path),
                "sha256": manifest["engine_sha256"],
                "environment": environment,
                "compatible": True,
            })
    return {"profile": settings.deployment_profile, "checked": True, "assets": results}
