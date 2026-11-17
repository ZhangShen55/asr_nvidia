"""昇腾部署的就绪前检查；检查资产与设备，不触发懒加载模型。"""
from __future__ import annotations

import importlib
from importlib import metadata
import os
from pathlib import Path


class DeploymentPreflightError(RuntimeError):
    """当前配置的必需运行条件不满足。"""


def _runtime_versions():
    expected = {"torch": "2.5.1", "torch-npu": "2.5.1", "torchaudio": "2.5.1",
                "transformers": "5.13.0", "numpy": "1.26.4", "funasr": "1.2.6"}
    versions = {}
    for name, wanted in expected.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise DeploymentPreflightError(f"缺少运行依赖 {name}=={wanted}") from exc
        if actual.split("+", 1)[0].split(".post", 1)[0] != wanted:
            raise DeploymentPreflightError(f"{name} 版本不匹配: {actual}，目标 {wanted}")
        versions[name] = actual
    if importlib.util.find_spec("attr") is None:
        raise DeploymentPreflightError("缺少 attrs，CANN 编译器需要可导入的 attr 模块")
    return versions


def validate_npu_runtime(settings) -> dict:
    """CPU 配置不导入 NPU；Qwen 文件缺失留给首次请求报告 4011。"""
    settings.validate_model_backends()
    devices = list(settings.vad_device)
    devices += settings.model_device_speech_campplus_devices + settings.model_device_emotion2vec_devices
    if settings.qwen_asr_enabled:
        devices.append(settings.qwen_local_device)
    npu_devices = sorted({device for device in devices if device.startswith("npu:")})
    assets = []
    required = [
        ("vad", settings.vad_model_dir, ("model.pt", "config.yaml", "am.mvn")),
        ("speech_campplus", settings.spk_model_dir, ("campplus_cn_en_common.pt", "config.yaml")),
        ("emotion2vec", settings.emotion_model_dir, ("model.pt", "config.yaml", "tokens.txt")),
    ]
    for name, directory, files in required:
        root = Path(directory or "")
        missing = [file for file in files if not (root / file).is_file() or (root / file).stat().st_size == 0]
        if missing:
            raise DeploymentPreflightError(f"{name} 资产缺失或为空: {root}: {', '.join(missing)}")
        assets.append({"model": name, "directory": str(root.resolve()), "files_checked": list(files)})
    versions = {}
    if npu_devices:
        versions = _runtime_versions()
        try:
            torch = importlib.import_module("torch")
            extension = importlib.import_module("torch_npu")
            api = getattr(torch, "npu", None) or extension.npu
            if not api.is_available():
                raise DeploymentPreflightError("没有可用 NPU；检查 CANN 环境和容器设备映射")
            count = api.device_count()
            for device in npu_devices:
                if int(device.split(":", 1)[1]) >= count:
                    raise DeploymentPreflightError(f"设备 {device} 不可见，可见逻辑设备数 {count}")
        except DeploymentPreflightError:
            raise
        except Exception as exc:
            raise DeploymentPreflightError(
                f"NPU 运行时不可用: {exc}；请先 source CANN set_env.sh 并使用 conda asr 解释器"
            ) from exc
    return {"profile": settings.deployment_profile, "checked": True, "devices": npu_devices,
            "visible_physical_devices": os.getenv("ASCEND_VISIBLE_DEVICES"),
            "versions": versions, "assets": assets, "models_loaded_by_preflight": False}
