"""昇腾部署的就绪前检查；检查资产与设备，不触发懒加载模型。"""
from __future__ import annotations

import importlib
from importlib import metadata
import os
from pathlib import Path


class DeploymentPreflightError(RuntimeError):
    """当前配置的必需运行条件不满足。"""


def _source_container_enabled() -> bool:
    value = os.getenv("ASR_CONTAINER_RUNTIME", "")
    if value not in ("", "cann82-source"):
        raise DeploymentPreflightError(f"不支持的 ASR_CONTAINER_RUNTIME: {value!r}")
    return value == "cann82-source"


def validate_source_container_config() -> dict | None:
    """源码镜像必须显式挂载配置，不能读到工作目录中的开发配置兜底。"""
    if not _source_container_enabled():
        return None
    configured = os.getenv("CONFIG_PATH")
    path = Path(configured) if configured else None
    if path is None or not path.is_file() or not os.access(path, os.R_OK):
        raise DeploymentPreflightError("CANN8.2源码容器要求 CONFIG_PATH 指向可读的普通挂载配置文件")
    try:
        import tomli
        with path.open("rb") as stream:
            config = tomli.load(stream)
    except (OSError, ValueError) as exc:
        raise DeploymentPreflightError(f"无法读取容器配置 {path}: {exc}") from exc
    # Settings的通用instance_count属性会int转换；这里拒绝true、字符串及小数，
    # 避免错误值被悄悄变为1，以及误用完整HTTP副本放大所有模型资源。
    count = config.get("instance_count", 1)
    if type(count) is not int or count != 1:
        raise DeploymentPreflightError(
            "本版源码容器仅支持 instance_count=1（真实整数）；"
            "Qwen模型数量请配置 qwen_asr.local_instance_count"
        )
    return {"runtime": "cann82-source", "config_path": str(path.resolve()), "instance_count": count}


def _runtime_versions():
    expected = {"torch": "2.5.1", "torch-npu": "2.5.1", "torchaudio": "2.5.1",
                "transformers": "5.13.0", "numpy": "1.26.4", "funasr": "1.2.6"}
    source_container = _source_container_enabled()
    if source_container:
        expected["torch-npu"] = "2.5.1.post1"
    versions = {}
    for name, wanted in expected.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise DeploymentPreflightError(f"缺少运行依赖 {name}=={wanted}") from exc
        # CANN8.2的post1是配套版本，不可像宿主兼容检查一样抹去.post。
        # 本地构建后缀保留在诊断中；仅允许相同distribution版本的+后缀。
        actual_version = actual.split("+", 1)[0]
        if not (source_container and name == "torch-npu"):
            actual_version = actual_version.split(".post", 1)[0]
        if actual_version != wanted:
            raise DeploymentPreflightError(f"{name} 版本不匹配: {actual}，目标 {wanted}")
        versions[name] = actual
    if importlib.util.find_spec("attr") is None:
        raise DeploymentPreflightError("缺少 attrs，CANN 编译器需要可导入的 attr 模块")
    return versions


def validate_npu_runtime(settings) -> dict:
    """CPU 配置不导入 NPU；Qwen 文件缺失留给首次请求报告 4011。"""
    container_config = validate_source_container_config()
    settings.validate_model_backends()
    versions = _runtime_versions() if container_config is not None else {}
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
    if npu_devices:
        if not versions:
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
            "versions": versions, "assets": assets, "models_loaded_by_preflight": False,
            "container_runtime": container_config}
