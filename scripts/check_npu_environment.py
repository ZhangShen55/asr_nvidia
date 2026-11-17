#!/usr/bin/env python3
"""NPU 环境诊断；metadata-only 不初始化设备，不能作为前向通过证据。"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import sys

REQUIRED = {
    "torch": "2.5.1", "torch-npu": "2.5.1", "torchaudio": "2.5.1",
    "torchvision": "0.20.1", "transformers": "5.13.0", "numpy": "1.26.4",
    "funasr": "1.2.6", "av": "14.2.0", "attrs": "26.1.0",
    "WeTextProcessing": "1.2.0", "pynini": "2.1.7",
}


def inspect_packages(version=importlib.metadata.version):
    packages, errors = {}, []
    for name, expected in REQUIRED.items():
        try:
            actual = version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        packages[name] = actual
        # wheel 的本地版本后缀不改变目标 torch ABI 版本。
        if actual is None or actual.split("+", 1)[0] != expected:
            errors.append(f"{name}: 需要 {expected}, 实际 {actual or '未安装'}")
    return packages, errors


def inspect_compiler_modules(importer=None):
    """CANN Python 编译器路径与共享库同样必要；不要覆盖 set_env.sh 的 PYTHONPATH。"""
    if importer is None:
        from importlib import import_module
        importer = import_module
    checked = {}
    for name in ("tbe", "te", "attr"):
        module = importer(name)
        checked[name] = str(getattr(module, "__file__", "namespace"))
    return checked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.getenv("CONFIG_PATH", "config.toml"))
    parser.add_argument("--metadata-only", action="store_true",
                        help="只检查包元数据，不声称运行时或前向通过")
    args = parser.parse_args(argv)
    packages, errors = inspect_packages()
    result = {"evidence": "metadata", "python": sys.executable,
              "python_version": platform.python_version(), "architecture": platform.machine(),
              "packages": packages, "errors": errors, "runtime_checked": False}
    if not args.metadata_only and not errors:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            result["cann_compiler_modules"] = inspect_compiler_modules()
            from core.config import Settings
            from core.deployment import validate_npu_runtime
            result["runtime"] = validate_npu_runtime(Settings(config_path=args.config))
            result["runtime_checked"] = bool(result["runtime"].get("checked", False))
            result["evidence"] = "runtime_preflight"
        except Exception as exc:
            errors.append(f"运行时检查失败: {exc}; 请先 source CANN set_env.sh 并保留其 PYTHONPATH（tbe/te），使用 asr 自带解释器")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
