#!/usr/bin/env bash
# 在 Docker builder 阶段或临时 staging 目录执行 Cython 保护构建。
set -euo pipefail

ROOT="${1:-/build}"
cd "$ROOT"

python -m pip install --no-cache-dir Cython
python setup_cython.py build_ext --inplace

python - <<'PY'
from pathlib import Path

for directory in (Path("core"), Path("api"), Path("utils"), Path("entity")):
    if not directory.is_dir():
        raise SystemExit(f"缺少 Cython 目录: {directory}")
    for source in directory.rglob("*.py"):
        if source.name == "__init__.py":
            continue
        stem = source.with_suffix("").name
        siblings = list(source.parent.glob(stem + ".*.so")) + list(source.parent.glob(stem + ".so"))
        if not siblings:
            raise SystemExit(f"未生成 Cython 扩展: {source}")

# 仅保留包初始化入口；业务源码在 runtime 层不可见。
for directory in (Path("core"), Path("api"), Path("utils"), Path("entity")):
    for source in directory.rglob("*.py"):
        if source.name != "__init__.py":
            source.unlink()
PY

find core api utils entity -type f -name '*.c' -delete
find . -type d -name '__pycache__' -prune -exec rm -rf {} +

python - <<'PY'
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

packages = {}
for name in ("Cython", "setuptools", "numpy", "torch", "torchaudio", "funasr", "tensorrt"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None

metadata = {
    "built_at": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "platform": platform.platform(),
    "compiler": platform.python_compiler(),
    "packages": packages,
}
Path("cython-build.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
PY

echo "[build_cython] 编译并校验完成，运行时仅保留扩展和 __init__.py。"
