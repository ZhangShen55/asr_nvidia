#!/bin/bash
# 在 Docker builder 阶段或本地执行：将业务模块编译为 .so 并移除对应 .py 源码
set -euo pipefail

ROOT="${1:-/build}"
cd "$ROOT"

echo "[build_cython] 安装 Cython ..."
pip install --no-cache-dir Cython

echo "[build_cython] 编译 core / utils / api / entity ..."
python setup_cython.py build_ext --inplace

echo "[build_cython] 移除已编译的 .py 源码（保留 __init__.py）..."
find core utils api entity -type f -name '*.py' ! -name '__init__.py' -delete

echo "[build_cython] 清理 Cython 生成的 .c 文件 ..."
find core utils api entity -type f -name '*.c' -delete 2>/dev/null || true

echo "[build_cython] 完成。入口 main.py 保持明文。"
