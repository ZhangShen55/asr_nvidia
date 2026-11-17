#!/usr/bin/env bash
# 仅在独立 staging 中编译并移除已生成扩展对应的源码。
set -euo pipefail
ROOT="${1:-/build}"
cd "$ROOT"
[[ -f .asr-build-stage ]] || { echo '[ERROR] 缺少 .asr-build-stage 标记，拒绝在源工作区进行保护构建' >&2; exit 2; }
PYTHON="${PYTHON:-python}"
"$PYTHON" - <<'PYCODE'
from importlib.metadata import version,PackageNotFoundError
import platform,sys
if platform.machine() != "aarch64" or sys.version_info[:2] != (3,10):
    raise SystemExit("保护交付要求 Linux aarch64/Python3.10，禁止混用其他ABI产物")
import numpy
if numpy.__version__ != "1.26.4" or version("numpy") != numpy.__version__:
    raise SystemExit(f"NumPy payload/metadata不一致或版本错误: {numpy.__version__}/{version('numpy')}；请按锁重建，不能仅依赖conda clone")
try: actual=version("Cython")
except PackageNotFoundError: raise SystemExit("缺少Cython；请在独立构建环境安装requirements-build.txt")
if actual != "3.0.12": raise SystemExit(f"Cython版本不匹配: {actual} != 3.0.12")
PYCODE
ASR_BUILD_JOBS="${CYTHON_BUILD_JOBS:-4}"
[[ "$ASR_BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || { echo '[ERROR] CYTHON_BUILD_JOBS 必须为正整数' >&2; exit 2; }
"$PYTHON" setup_cython.py build_ext --inplace --parallel "$ASR_BUILD_JOBS"
"$PYTHON" - <<'PYCODE'
from pathlib import Path
import shutil
roots=[Path(name) for name in ("core","api","utils","entity")]
for directory in roots:
    if not directory.is_dir(): raise SystemExit(f"缺少Cython目录: {directory}")
    for source in directory.rglob("*.py"):
        if source.name == "__init__.py": continue
        stem=source.with_suffix("").name
        if not list(source.parent.glob(stem+".*.so"))+list(source.parent.glob(stem+".so")):
            raise SystemExit(f"未生成Cython扩展: {source}")
for directory in roots:
    for source in directory.rglob("*.py"):
        if source.name != "__init__.py": source.unlink()
    for generated in directory.rglob("*.c"): generated.unlink()
    for cache in list(directory.rglob("__pycache__")): shutil.rmtree(cache)
PYCODE
"$PYTHON" - <<'PYCODE'
import importlib.metadata,json,platform,sys,sysconfig
from datetime import datetime,timezone
from pathlib import Path
packages={}
for name in ("Cython","setuptools","numpy","torch","torch-npu","torchaudio","transformers","funasr"):
    try: packages[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: packages[name]=None
Path("cython-build.json").write_text(json.dumps({"built_at":datetime.now(timezone.utc).isoformat(),"python":sys.version,"architecture":platform.machine(),"soabi":sysconfig.get_config_var("SOABI"),"platform":platform.platform(),"compiler":platform.python_compiler(),"packages":packages,"compiler_directives":{"boundscheck":True,"wraparound":True,"initializedcheck":True,"cdivision":False},"cann_version_files":[str(p) for p in Path("/usr/local/Ascend").glob("**/version.cfg")]},ensure_ascii=False,indent=2)+"\n")
PYCODE
echo '[build_cython] 保护构建完成；真实NPU运行验收另行执行。'
