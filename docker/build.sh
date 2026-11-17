#!/usr/bin/env bash
# 只在临时 staging 中组装原始权重和可重定位 asr 环境；不清理镜像或缓存。
set -euo pipefail
ASR_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASR_BUILD_TIME="${BUILD_TIME:-$(date +%Y%m%d%H%M%S)}"
ASR_IMAGE="${ASR_IMAGE:-asr_offline:v3_${ASR_BUILD_TIME}}"
BASE_IMAGE="${BASE_IMAGE:-jy-algorithm-app-asr-ascend_cann8.1.rc1-ub2204py310:v1.1.9_260709}"
ASR_ENV_PREFIX="${ASR_ENV_PREFIX:-/home/zhangs/conda-envs/asr}"
ASR_PYTHON="${ASR_PYTHON:-$ASR_ENV_PREFIX/bin/python}"
[[ "$ASR_IMAGE" =~ :v3_[0-9]{14}$ ]] || { echo '[ERROR] ASR_IMAGE tag 必须为 v3_YYYYMMDDHHMMSS' >&2; exit 2; }
[[ -x "$ASR_PYTHON" ]] || { echo "[ERROR] 缺少环境解释器: $ASR_PYTHON" >&2; exit 2; }
# 构建前先明确文件缺失，不生成或改动模型资产。
"$ASR_PYTHON" - "$ASR_REPO_ROOT" <<'PYCODE'
from pathlib import Path
import sys
root=Path(sys.argv[1])/"model"
required={"speech_fsmn_vad_zh-cn-16k-common-pytorch":["model.pt","config.yaml"],
          "speech_campplus_sv_zh_en_16k-common_advanced":["campplus_cn_en_common.pt","config.yaml"],
          "emotion2vec_plus_large":["model.pt","config.yaml","tokens.txt"]}
for name,files in required.items():
    for filename in files:
        p=root/name/filename
        if not p.is_file() or not p.stat().st_size:
            raise SystemExit(f"缺少必需模型文件: {p}")
PYCODE
ASR_STAGE="$(mktemp -d "${TMPDIR:-/tmp}/asr-npu-build.XXXXXX")"
cleanup() {
  "$ASR_PYTHON" - "$ASR_STAGE" <<'PYCODE'
from pathlib import Path
import shutil,sys
p=Path(sys.argv[1])
if p.name.startswith("asr-npu-build."):
    shutil.rmtree(p)
PYCODE
}
trap cleanup EXIT
mkdir -p "$ASR_STAGE/scripts" "$ASR_STAGE/model"
for file in main.py model_status_runtime.py telemetry_runtime.py setup_cython.py; do
  cp "$ASR_REPO_ROOT/$file" "$ASR_STAGE/$file"
done
cp "$ASR_REPO_ROOT/docker/config.toml" "$ASR_STAGE/config.toml"
cp "$ASR_REPO_ROOT/docker/start.sh" "$ASR_STAGE/start.sh"
cp "$ASR_REPO_ROOT/docker/nginx.conf" "$ASR_STAGE/nginx.conf"
cp "$ASR_REPO_ROOT/scripts/build_cython.sh" "$ASR_STAGE/scripts/build_cython.sh"
cp "$ASR_REPO_ROOT/scripts/check_npu_environment.py" "$ASR_STAGE/scripts/check_npu_environment.py"
for directory in core api utils entity; do cp -a "$ASR_REPO_ROOT/$directory" "$ASR_STAGE/$directory"; done
"$ASR_PYTHON" - "$ASR_REPO_ROOT/model" "$ASR_STAGE/model" <<'PYCODE'
import shutil,sys
from pathlib import Path
source,target=map(Path,sys.argv[1:])
for name in ("speech_fsmn_vad_zh-cn-16k-common-pytorch","speech_campplus_sv_zh_en_16k-common_advanced","emotion2vec_plus_large"):
    shutil.copytree(source/name,target/name,ignore=shutil.ignore_patterns("tensorrt","__pycache__",".git"))
PYCODE
if [[ -n "${ASR_ENV_ARCHIVE:-}" ]]; then
  [[ -r "$ASR_ENV_ARCHIVE" ]] || { echo '[ERROR] ASR_ENV_ARCHIVE 不可读' >&2; exit 2; }
  cp "$ASR_ENV_ARCHIVE" "$ASR_STAGE/asr-env.tar.gz"
else
  "$ASR_PYTHON" -c 'import conda_pack' 2>/dev/null || {
    echo '[ERROR] 缺少 conda-pack；请在独立构建环境安装 requirements-build.txt，或提供 ASR_ENV_ARCHIVE。不会自动安装。' >&2; exit 2;
  }
  "$ASR_PYTHON" -m conda_pack --prefix "$ASR_ENV_PREFIX" --output "$ASR_STAGE/asr-env.tar.gz"
fi
# 外部归档也必须含可重定位入口；实际 ABI/依赖验证由 builder 完成。
"$ASR_PYTHON" - "$ASR_STAGE/asr-env.tar.gz" <<'PYCODE'
import tarfile,sys
with tarfile.open(sys.argv[1]) as f:
    names={p.name.lstrip("./") for p in f.getmembers()}
    for required in ("bin/python","bin/conda-unpack"):
        if required not in names: raise SystemExit(f"环境归档缺少 {required}")
PYCODE
DOCKER_BUILDKIT=1 docker build --progress=plain --pull=false \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" --build-arg BUILDKIT_INLINE_CACHE=1 \
  -f "$ASR_REPO_ROOT/docker/Dockerfile" -t "$ASR_IMAGE" "$ASR_STAGE"
echo "[INFO] built image: $ASR_IMAGE"
