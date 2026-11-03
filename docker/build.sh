#!/usr/bin/env bash
# 构建最小 staging 上下文并生成 asr_offline:v3_<时间> 镜像。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_TIME="${BUILD_TIME:-$(date +%Y%m%d%H%M%S)}"
IMAGE="${ASR_IMAGE:-asr_offline:v3_${BUILD_TIME}}"

if [[ ! "$IMAGE" =~ ^asr_offline:v3_[0-9]{14}$ ]]; then
  echo "[ERROR] ASR_IMAGE 必须符合 asr_offline:v3_<YYYYMMDDHHMMSS>: $IMAGE" >&2
  exit 2
fi

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/asr-offline-v3.XXXXXX")"

cleanup() { python - "$STAGE" <<'PY'
import shutil, sys
shutil.rmtree(sys.argv[1], ignore_errors=True)
PY
}
trap cleanup EXIT

mkdir -p "$STAGE"/model "$STAGE"/scripts
cp "$REPO_ROOT/main.py" "$STAGE/main.py"
cp "$REPO_ROOT/model_status_runtime.py" "$STAGE/model_status_runtime.py"
cp "$REPO_ROOT/telemetry_runtime.py" "$STAGE/telemetry_runtime.py"
cp "$REPO_ROOT/setup_cython.py" "$STAGE/setup_cython.py"
cp "$REPO_ROOT/config.toml" "$STAGE/config.toml" 2>/dev/null || true
cp "$REPO_ROOT/docker/config.toml" "$STAGE/config.toml"
cp "$REPO_ROOT/docker/start.sh" "$STAGE/start.sh"
cp "$REPO_ROOT/docker/nginx.conf" "$STAGE/nginx.conf"
cp "$REPO_ROOT/scripts/build_cython.sh" "$STAGE/scripts/build_cython.sh"
cp "$REPO_ROOT/scripts/check_tensorrt_assets.py" "$STAGE/scripts/check_tensorrt_assets.py"
cp "$REPO_ROOT/requirements-pip.txt" "$STAGE/requirements-pip.txt"
cp "$REPO_ROOT/requirements-pip-runtime.txt" "$STAGE/requirements-pip-runtime.txt"
cp "$REPO_ROOT/requirements-pip-extra.txt" "$STAGE/requirements-pip-extra.txt"
cp "$REPO_ROOT/requirements-tensorrt.txt" "$STAGE/requirements-tensorrt.txt"
for directory in core api utils entity; do
  cp -a "$REPO_ROOT/$directory" "$STAGE/$directory"
done

python - "$REPO_ROOT" "$STAGE/model" <<'PY'
import json
import shutil
import sys
from pathlib import Path

repo = Path(sys.argv[1])
out = Path(sys.argv[2])
model_names = {
    "speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "speech_campplus_sv_zh_en_16k-common_advanced",
    "emotion2vec_plus_large",
}
for name in model_names:
    source = repo / "model" / name
    target = out / name
    if not source.is_dir():
        raise SystemExit(f"缺少必需模型目录: {source}")
    shutil.copytree(source, target)
    trt = target / "tensorrt"
    if trt.exists():
        for child in list(trt.iterdir()):
            # manifest.json 对 fp16 engine 的导出来源做 SHA256 校验；保留
            # sources 使运行时仍能验证来源，其他精度目录和缓存不进入镜像。
            if child.name not in {"fp16", "sources"}:
                if child.is_dir(): shutil.rmtree(child)
                else: child.unlink()
        fp16 = trt / "fp16"
        if fp16.exists():
            for child in list(fp16.iterdir()):
                if child.name not in {"engine.plan", "manifest.json", "sources"}:
                    if child.is_dir(): shutil.rmtree(child)
                    else: child.unlink()
    # README、下载缓存和模型 requirements 不参与运行时。
    for child in list(target.iterdir()):
        if child.name in {"README.md", "requirements.txt", "tokens_copy.txt"}:
            if child.is_dir(): shutil.rmtree(child)
            else: child.unlink()
    manifest = target / "tensorrt" / "fp16" / "manifest.json"
    if manifest.exists():
        meta = json.loads(manifest.read_text())
        required = {item["path"] for item in meta.get("model_sources", []) + meta.get("export_sources", [])}
        required.add("tensorrt/fp16/engine.plan")
        required.add("tensorrt/fp16/manifest.json")
        for relative in required:
            if not (target / relative).exists():
                raise SystemExit(f"TensorRT manifest 来源缺失: {target / relative}")
PY

printf '[INFO] staging=%s\n' "$STAGE"
printf '[INFO] image=%s\n' "$IMAGE"
docker build --pull -f "$REPO_ROOT/docker/Dockerfile" -t "$IMAGE" "$STAGE"
printf '[INFO] built image: %s\n' "$IMAGE"
