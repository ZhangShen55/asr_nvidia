#!/usr/bin/env bash
# 按目标 GPU profile 构建最小 staging 上下文，并保留 BuildKit 本地缓存。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_TIME="${BUILD_TIME:-$(date +%Y%m%d%H%M%S)}"
GPU_PROFILE="${GPU_PROFILE:-rtx4090-sm89}"
IMAGE="${ASR_IMAGE:-asr_offline-rtx4090:v3_${BUILD_TIME}}"
BASE_IMAGE="${BASE_IMAGE:-docker.1ms.run/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04}"
CACHE_DIR="${DOCKER_CACHE_DIR:-$REPO_ROOT/.docker-cache/shared}"

case "$GPU_PROFILE" in
  rtx3090-sm86) DEFAULT_DOCKERFILE="Dockerfile.rtx3090" ;;
  rtx4070tisuper-sm89) DEFAULT_DOCKERFILE="Dockerfile.rtx4070tisuper" ;;
  rtx4090-sm89) DEFAULT_DOCKERFILE="Dockerfile.rtx4090" ;;
  *)
    echo "[ERROR] 不支持的 GPU_PROFILE: $GPU_PROFILE" >&2
    echo "        可选值: rtx3090-sm86, rtx4070tisuper-sm89, rtx4090-sm89" >&2
    exit 2
    ;;
esac
DOCKERFILE="${DOCKERFILE:-$DEFAULT_DOCKERFILE}"

[[ "$IMAGE" == *:* ]] || {
  echo "[ERROR] ASR_IMAGE 必须是合法的 image:tag: $IMAGE" >&2
  exit 2
}
[[ -f "$REPO_ROOT/docker/$DOCKERFILE" ]] || {
  echo "[ERROR] Dockerfile 不存在: $REPO_ROOT/docker/$DOCKERFILE" >&2
  exit 2
}

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

python - "$REPO_ROOT" "$STAGE/model" "$GPU_PROFILE" <<'PY'
import json
import shutil
import sys
from pathlib import Path

repo = Path(sys.argv[1])
out = Path(sys.argv[2])
profile = sys.argv[3]
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

    is_trt_model = name in {
        "speech_campplus_sv_zh_en_16k-common_advanced",
        "emotion2vec_plus_large",
    }
    if not is_trt_model:
        shutil.copytree(source, target)
    else:
        # 只复制原始权重/配置；TensorRT 资产单独选出一个 profile，避免
        # 把其他 GPU、fp32 engine 和 ONNX 文件带入目标镜像。
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("tensorrt"))
        source_trt = source / "tensorrt"
        selected = source_trt / profile / "fp16"
        if not selected.is_dir() and profile == "rtx4090-sm89":
            # 4090 的正式资产历史上位于顶层 tensorrt/fp16。
            selected = source_trt / "fp16"
        if not (selected / "engine.plan").is_file() or not (selected / "manifest.json").is_file():
            raise SystemExit(f"{name} 缺少 profile={profile} 的 fp16 engine/manifest: {selected}")
        target_trt = target / "tensorrt"
        target_fp16 = target_trt / "fp16"
        target_fp16.mkdir(parents=True)
        shutil.copy2(selected / "engine.plan", target_fp16 / "engine.plan")
        shutil.copy2(selected / "manifest.json", target_fp16 / "manifest.json")
        source_sources = source_trt / "sources"
        if source_sources.is_dir():
            shutil.copytree(source_sources, target_trt / "sources")

        manifest = json.loads((target_fp16 / "manifest.json").read_text())
        if manifest.get("precision_policy") != "fp16":
            raise SystemExit(f"{selected}/manifest.json 不是 fp16 资产")
        required = {item["path"] for item in manifest.get("model_sources", []) + manifest.get("export_sources", [])}
        required.update({"tensorrt/fp16/engine.plan", "tensorrt/fp16/manifest.json"})
        for relative in required:
            if not (target / relative).exists():
                raise SystemExit(f"TensorRT manifest 来源缺失: {target / relative}")

    # README、下载缓存和模型 requirements 不参与运行时。
    for child in list(target.iterdir()):
        if child.name in {"README.md", "requirements.txt", "tokens_copy.txt"}:
            if child.is_dir(): shutil.rmtree(child)
            else: child.unlink()
PY

printf '[INFO] staging=%s\n' "$STAGE"
printf '[INFO] image=%s\n' "$IMAGE"
printf '[INFO] gpu_profile=%s\n' "$GPU_PROFILE"
printf '[INFO] dockerfile=%s\n' "$DOCKERFILE"
printf '[INFO] base_image=%s\n' "$BASE_IMAGE"
printf '[INFO] build_cache=%s\n' "$CACHE_DIR"
mkdir -p "$CACHE_DIR"

build_args=(
  --progress=plain
  --pull=false
  --build-arg "BASE_IMAGE=$BASE_IMAGE"
  --build-arg "GPU_PROFILE=$GPU_PROFILE"
  --build-arg "BUILDKIT_INLINE_CACHE=1"
  -f "$REPO_ROOT/docker/$DOCKERFILE"
  -t "$IMAGE"
)
# The host's default Docker driver keeps intermediate layers in the daemon's
# local cache.  It does not support type=local cache export; inline cache is
# enabled above so the image also carries cache metadata without requiring a
# network pull or a separate buildkit daemon.
DOCKER_BUILDKIT=1 docker build "${build_args[@]}" "$STAGE"
printf '[INFO] built image: %s\n' "$IMAGE"
