#!/usr/bin/env bash
set -euo pipefail

export CONFIG_PATH="${CONFIG_PATH:-/config.toml}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

APP_ROOT=/app
PYTHON=/opt/venv/bin/python
UVICORN=/opt/venv/bin/uvicorn

cd "$APP_ROOT"
test -r "$CONFIG_PATH" || { echo "[ERROR] CONFIG_PATH 不可读: $CONFIG_PATH" >&2; exit 1; }
test -x "$PYTHON" || { echo "[ERROR] runtime venv 不存在: $PYTHON" >&2; exit 1; }

instance_count=$(
  CONFIG_PATH="$CONFIG_PATH" "$PYTHON" - <<'PY'
import os
from core.config import Settings
settings = Settings(config_path=os.environ["CONFIG_PATH"])
settings.validate_model_backends()
value = settings.instance_count
if value < 1:
    raise ValueError("instance_count 必须是正整数")
print(value)
PY
)

printf '[INFO] CONFIG_PATH=%s\n' "$CONFIG_PATH"
printf '[INFO] deployment_profile=%s\n' "${DEPLOYMENT_PROFILE:-cuda-tensorrt-fp16}"
printf '[INFO] 启动 %s 个 uvicorn 实例\n' "$instance_count"

upstream=/etc/nginx/conf.d/backend_upstream.conf
{
  echo 'upstream backend {'
  for ((i=0; i<instance_count; i++)); do
    printf '    server 127.0.0.1:%s;\n' "$((8000 + i))"
  done
  echo '}'
} > "$upstream"

nginx -t
nginx

children=()
cleanup() {
  trap - TERM INT EXIT
  for pid in "${children[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait || true
  nginx -s quit 2>/dev/null || true
}
trap cleanup TERM INT EXIT

for ((i=0; i<instance_count; i++)); do
  port=$((8000 + i))
  echo "[INFO] 启动 uvicorn 端口: $port"
  "$UVICORN" main:app --host 127.0.0.1 --port "$port" --workers 1 &
  children+=("$!")
done

wait -n "${children[@]}"
status=$?
echo "[ERROR] uvicorn 实例退出，status=$status" >&2
exit "$status"
