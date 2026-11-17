#!/usr/bin/env bash
# 容器内入口；设备映射只由 compose/宿主启动入口负责。
set -euo pipefail
export CONFIG_PATH="${CONFIG_PATH:-/config.toml}"
export ASR_DEPLOYMENT_PROFILE="${ASR_DEPLOYMENT_PROFILE:-npu-910b}"
export ASR_RUNTIME_DIR="${ASR_RUNTIME_DIR:-/tmp/asr-offline}"
export ASR_STATS_PATH="${ASR_STATS_PATH:-${ASR_RUNTIME_DIR}/asr_stats.json}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONNOUSERSITE=1
export PYTHONFAULTHANDLER=1
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
APP_ROOT="${ASR_APP_ROOT:-/app}"
ASR_PYTHON="${ASR_PYTHON:-/opt/asr/bin/python}"
ASR_CANN_ENV="${ASR_CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
[[ -x "$ASR_PYTHON" ]] || { echo "[ERROR] asr 解释器不存在: $ASR_PYTHON" >&2; exit 1; }
[[ -r "$ASR_CANN_ENV" ]] || { echo "[ERROR] CANN 脚本不存在: $ASR_CANN_ENV" >&2; exit 1; }
# 禁止继承镜像系统 site-packages；CANN 脚本随后仅加入其自身支持路径。
unset PYTHONPATH
set +u
source "$ASR_CANN_ENV"
set -u
cd "$APP_ROOT"
mkdir -p "$ASR_RUNTIME_DIR"
[[ -r "$CONFIG_PATH" ]] || { echo "[ERROR] CONFIG_PATH 不可读: $CONFIG_PATH" >&2; exit 1; }
"$ASR_PYTHON" scripts/check_npu_environment.py --config "$CONFIG_PATH"
instance_count=$("$ASR_PYTHON" - <<'PYCODE'
import os
from core.config import Settings
s=Settings(config_path=os.environ["CONFIG_PATH"])
s.validate_model_backends()
if s.instance_count < 1: raise ValueError("instance_count 必须是正整数")
print(s.instance_count)
PYCODE
)
echo "[INFO] instance_count=${instance_count}；每个完整进程各自加载 Qwen 和模型池"
mkdir -p /etc/nginx/asr-conf.d
upstream=/etc/nginx/asr-conf.d/backend_upstream.conf
{
  echo 'upstream backend {'
  for ((i=0; i<instance_count; i++)); do printf '    server 127.0.0.1:%s;\n' "$((8000+i))"; done
  echo '}'
} > "$upstream"
nginx -t
children=()
nginx_pid=''
ASR_STOP_STATUS=''
ASR_SHUTDOWN_TIMEOUT_S="${ASR_SHUTDOWN_TIMEOUT_S:-150}"
ASR_SHUTDOWN_TERM_TIMEOUT_S="${ASR_SHUTDOWN_TERM_TIMEOUT_S:-10}"
for timeout in "$ASR_SHUTDOWN_TIMEOUT_S" "$ASR_SHUTDOWN_TERM_TIMEOUT_S"; do
  [[ "$timeout" =~ ^[0-9]+$ ]] || { echo '[ERROR] 关闭超时必须为非负整数' >&2; exit 2; }
done
owned_group_alive() {
  ps -eo pgid=,stat= | awk -v group="$1" '$1 == group && $2 !~ /^Z/ { found=1 } END { exit !found }'
}
wait_owned_groups() {
  local deadline=$((SECONDS + $1)) alive pid
  while true; do
    alive=0
    for pid in "${groups[@]}"; do
      if owned_group_alive "$pid"; then alive=1; fi
    done
    (( alive == 0 )) && return 0
    (( SECONDS >= deadline )) && return 1
    sleep 1
  done
}
cleanup() {
  trap '' TERM INT
  local exit_status="${1:-1}" pid child_status
  trap - EXIT
  # 正常只通知main.py服务父进程TERM；统一入口使lifespan结束后的信号重放走Python atexit。
  # 组级信号仅在超过有界宽限期时使用，不会抢先中止子进程清理。
  for pid in "${children[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  if [[ -n "$nginx_pid" ]]; then kill -QUIT "$nginx_pid" 2>/dev/null || true; fi
  groups=("${children[@]}")
  [[ -z "$nginx_pid" ]] || groups+=("$nginx_pid")
  if ! wait_owned_groups "$ASR_SHUTDOWN_TIMEOUT_S"; then
    echo '[ERROR] 自有服务未在宽限期完成关闭；本次测试/服务收尾失败，升级组级TERM' >&2
    (( exit_status != 0 )) || exit_status=1
    for pid in "${groups[@]}"; do
      if owned_group_alive "$pid"; then kill -TERM -- "-$pid" 2>/dev/null || true; fi
    done
    if ! wait_owned_groups "$ASR_SHUTDOWN_TERM_TIMEOUT_S"; then
      for pid in "${groups[@]}"; do
        if owned_group_alive "$pid"; then
          echo "[ERROR] 自有进程组 $pid TERM后仍存活，升级KILL" >&2
          kill -KILL -- "-$pid" 2>/dev/null || true
        fi
      done
    fi
  fi
  # 监督阶段不调用wait消费状态，收尾时逐PID完整取得真实退出码。
  # 127表示状态不可归属或未知，必须失败，不能伪造正常回收。
  for pid in "${groups[@]}"; do
    if wait "$pid" 2>/dev/null; then child_status=0; else child_status=$?; fi
    if (( child_status != 0 )); then
      echo "[ERROR] 自有子进程 $pid 退出异常 exit=${child_status}（未知127同样失败）" >&2
      (( exit_status != 0 )) || exit_status=1
    fi
  done
  exit "$exit_status"
}
trap 'cleanup "$?"' EXIT
# 启动期信号只置标记，后台启动与PID登记之间不会提前退出而遗弃子进程。
# TERM保留已有取消状态；INT明确返回130，不能被后续TERM洗成成功。
trap 'ASR_STOP_STATUS=${ASR_STOP_STATUS:-0}' TERM
trap 'ASR_STOP_STATUS=130' INT
setsid nginx -g 'daemon off;' &
nginx_pid=$!
for ((i=0; i<instance_count; i++)); do
  setsid "$ASR_PYTHON" main.py --host 127.0.0.1 --port "$((8000+i))" --workers 1 &
  children+=("$!")
done
# 标记检查后只允许最多0.2秒延迟；窗口内收到信号会很快再次检查，
# 不会进入无期限wait丢失关闭请求。所有PID登记后才开始监督。
while [[ -z "$ASR_STOP_STATUS" ]]; do
  child_finished=0
  for pid in "$nginx_pid" "${children[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then child_finished=1; break; fi
  done
  if (( child_finished != 0 )); then break; fi
  if sleep 0.2; then :; elif [[ -z "$ASR_STOP_STATUS" ]]; then
    echo '[ERROR] 尚未请求关闭时，监督轮询延迟失败' >&2
    exit 1
  fi
done
exit "${ASR_STOP_STATUS:-0}"
