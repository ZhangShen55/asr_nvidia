# syntax=docker/dockerfile:1.7
# 本机已有 CANN 8.1.RC1/aarch64/Python3.10 底座；新环境隔离系统 site-packages。
ARG BASE_IMAGE=jy-algorithm-app-asr-ascend_cann8.1.rc1-ub2204py310:v1.1.9_260709
FROM ${BASE_IMAGE} AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc g++ libsndfile1 libgomp1 ffmpeg ca-certificates \
    && apt-get clean
WORKDIR /src
COPY asr-env.tar.gz /tmp/asr-env.tar.gz
RUN mkdir -p /opt/asr && tar -xzf /tmp/asr-env.tar.gz -C /opt/asr \
    && /opt/asr/bin/python /opt/asr/bin/conda-unpack
ENV PATH=/opt/asr/bin:$PATH PYTHONNOUSERSITE=1 PYTHONPATH=""
COPY . .
# 该构建入口只允许在临时 staging 内删除已编译源码。
RUN touch .asr-build-stage && chmod +x scripts/build_cython.sh \
    && PYTHON=/opt/asr/bin/python ./scripts/build_cython.sh /src
# 无设备的构建层仅做导入闭包检查，不能充当 NPU 前向验收。
RUN TORCH_DEVICE_BACKEND_AUTOLOAD=0 /opt/asr/bin/python - <<'PYCODE'
import importlib
for name in ("main", "api.routes.asr_v3", "core.deployment", "core.models", "utils.audio_utils", "entity.data"):
    importlib.import_module(name)
print("protected runtime import smoke: OK (no hardware validation)")
PYCODE

FROM ${BASE_IMAGE} AS runtime
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/asr/bin:$PATH \
    PYTHONNOUSERSITE=1 \
    CONFIG_PATH=/config.toml \
    ASR_DEPLOYMENT_PROFILE=npu-910b \
    ASR_RUNTIME_DIR=/tmp/asr-offline \
    ASR_STATS_PATH=/tmp/asr-offline/asr_stats.json \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libsndfile1 libgomp1 nginx curl ca-certificates util-linux procps \
    && apt-get clean
WORKDIR /app
COPY --from=builder /opt/asr /opt/asr
# 候选底座含历史应用；只清理镜像构建层 /app，避免旧代码/模型混入交付。
RUN /opt/asr/bin/python -c "import shutil; shutil.rmtree('/app', ignore_errors=True)" \
    && mkdir -p /app /etc/nginx/asr-conf.d
COPY --from=builder /src/main.py /src/start.sh /app/
COPY --from=builder /src/config.toml /config.toml
COPY --from=builder /src/model_status_runtime.py /app/model_status_runtime.py
COPY --from=builder /src/telemetry_runtime.py /app/telemetry_runtime.py
COPY --from=builder /src/core /app/core
COPY --from=builder /src/api /app/api
COPY --from=builder /src/utils /app/utils
COPY --from=builder /src/entity /app/entity
COPY --from=builder /src/model /app/model
COPY --from=builder /src/cython-build.json /app/cython-build.json
COPY --from=builder /src/scripts/check_npu_environment.py /app/scripts/check_npu_environment.py
COPY nginx.conf /etc/nginx/nginx.conf
RUN mkdir -p "$ASR_RUNTIME_DIR" && chmod +x /app/start.sh
EXPOSE 9000
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl --fail --silent http://127.0.0.1:9000/get_status >/dev/null || exit 1
ENTRYPOINT ["/app/start.sh"]
