# syntax=docker/dockerfile:1.7
# BASE_IMAGE 在正式构建中固定为本机已核验底座的任务私有 registry digest。
ARG BASE_IMAGE=cann:8.2.rc1-910b-ubuntu22.04-py3.10
FROM ${BASE_IMAGE} AS environment
ENV PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
# 归档/wheel以命名上下文只读挂载，不把大归档或工具写入任何镜像层。
RUN --mount=type=bind,from=runtime_env,source=.,target=/input \
    mkdir -p /opt/asr \
    && tar -xzf /input/asr-env.tar.gz -C /opt/asr \
    && /opt/asr/bin/python /opt/asr/bin/conda-unpack \
    && find /opt/asr/bin -maxdepth 1 \( -type f -o -type l \) -name "python*-config" -delete \
    && PYTHONPATH= /opt/asr/bin/python -m pip install --no-index --no-deps --no-compile /input/*.whl \
    && PYTHONPATH= /opt/asr/bin/python -m pip check \
    && /opt/asr/bin/python -c "from importlib.metadata import version; assert version('torch-npu') == '2.5.1.post1'; assert version('torch').split('+')[0] == '2.5.1'" \
    && /opt/asr/bin/python -c "from pathlib import Path; Path('/opt/asr/bin/conda-unpack').unlink()"

FROM ${BASE_IMAGE} AS runtime-deps
ENV DEBIAN_FRONTEND=noninteractive
# 仅补运行库。业务编译工具只安装到独立编译阶段。
# 底座遗留01proxy指向旧网络，构建只直连已验证的官方HTTPS源，不继承该代理。
RUN --mount=type=cache,id=asr-cann82-apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=asr-cann82-apt-lists,target=/var/lib/apt/lists,sharing=locked \
    find /etc/apt/apt.conf.d -maxdepth 1 -type f -name 01proxy -delete \
    && sed -i 's|http://ports.ubuntu.com|https://ports.ubuntu.com|g' /etc/apt/sources.list \
    && apt-get -o Acquire::http::Proxy=DIRECT -o Acquire::https::Proxy=DIRECT -o Acquire::Retries=3 update \
    && apt-get -o Acquire::http::Proxy=DIRECT -o Acquire::https::Proxy=DIRECT install -y --no-install-recommends ffmpeg libsndfile1 libgomp1 ca-certificates

FROM runtime-deps AS protection
COPY --from=environment /opt/asr /opt/asr
RUN --mount=type=cache,id=asr-cann82-apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=asr-cann82-apt-lists,target=/var/lib/apt/lists,sharing=locked \
    apt-get -o Acquire::https::Proxy=DIRECT -o Acquire::Retries=3 update \
    && apt-get -o Acquire::https::Proxy=DIRECT -o Acquire::Retries=3 install -y --no-install-recommends gcc binutils libc6-dev
RUN --mount=type=bind,from=cython_tools,source=.,target=/input \
    echo "b988bb297ce76c671e28c97d017b95411010f7c77fa6623dd0bb47eed1aee1bc  /input/cython-3.0.12.tar.gz" | sha256sum -c - \
    && PYTHONPATH= NO_CYTHON_COMPILE=true /opt/asr/bin/python -m pip install \
       --no-build-isolation --no-index --no-deps --no-compile \
       --target /opt/cython-tools /input/cython-3.0.12.tar.gz
WORKDIR /build
COPY core/ /build/core/
COPY api/ /build/api/
COPY utils/ /build/utils/
COPY entity/ /build/entity/
# 删除操作仅对本阶段的独立目录生效。限制编译产物 ABI 后缀，并剥离调试符号。
RUN --mount=type=bind,source=setup_cython.py,target=/build/setup_cython.py \
    --mount=type=bind,source=scripts/build_cython.sh,target=/build/build_cython.sh \
    touch .asr-build-stage \
    && PYTHON=/opt/asr/bin/python PYTHONPATH=/opt/cython-tools \
       PYTHONDONTWRITEBYTECODE=1 CC=gcc LDSHARED='gcc -shared' CFLAGS=-g0 \
       bash /build/build_cython.sh /build \
    && /opt/asr/bin/python -c 'import sysconfig; from pathlib import Path; suffix=sysconfig.get_config_var("EXT_SUFFIX"); roots=tuple(Path(p) for p in ("core","api","utils","entity")); assert all(any(root.rglob("*"+suffix)) for root in roots); assert all(p.is_dir() or p.name=="__init__.py" or p.name.endswith(suffix) for root in roots for p in root.rglob("*"))' \
    && find core api utils entity -type f -name '*.so' -exec strip --strip-unneeded {} +

FROM runtime-deps AS runtime
COPY --from=environment /opt/asr /opt/asr
ENV PATH=/opt/asr/bin:$PATH \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONFIG_PATH=/config.toml \
    ASR_DEPLOYMENT_PROFILE=npu-910b \
    ASR_CONTAINER_RUNTIME=cann82-source \
    ASR_RUNTIME_DIR=/tmp/asr-offline \
    TMPDIR=/tmp/asr-offline \
    ASR_STATS_PATH=/tmp/asr-offline/asr_stats.json \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
WORKDIR /app
# 运行层从独立编译阶段复制扩展。三个 Python 入口保留启动和解释器清理语义。
COPY main.py model_status_runtime.py telemetry_runtime.py /app/
COPY --from=protection /build/core/ /app/core/
COPY --from=protection /build/api/ /app/api/
COPY --from=protection /build/utils/ /app/utils/
COPY --from=protection /build/entity/ /app/entity/
# 模型上下文必须由准备工具先筛选，原始模型文件始终只读。
COPY --from=model_assets /speech_fsmn_vad_zh-cn-16k-common-pytorch/ /app/model/speech_fsmn_vad_zh-cn-16k-common-pytorch/
COPY --from=model_assets /speech_campplus_sv_zh_en_16k-common_advanced/ /app/model/speech_campplus_sv_zh_en_16k-common_advanced/
COPY --from=model_assets /emotion2vec_plus_large/ /app/model/emotion2vec_plus_large/
COPY --from=protection /build/cython-build.json /opt/asr-metadata/
COPY --from=runtime_env /runtime-packages.json /wheel-source.json /opt/asr-metadata/
RUN mkdir -p /tmp/asr-offline
# 追溯元数据置于依赖/模型层之后，修改发布信息不会作废昂贵的构建缓存。
ARG SOURCE_REVISION=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="asr_offline_910b" \
      org.seacraftasr.code-format="cython" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.description="910B CANN8.2 Cython-protected runtime; single HTTP; external readonly config"
EXPOSE 9000
HEALTHCHECK --interval=15s --timeout=5s --start-period=360s --retries=6 \
    CMD /opt/asr/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9000/get_status',timeout=4).read()" || exit 1
# CANN环境脚本有未定义变量，不能在set -u下source；exec保留TERM及正常NPU退出链。
ENTRYPOINT ["/bin/bash", "-c", "set -e; set +u; source /usr/local/Ascend/ascend-toolkit/set_env.sh; export ASCEND_RT_VISIBLE_DEVICES=\"${ASCEND_RT_VISIBLE_DEVICES:-${ASCEND_VISIBLE_DEVICES:-7}}\"; exec /opt/asr/bin/python /app/main.py \"$@\"", "asr"]
CMD ["--host", "0.0.0.0", "--port", "9000", "--workers", "1", "--timeout-graceful-shutdown", "30"]
