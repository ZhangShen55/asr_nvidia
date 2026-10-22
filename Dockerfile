# ================== 依赖层：系统 & Conda & PIP 依赖（无业务源码） ==================
FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-runtime-centos7 AS deps

RUN mv /etc/yum.repos.d/CentOS-Base.repo /etc/yum.repos.d/CentOS-Base.repo.bak && \
    curl -o /etc/yum.repos.d/CentOS-Base.repo http://mirrors.aliyun.com/repo/Centos-7.repo && \
    yum clean all && yum makecache && \
    yum -y install epel-release git wget bzip2 ca-certificates && \
    yum -y install nginx && \
    rpm --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro && \
    rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm && \
    yum clean all && yum install -y vim

# CentOS 7 仅 glibc 2.17；latest Miniconda 安装器要求 glibc >= 2.28，须固定旧版
ARG MINICONDA_INSTALLER=Miniconda3-py310_23.11.0-2-Linux-x86_64.sh
RUN wget -q "https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}" -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && rm -f /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

RUN conda config --system --set show_channel_urls true && \
    conda config --system --set channel_priority strict && \
    conda config --system --remove channels defaults || true && \
    conda config --system --add channels conda-forge && \
    conda config --system --remove-key default_channels || true && \
    conda config --system --add default_channels https://conda.anaconda.org/conda-forge && \
    conda info

# av 须从 conda 安装（PyPI 在 CentOS7 上常无 wheel，源码编译需 ffmpeg pkg-config）
RUN conda create -n seacraftasr -y -c conda-forge \
    python=3.10 \
    "ffmpeg>=6,<8" \
    av=14.4.0 \
    libsndfile \
    pysoundfile \
    pkg-config \
    && conda clean -afy
SHELL ["bash", "-lc"]
ENV PATH="/opt/conda/envs/seacraftasr/bin:$PATH"
RUN python --version && ldd --version

RUN PIP_EXTRA_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    pip install --only-binary=sentencepiece --prefer-binary sentencepiece==0.2.0
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

WORKDIR /tmp/build
COPY requirements-pip.txt /tmp/requirements-pip.txt
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install --retries 10 --timeout 600 \
      --prefer-binary \
      -r /tmp/requirements-pip.txt

# 小语种与文本规范化运行时单独成层：基础依赖清单变化时不必重下 torch 等大包
# pynini 由 conda-forge 提供（PyPI 的 cp310 manylinux wheel 与 glibc 2.17 不兼容）
COPY requirements-pip-extra.txt /tmp/requirements-pip-extra.txt
RUN conda install -y -n seacraftasr -c conda-forge pynini && \
    conda clean -afy
RUN pip install --retries 10 --timeout 600 \
      --prefer-binary \
    -r /tmp/requirements-pip-extra.txt && \
    pip install --retries 10 --timeout 600 --no-deps \
      WeTextProcessing==1.2.0


# ================== 运行层（源码；Cython 在 CentOS7+FunASR 下易 segfault，暂不启用）==================
# 若需 Cython：取消注释 builder 阶段，并将下方 COPY 改为 --from=builder
# ARG ENABLE_CYTHON=1 时见 scripts/build_cython.sh
FROM deps AS runtime

WORKDIR /app
ENV CONFIG_PATH=/config.toml

# 先落地默认容器配置，模型校验需要据此判断是否启用小语种能力
COPY config.docker.toml /config.toml

# 模型独立成层，模型变化不触发依赖重装
RUN --mount=type=bind,source=model,target=/tmp/model,readonly \
    set -e; \
    for d in \
      speech_fsmn_vad_zh-cn-16k-common-pytorch \
      punc_ct-transformer_cn-en-common-vocab471067-large \
      speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
      speech_campplus_sv_zh_en_16k-common_advanced \
      emotion2vec_plus_large; do \
        test -d "/tmp/model/$d" || { echo "[ERROR] 缺少模型目录: $d"; exit 1; }; \
        test -n "$(find "/tmp/model/$d" -type f -print -quit)" || { echo "[ERROR] 模型目录为空: $d"; exit 1; }; \
    done; \
    if grep -Eq '^[[:space:]]*open_mul_lang[[:space:]]*=[[:space:]]*true' /config.toml; then \
        test -d "/tmp/model/faster-whisper-large-v3" || { echo "[ERROR] 缺少模型目录: faster-whisper-large-v3"; exit 1; }; \
        test -n "$(find "/tmp/model/faster-whisper-large-v3" -type f -print -quit)" || { echo "[ERROR] 模型目录为空: faster-whisper-large-v3"; exit 1; }; \
        echo "[INFO] 小语种模型目录检查通过"; \
    else \
        echo "[INFO] 未开启小语种能力，跳过 faster-whisper-large-v3 校验"; \
    fi; \
    echo "[INFO] 必需模型目录检查通过"

COPY model/ /app/model/

COPY main.py start.sh ./
COPY core/ core/
COPY utils/ utils/
COPY api/ api/
COPY entity/ entity/

# 配置与 Nginx
COPY nginx.conf /etc/nginx/nginx.conf

RUN chmod +x ./start.sh

EXPOSE 9000
CMD ["/bin/bash", "-c", "./start.sh"]
