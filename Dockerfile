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
COPY wheel/pyarrow-20.0.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl /tmp/wheels/pyarrow-20.0.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
COPY requirements-pip.txt /tmp/requirements-pip.txt
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install /tmp/wheels/pyarrow-20.0.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl && \
    pip install --no-cache-dir --retries 10 --timeout 600 \
      --prefer-binary \
      -r /tmp/requirements-pip.txt


# ================== 运行层（源码；Cython 在 CentOS7+FunASR 下易 segfault，暂不启用）==================
# 若需 Cython：取消注释 builder 阶段，并将下方 COPY 改为 --from=builder
# ARG ENABLE_CYTHON=1 时见 scripts/build_cython.sh
FROM deps AS runtime

WORKDIR /app
ENV CONFIG_PATH=/config.toml

COPY main.py start.sh ./
COPY core/ core/
COPY utils/ utils/
COPY api/ api/
COPY entity/ entity/

# 配置与 Nginx
COPY config.toml /config.toml
COPY nginx.conf /etc/nginx/nginx.conf

RUN chmod +x ./start.sh

EXPOSE 9000
CMD ["/bin/bash", "-c", "./start.sh"]
