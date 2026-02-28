# ================== 依赖层：系统 & Conda & PIP 依赖（无源码） ==================
FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-runtime-centos7 AS deps

# 基础工具 & nginx
RUN mv /etc/yum.repos.d/CentOS-Base.repo /etc/yum.repos.d/CentOS-Base.repo.bak && \
    curl -o /etc/yum.repos.d/CentOS-Base.repo http://mirrors.aliyun.com/repo/Centos-7.repo && \
    yum clean all  && yum makecache && \
    yum -y install epel-release git wget bzip2 ca-certificates && \
    yum -y install nginx jq && \
    rpm --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro && \
    rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm && \
    yum clean all && yum install -y vim

# Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# conda-forge
RUN conda config --system --set show_channel_urls true && \
    conda config --system --set channel_priority strict && \
    conda config --system --remove channels defaults || true && \
    conda config --system --add channels conda-forge && \
    conda config --system --remove-key default_channels || true && \
    conda config --system --add default_channels https://conda.anaconda.org/conda-forge && \
    conda info

# Python 环境
RUN conda create -n seacraftasr -y -c conda-forge python=3.10 "ffmpeg>=6,<7" libsndfile pysoundfile && conda clean -afy
SHELL ["bash", "-lc"]
ENV PATH="/opt/conda/envs/seacraftasr/bin:$PATH"
RUN python --version && ldd --version

# pip 源 + sentencepiece（wheel 优先）
RUN PIP_EXTRA_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    pip install --only-binary=sentencepiece --prefer-binary sentencepiece==0.2.0
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 将 pyarrow.whl 与 requirements.txt 单独拷入依赖层并安装（不引入源码）
WORKDIR /tmp/build
COPY wheel/pyarrow-20.0.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl /tmp/wheels/pyarrow-20.0.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
COPY requirements.txt /tmp/requirements.txt
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install /tmp/wheels/pyarrow-20.0.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl && \
    pip install --no-cache-dir --retries 10 --timeout 60 \
      --prefer-binary \
      -r /tmp/requirements.txt

# nginx 配置放到运行层去 COPY
# config.json 也放到运行层去 COPY


# ================== 运行层：直接使用源码 ==================
FROM deps AS runtime
WORKDIR /app

# 拷贝源码
COPY . /app

# 配置文件与 Nginx
COPY config.json /
COPY nginx.conf /etc/nginx/nginx.conf

# 权限 & 暴露端口
RUN chmod +x ./start.sh
EXPOSE 9000

# 运行
CMD ["/bin/bash","-c","./start.sh"]
