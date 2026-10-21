# Docker 构建与运行说明

## 镜像内容

镜像基于 CentOS 7 + CUDA 12.1.1，使用 Python 3.10 Conda 环境。业务只保留离线 ASR、状态查询和任务列表清理。

五个模型目录随镜像复制到 `/app/model/`：

- `speech_fsmn_vad_zh-cn-16k-common-pytorch`
- `punc_ct-transformer_cn-en-common-vocab471067-large`
- `speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`
- `speech_campplus_sv_zh_en_16k-common_advanced`
- `emotion2vec_plus_large`

## 构建

从仓库根目录执行：

```bash
docker build -t jy-algorithm-app-asr-offline:v3.0 .
```

构建要求：

1. 五个模型目录完整存在于 `model/`。
2. 构建过程保留 Docker 缓存，不执行 `--no-cache`。
3. 不要求提供 `wheel/pyarrow-*.whl`。

## 依赖处理

- `av` 由 conda-forge 安装，避免在 CentOS 7 上源码编译。
- `pyarrow==20.0.0` 由 `requirements-pip.txt` 显式锁定；CP310 wheel 标签兼容 CentOS 7 的 glibc 2.17。
- Cython 在容器内运行业务模块时曾出现段错误，因此镜像继续使用 Python 源码。

## 运行

默认镜像不需要挂载模型：

```bash
docker run -d \
  --name jy-asr-offline-v3 \
  --gpus '"device=1"' \
  -p 9000:9000 \
  jy-algorithm-app-asr-offline:v3.0
```

镜像内 `/config.toml` 来自 `config.docker.toml`。如需调整并发或设备，可挂载配置文件；模型路径继续使用 `/app/model/`。

## 已验证行为

验证时记录以下结果：

- 构建成功，五个模型目录检查通过。
- 不挂载模型时 `/get_status` 返回 `status=living`。
- OpenAPI 仅包含 `/v1.1.8/seacraft_asr`、`/get_status` 和 `/clear_tasks_list`。
- 已移除接口返回 `404`。
- 真实音频请求 `/v1.1.8/seacraft_asr` 成功。
