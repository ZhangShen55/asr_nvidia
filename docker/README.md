# ASR Offline v3 CUDA 部署

本目录提供 NVIDIA CUDA 12.4 的 v3 生产部署入口。镜像基于
`docker.1ms.run/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`，内置 VAD、CAM++、
emotion2vec 模型和已验证的 fp16 TensorRT engine。运行时配置通过挂载注入，模型不需要
再挂载到宿主机。

## 前置条件

- Docker、Docker Compose v2 和 NVIDIA Container Toolkit 已安装。
- 宿主机 NVIDIA 驱动能提供 CUDA 12.4 兼容运行时，并能看到 manifest 声明的 GPU 架构（当前资产验证环境为 RTX 4090 / SM89）。
- 可以拉取目标基础镜像；首次构建会下载 Python、PyTorch 2.6.0 cu124、TensorRT 10.7.0.post1 和 ASR 依赖。
- 宿主机准备好上游 Qwen-ASR 地址，并在挂载配置中修改 `[qwen_asr].base_url`。

构建阶段会校验两个模型的 fp16 engine、manifest、模型来源 SHA256 和 TensorRT 资产；运行阶段还会校验真实 GPU、CUDA 与 TensorRT 版本。任何必需资产不完整或不兼容都会阻止服务就绪，不能自动回退 PyTorch。

## 构建镜像

在仓库根目录执行：

```bash
cd docker
ASR_IMAGE=asr_offline:v3_$(date +%Y%m%d%H%M%S) ./build.sh
```

`build.sh` 先建立临时白名单 staging 上下文，只复制应用、三个模型和 fp16 校验来源；测试、样本音频、缓存、fp32 engine 和无关模型不会进入构建上下文。未显式设置 `ASR_IMAGE` 时，脚本自动生成 `asr_offline:v3_<YYYYMMDDHHMMSS>`。

由于基础镜像尚未拉取完成时构建无法开始，请先单独执行：

```bash
docker pull docker.1ms.run/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
```

## 配置挂载与启动

复制并编辑配置模板：

```bash
cp docker/config.toml /path/to/asr-config.toml
# 修改 qwen_asr.base_url、队列、日志等运行参数；保留镜像内 /app/model 路径
```

使用 compose 启动时，必须显式提供符合 v3 时间格式的镜像 tag：

```bash
cd docker
ASR_IMAGE=asr_offline:v3_20261103120000 \
ASR_CONFIG=/path/to/asr-config.toml \
docker compose -f docker-compose.yaml up -d
```

compose 只读挂载一个配置文件到 `/config.toml`，并设置 `CONFIG_PATH=/config.toml`。镜像内默认配置仍位于 `/config.toml`，但生产部署应使用挂载文件。模型默认位于 `/app/model/`，不要把模型目录作为宿主机卷覆盖。

默认 profile 要求：

- `deployment_profile = "cuda-tensorrt-fp16"`
- `speech_campplus_use_tensorrt = true`、`emotion2vec_use_tensorrt = true`
- 两个 TensorRT 精度均为 `fp16`
- `audio_concurrency` 默认 8，`audio_spk_emo_concurrency` 默认 4

## 健康检查和接口

```bash
docker compose -f docker-compose.yaml ps
curl --fail http://127.0.0.1:9000/get_status
```

Qwen 转写接口为 `POST /v3/seacraft_asr`。本次镜像不注册 `/v2/seacraft_asr`；旧客户端会收到 404。`GET /get_status` 的 `deployment` 字段显示实际配置路径和 profile，`localModels.instance_details` 显示每个 TensorRT 实例的 engine、精度和兼容状态。

普通转写请求（A）不携带增强参数；请求 `showSpk=true`、`showEmotion=true` 或 `showRoleIdentify=true` 属于 B。A/B 共用 `audio_concurrency`，B 还受 `audio_spk_emo_concurrency` 限制。配置关闭能力时返回既有能力错误码，不以 `null` 伪装成功。

## 停止、日志和回滚

```bash
docker compose -f docker-compose.yaml logs -f asr-offline
docker compose -f docker-compose.yaml stop
docker compose -f docker-compose.yaml down
```

回滚时停止当前 compose，使用上一份完整的 `asr_offline:v3_<时间>` 镜像和上一份配置重新启动。v2 不在新镜像中保留兼容别名，接口兼容回滚依赖上一完整镜像。
