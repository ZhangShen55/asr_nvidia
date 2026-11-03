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

## 部署 Qwen-ASR 上游服务

Qwen-ASR 是独立的 GPU 服务，不放入本项目的 `asr-offline` 镜像，也不由本项目的
`docker-compose.yaml` 自动创建。这样可以独立调整 Qwen 的模型、KV cache 和 API worker
数量，避免和 CAM++/emotion2vec 共用一个容器生命周期。

下面是使用 Qwen 镜像默认入口的示例。具体镜像 tag、模型大小和启动参数应以实际使用的
Qwen-ASR 镜像说明为准：

```bash
docker run -d \
  --name qwen3-asr-0.6b-8806 \
  --restart unless-stopped \
  --gpus all \
  --shm-size 16g \
  -p 8806:8000 \
  qwenllm/qwen3-asr:latest
```

确认 Qwen 服务就绪：

```bash
curl --fail http://127.0.0.1:8806/v1/models
```

然后在挂载给 `asr-offline` 的配置中设置：

```toml
[qwen_asr]
base_url = "http://宿主机地址:8806"
```

如果两个容器使用同一个自定义 Docker 网络，可以改为 Qwen 容器名，例如
`http://qwen3-asr-0.6b-8806:8000`；如果使用当前 compose 的默认网络和宿主机端口映射，
则使用宿主机可访问的 IP 和 `8806` 端口。启动 `asr-offline` 前应先确认该 `/v1/models`
接口可访问。

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

compose 只读挂载一个配置文件到 `/config.toml`，并设置 `CONFIG_PATH=/config.toml`。镜像内也提供同路径的默认模板，生产部署应使用挂载文件。模型默认位于 `/app/model/`，不要把模型目录作为宿主机卷覆盖。

CUDA TensorRT profile 由镜像内部的 `ASR_DEPLOYMENT_PROFILE=cuda-tensorrt-fp16` 环境变量启用，
不需要也不应该写进挂载的 `config.toml`。这样用户配置只负责业务运行参数，镜像仍能强制校验
fp16 TensorRT 资产；Compose 已显式传入该内部变量。

profile 要求：

- `speech_campplus_use_tensorrt = true`、`emotion2vec_use_tensorrt = true`
- 两个 TensorRT 精度均为 `fp16`
- `audio_concurrency` 默认 8，`audio_spk_emo_concurrency` 默认 4

应用日志和任务统计写入容器的 `/tmp/asr-offline/`，不会在 `/app` 下生成
`asr_service.log` 或 `asr_stats.json`。运行镜像的 `/app` 只包含入口、运行时辅助模块、
Cython 扩展和模型，不包含配置副本、构建脚本、测试文件或 Python 缓存。

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

## 是否在容器启动时自动构建 TensorRT engine

技术上可行，但当前镜像不能直接执行这件事。容器内构建需要同时具备：

- `--gpus all` 提供实际目标 GPU；
- TensorRT builder、CUDA、PyTorch 和与目标环境匹配的运行库；
- `onnx==1.17.0` 等导出依赖，以及 `scripts/export_local_tensorrt.py` 和完整模型权重；
- 可写的 engine 缓存目录；
- 足够的磁盘、主机内存和 GPU 显存用于导出、构图和 tactic 搜索。

当前生产 Dockerfile 使用 runtime 依赖，构建脚本只把已验证的 fp16 资产复制进镜像，并且不会把
TensorRT 导出依赖和导出脚本放入运行层。因此，直接在当前容器启动时运行导出命令会失败或产生
不完整资产，需要另外制作 builder 镜像或修改镜像分层。

即使补齐依赖，也不建议每次服务重启都自动转换，原因是：

1. 首次启动会变成长时间的构建阶段，构建期间服务无法就绪；模型越大，TensorRT tactic 搜索越耗时。
2. 容器重建或缓存丢失会重复构建，导致发布时间和启动时间不可预测。
3. 构建时可能遇到显存不足、磁盘不足、TensorRT 算子不支持或驱动异常；若与服务进程同时启动，还会发生资源争用。
4. 当前服务按 `tensorrt/<precision>/engine.plan` 读取资产，尚未实现按 GPU UUID 自动选择多套 engine；不同 GPU 的资产不能安全地互相覆盖。
5. 自动构建后的 engine 仍需完成数值、角色聚类、情绪标签和并发压测，不能只以“构建成功”作为上线条件。

更稳妥的部署方式是“目标 GPU 上一次性构建，运行镜像只加载已验收资产”：

1. 在目标 4070 Ti Super 主机上启动一次带 GPU 的 builder 容器，安装 `requirements-tensorrt-export.txt`，执行模型 README 中的构建命令。
2. 将生成的 `engine.plan`、`manifest.json` 和 `tensorrt/sources/` 持久化到模型目录或制品仓库。
3. 运行 `asr-offline` 前执行 `scripts/check_tensorrt_assets.py` 和目标机实测验证。
4. 验收通过后再构建或启动生产 runtime 镜像；后续普通重启不再重新构建。

如果未来需要支持多种 NVIDIA GPU，可以按 GPU 架构、TensorRT/CUDA 版本维护不同 engine 制品，
在部署阶段选择对应目录。将自动构建限定为“首次安装或显式升级动作”，并采用临时文件写入后原子替换，
稳定性会明显高于每次入口启动都转换。
