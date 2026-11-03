# Proposal

## Why

当前仓库的容器入口仍基于旧 CUDA/CentOS 运行时，Docker 部署文件分散在仓库根目录，接口和配置文档继续以 v2 为主，无法直接形成可复制的 NVIDIA CUDA 生产镜像。现有 TensorRT engine 与 Cython 构建基础已经具备，但缺少一个将模型、强制 fp16 TensorRT、代码保护、配置挂载和 v3 接口统一起来的最小部署方案，因此需要在本次 CUDA 部署中一次性固化这些运行契约。

## What Changes

- 新增面向 NVIDIA GPU 的 Docker 部署 profile，基础镜像固定为 `docker.1ms.run/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`，镜像内置 VAD、CAM++、emotion2vec 模型及其 TensorRT 资产。
- CUDA 部署对 CAM++ 和 emotion2vec 强制使用已有的 fp16 TensorRT engine；engine、manifest、GPU/CUDA/TensorRT 兼容性或校验失败时，构建或启动明确失败，不静默回退 PyTorch。
- 通过多阶段构建安装依赖并使用 Cython 编译 `core/`、`api/`、`utils/`、`entity/`；运行镜像删除这些目录中的业务 `.py` 源文件，保留 `__init__.py`，`main.py`、启动脚本、配置模板和部署文件保持明文。
- **BREAKING** 将 Qwen ASR 对外接口从 `POST /v2/seacraft_asr` 替换为 `POST /v3/seacraft_asr`，不保留 v2 路由；同步更新响应、错误、示例、脚本和测试中的接口引用，v1.1.8 行为保持既有约定。
- 配置文件不打入运行镜像作为唯一运行配置；容器通过 `CONFIG_PATH` 或约定挂载路径读取宿主机配置，模型路径默认指向镜像内 `/app/model/`。
- 将 Dockerfile、构建忽略规则、compose 编排和部署说明集中到 `docker/`，提供 `docker/README.md` 与 `docker/docker-compose.yaml`；compose 声明 NVIDIA GPU、配置挂载、端口和镜像内模型使用方式。
- 镜像标签采用 `asr_offline:v3_<时间>` 格式，并在文档中说明构建、启动、健康检查、配置迁移和 TensorRT 资产前置条件。
- 运行镜像只复制启动所需代码、依赖、模型和部署元数据，排除测试、样本音频、源码构建产物、缓存及无关模型，构建阶段保留可追溯的版本与资产校验信息。

## Capabilities

### New Capabilities

无。现有能力边界已覆盖容器模型交付、后端选择、Qwen 适配和服务运行期部署契约。

### Modified Capabilities

- `container-model-packaging`: 增加 CUDA 12.4 Ubuntu 运行时、镜像内置模型与 TensorRT fp16 资产、Cython 精简运行时、配置挂载、最小构建上下文和 `docker/` 部署文件契约。
- `local-model-inference-backends`: 增加 CUDA Docker profile 对 CAM++ 与 emotion2vec 强制使用 fp16 TensorRT、资产不完整或不兼容时显式失败且禁止回退的要求。
- `qwen-asr-v2-adapter`: 将公开 Qwen 接口、默认值说明、示例和相关调用从 v2 迁移到 v3，保留请求/响应及增强能力语义。
- `offline-asr-service`: 将最小业务接口清单中的 Qwen 路由更新为 `POST /v3/seacraft_asr`，并明确 v2 路由移除。
- `service-operations`: 增加 v3 镜像标签、配置挂载、compose GPU 运行契约、Docker 文档位置和启动前检查要求。

## Impact

- 影响 `api/routes`、应用入口、配置解析、模型加载及 TensorRT 资产校验；所有 v2 客户端必须迁移到 v3。
- 影响根目录 Docker 构建入口、忽略规则、启动脚本和依赖安装流程；新的部署入口集中在 `docker/`。
- 依赖 CUDA 12.4、兼容的 NVIDIA 驱动、TensorRT 10.7.0.post1 运行时及已验证的 fp16 engine；构建和运行必须在可用 NVIDIA GPU 环境完成验证。
- Cython 运行镜像不再提供指定目录的业务 Python 源码，调试和开发仍需使用源码构建上下文。
- 配置、模型目录、端口、GPU 可见性和镜像 tag 成为部署文档与 compose 的外部运维契约。
