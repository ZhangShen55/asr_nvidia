# Design

## Context

当前仓库已有模型目录、CAM++/emotion2vec TensorRT runner、TensorRT 资产校验脚本和 Cython 构建脚本，但根目录 Dockerfile 仍使用旧 CUDA/CentOS 运行时，启动脚本依赖 Conda，业务路由、脚本和测试以 v2 命名。已有 fp16 engine 位于两个增强模型目录下，运行时还必须读取宿主机配置并保留 Qwen、VAD 和状态接口的现有行为。

本设计只描述本次 CUDA 部署 profile 的实现边界；通用 CPU/NPU 或源码开发环境不随该 profile 改成强制 TensorRT。

## Goals / Non-Goals

**Goals:**

- 产出基于 CUDA 12.4.1 cuDNN runtime Ubuntu 22.04 的可复现 NVIDIA 镜像，模型和 TensorRT 资产随镜像交付。
- 在构建或启动阶段验证模型目录、fp16 engine、manifest、SHA256 和运行环境兼容性，失败时阻止服务就绪。
- 通过 Cython 多阶段构建保护指定业务目录，运行层只保留扩展、入口和必需资源。
- 把 Dockerfile、compose、构建脚本和部署说明集中到 docker/，配置以卷挂载注入，镜像 tag 可追溯到 v3 时间标识。
- 将 Qwen 路由和所有面向客户端的示例、脚本、测试迁移到 v3，并删除 v2 注册。

**Non-Goals:**

- 不重新设计 Qwen 请求/响应字段、A/B 音频准入、role/emotion 错误码或 v1.1.8 链路。
- 不把 TensorRT 强制策略扩展到非 CUDA Docker 运行、CPU、NPU 或源码开发启动。
- 不在镜像中打包测试音频、测试工具、模型导出源码、VAD 评估产物或历史无关模型。
- 不在本 change 中构建新的 TensorRT engine；只验收仓库已有且 manifest 完整的 fp16 资产。

## Decisions

### 1. 使用多阶段 Ubuntu CUDA 构建

builder 和 runtime 均以指定的 NVIDIA CUDA 12.4.1 cuDNN runtime Ubuntu 22.04 镜像为基线；builder 额外安装编译器、Python 3.10、Cython 和完整 Python 依赖，runtime 只安装运行库和已锁定的依赖。这样可保证编译环境与运行时 CUDA ABI 一致，同时避免把编译器、头文件和 pip 缓存带入生产层。

考虑过继续使用根目录的 Conda/CentOS Dockerfile，但它与目标基础镜像和 Cython 运行层不一致，且会保留大量无关工具，因此不作为新 profile 的入口。

### 2. 使用显式 staging 上下文保持 docker/ 集中

实现阶段在 docker/ 提供构建脚本，按白名单把 main.py、启动所需包、配置模板、nginx 配置和三个模型目录复制到临时 staging 目录，再以 staging 作为 docker build 上下文；Dockerfile 和 compose 仍位于 docker/。staging 同时排除测试、样本、缓存、TensorRT sources 和未引用模型，避免依赖仓库根目录的隐式上下文。

考虑过直接以仓库根目录作为 compose build context，但这会让根目录 .dockerignore 和旧 Dockerfile 继续成为部署契约，也容易把无关文件送入构建上下文；白名单 staging 的文件集合更容易审计。

### 3. 在 builder 中编译并在 runtime 中复制 Cython 产物

复用 setup_cython.py 的模块发现逻辑，并补齐 api 目录下所有业务模块的覆盖检查。编译产物、各目录的 __init__.py、main.py、启动脚本和必要静态资源复制到 runtime；业务 .py、生成的 .c、build 目录和缓存不复制。构建结束执行导入检查，确保每个公开模块可以在无源码的 runtime 中导入。

main.py 保持明文作为进程入口，配置、nginx 和部署文档保持明文，方便运维定位启动问题。

### 4. 将 fp16 TensorRT 校验设为 profile 闸门

构建阶段先检查两个模型的固定目录和 fp16 engine/manifest 文件，运行阶段再按实际 GPU、CUDA 和 TensorRT 版本执行兼容校验。配置模板在 Docker profile 中将两个 TensorRT 开关设为 true、精度固定为 fp16；模型实例加载失败直接使健康检查失败。不得捕获该错误后切换到 PyTorch，通用后端的非 Docker 回滚逻辑仍由既有配置控制。

由于构建阶段不一定拥有 GPU，构建时执行结构、SHA256 和 manifest 一致性校验，启动时执行设备与运行时兼容性校验；两阶段的结果都写入日志，避免把无 GPU 的构建检查误报为实机验收。

### 5. 以 v3 作为唯一 Qwen 路由

把路由模块、FastAPI 注册、内部调用、脚本、文档和测试中的 /v2/seacraft_asr 统一替换为 /v3/seacraft_asr，保留处理函数的请求参数、响应结构、A/B 准入和错误语义。删除 v2 路由注册并增加路由回归检查，防止兼容别名被意外保留。

### 6. 使用 CONFIG_PATH 和 compose GPU 声明

runtime 默认提供镜像内模板作为兜底，但 compose 默认把宿主机配置只读挂载到固定容器路径并设置 CONFIG_PATH；模型目录不作为卷挂载。compose 使用 NVIDIA GPU 设备声明、端口映射、健康检查和可覆盖的 ASR_IMAGE 变量。构建脚本生成 asr_offline:v3_YYYYMMDDHHMMSS 形式的默认 tag，允许 CI 或运维显式传入其他同格式 tag。

### 7. 以 venv 替代 Conda 启动依赖

新启动脚本使用 runtime 内固定 venv 的 uvicorn、nginx 和 Python 路径，继续读取 instance_count 并生成 upstream；不再 source /opt/conda。这样启动行为与 Ubuntu 基础镜像一致，且 venv 的包集合可以在 builder 阶段冻结并复制到 runtime。

## Risks / Trade-offs

- [基础镜像或镜像仓库不可达] → 构建前检查基础镜像可拉取；README 记录镜像 digest/版本和离线预拉取方式，失败时不使用旧基础镜像静默替换。
- [fp16 engine 与实际 GPU 架构不匹配] → 启动时强制读取 manifest 的 compute capability、CUDA 和 TensorRT 约束，失败时服务不就绪，并在文档中列出已验证硬件范围。
- [Cython 扩展缺少运行时依赖或模块遗漏] → builder 运行全量导入/启动 smoke test，再复制到 runtime；任何导入失败都使镜像构建失败。
- [Ubuntu Python/系统库与现有依赖不兼容] → 在 builder 中固定 Python 和依赖版本，执行 v3 路由、模型加载和状态接口测试后才生成 tag；保留上一镜像用于回滚。
- [v2 客户端无法直接升级] → 将破坏性迁移写入 README、健康检查和发布说明；部署前用 v3 探活，回滚时恢复上一完整镜像，而不是在新镜像中保留隐式 v2 别名。
- [staging 白名单漏复制运行资源] → 构建脚本维护显式清单并在 runtime 中执行文件存在性、模型资产和导入检查；新增运行依赖必须同时更新清单和测试。
- [配置挂载覆盖了镜像内必要路径] → compose 只挂载单一配置文件为只读，启动时打印实际 CONFIG_PATH 和模型路径，配置中的模型路径校验必须拒绝宿主机不存在的路径。

## Migration Plan

1. 在具备 NVIDIA 驱动的构建机上预拉取目标基础镜像，确认 CUDA、GPU 架构和现有 fp16 manifest 的兼容范围。
2. 运行 docker 构建脚本生成 staging、执行 Cython 编译和资产校验，按本地时间生成 asr_offline:v3_YYYYMMDDHHMMSS tag。
3. 使用 compose 挂载生产配置启动单容器，先检查 /get_status、TensorRT 实例状态和 v3 探活，再执行普通 A 请求与全增强 B 请求。
4. 对八路完整音频执行既有 TensorRT/并发验收，保存镜像 tag、配置、GPU、资产摘要和日志；所有结果通过后再推广到生产。
5. 回滚时停止新 compose 服务并启动上一 v3 镜像和上一配置；由于 v2 不保留在新镜像中，接口兼容回滚依赖上一完整镜像。
