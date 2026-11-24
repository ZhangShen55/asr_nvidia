# Proposal

## Why

当前 Docker 交付仍混有旧部署脚本、Nginx 和强制 Cython 保护构建，与现有单 HTTP、四 Qwen 模型进程的运行方式及本次源码镜像要求不一致。需要基于服务器已有 CANN 8.2 RC1 镜像交付一个经过真实 B4 完整音频验证的可用镜像，同时限制镜像内容和本轮构建资源的生命周期。

## What Changes

- 使用本机 `cann:8.2.rc1-910b-ubuntu22.04-py3.10` 作为候选底座，锁定实际 image ID；配套容器专用 `torch 2.5.1 / torch-npu 2.5.1.post1`，不修改宿主 asr 环境或原 CANN 8.1 依赖锁。先验证驱动、底座、依赖及真实 NPU 前向，再执行完整服务验收。
- **BREAKING** 本版直接运行必要 Python 源码，不执行 Cython、不带入旧业务扩展或保护构建工具。运行入口为一个 HTTP 服务，内部仍按配置创建 Qwen4/CAM4/emo1；删除 Nginx 与旧启动/构建脚本，容器拒绝 `instance_count != 1`，模型数量使用 `qwen_asr.local_instance_count`。
- **BREAKING** `docker/` 最终恰好四个文件：`Dockerfile`、`docker-compose.yml`、`config.toml`、`README.md`。历史验证报告迁到 `docs/` 保留；忽略规则、依赖锁和验收报告在目录外，构建临时材料不得留在该目录。
- Compose 必须将宿主 `config.toml` 只读挂载到容器 `/config.toml`，设置 `CONFIG_PATH`；不烘焙配置，修改后重启生效。三个小模型随镜像交付，Qwen HF 权重独立只读挂载。
- 采用显式运行文件白名单及最小构建上下文。测试、文档、项目脚本、样本、Git/OpenSpec 数据和旧编译产物不得进入任何构建阶段或最终应用目录；运行依赖和必要模型资产单独审核。
- **BREAKING** 镜像命名为 `asr_offline_910b:v3_YYYYMMDD`，首版示例 `asr_offline_910b:v3_20261125`；日期为实际发布日期，记录源码提交、配置和实际镜像 ID，避免验收后重建。
- 构建失败保留缓存和原始日志，定位后复用缓存重试。成功后使用确切候选镜像进行四条完整音频同时请求的 B4 验收，报告完成率、严格无警告率、P50/P95、墙钟、HBM、段级异常及增强真实调用。
- 仅在验收和正常关闭通过后，将已测 image ID 发布为最终日期标签，再清理本轮拥有的中间镜像、构建缓存和临时构建资源；保留最终/基础镜像、其它原有镜像、共享容器、日志与报告，禁止全局 prune。

## Capabilities

### New Capabilities

无；沿用现有模型交付、运维和 NPU 运行时能力边界。

### Modified Capabilities

- `container-model-packaging`：源码运行、配置必须挂载、必要文件白名单、四文件目录与 CANN 8.2 交付边界；同步增强池启动预加载语义。
- `service-operations`：单 HTTP 无代理部署入口、日期镜像名、真实 B4 验收及构建缓存/镜像的归属清理流程。
- `npu-910b-runtime`：增加容器专用 CANN 8.2 配套依赖路径，保持宿主 CANN 8.1 环境和物理7/逻辑0的隔离。
- `local-model-runtime`：同步本版容器只支持一个完整 HTTP 服务的限制，保留模型实例数与资源放大的独立语义。

## Impact

主要修改 Docker 交付文件、根构建忽略规则、容器专用依赖交付材料、部署说明与对应验收工具/报告。仅在必要时收窄调整应用的容器入口校验及版本诊断，不改变 VAD、调度、复读恢复、精度、接口字段、错误码或既定模型数量/阈值。用户根 `config.toml` 的未提交 VAD 修改、原始权重与所有共享服务保持；本 proposal 阶段不构建、不启动、不清理 Docker 资源，不宣称生产上线或人工准确率验收完成。
