# Proposal

## Why

上一版镜像交付了可直接阅读的业务 Python 源码。新构建需要使用 Cython 扩展交付核心业务实现，降低镜像内业务源码的可读性。

## What Changes

- 复用仓库现有 Cython 3.0.12 保护脚本，隔离编译 core、api、utils、entity。
- 运行镜像仅从编译阶段复制通过 ABI 和文件清单校验的扩展、包入口及必要启动文件。
- 基于同一 CANN8.2 RC1 底座与运行依赖新建保护镜像，使用独立标签。

## Capabilities

### New Capabilities

### Modified Capabilities

- `container-model-packaging`：当前 Docker 构建默认交付 Cython 保护产物。

- `service-operations`：保护候选沿用完整 B4 验收口径。

## Impact

涉及 Dockerfile、构建上下文白名单和交付验证。HTTP 路由、配置格式、模型调度与挂载契约保持原义。既有源码镜像标签继续可用。
