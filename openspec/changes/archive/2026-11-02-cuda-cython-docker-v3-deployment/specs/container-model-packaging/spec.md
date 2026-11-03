# Spec Delta

## ADDED Requirements

### Requirement: 提供 CUDA 12.4 TensorRT 部署镜像
CUDA 部署 profile MUST 使用 docker.1ms.run/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 作为运行时基础，并将 VAD、CAM++、emotion2vec 模型及其已验证的 TensorRT 资产内置到镜像；未挂载模型目录时服务仍 MUST 能按默认路径加载模型。

#### Scenario: 使用镜像内模型启动
- **WHEN** 运维使用 NVIDIA 容器运行时启动镜像且只挂载配置文件
- **THEN** 服务从镜像内默认模型路径加载三个必需模型并进入就绪状态

#### Scenario: 缺少必需模型
- **WHEN** 构建上下文缺少 VAD、CAM++ 或 emotion2vec 任一模型目录
- **THEN** 构建或启动检查明确失败并指出缺失目录

### Requirement: CUDA 部署强制交付 fp16 engine
CUDA 部署 profile MUST 使用 CAM++ 与 emotion2vec 对应模型目录中的 fp16 engine.plan 和 manifest.json；资产缺失、SHA256 不匹配或与目标 GPU/CUDA/TensorRT 不兼容时，构建或启动 MUST 失败并给出可定位原因，MUST NOT 静默回退 PyTorch。

#### Scenario: fp16 资产完整且兼容
- **WHEN** 两个模型均存在完整 fp16 engine、manifest 和匹配的兼容信息
- **THEN** 镜像检查通过，服务按 TensorRT fp16 后端启动

#### Scenario: fp16 资产缺失或不兼容
- **WHEN** 任一模型缺少 fp16 engine/manifest 或校验、兼容检查失败
- **THEN** 部署报告对应模型和资产路径的错误并阻止就绪

### Requirement: 运行镜像只保留 Cython 产物
部署构建 MUST 编译 core/、api/、utils/、entity/ 中的业务 Python 模块并在运行镜像删除对应 .py 源文件，保留各目录的 __init__.py；main.py、启动脚本、配置模板和 Docker 部署文件 MUST 保持可读。

#### Scenario: 检查运行镜像源码
- **WHEN** 构建完成并检查运行镜像文件
- **THEN** 指定四个目录包含可导入的 Cython 扩展且不包含业务 .py 源文件，main.py 仍存在

#### Scenario: 构建阶段失败
- **WHEN** 任一指定模块无法编译为扩展
- **THEN** 构建失败，不生成缺少业务模块的运行镜像

### Requirement: 配置通过挂载注入
运行镜像 MUST 支持通过 CONFIG_PATH 指向挂载配置文件；未设置时使用镜像内默认模板，模型路径默认保持指向 /app/model/，配置挂载不得要求重新挂载模型。

#### Scenario: 挂载运行配置
- **WHEN** 运维把宿主机配置挂载到容器并设置 CONFIG_PATH
- **THEN** 服务读取挂载配置中的设备、并发和上游参数，同时继续使用镜像内模型

#### Scenario: 未提供挂载配置
- **WHEN** 容器未设置 CONFIG_PATH 且镜像含默认模板
- **THEN** 服务使用默认模板启动并保持 /app/model/ 路径

### Requirement: Docker 交付文件集中且构建上下文最小
Dockerfile、compose 文件、忽略规则和部署说明 MUST 集中在 docker/；运行镜像 MUST 排除测试样本、缓存、源码构建中间物和无关模型，只复制启动所需的应用、依赖、模型与元数据。

#### Scenario: 查看部署入口
- **WHEN** 运维进入 docker/ 目录
- **THEN** 能找到 README、Dockerfile、compose 配置及构建/启动所需说明

#### Scenario: 检查运行镜像内容
- **WHEN** 构建镜像并列出运行时文件
- **THEN** 不包含测试音频、测试代码、Cython .c 文件、Python 缓存或未被服务引用的模型目录
