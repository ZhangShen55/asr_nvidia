# Spec Delta

## ADDED Requirements

### Requirement: 提供可复现的 v3 Docker 部署入口
Docker 部署 MUST 在 docker/ 下提供 README.md、Dockerfile 和 docker-compose.yaml；文档 MUST 给出构建、启动、健康检查、停止和配置挂载命令，并说明 v3 路由及模型内置策略。

#### Scenario: 查看部署目录
- **WHEN** 运维只查看 docker/ 目录
- **THEN** 可以获得构建镜像、启动 GPU 容器、挂载配置和验证服务状态所需的完整说明

#### Scenario: compose 启动服务
- **WHEN** 运维使用 docker-compose.yaml 启动服务并提供 NVIDIA 容器运行时
- **THEN** 容器声明 GPU 使用、配置文件挂载、服务端口和 v3 镜像入口，服务可通过健康检查确认就绪

### Requirement: 镜像标签符合 v3 时间格式
部署构建 MUST 生成或接受 asr_offline:v3_<时间> 格式的镜像标签；compose 默认引用该格式的镜像变量，文档 MUST 说明如何覆盖标签。

#### Scenario: 构建并启动时间标签镜像
- **WHEN** 运维使用 asr_offline:v3_ 加时间标识构建镜像并启动 compose
- **THEN** compose 使用同一 tag 启动，状态和日志可关联到该构建

#### Scenario: 标签格式错误
- **WHEN** 镜像 tag 不符合 asr_offline:v3_<时间> 且未显式覆盖部署变量
- **THEN** 构建或启动说明明确提示格式错误，不把旧 v2 tag 当作默认镜像

### Requirement: 启动前校验部署前置条件
CUDA Docker 部署 MUST 在服务就绪前校验 NVIDIA GPU 可见性、模型目录、fp16 TensorRT engine/manifest、兼容性和 CONFIG_PATH 可读性；任一必需条件失败 MUST 明确报告并阻止就绪。

#### Scenario: GPU 或资产不可用
- **WHEN** 容器无可用 GPU、fp16 资产缺失/不兼容或配置文件不可读
- **THEN** 启动日志指出具体前置条件并以失败状态结束，不接受请求

#### Scenario: 前置条件通过
- **WHEN** GPU、资产和配置检查均通过
- **THEN** 服务启动 v3 路由并报告镜像 tag、实际配置路径和 TensorRT fp16 后端状态

### Requirement: 配置挂载和镜像模型路径可观测
运维文档和状态输出 MUST 区分挂载配置路径与镜像内模型路径，说明配置挂载只覆盖运行参数；默认模型仍来自 /app/model/，不得要求将模型目录作为宿主机卷挂载。

#### Scenario: 挂载自定义配置
- **WHEN** compose 把宿主机配置挂载到容器并设置 CONFIG_PATH
- **THEN** 日志或状态可确认实际配置路径，模型加载路径仍为镜像内 /app/model/
