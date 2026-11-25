# Spec Delta

## MODIFIED Requirements

### Requirement: Docker 构建不依赖仓库外 PyArrow wheel
构建 MUST 使用匹配本轮 CANN 8.2 RC1、Linux aarch64、Python3.10 的容器专用依赖锁及可重定位运行环境，不依赖旧 `docker/build.sh`或仓库外 PyArrow wheel。运行依赖与构建工具来源 MUST 分开记录；MUST NOT 从未知系统 site-packages 补齐包或修改宿主 asr 环境。文件准备与实际镜像、推理验收分别记录。

#### Scenario: 干净构建上下文构建镜像
- **WHEN** 无 wheel 目录，但具备已核验环境归档或可复现依赖来源及三个必需小模型
- **THEN** 独立构建环境完成重定位、版本/ABI 和运行导入检查，使用固定版本 Cython 完成保护编译；缺少依赖或资产明确失败，不要求外置 PyArrow wheel

### Requirement: 运行镜像只保留 Cython 产物
当前 Docker 构建 MUST 编译 core/、api/、utils/、entity/ 业务模块并删除对应业务.py，保留__init__.py和可读main.py，运行层使用匹配ABI的扩展；必要启动与配置材料按该模式交付。本次构建 MUST 只交付与运行解释器 ABI 一致的当前扩展；编译器安装、Cython 和中间 C 只留在独立阶段。

#### Scenario: 检查运行镜像源码
- **WHEN** 显式选择保护交付模式并构建运行镜像
- **THEN** 四个业务目录包含可导入Cython扩展，不含对应业务.py，保留__init__.py和main.py及必要轻量运行入口

#### Scenario: 构建阶段失败
- **WHEN** 保护交付模式中任一指定模块无法编译
- **THEN** 该保护构建明确失败，不生成缺少业务模块的镜像

### Requirement: Docker 交付文件集中且构建上下文最小
`docker/` MUST 最终恰好包含 `Dockerfile`、`docker-compose.yml`、`config.toml`、`README.md` 四个文件。忽略规则、锁文件及报告 MUST 位于目录外。构建 MUST 使用显式运行文件白名单，禁止测试、文档、无关项目脚本、样本、Git/OpenSpec 数据、缓存及旧业务扩展进入构建上下文或镜像。现有保护脚本及其生成物仅允许进入编译阶段；不得靠最后一层删除掩盖已带入的资料。

#### Scenario: 查看部署入口
- **WHEN** 运维查看完成后的 docker 目录
- **THEN** 仅见上述四个文件，README 给出完整构建与运行命令；旧脚本和 Nginx 配置移除，历史报告在目录外保留

#### Scenario: 检查运行镜像内容
- **WHEN** 审核构建上下文、各阶段及最终应用目录
- **THEN** 只包含必要启动源码、当前编译扩展、运行依赖、三个模型必要资产和追溯元数据；测试/文档/无关项目脚本等禁入内容从未被 COPY/ADD 到任何阶段

### Requirement: 提供昇腾 NPU 部署镜像
本版 MUST 基于已核验的本机 CANN 8.2 RC1 底座交付保护镜像，内置三个小模型，单 HTTP 直接启动、不使用 Nginx 或项目启动脚本。Compose MUST 声明昇腾 runtime、特权模式、host IPC、可见物理卡及必要挂载。默认 Qwen 和非空增强数组 MUST 按现有契约在 HTTP 就绪前预加载；完整构建、真实前向、B4 和正常关闭均通过后方可宣称本版镜像可用。

#### Scenario: 使用镜像内模型启动
- **WHEN** 通过昇腾 runtime 提供只读配置及 Qwen 权重挂载
- **THEN** 单 HTTP 从镜像读取三个小模型，VAD 预热、Qwen 预加载及增强预加载按既有契约完成，再报告健康；无外部代理或 Qwen 服务依赖

#### Scenario: 缺少必需模型
- **WHEN** 构建上下文缺少任一必需小模型
- **THEN** 构建或启动检查明确失败并指出目录，不以能力关闭掩盖缺失资产

#### Scenario: 设备绑定方式随镜像交付
- **WHEN** 运维查看 Compose 和 README
- **THEN** 可获得 runtime、特权、IPC、可见卡及挂载配置，不需要从旧 start.sh 推断宿主参数

#### Scenario: 物理卡与容器逻辑编号分离
- **WHEN** 宿主开发为物理 npu:7，而容器仅暴露物理卡7
- **THEN** 独立验证映射并使用容器逻辑 npu:0，不修改宿主开发设备配置

#### Scenario: 资产检查和部署验收分级
- **WHEN** 仅确认资产、依赖导入或 HTTP READY
- **THEN** 分别记录已完成等级，不将其当作模型真实 forward、四并发完整音频或生产上线验收

## ADDED Requirements

### Requirement: 保护运行层保留必要入口和本版扩展
保护运行镜像 MUST 仅保留必要启动和包入口 Python 文件、编译扩展及运行资产。构建 MUST NOT 复制旧业务扩展或将保护编译工具带入运行层；核心模块导入路径 MUST 可验证为本版 ABI 匹配的 `.so`。第三方运行依赖自带的必要原生扩展不属于此禁止范围。

#### Scenario: 核心模块由扩展运行
- **WHEN** 在候选镜像中检查核心模块导入来源并执行真实请求
- **THEN** 使用本版编译扩展，无同名明文源码或旧扩展遮蔽；项目 tests、docs、scripts 等不作为运行依赖复制进镜像

## REMOVED Requirements

### Requirement: 本版镜像采用必要源码交付
**Reason**: 当前构建改为独立编译阶段交付核心扩展。
**Migration**: 源码版继续使用原独立标签。
