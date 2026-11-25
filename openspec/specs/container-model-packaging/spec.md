# container-model-packaging Specification

## Purpose

定义离线 ASR 的昇腾模型与运行环境交付方式：VAD、CAM++ 和 emotion2vec 三个小模型随镜像内置，Qwen 原始 HF 权重通过独立只读挂载提供，运行参数可通过配置文件覆盖。交付材料须明确目标依赖、解释器和模型资产边界，并区分文件准备与实际镜像构建、启动和上线验收。

## Requirements

### Requirement: 随镜像交付完整模型集合
镜像 MUST 内置 `/v3/seacraft_asr` 所需的 VAD、CAM++ 和 emotion2vec 三个模型目录。MUST NOT 包含已下线的 SeACo-Paraformer、PUNC 或 Whisper 权重。Qwen 原始 HF 权重 MUST 通过独立只读挂载提供。非空增强设备数组 MUST 按现有启动契约预加载，空数组不加载；资产存在、加载成功、真实前向与端到端验收分别记录。

#### Scenario: 不挂载模型启动容器
- **WHEN** 运维使用镜像且未挂载任何模型目录
- **THEN** 三个小模型资产可从镜像找到；默认预加载的 Qwen 因未提供独立权重不能就绪，已启动资源统一收尾，不要求在初始化提前失败时三个小模型都已实际加载，不把资产存在宣称为完整转写成功

#### Scenario: 模型目录缺失
- **WHEN** 构建上下文缺少任一必需小模型目录
- **THEN** 构建或启动检查明确失败并指出缺失目录

#### Scenario: 已下线模型目录缺失
- **WHEN** 模型目录不含 SeACo-Paraformer、PUNC 或 faster-whisper-large-v3
- **THEN** 不因此阻止构建或启动，这些目录也不得进入本轮构建上下文

### Requirement: 使用固定模型目录结构
`model/` 下 MUST 使用以下目录名：`speech_fsmn_vad_zh-cn-16k-common-pytorch`、`speech_campplus_sv_zh_en_16k-common_advanced` 和 `emotion2vec_plus_large`。

#### Scenario: 使用约定目录构建
- **WHEN** 三个模型按约定目录名放入仓库根目录的 `model/`
- **THEN** 镜像构建无需额外重命名即可将模型复制到 `/app/model/`

#### Scenario: 提供法语模型目录
- **WHEN** `model/` 下仍存在历史遗留的 `faster-whisper-large-v3` 目录
- **THEN** 镜像构建忽略该目录，不将其复制到 `/app/model/`，法语转写能力已随 `/v1.1.8/seacraft_asr` 下线

### Requirement: 容器默认使用镜像内模型路径
容器专用配置 MUST 由宿主 `docker/config.toml` 只读挂载为 `/config.toml`。三个内置小模型路径 MUST 分别为 `/app/model/speech_fsmn_vad_zh-cn-16k-common-pytorch`、`/app/model/speech_campplus_sv_zh_en_16k-common_advanced`、`/app/model/emotion2vec_plus_large`。Qwen MUST 由 `QWEN_MODEL_DIR` 独立只读挂载至 `/app/qwen-model`，配置对应此路径；配置文件 MUST NOT 烘焙进镜像。

#### Scenario: 检查默认容器配置
- **WHEN** 运维查看 Docker 配置和实际挂载
- **THEN** 三个小模型使用上述镜像内路径，Qwen 使用独立只读挂载，配置来自宿主可读文件而非镜像副本，无 `/var/model_zoo` 或开发机路径依赖

### Requirement: 允许挂载配置但不要求挂载模型
容器 MUST 通过 `CONFIG_PATH` 读取只读挂载配置；三个小模型默认读取镜像副本，不要求重新挂载。Qwen 原始 HF 权重仍 MUST 独立提供，不能由配置挂载替代。配置修改 MUST 在服务重启后生效，不承诺热加载。

#### Scenario: 挂载调整并发参数的配置
- **WHEN** 运维修改宿主配置并重启容器
- **THEN** 服务按新配置运行，三个小模型仍读取镜像副本，Qwen 从独立挂载读取，不要求重建镜像

### Requirement: 模型文件不进入版本控制
模型权重及其原始配置 MUST NOT 被 Git 跟踪；仓库 MUST 保留模型放置与来源说明。三个小模型的构建副本与独立提供的 Qwen HF checkpoint MUST 分别按既定路径准备，不以不入 Git 为理由省略资产校验或误删原始资产。

#### Scenario: 放入模型后检查 Git 状态
- **WHEN** 开发者将三个小模型放入约定 `model/` 目录，并通过独立路径提供 Qwen 权重
- **THEN** `git status` 不把这些权重/原始配置列为待提交，`model/README.md` 等项目说明仍可跟踪；不要求历史五个模型目录存在

### Requirement: 模型目录进入 Docker 构建上下文
构建上下文 MUST 仅包含三个必需小模型的必要权重、配置及其运行所需资产，即使这些文件被 Git 忽略。MUST NOT 将其它模型、模型目录中的项目报告、实验音频或无关文件整体带入任一构建阶段。

#### Scenario: 构建镜像
- **WHEN** 已准备好三个小模型并构建镜像
- **THEN** 必需资产进入上下文及运行镜像，无关模型和实验资料在发送上下文前排除；保存模型资产清单供核验

### Requirement: Docker 构建不依赖仓库外 PyArrow wheel
构建 MUST 使用匹配本轮 CANN 8.2 RC1、Linux aarch64、Python3.10 的容器专用依赖锁及可重定位运行环境，不依赖旧 `docker/build.sh`或仓库外 PyArrow wheel。运行依赖与构建工具来源 MUST 分开记录；MUST NOT 从未知系统 site-packages 补齐包或修改宿主 asr 环境。文件准备与实际镜像、推理验收分别记录。

#### Scenario: 干净构建上下文构建镜像
- **WHEN** 无 wheel 目录，但具备已核验环境归档或可复现依赖来源及三个必需小模型
- **THEN** 独立构建环境完成重定位、版本/ABI 和运行导入检查，使用固定版本 Cython 完成保护编译；缺少依赖或资产明确失败，不要求外置 PyArrow wheel

### Requirement: 默认容器使用容器专用配置
默认 Compose MUST 使用宿主 `docker/config.toml` 的只读挂载和显式 `CONFIG_PATH`，MUST NOT 使用宿主开发根配置或镜像默认模板兜底。仅暴露物理卡7时配置 MUST 按验证映射使用逻辑 `npu:0`，保留 HTTP1、Qwen4/S16、CAM4/emo1 及现有有效推理参数。

#### Scenario: 启动默认镜像
- **WHEN** 通过默认 Compose 启动并提供必需挂载
- **THEN** 服务读取容器专用配置、逻辑 npu:0 和约定模型路径；配置文件缺失或不可读则在创建 worker 前明确失败，不自动生成目录或回退根配置

### Requirement: 镜像内置转写与规范化运行时
镜像 MUST 包含文本规范化所需的规则包运行时，使开启该能力时无需额外安装依赖；文本规范化依赖缺失时 MUST 按透传策略降级而不影响请求成功。镜像 MUST NOT 因内置小语种转写运行时而增加体积。

#### Scenario: 容器内文本规范化可用
- **WHEN** 容器内文本规范化依赖可用
- **THEN** 转写请求的文本按配置执行规范化

#### Scenario: 容器内启用法语能力
- **WHEN** 部署配置中残留多语言或法语相关开关
- **THEN** 服务忽略该开关，不加载 Whisper 模型，法语能力已随 `/v1.1.8/seacraft_asr` 下线

#### Scenario: 容器内文本规范化不可用
- **WHEN** 容器内文本规范化依赖不可用
- **THEN** 转写请求仍返回成功结果，文本保持 ASR 原始输出

### Requirement: 目标环境兼容信息必须可追溯
模型交付 MUST 说明每套模型已验证的硬件架构、运行时环境及输入范围；加载时 MUST 针对实际实例设备校验兼容性，不能仅凭设备编号推断架构，未验证环境 MUST NOT 被宣称可直接复用。

#### Scenario: 使用当前已验证产物
- **WHEN** 交付本轮昇腾 910B 候选
- **THEN** 说明明确其验证环境、CAM++ fbank profile 和 emotion2vec 1～10 秒 profile

#### Scenario: engine 与实际设备不兼容
- **WHEN** 目标实例的架构或运行时不满足已验证条件
- **THEN** 记录兼容检查失败，不加载该模型；通过修正配置或更换匹配的权重来恢复

### Requirement: 运行镜像只保留 Cython 产物
当前 Docker 构建 MUST 编译 core/、api/、utils/、entity/ 业务模块并删除对应业务.py，保留__init__.py和可读main.py，运行层使用匹配ABI的扩展；必要启动与配置材料按该模式交付。本次构建 MUST 只交付与运行解释器 ABI 一致的当前扩展；编译器安装、Cython 和中间 C 只留在独立阶段。

#### Scenario: 检查运行镜像源码
- **WHEN** 显式选择保护交付模式并构建运行镜像
- **THEN** 四个业务目录包含可导入Cython扩展，不含对应业务.py，保留__init__.py和main.py及必要轻量运行入口

#### Scenario: 构建阶段失败
- **WHEN** 保护交付模式中任一指定模块无法编译
- **THEN** 该保护构建明确失败，不生成缺少业务模块的镜像

### Requirement: 配置通过挂载注入
运行容器 MUST 从宿主只读挂载配置读取运行参数，默认路径 `/config.toml`，由 `CONFIG_PATH` 明示；MUST NOT 使用镜像内置配置兜底。三个小模型默认保持镜像内 `/app/model/`，Qwen 独立只读挂载；自定义 Qwen 路径必须与挂载一致。配置改变需重启，不要求连带重挂小模型或重建镜像。

#### Scenario: 挂载运行配置
- **WHEN** 宿主配置文件正确挂载且 CONFIG_PATH 指向该文件
- **THEN** 服务使用挂载参数及约定模型路径，实际配置来源可核对

#### Scenario: 未提供挂载配置
- **WHEN** 未提供配置文件或该路径是目录、不可读文件
- **THEN** 容器在模型和 worker 创建前明确失败，不回退到镜像模板或开发配置

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

### Requirement: 保护运行层保留必要入口和本版扩展
保护运行镜像 MUST 仅保留必要启动和包入口 Python 文件、编译扩展及运行资产。构建 MUST NOT 复制旧业务扩展或将保护编译工具带入运行层；核心模块导入路径 MUST 可验证为本版 ABI 匹配的 `.so`。第三方运行依赖自带的必要原生扩展不属于此禁止范围。

#### Scenario: 核心模块由扩展运行
- **WHEN** 在候选镜像中检查核心模块导入来源并执行真实请求
- **THEN** 使用本版编译扩展，无同名明文源码或旧扩展遮蔽；项目 tests、docs、scripts 等不作为运行依赖复制进镜像
