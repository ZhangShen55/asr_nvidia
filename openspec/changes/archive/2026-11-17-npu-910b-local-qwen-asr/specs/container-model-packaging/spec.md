# Spec Delta

## MODIFIED Requirements

### Requirement: 随镜像交付完整模型集合
镜像 MUST 内置 `/v3/seacraft_asr` 所需的三个模型目录：VAD、CAM++ 和 emotion2vec。镜像 MUST NOT 要求提供已下线接口专用的 SeACo-Paraformer、PUNC 与 `faster-whisper-large-v3` 目录。转写模型权重 MUST 可通过挂载提供，MUST NOT 强制打进镜像。

#### Scenario: 不挂载模型启动容器
- **WHEN** 运维人员使用默认镜像和默认配置启动容器且未挂载任何模型目录
- **THEN** 容器能够找到三个小模型的原始权重及配置文件，VAD 按启动契约预热；CAM++、emotion2vec 在能力首次请求时加载，Qwen 须使用另行提供的本地权重目录。此场景不表示未提供 Qwen 权重时已通过端到端转写验收

#### Scenario: 模型目录缺失
- **WHEN** 构建上下文中的 `model/` 缺少任一必需模型目录
- **THEN** 构建或启动验证必须明确失败并指出缺失的模型目录

#### Scenario: 已下线模型目录缺失
- **WHEN** 构建上下文中的 `model/` 不含 SeACo-Paraformer、PUNC 或 `faster-whisper-large-v3`
- **THEN** 构建与启动均正常完成

### Requirement: 默认容器使用容器专用配置
镜像 MUST 默认使用容器专用配置生成 `/config.toml`，MUST NOT 将面向宿主机开发的 `config.toml` 作为默认容器配置。默认容器配置的设备标识 MUST 为 NPU 形态。

#### Scenario: 启动默认镜像
- **WHEN** 容器未设置外部配置文件
- **THEN** 服务使用容器专用配置中的 `npu:0` 落卡与 `/app/model/` 模型路径

### Requirement: 目标环境兼容信息必须可追溯
模型交付 MUST 说明每套模型已验证的硬件架构、运行时环境及输入范围；加载时 MUST 针对实际实例设备校验兼容性，不能仅凭设备编号推断架构，未验证环境 MUST NOT 被宣称可直接复用。

#### Scenario: 使用当前已验证产物
- **WHEN** 交付本轮昇腾 910B 候选
- **THEN** 说明明确其验证环境、CAM++ fbank profile 和 emotion2vec 1～10 秒 profile

#### Scenario: engine 与实际设备不兼容
- **WHEN** 目标实例的架构或运行时不满足已验证条件
- **THEN** 记录兼容检查失败，不加载该模型；通过修正配置或更换匹配的权重来恢复

### Requirement: 容器默认使用镜像内模型路径
默认容器配置 MUST 由 `docker/config.toml` 提供，并在镜像内作为 `/config.toml` 使用。三个内置小模型路径 MUST 分别为 `/app/model/speech_fsmn_vad_zh-cn-16k-common-pytorch`、`/app/model/speech_campplus_sv_zh_en_16k-common_advanced`、`/app/model/emotion2vec_plus_large`。Qwen 原始 HF 权重 MUST 与三个小模型区分：由独立只读挂载提供，默认 compose 使用 `QWEN_MODEL_DIR` 挂载至 `/app/qwen-model`，配置的 `local_model_dir` 对应此路径，MUST NOT 要求五个模型都内置到 `/app/model/`。

#### Scenario: 检查默认容器配置
- **WHEN** 运维查看 `docker/config.toml`、镜像内 `/config.toml` 与默认 compose
- **THEN** 三个小模型使用上述 `/app/model/` 路径，Qwen 使用独立 `/app/qwen-model` 原始 HF 目录；compose可挂载 `ASR_CONFIG` 覆盖 `/config.toml`，三个内置模型不因此要求重新挂载，路径中不存在 `/var/model_zoo` 依赖

### Requirement: 模型文件不进入版本控制
模型权重及其原始配置 MUST NOT 被 Git 跟踪；仓库 MUST 保留模型放置与来源说明。三个小模型的构建副本与独立提供的 Qwen HF checkpoint MUST 分别按既定路径准备，不以不入 Git 为理由省略资产校验或误删原始资产。

#### Scenario: 放入模型后检查 Git 状态
- **WHEN** 开发者将三个小模型放入约定 `model/` 目录，并通过独立路径提供 Qwen 权重
- **THEN** `git status` 不把这些权重/原始配置列为待提交，`model/README.md` 等项目说明仍可跟踪；不要求历史五个模型目录存在

### Requirement: Docker 构建不依赖仓库外 PyArrow wheel
构建 MUST 使用匹配 CANN8.1.RC1、Linux aarch64、Python3.10 的目标依赖锁和可重建 asr 环境，通过 `docker/build.sh` 打包已核验前缀或显式提供的 conda-pack 环境归档供 builder 解包。MUST NOT 要求预先放置仓库外 `wheel/pyarrow-*.whl`，也 MUST NOT 将历史 CentOS7、特定 manylinux wheel 或未列入当前锁的 PyArrow20.0.0 安装假设作为本轮 NPU 构建前置条件。运行与构建工具、pip与conda来源 MUST 分开记录；真实镜像构建/上线是否执行仍按部署阶段记录，不由交付文件存在推断。

#### Scenario: 干净构建上下文构建镜像
- **WHEN** 仓库无 `wheel/` 目录，但已按目标锁准备可读的 asr 环境前缀或含 `bin/python`、`bin/conda-unpack` 的环境归档及三个必需模型
- **THEN** 构建入口不查找外置 PyArrow wheel，而在独立staging准备环境、代码和原始模型，由匹配解释器校验依赖/ABI并编译；缺少环境或必需资产时明确失败，不从未知系统site-packages补齐

### Requirement: 允许挂载配置但不要求挂载模型
容器 MUST 继续支持通过 `CONFIG_PATH` 和配置文件挂载修改并发、设备及其它运行参数；VAD、CAM++、emotion2vec 三个小模型默认继续使用镜像内副本，不要求重新挂载。此免挂载约定 MUST NOT 被解释为Qwen权重已内置；Qwen原始HF目录仍通过独立只读挂载提供，并由本地转写的加载/4011契约验证。

#### Scenario: 挂载调整并发参数的配置
- **WHEN** 运维挂载新并发或设备配置，三个小模型仍使用 `/app/model/` 路径，Qwen配置指向独立挂载目录
- **THEN** 服务使用挂载配置并读取镜像内三个小模型；不要求再次挂载这三个目录，也不从配置文件推断Qwen资产已存在或完整转写已通过

### Requirement: 配置通过挂载注入
运行镜像 MUST 支持 `CONFIG_PATH` 指向挂载配置文件，未设置时使用镜像内 `/config.toml` 默认模板。配置挂载只修改运行参数，三个小模型默认保持 `/app/model/`，Qwen默认保持独立 `/app/qwen-model` 只读挂载；自定义Qwen路径 MUST 与实际挂载对应。MUST NOT 要求配置挂载连带重挂三个内置模型，也不能将Qwen权重包含在配置挂载中。

#### Scenario: 挂载运行配置
- **WHEN** 运维把宿主配置挂载至容器并设置 `CONFIG_PATH`
- **THEN** 服务读取该配置的设备、并发与本地模型路径，三个内置小模型仍可读取镜像副本，Qwen从另行提供的HF目录加载

#### Scenario: 未提供挂载配置
- **WHEN** 容器未设置外部配置且镜像含默认模板
- **THEN** 服务使用 `/config.toml`，三个小模型为 `/app/model/`、Qwen为独立 `/app/qwen-model`；此配置存在不代替各资产与能力验证

## REMOVED Requirements

### Requirement: TensorRT engine 使用可校验的模型资产结构
**Reason**: 该需求定义 `tensorrt/<precision>/engine.plan` 与 `manifest.json` 的资产结构，是 NVIDIA 专属加速产物；本分支只负责昇腾 910B，CAM++ 与 emotion2vec 在 NPU 上使用原始 PyTorch 权重经 torch_npu 执行，不产生 engine。

**Migration**: 原生基线直接使用 PyTorch checkpoint，不再要求 NVIDIA engine manifest；仍须记录模型来源、revision、关键文件及完整性证据。目录和关键文件存在仅为资产检查，不能代替加载、前向或业务验收。

### Requirement: engine 和运行依赖只对需要的后端生效
**Reason**: 该需求用于保证只有选择 TensorRT 的 CUDA 实例才要求 engine 与 TensorRT 运行时；移除 TensorRT 后端后不存在按需加载的加速器运行时资产。

**Migration**: 不适用。torch_npu 依赖对 NPU 实例是必需的，对 CPU 实例不需要——该隔离由「后端依赖和设备 API 必须隔离」需求承接。

### Requirement: 提供 CUDA 12.4 TensorRT 部署镜像
**Reason**: NVIDIA CUDA 12.4 运行时不适用于昇腾 910B，本分支只交付昇腾 NPU 镜像。

**Migration**: 不适用。由「提供昇腾 NPU 部署镜像」需求承接同等职责。

### Requirement: CUDA 部署强制交付 fp16 engine
**Reason**: fp16 TensorRT engine 是 NVIDIA 专属产物，在 NPU 上不存在对应资产与兼容性校验链路。

**Migration**: 不适用。NPU 实例的计算精度由 torch_npu 运行时与权重 dtype 决定，状态接口如实报告实际精度，不引入额外的 engine 精度配置。

## ADDED Requirements

### Requirement: 提供昇腾 NPU 部署镜像
NPU 部署镜像的交付文件 MUST 采用匹配已验证依赖的昇腾 CANN 底座，并内置 VAD、CAM++、emotion2vec 三个小模型目录。Compose 或宿主容器启动入口 MUST 配置 `--privileged --runtime=ascend --ipc=host` 对应参数及 `ASCEND_VISIBLE_DEVICES`；MUST NOT 把这些宿主启动参数放进容器内 docker/start.sh 冒充设备绑定。docker/start.sh MUST 负责容器内解释器、适用预检与服务启动。交付文件适配属于本次迁移，实际镜像构建与上线 MUST 在完整服务稳定并完成适用验收之后执行，文件准备完成 MUST NOT 被报告为构建或部署成功。

#### Scenario: 使用镜像内模型启动
- **WHEN** 运维使用昇腾容器运行时启动镜像且只挂载配置文件
- **THEN** 服务从镜像默认路径找到三个小模型资产并预热 VAD 后就绪；增强模型保留按需加载，本地 Qwen 权重及完整转写能力另行验证

#### Scenario: 缺少必需模型
- **WHEN** 构建上下文缺少 VAD、CAM++ 或 emotion2vec 任一模型目录
- **THEN** 构建或启动检查明确失败并指出缺失目录

#### Scenario: 设备绑定方式随镜像交付
- **WHEN** 运维查看镜像部署说明
- **THEN** 可以获得 `--privileged --runtime=ascend --ipc=host` 与 `ASCEND_VISIBLE_DEVICES` 的完整启动参数，无需自行推断设备映射方式

#### Scenario: 物理卡与容器逻辑编号分离
- **WHEN** 本轮宿主开发使用 conda asr 和物理 `npu:7`，后续容器仅暴露物理卡 7
- **THEN** 宿主使用 `/home/zhangs/conda-envs/asr/bin/python`；宿主启动入口设置 `ASCEND_VISIBLE_DEVICES=7`，容器配置按实际可见设备映射使用逻辑编号（仅映射该卡时通常为 `npu:0`），不直接沿用宿主 `npu:7`

#### Scenario: 资产检查和部署验收分级
- **WHEN** 已确认三个模型目录及关键文件存在，但尚未加载权重或执行前向
- **THEN** 仅记录资产存在，MUST NOT 把该检查计为加载、推理、镜像构建或上线验收通过；实际构建与上线保持后置
