# npu-910b-runtime Specification

## Purpose

定义 SeaCraftASR 在华为昇腾 910B 服务器上运行时的可验证契约：依赖栈版本矩阵、Python 虚拟环境边界、容器 NPU 设备绑定方式、容器内设备编号规则、推理 dtype 与算子约束，以及 NPU 资源的观测口径，确保迁移后的服务不会退回到不可用的混合运行时状态。

## Requirements

### Requirement: 固定 NPU 依赖栈版本矩阵
宿主 asr 环境 MUST 保持 CANN8.1.RC1 / torch2.5.1 / torch_npu2.5.1；本版独立容器 MUST 使用指定本机 CANN8.2.RC1 底座与 torch2.5.1 / torch-npu2.5.1.post1。两者均锁定 transformers5.13.0、numpy1.26.4、funasr1.2.6、torchaudio2.5.1、av14.2.0、torchvision0.20.1、attrs26.1.0及ITN依赖，记录来源、Python/aarch64、CANN/驱动及模型/processor revision。MUST NOT 混用不同 torch 大版本，或为本轮容器修改宿主环境/驱动。

#### Scenario: 导入运行时检查版本
- **WHEN** 在原宿主 asr 环境执行检查
- **THEN** torch 与 torch_npu 均为2.5.1、transformers为5.13.0，保留已验宿主组合，不因容器交付覆盖其锁或包

#### Scenario: 版本不匹配时启动失败
- **WHEN** torch_npu 与 torch 主次版本不一致或不满足所选 CANN 配套
- **THEN** 就绪前明确报告具体版本和运行环境，阻止启动，不以部分能力失效或 CPU 回退掩盖不匹配

#### Scenario: CANN8.2源码容器配套
- **WHEN** 使用本机指定 CANN8.2.RC1 底座构建本版容器
- **THEN** 容器独立环境使用 torch2.5.1 / torch-npu2.5.1.post1，保存精确锁、包来源/摘要和依赖一致性结果；真实导入、NPU矩阵计算与模型前向分别通过后才进入完整验收

### Requirement: 运行时必须使用虚拟环境自带解释器
服务进程 MUST 使用 `asr` 虚拟环境自带的 Python 解释器启动，MUST NOT 通过 `PYTHONPATH` 注入方式把虚拟环境的包叠加到镜像系统解释器之上。镜像系统 site-packages 中的 `torchvision` 与 `torch 2.5.1` ABI 不兼容，系统解释器叠加模式 MUST 被判定为不可用运行时。

#### Scenario: 使用系统解释器叠加虚拟环境包
- **WHEN** 以镜像系统 Python 启动并把虚拟环境 site-packages 加入 `PYTHONPATH`
- **THEN** 该组合被预检拒绝并给出解释器/包来源诊断；历史曾出现 `operator torchvision::nms does not exist`，但验收不要求特定报错字符串

#### Scenario: 使用虚拟环境解释器
- **WHEN** 以虚拟环境自带解释器启动服务
- **THEN** `torch`、`torch_npu`、`transformers` 与 `torchvision` 均可正常导入并完成一次 NPU 前向计算

### Requirement: 补齐 CANN 编译器依赖模块
运行环境 MUST 提供 `attrs` 模块，使 CANN 算子编译器子进程可以导入其 Python 适配层。缺失该模块 MUST 在启动阶段以可定位的形式失败，MUST NOT 表现为推理中途的静默降级。

#### Scenario: 缺失 attrs 模块
- **WHEN** 运行环境缺少 `attrs` 且服务首次在 NPU 上放置权重
- **THEN** 启动或预热阶段报出 `ModuleNotFoundError: No module named 'attr'` 并伴随 CANN 精度模式设置失败，运维可据此直接定位缺失依赖

#### Scenario: attrs 齐备
- **WHEN** 运行环境已提供 `attrs`
- **THEN** CANN 算子编译器初始化成功，权重放置到 NPU 后不出现编译相关异常

### Requirement: 容器必须正确绑定 NPU 设备
服务容器 MUST 以特权模式配合昇腾运行时启动并共享宿主 IPC，同时将宿主物理卡号通过 `ASCEND_VISIBLE_DEVICES` 传入容器。CANN8.2源码容器 MUST 同时设置 `ASCEND_RT_VISIBLE_DEVICES` 限定进程可见卡；特权模式下 MUST NOT 仅凭设备节点或一个环境变量推断已隔离单卡。容器 MUST NOT 在缺少 NPU 设备节点映射的情况下启动并声称具备 NPU 能力。

#### Scenario: 缺少设备节点映射
- **WHEN** 容器未映射 `/dev/davinci*` 设备节点却设置了 `ASCEND_VISIBLE_DEVICES`
- **THEN** 运行时报告 `Invalid device ID ... Expected value: [0, 0)`，服务不得进入就绪状态

#### Scenario: 正确绑定设备
- **WHEN** 容器以特权模式与昇腾运行时启动并映射设备节点
- **THEN** 容器内可见 NPU 数量与预期一致，NPU 矩阵计算通过

### Requirement: 容器内设备编号按可见设备重新映射
当容器通过设备绑定和实际进程可见卡限定了可见物理卡时，容器内 MUST 使用从 0 开始的重新映射编号访问设备，MUST NOT 继续使用宿主物理卡号。

#### Scenario: 限定单张物理卡
- **WHEN** CANN8.2源码容器设置 `ASCEND_VISIBLE_DEVICES=7` 与 `ASCEND_RT_VISIBLE_DEVICES=7` 后访问设备
- **THEN** 启动预检在任何设备放置/模型worker创建前确认进程仅可见1卡，配置仅允许 `npu:0`，并以宿主进程身份核对物理7；配置 `npu:7` 或实际可见数量不为1时就绪前失败

#### Scenario: 未限定可见设备
- **WHEN** 容器不设置可见卡限定且宿主机共有多张卡
- **THEN** 系统检查实际可见设备与物理卡映射并记录结果，不仅凭未设置单一变量就推断编号一致

### Requirement: 推理输入必须显式对齐模型精度
调用 NPU 推理时，MUST 将浮点输入张量显式转换为模型权重的计算精度，MUST NOT 依赖框架隐式类型提升。设备迁移 MUST 覆盖必要输入；dtype 转换 MUST 仅作用于浮点输入，整型索引与布尔 mask MUST 保持原 dtype 和数值。

#### Scenario: 浮点输入与半精度权重不匹配
- **WHEN** 音频特征为 float32 而卷积权重为 bfloat16，且未做显式转换
- **THEN** 推理报 `Input type (float) and bias type (c10::BFloat16) should not be the same` 类型的 dtype 不匹配错误

#### Scenario: 显式转换后推理
- **WHEN** 浮点输入按模型 dtype 显式转换后调用推理
- **THEN** 推理成功产出结果，整型索引张量数值保持不变

### Requirement: 观测 NPU 资源占用口径
系统 MUST 以 NPU 设备上的 HBM 占用作为资源观测口径，MUST NOT 以 NVIDIA 显存指标描述 NPU 资源。观测结果 MUST 能区分已加载模型占用与未加载模型占用，且 MUST NOT 把 PyTorch 之外的分配器统计为零解读为无资源占用。

#### Scenario: 查询未加载模型的状态
- **WHEN** 某模型尚未被任何请求触发加载
- **THEN** 状态接口报告该模型未加载且设备占用为 0，并保留其配置实例数与设备标识

#### Scenario: 查询已加载模型的状态
- **WHEN** 模型已加载到 NPU
- **THEN** 状态接口报告该实例的 NPU 设备标识与 HBM 占用数值，可与进程级聚合占用核对

### Requirement: 验证等级与依赖交付证据可区分
系统 MUST 分别记录安装元数据、依赖一致性、导入、NPU 真实前向、v3 端到端和性能验收结果。包版本存在 MUST NOT 被表述为前向成功；模型目录缺失 MUST 明确标记相关验证尚未执行。环境重建材料 MUST 区分业务运行、NPU、构建/测试和独立候选依赖，不以删除无直接 import 的包替代传递依赖审计。

#### Scenario: 仅查询包版本
- **WHEN** importlib.metadata 返回目标包版本
- **THEN** 报告仅确认安装记录，仍分别跟踪实际导入、依赖完整性和真实前向，不能勾选模型推理验收

#### Scenario: 构建扩展并交付
- **WHEN** 在 aarch64 上进行 Cython 保护构建
- **THEN** 构建工具版本可追溯，扩展与运行层 Python ABI 匹配，记录 torch_npu/transformers 等版本，并在无业务源码运行层验证导入及关键链路

### Requirement: 本轮开发使用指定物理设备与解释器
本轮宿主开发 MUST 使用 `npu:7` 和 `/home/zhangs/conda-envs/asr/bin/python`，MUST NOT 调度其它卡或停止共享卡已有服务。容器可见卡设置与逻辑编号 MUST 单独记录，不能把物理卡号直接当容器编号。

#### Scenario: 宿主开发与单卡容器切换
- **WHEN** 开发从宿主 npu:7 切换到仅映射物理7号卡的容器
- **THEN** 先检查可见设备映射；按实际重新编号配置逻辑设备，并记录宿主物理7号卡与容器逻辑卡的对应关系

### Requirement: 本地CANN底座身份及宿主兼容性可验证
构建 MUST 固定已核验的本地 CANN 镜像身份，记录架构、实际 toolkit/910B kernel、驱动/固件及配套检查。独立构建器无法直接读取 daemon 本地镜像时 MUST 显式传输并核对身份，不能从外网拉同名镜像替代。依赖调整仅在容器或独立构建环境执行；驱动兼容阻塞 MUST 明确报告，不擅自升级驱动或静默换底座。

#### Scenario: 镜像标签存在但实际组件不同
- **WHEN** 底座标签存在，但架构、实际 CANN/910B kernel 或驱动组合不符合目标
- **THEN** 停在对应兼容检查，记录真实版本和失败原因，不凭镜像名宣称可用

#### Scenario: 独立构建器读取本地底座
- **WHEN** 构建器需要通过任务私有传输入口获取已有底座
- **THEN** 核对传输前后 config digest、平台及内容身份，不对外发布镜像；保留原底座，传输资源登记为本轮临时资源
