# Spec Delta

## MODIFIED Requirements

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

## ADDED Requirements

### Requirement: 本地CANN底座身份及宿主兼容性可验证
构建 MUST 固定已核验的本地 CANN 镜像身份，记录架构、实际 toolkit/910B kernel、驱动/固件及配套检查。独立构建器无法直接读取 daemon 本地镜像时 MUST 显式传输并核对身份，不能从外网拉同名镜像替代。依赖调整仅在容器或独立构建环境执行；驱动兼容阻塞 MUST 明确报告，不擅自升级驱动或静默换底座。

#### Scenario: 镜像标签存在但实际组件不同
- **WHEN** 底座标签存在，但架构、实际 CANN/910B kernel 或驱动组合不符合目标
- **THEN** 停在对应兼容检查，记录真实版本和失败原因，不凭镜像名宣称可用

#### Scenario: 独立构建器读取本地底座
- **WHEN** 构建器需要通过任务私有传输入口获取已有底座
- **THEN** 核对传输前后 config digest、平台及内容身份，不对外发布镜像；保留原底座，传输资源登记为本轮临时资源
