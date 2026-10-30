# 规格增量

## Purpose

定义 VAD 切分环节的执行承载方式与并发治理契约：VAD 模型可部署在 CPU 或 GPU，由一组拥有独立解释器的常驻 worker 进程承载；请求在 worker 不足时进入等待队列并按排队深度自适应扩缩容；切分产生的分段临时文件在单个分段用尽后立即清除，使峰值磁盘占用与音频时长解耦。

## ADDED Requirements

### Requirement: VAD 设备可配置且默认使用 CPU
系统 MUST 支持通过配置项 `vad_device` 指定 VAD 模型的运行设备，取值为 `"cpu"` 或形如 `"cuda:0"` 的设备串，默认值 MUST 为 `"cpu"`。当取值为 CPU 时，系统 MUST NOT 为 VAD 模型建立 CUDA 上下文；当取值为 GPU 设备串时，系统 MUST 在该设备上加载 VAD 模型。VAD 的设备选择 MUST 独立于 CAM++ 与 emotion2vec 的落卡配置。

#### Scenario: 默认配置下 VAD 运行在 CPU
- **当** 部署配置未显式覆盖 `vad_device`
- **则** VAD 模型在 CPU 上加载，且不占用任何 GPU 显存

#### Scenario: 显式配置 VAD 使用 GPU
- **当** 部署配置把 `vad_device` 设为 `"cuda:0"`
- **则** VAD 模型加载到 GPU 0，每个 VAD worker 各自持有一份 CUDA 上下文

#### Scenario: VAD 设备与本地推理模型解耦
- **当** 部署配置把 `vad_device` 设为 `"cpu"` 而把 CAM++ 的落卡索引设为 0
- **则** VAD 不占用 GPU 显存，CAM++ 仍加载到 GPU 0

### Requirement: VAD 由拥有独立解释器的常驻 worker 进程承载
系统 MUST 由一组常驻 worker 进程承载 VAD 模型，每个 worker MUST 持有一份独立的 FSMN-VAD 模型实例并拥有独立的解释器执行环境，使得多个音频的 VAD 切分能够真正并行而非串行。worker 数量 MUST 由配置项 `vad_pool_size` 决定，默认值为 8。系统 MUST 在服务启动阶段预热全部 worker，避免首个请求承担模型加载延迟。

#### Scenario: 多个音频的 VAD 切分并行执行
- **当** 同时有 4 个请求进入且 `vad_pool_size` 不小于 4
- **则** 这 4 次 VAD 切分由 4 个 worker 同时执行，总耗时不随请求数线性增长

#### Scenario: 启动期预热
- **当** 服务启动完成并收到第一个音频请求
- **则** 该请求的 VAD 阶段不包含模型加载耗时

#### Scenario: worker 之间互不共享模型状态
- **当** 两个 worker 并发处理两个不同音频
- **则** 两次切分各自使用独立的模型状态缓存，输出结果不相互影响

### Requirement: 按排队深度自适应扩缩容 worker 数量
系统 MUST 在 VAD 等待队列深度超过配置项 `vad_queue_watermark`（默认 4）时临时拉起额外的 worker 进程，拉起后的常驻数量 MUST NOT 超过配置项 `vad_pool_max`（默认 16）。空闲时间超过配置项 `vad_collapse_timeout_s`（默认 120 秒）的额外 worker MUST 被回收，常驻数量回落至 `vad_pool_size`。扩容的触发判据 MUST 是等待队列深度，MUST NOT 依据显存占用。

#### Scenario: 请求量超过常驻容量时扩容
- **当** 等待队列深度达到 `vad_queue_watermark` 且常驻数量低于 `vad_pool_max`
- **则** 系统临时拉起额外 worker，使等待中的请求尽快获得执行槽位

#### Scenario: 达到扩容上限后不再拉起
- **当** 常驻 worker 数量已达 `vad_pool_max` 且仍有请求在等待
- **则** 系统不再拉起新 worker，等待中的请求继续排队

#### Scenario: 空闲 worker 被回收
- **当** 额外拉起的 worker 空闲时间超过 `vad_collapse_timeout_s`
- **则** 该 worker 进程被回收，常驻数量回到 `vad_pool_size`

### Requirement: VAD 等待超时以独立错误码返回
系统 MUST 在请求等待 VAD worker 超过配置项 `vad_queue_timeout_s`（默认 120 秒）时终止该请求，返回错误码 4014 与可读的错误描述，且 MUST 立即释放该请求占用的音频处理槽位。MUST NOT 因单个请求超时而导致 worker 泄漏或服务不可用。

#### Scenario: VAD 排队超时
- **当** 请求等待 VAD worker 超过 `vad_queue_timeout_s`
- **则** 响应返回错误码 4014，音频处理槽位被释放

#### Scenario: 超时后服务仍可用
- **当** 某个请求因 VAD 排队超时而失败
- **则** 其余请求与后续新请求仍能正常获得 VAD worker 并完成转写

### Requirement: 分段临时文件按需创建且用尽即删
系统 MUST NOT 一次性为整份音频生成全部分段临时文件，而 MUST 在需要转写某个分段时才创建该分段的文件；该分段的转写请求一旦返回，MUST 立即删除该分段文件。MUST NOT 出现分段文件在其所属请求仍在处理期间被长期保留的现象。系统 MUST 保证任意时刻单个请求持有的分段临时文件数量与音频的分段总数解耦。

#### Scenario: 长音频的分段文件不同时驻留
- **当** 一个 60 分钟音频被切分为数百个分段并逐段转写
- **则** 任意时刻该请求同时存在的分段文件数量不超过少量在途分段，而非全部分段

#### Scenario: 分段转写完成后立即释放
- **当** 某个分段的转写请求返回结果
- **则** 该分段对应的临时文件被立即删除，不等到整个请求结束

#### Scenario: 请求异常时文件同样被清理
- **当** 请求在转写过程中抛出异常
- **则** 该请求已创建的全部临时文件（上传文件、转换文件、分段文件）均被删除

### Requirement: worker 进程只读使用主进程已落盘的音频文件
系统 MUST 由主进程负责把上传音频落盘为临时文件，并把该文件路径传递给 VAD worker；worker MUST NOT 另行写入音频文件，MUST NOT 与主进程争用同一临时文件路径。音频数据在进程间的传递 MUST NOT 引入额外的编解码或内存拷贝。

#### Scenario: worker 直接读取已有临时文件
- **当** 主进程已把音频落盘并发起 VAD 切分
- **则** worker 直接对该文件路径执行切分，主进程不重复转码、不重复写盘

#### Scenario: 临时文件写入方唯一
- **当** 一个请求的整个生命周期被跟踪
- **则** 该请求的音频临时文件只由主进程写入，worker 进程不产生新的音频文件

### Requirement: 容器 CPU 配额与 VAD 池规模的关系可预期
系统 MUST 保证 VAD worker 数量与容器 CPU 消耗之间存在明确、可预期的关系：每个 VAD worker 处理 60 分钟音频时平均占用 2.2~2.5 个 CPU 核，worker 数量为 N 时 VAD 阶段的 CPU 占用约为 N 倍的 2.5 核。系统 MUST NOT 在 CPU 配额不足时仍按配置拉起全部 worker 而不加约束地超订 CPU。

#### Scenario: 池规模与 CPU 占用的线性关系
- **当** `vad_pool_size` 从 4 调整为 8
- **则** VAD 阶段的 CPU 占用由约 10 核增至约 20 核

#### Scenario: 部署方按配额选择池规模
- **当** 部署方为容器分配 16 核 CPU 配额
- **则** 其可将 `vad_pool_size` 设为 4 以避免超订，VAD 阶段仍能提供 4 路并行切分
