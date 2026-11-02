# Spec Delta

## MODIFIED Requirements

### Requirement: 异步并行计算真实角色与情绪
系统 MUST 支持使用 `speech_campplus_sv_zh_en_16k-common_advanced` 推断说话人角色、使用 `emotion2vec_plus_large` 推断情绪，并与 Qwen 转写并行执行：每个 VAD 段的转写与所请求的 role/emotion 任务 MUST 同时发起。CAM++ 与 emotion2vec 的段级并发容量 MUST 分别由各自设备数组的成功实例数决定，不增加共享模型推理闸门；B 类整音频准入限制不改变模型池实例容量。两个模型 MUST 分别由独立配置开关控制，emotion2vec 的 PyTorch CUDA 路径 MUST 沿用既有半精度加载策略，TensorRT 路径 MUST 按 emotion2vec_tensorrt_precision 选择 fp16 或 fp32，NPU/CPU 精度 MUST 独立于 TensorRT 配置；请求关闭的能力 MUST 在音频准入前报错。

#### Scenario: 开启角色与情绪能力
- **WHEN** 两个功能开关均为 `true`，请求 `showSpk=true`、`showEmotion=true` 且模型推断成功
- **THEN** 响应分段的 `role` 与 `emotion` 为真实推断值，且与转写同时并发执行并分别使用对应模型池

#### Scenario: 角色与情绪能力未开启
- **WHEN** 请求未携带 `showSpk`、`showEmotion`、`showRoleIdentify`（均默认 `false`）
- **THEN** 每个分段的 `role` 与 `emotion` 均为 `null`，两个模型池均不因该请求加载

#### Scenario: 配置未开启说话人能力
- **WHEN** `open_v2_role=false` 且请求需要说话人能力，没有同时请求另一项已关闭的能力
- **THEN** 系统在音频准入前返回 `4015`，不开始该请求的模型加载或转写；未请求说话人能力的请求不因该开关报错

#### Scenario: 配置未开启情绪能力
- **WHEN** `open_v2_emotion=false` 且请求需要情绪能力，没有同时请求另一项已关闭的能力
- **THEN** 系统在音频准入前返回 `4016`，不开始该请求的模型加载或转写；未请求情绪能力的请求不因该开关报错

#### Scenario: 配置未同时开启说话人与情绪能力
- **WHEN** `open_v2_role` 与 `open_v2_emotion` 中至少一项为 `false`
- **THEN** 仅请求已开启能力时正常进入 B 处理；请求任何关闭的能力时整个请求失败；同时请求两项且两项都关闭时返回 `4018`

#### Scenario: 角色或情绪推断失败
- **WHEN** 请求的增强能力均已开启且存在可用模型实例，但某个分段的说话人或情绪推理失败
- **THEN** 该分段的 `role` 或 `emotion` 回退为 `null`，转写结果仍正常返回

#### Scenario: 情绪模型精度
- **WHEN** emotion2vec 使用 PyTorch CUDA 后端
- **THEN** 沿用既有 fp16 加载和推理策略，TensorRT 精度配置不改变该路径

#### Scenario: 情绪 TensorRT 精度
- **WHEN** emotion2vec CUDA 实例启用 TensorRT 并显式配置 fp32
- **THEN** 加载该模型 FP32 engine，若精度未配置则选择 FP16 engine，业务情绪映射不变

#### Scenario: 两个模型使用不同后端
- **WHEN** CAM++ CUDA 实例启用 TensorRT，emotion2vec CUDA 实例关闭 TensorRT
- **THEN** 两种增强任务仍与该段 Qwen 任务并行发起，各自按模型池容量执行，emotion2vec 继续使用 PyTorch CUDA

## ADDED Requirements

### Requirement: 本地后端切换保持音频准入和取消契约
系统 MUST 在切换 CAM++/emotion2vec 后端时保留既有 A/B 分类、总音频/B 音频上限、段/Qwen 并发、能力开关、错误优先级和清理规则。请求取消 MUST 在已启动的线程/设备工作、结果读取和必要清理完成后归还对应资源，MUST NOT 因 TensorRT 异步提交返回而提前释放租约。

#### Scenario: B 上限不因后端切换而变化
- **WHEN** 总音频上限为 8、B 上限为 4，已有 3A+4B，后端从 PyTorch 改为 TensorRT
- **THEN** 新 B 仍返回 4017，新 A 仍可获准，模型实例容量仍由各自设备数组决定

#### Scenario: 普通转写不触发增强模型
- **WHEN** 三个增强请求参数均为 false，模型 TensorRT 开关为 true
- **THEN** 请求仍为 A，不因后端配置加载或调用 CAM++/emotion2vec，也不要求其 engine

#### Scenario: 启用 TensorRT 后请求关闭能力
- **WHEN** 模型 TensorRT 开关为 true，但请求要求的功能被 features 配置关闭
- **THEN** 仍在音频准入前按既有 4015/4016/4018 返回，不因存在 engine 而绕过校验

#### Scenario: CUDA 工作尚未完成时取消
- **WHEN** 已准入 B 请求取消，而某实例已经提交 TensorRT 工作且设备尚未完成
- **THEN** 请求仍持有该模型及整音频租约，完成设备工作和必要清理后才归还，最终计数一致

#### Scenario: engine 初始化失败
- **WHEN** 已准入请求所需模型的全部 TensorRT 实例初始化失败
- **THEN** 返回对应模型不可用 4015 或 4016，并完成请求清理和音频租约释放，不自动切为 PyTorch 或改用组合能力关闭错误
