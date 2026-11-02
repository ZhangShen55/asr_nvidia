# Tasks

实施前提：以 `2026-11-02-split-audio-concurrency-gates` 完成后的代码为基线。此次规划仅包含 CAM++/emotion2vec 的四个配置项、按设备数组选择实际后端及必要的推理接入，不包含 VAD 替换或新的并发闸门。

已有证据：四个 FP32/FP16 候选 engine 已放入各模型 `tensorrt/` 子目录并校验 SHA256，独立实验见 `model/tensorrt-validation-20261102/evidence/REPORT.md`。以下是尚待完成的项目集成与验收任务，不能因初步实验成功直接勾选。

## 1. 四个配置项与实例设备契约

- [x] 1.1 在 `[model_device_pool]` 解析 `speech_campplus_use_tensorrt=false`、`speech_campplus_tensorrt_precision="fp16"`、`emotion2vec_use_tensorrt=false`、`emotion2vec_tensorrt_precision="fp16"`，同步本地/容器模板及参数说明；用临时 TOML 验证缺省、自定义、严格布尔值和 fp16/fp32 枚举，非法值在就绪前报字段级错误，不恢复旧配置表。
- [x] 1.2 按每个 CAM++/emotion2vec 设备数组元素解析 cpu/cuda:N/npu:N 和实际后端，保持重复设备、多实例和非对称池；用配置与运行器桩验证 CUDA+true/false、NPU/CPU 忽略 TensorRT、混合池及空数组，四个参数不覆盖功能开关或数组容量。
- [x] 1.3 冻结当前代码、模型权重、软件/GPU 版本、样本 manifest 和总/B/段/Qwen 并发与设备数组，整理已有独立实验及其限制；核对所有来源/engine 哈希可追溯，明确服务端基线和完整课堂验收尚未完成。

## 2. 后端运行器与模型池接入

- [x] 2.1 建立 PyTorch CUDA/CPU、torch_npu 和 TensorRT 的运行器选择与统一输入输出边界，隔离设备、AMP 和后端依赖；验证关闭 TensorRT 的 CUDA、CPU 及 NPU 路径不依赖 TensorRT，NPU 不调用/初始化 CUDA，用桩验证选择与导入隔离并明确 NPU 尚未实机验收。
- [x] 2.2 在独立模型池实例中加载匹配运行器和 engine，保留懒加载、FIFO 租约及容量；每个 TensorRT 实例独占 context/stream/可写缓冲区，验证 CAM++ 3、emotion2vec 1 的并发容量、部分加载失败隔离及全部失败的 4015/4016，不增加全局模型锁或自动 PyTorch 回退。
- [x] 2.3 将线程与设备推理完成纳入租约和资源收尾，保留 `run_in_thread`/`await_completion` 语义；用阻塞线程、模拟异步提交、延迟设备完成与重复取消证明设备完成前不归还模型和 A/B 音频租约，并验证加载取消及关闭时资源正确回收。
- [x] 2.4 保留路由和运行入口的能力校验及错误分层；测试关闭能力不加载/租用模型、4018 只表示同时请求两项已关闭能力、engine 失败显式可观测、普通健康实例段级失败可 null，以及实例级故障更新健康状态。
- [x] 2.5 扩展按实例的实际后端、配置/实际精度、engine、兼容/健康/失败原因和内存口径，并同步运维说明及 CUDA/NPU 依赖约束；验证既有 A/B、模型池、VAD 状态字段保持含义，查询不加载模型，非 PyTorch 分配内存不会被误报为 0，无需实际导出即可完成状态/导入隔离测试。

## 3. CAM++ TensorRT 适配与验证

- [x] 3.1 整理可复现 CAM++ 导出/构建工具和正式 manifest，复用已交付候选而非无条件重复构建；核验 80 维 fbank、batch=1、T=50/200/3000 profile、FP32 I/O、192 维 embedding 和 52 个 AveragePool 尾窗口修正，针对目标架构验证哈希/兼容信息并补齐权重与代码版本。
- [x] 3.2 接入与现有 FunASR 一致的按段去均值 fbank 和结果读取，保留不足 1 秒跳过 CAM++；用固定音频验证预处理与输出契约、空/非法/超 profile 输入、动态 shape 连续变化及不同实例缓冲隔离，不擅自截断、跨段合并或静默切后端。
- [x] 3.3 先复核 FP32 embedding 数值、维度、有限值和完整课堂聚类，再对 FP16 重复验证并记录逐段差异；输出与当前 PyTorch 的聚类划分对比和边界失败样本，明确现有 33 个裁剪及抽样聚类不能代替整份录音验收，更新模型使用说明。

## 4. emotion2vec TensorRT 适配与验证

- [x] 4.1 整理完整 emotion2vec 导出/构建工具及正式 manifest，保留最大 10 秒 ALiBi 缓存与动态切片；核验 batch=1、16 kHz、N=16000/48000/160000、FP32 I/O、9 类 logits 与 7 类 scores，包含归一化至分类 Softmax 全路径和权重/代码追溯。
- [x] 4.2 接入原始波形和原业务情绪映射，按配置加载 fp32/fp16 engine；测试 1 秒、10 秒及非整秒输入，1～10 秒外仍走现有“平淡”分支且先校验能力开关，空/非法输入与越界失败不触发隐式后端回退，不重复归一化或更改类别顺序。
- [x] 4.3 先验证 FP32，再验证 FP16 的概率/标签与实际 FunASR generate 输出，扩充低置信度、类别接近及完整课堂片段；记录逐段最大概率差、业务字段一致性和失败样本，明确已有 27/27 标签一致及约 1.60 个百分点最大概率差只是初步证据，并更新模型说明。

## 5. 打包、并发回归与验收

- [x] 5.1 更新模型交付说明、配置示例、依赖和容器资产检查，模型目录下提供 tensorrt/<precision>/engine.plan 与 manifest.json，并保留 PyTorch 权重；验证所需 CUDA TensorRT 实例缺资产时明确不可用，其他后端/关闭能力不要求 engine 或 TensorRT 包，engine 仅针对匹配环境，不更改现有 VAD 交付要求。
- [x] 5.2 完成四个配置组合（全关闭、仅 CAM++、仅 emotion2vec、两者开启）的路由、模型池和 A/B 准入回归；验证文本/边界/聚类/情绪契约、默认开关回归、能力错误、4013/4017、等待/取消/清理/重复释放及最终计数归零，真实 CUDA 下补充设备完成同步检查。
- [x] 5.3 固定同卡 Qwen 负载、VAD、总/B/段/Qwen 并发、模型数组及 A/B 请求比例，对比 PyTorch 与 TensorRT FP32/FP16 的单段和端到端 P50/P95、吞吐、等待、显存、GPU、错误及准入拒绝/等待；默认 B=4 时测试 1/2/4 个活动 B，八路提交单列限流结果，八个活动 B 必须所有对照组都显式设 B>=8，不以改变限流或实例数冒充后端收益。
- [x] 5.4 按 design 第 8 节固定的 8 门不同课程完整 WAV，执行 CAM++ 3 个同卡 CUDA 实例、emotion2vec 1 个同卡 CUDA 实例、showSpk=true、showEmotion=true 的八路完整音频验收；两个功能开启，测试总音频/B 上限均为 8、段并发 16、Qwen 全局并发 128，在一致配置和外部负载下分别测量当前 PyTorch、TensorRT FP32、TensorRT FP16。冻结样本路径/时长/大小/SHA256，分开冷加载/预热和实测；同时起跑并核对全部准入、处理重叠与活动 B 峰值，交付逐音频耗时/RTF/段数/状态和实际模型调用/跳过/失败/null、整批耗时/成功数/吞吐及资源指标；每轮以八个成功完整请求的客户端端到端耗时按线性插值计算 P50/P95，标注秒单位及 N=8，失败/拒绝/超时完整列出并标记该轮验收不完整，保留原始证据可复算，不用单段模型耗时替代。
- [x] 5.5 分模型验证关闭 TensorRT 开关并重启后的 PyTorch 回滚、engine 缺失/损坏/不兼容和单实例故障；确认另一模型选择独立、状态与报告一致，运行相关测试及 `openspec validate 2026-10-29-fireredvad-cam-emo-tensorrt-backends --strict`，交付可复算报告和可审阅变更。

## Workflow follow-up

- 默认开关保持 false，质量和目标同卡负载验证通过后再由用户决定部署与灰度；本 change 不自动操作正在运行的服务。
- NPU 环境具备后再做真实 torch_npu 全链路质量与性能专项验收；当前只证明路由/依赖隔离的桩测试不能宣称 NPU 实机可用。
- 实现及验收完成后再归档；模型层加速不作为完整请求加速倍数。
