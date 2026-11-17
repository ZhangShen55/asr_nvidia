# 迁移补充任务清单

本清单补充远程原有 47 项任务；未执行实施，不勾选任何任务。23 项已合并进同目录 tasks.md；本文件保留补充任务来源快照，不重复计算为另一组任务。0.x 为前置工作，10.x 应在相应原任务验收前完成，不能等归档后再补。

## 0. 迁移前置条件与 v3 契约冻结（补充）

- [ ] 0.1 固化 `/v3/seacraft_asr` 的请求、响应、错误码和默认值基准；覆盖 `bg`/`ed` 两位小数字符串、`load_audio_time_ms`/`gpu_time_ms` 字符串、增强功能默认 false，以及 `speed_info`、`language_ratio`；保持 `gpu_time_ms` 原计时起止与统计口径，不改成纯 NPU kernel 时间。
- [ ] 0.2 核验用户已补充的 emotion2vec_plus_large、speech_fsmn_vad_zh-cn-16k-common-pytorch、speech_campplus_sv_zh_en_16k-common_advanced 三个目录的权重、配置、标签和 checkpoint/revision；另行核对 Qwen 权重，分别记录缺失项。文件到位后仍须通过实际加载与前向验证，才能勾选对应验收任务。
- [ ] 0.3 建立测试音频递归清单：已发现 52 WAV、2 MP3，共 5,144,064,354 字节；WAV 文件头合计约 43.66 小时，均为 16 kHz/单声道/16 位。记录 SHA256、音频格式、时长及样本用途，区分文件头检查、完整解码和人工标注的证据等级。
- [ ] 0.4 处理测试目录 GBK 原始文件名字节与 UTF-8 路径不匹配；扫描器支持现有文件，上传使用稳定样本 ID；若规范化名称，先记录原始字节到新路径的映射并检查重名，禁止覆盖原样本。
- [ ] 0.5 固定本次宿主开发为 `xjtu910b7`、`npu:7`、`/home/zhangs/conda-envs/asr/bin/python`；记录卡 7 上既有服务占用，不调度其他卡、不终止既有服务，不把共享卡结果称为独占性能。

## 10. 依赖、后端选型与构建闭包（补充）

- [ ] 10.1 统一全部 `requirements*.txt` 与环境重建入口：锁定 torch/torch-npu/torchaudio 2.5.1、transformers 5.13.0、numpy 1.26.4、funasr 1.2.6 的已验证目标组合；修正 av 14.4.0 与当前 conda av 14.2.0 的冲突，记录 Python/aarch64/CANN/驱动、分发来源及 pip/conda 分工。
- [ ] 10.2 分离通用运行、NPU、开发测试和可选优化依赖；保留 torchaudio fbank/音频读取、WeTextProcessing/pynini、attrs。依据实际 import/依赖元数据决定 requests/httpx/av/torchvision 的去留，覆盖测试客户端和压测脚本，不按核心代码无直接 import 就删除传递依赖。
- [ ] 10.3 在独立临时前缀验证可重建性和依赖一致性，分别记录 metadata、导入、NPU 单模型前向、端到端结果；不得原地升级/降级当前 asr 环境验证候选，也不得以安装成功替代实机验收。
- [ ] 10.4 建立 Qwen eager 预热后基线；评估长度分桶微批、匹配当前版本的图编译，以及隔离环境内锁定历史版本的 vLLM-Ascend。记录权重格式/模型 revision、raw 输出、语言契约、补丁、维护状态、质量和稳态性能；不预先宣称原生 eager 最优或 vLLM 路线永久不可用。
- [ ] 10.5 完成生产后端选择记录：默认原生实现是基线，候选通过不自动改变交付架构；若选择候选替换原生主链，先同步 proposal/design/specs/tasks 和依赖边界，再实施。正式交付仅保留选定执行路径及必要适配契约。
- [ ] 10.6 若启用微批，配置批大小、等待上限及长度策略；保持各段独立音频、语言、结果、VAD 时间轴和错误归属，明确实例/批次/段的容量计数及取消收尾。不得拼接音频或简单将不同长度输入补零后无 mask 求均值。
- [ ] 10.7 比较 CAM++/emotion2vec 的 eager FP32、质量合格的 BF16/FP16 和匹配运行时的 TorchAir；记录动态长度、冷编译、重编译、稳态延迟及同卡干扰。只有收益足够且算子/长度/精度可控时再选 ONNX→ATC→OM，不以原权重不是 ONNX 作为永久排除理由。
- [ ] 10.8 验证 CAM++ 池化算子在 T=99/100/101/199/200/201、短输入和负值下的等价性；平均池化尾段按有效元素计数，最大池化无效位置不得用零改变负值。适配放在项目中，不修改 conda site-packages；同时验证 embedding 与聚类。
- [ ] 10.9 审计 NPU worker 与模型池生命周期：保留 VAD spawn/FIFO/静态容量，验证实际落卡；查询状态不加载模型，NPU 加载失败不静默改走 CPU；超时/取消后等待设备工作结束再归还实例及音频租约。
- [ ] 10.10 修正 VAD 和增强模型内存采样中 NPU 返回 0 的路径；区分 PyTorch allocated/reserved、进程 HBM 与设备聚合 HBM，无法归因时明确标记，不把既有 uvicorn 占用算作本项目模型占用。
- [ ] 10.11 改写保留通用脚本：bench_local_model_latency、check_cam_cluster_boundary、speaker_threshold_sweep、profile_role_emotion、verify_vad_pool_regression、VAD 扩展压测及 CAM/emo 汇总；清理其 TRT 导入、CUDA autocast/device/synchronize 与报告中的硬编码。
- [ ] 10.12 补全测试迁移：test_docker_profile、test_model_pool、test_model_pool_route、test_local_model_runtime、test_vad_pool、test_config_contract；反转 test_tensorrt_config 中“VAD 拒绝 NPU”的旧断言。保留 test_inference_backends 中 NPU 不查询 CUDA、隔离、失败池与取消收尾场景，不整份删除带 TRT 替身的业务测试。
- [ ] 10.13 删除纯 TRT 模块/脚本后执行导入闭包检查，覆盖业务、tests、保留 scripts 和构建清单；可删除纯 TRT 的 test_model_assets。移除 validate_backend_admission 的 TRT 工装前，确认 tests/test_audio_admission* 仍覆盖其通用准入/容量/取消契约。
- [ ] 10.14 保留并适配 Cython 交付：锁定构建工具，更新 build_cython 元数据中的 TRT 为 NPU/transformers/架构信息，在目标 aarch64/Python ABI 下验证扩展导入，确保新模块纳入构建且不残留已删模块引用。
- [ ] 10.15 将 NPU 设备映射、runtime/IPC/可见物理卡参数放在 Compose 或宿主启动入口；docker/start.sh 只处理容器内解释器、预检和进程启动。初始单服务进程，明确 instance_count 复制本地 Qwen 权重的资源成本。
- [ ] 10.16 用同一份清单记录删/改/保留的资产、理由和替代验证；保留归档 OpenSpec、模型权重、原始音频及可复算标定证据。清理源文件不等于允许删除运行镜像、日志、缓存或既有服务。
- [ ] 10.17 分层验收：先短样本与失败路径，再课堂长录音，最后八路并发；记录质量、P50/P95、音频吞吐、RTF、HBM、CPU 与卡 7 共享占用。无人工参考时仅报告结果差异，不宣称 WER/CER 或有效对话召回达标。
- [ ] 10.18 复核规划一致性与状态：修正 VAD 同进程、v2 路径和固定 16/128 等历史表述；清理其他 NVIDIA open change 时保留当前 change 到验收归档。交付文件适配纳入本轮，实际镜像构建和上线待服务稳定后执行。
