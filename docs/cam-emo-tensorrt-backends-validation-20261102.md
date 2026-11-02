# CAM++ / emotion2vec TensorRT 服务集成验证报告（2026-11-02）

本次已接入四个后端参数、按设备数组分配运行器，并完成同卡八路完整课堂测试。两个 TensorRT 开关默认仍为 false。FireRedVAD、YAMNet、VAD 分段及 tqdm 均不属于本次改动。

## 1. 结果与适用范围

| 后端 | 成功数 | 完整请求 P50（秒） | P95（秒） | 整批墙钟（秒） | 吞吐（音频秒/墙钟秒） | 冷加载及预热（秒） |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch CUDA（原 FP16） | 8/8 | 407.003 | 428.395 | 431.900 | 56.599 | 28.942 |
| TensorRT FP32 | 8/8 | 115.822 | 123.924 | 124.841 | 195.810 | 41.959 |
| TensorRT FP16 | 8/8 | 97.016 | 107.217 | 108.291 | 225.737 | 22.918 |

FP16 相对当前 PyTorch 的 P50 加速为 **4.20 倍**，P95 为 4.00 倍，整批墙钟为 3.99 倍。各组只执行一轮，N=8，P95 是这批样本的描述值，不能视作稳定生产尾延迟或对任意负载的倍数保证。

FP16 在八门课共同输出片段上的角色聚类划分均与当前 PyTorch FP16 一致；情绪仍有 9 个业务标签差异。FP32 在高等传热学上从 3 簇变成 2 簇，固定输入复核时 PyTorch FP32 也得到相同 2 簇。因此已完成集成和比较，尚不能称为所有输入上的无损替换，保持默认关闭，由业务方结合边界样本决定启用策略。

## 2. 冻结配置与环境

```toml
[model_device_pool]
speech_campplus_devices = ["cuda:0", "cuda:0", "cuda:0"]
emotion2vec_devices = ["cuda:0"]
speech_campplus_use_tensorrt = false
speech_campplus_tensorrt_precision = "fp16"
emotion2vec_use_tensorrt = false
emotion2vec_tensorrt_precision = "fp16"

[features]
open_v2_role = true
open_v2_emotion = true

[qwen_asr]
audio_concurrency = 8
audio_spk_emo_concurrency = 8 # 仅完整八路测试；正式默认仍为 4
segment_concurrency = 16
request_concurrency = 128
vad_device = ["cpu", "cpu", "cuda:0", "cuda:0"]
```

三组都请求 `showSpk=true`、`showEmotion=true`；A/B 请求比例为 0:8。PyTorch 组两个开关 false，保留既有半精度策略；另外两组两个开关 true，分别选 fp32、fp16。除后端/精度和日志路径外，三个完整测试配置逐项相同，VAD、角色阈值和情绪业务映射保持一致。测试服务使用临时端口 18084；未重启正式服务，也未自动开启正式 TensorRT 开关。

本机实际服务解释器 `/root/anaconda3/envs/asr/bin/python`：Python 3.11.16、PyTorch 2.6.0+cu124、FunASR 1.2.6、TensorRT 10.7.0.post1、CUDA 12.4、glibc 2.34。TensorRT Python 3.11 绑定和已有原生库通过临时 PYTHONPATH/LD_LIBRARY_PATH 使用；没有为测试改装 ASR 环境。可选安装依赖已写入 `requirements-tensorrt.txt` / `requirements-tensorrt-export.txt`。

CUDA_VISIBLE_DEVICES 未设置，torch cuda:0 UUID 与 nvidia-smi 物理卡 0 一致：`GPU-d36718da-51c5-2724-1720-9cf0df1978f7`，RTX 4090、SM89、驱动 580.126.09。Qwen3-ASR-1.7B 使用同卡现有服务 `http://127.0.0.1:8806`。所有比较使用该部署和相同 Qwen 请求集合，没有把增强模型移到空卡。共享环境外部即时负载未完全隔离，未通过停止其他业务制造空载结果，单轮结果仍可能受调度、缓存和外部负载波动影响。

源代码起点 `5dfcb176d34caf21f72f89d48d582a03011a9c0d`；实施期间并发 change 在另一个会话提交归档至 `ae34ec4`，本次保留其 A/B 准入实现和原有工作区改动。初始副本/差异见 [源文件快照](/home/zhangs/workspace/asr_nvidia/test/tensorrt-backends-20261102/source-before)、[baseline.patch](/home/zhangs/workspace/asr_nvidia/test/tensorrt-backends-20261102/baseline.patch)；软硬件见 [service-environment.json](/home/zhangs/workspace/asr_nvidia/test/tensorrt-backends-20261102/service-environment.json)，最终文件与资产哈希见 [implementation-manifest.json](/home/zhangs/workspace/asr_nvidia/test/tensorrt-backends-20261102/implementation-manifest.json)。

当前 CentOS 7 / CUDA 12.1.1 Docker 基础镜像没有在本次构建或实机验收；不能把主机 CUDA 12.4 的结果等同为该镜像已兼容。NPU 只完成设备路由/导入隔离桩测试和原始网络 CPU 等价检查，**未进行 torch_npu 硬件质量或性能验收**。

## 3. 样本、起跑与计时

八门不同课程各取原始完整 WAV，未预先裁剪后冒充整课请求；16 kHz、单声道、16-bit PCM，总时长 24445.131 秒。完整路径/精确时长/字节数/SHA256 已冻结于 [baseline.json](/home/zhangs/workspace/asr_nvidia/test/tensorrt-backends-20261102/baseline.json)。

| 编号 | 课程 | 原始时长（秒） | 字节数 | SHA256 |
| --- | --- | --- | --- | --- |
| 1 | 通用学术英语Ⅱ | 2821.056000 | 90273870 | `5ae4039dea035b36b64c4dd4a0f1f774190a2080ec95d67aca91dceccf9db82a` |
| 2 | 公共管理学 | 2822.560000 | 90321998 | `0e9f4df99a73f4e5ea9b7415a9cac4db3a710dabe0beda5b722e7236ad36985e` |
| 3 | 多模态机器学习 | 3307.008000 | 105824334 | `0510efc2067f70eff7566fdc49ea5e8206f07f0b23a44dfc369b052d5baedd80` |
| 4 | 储能学术英语实践 | 3117.312000 | 99754062 | `16cdf4a725788a2d46594fa4d9aaac5e75f8d19e0303e04a894109d365b96b70` |
| 5 | 高等传热学 | 3308.224000 | 105863246 | `929aae58ffe8a17168260e82b2613a5eea73588c7156fb66d699bae1905ffc31` |
| 6 | 综合岩石学 | 2827.968000 | 90495054 | `2eef3c2e6630c1c5540a0b872caa94cb37249f8de32426086054e178e7e93756` |
| 7 | 科学的历程 | 3119.872000 | 99835982 | `c08d49f898b68a43be76f7af7594138c20920f20e850e7b3dcf51014caa1dcce` |
| 8 | 线性代数与解析几何II | 3121.130688 | 99876260 | `d2485c6e64bba4285458b8c442191349ccbff581c14d9ba0d8d206627f93fab6` |

客户端在屏障前已把八份 WAV 读入内存；8 个线程共同起跑。逐请求使用 perf_counter 单调时钟，从发起上传/HTTP 调用到读取完整响应，包含上传、准入、VAD、分段处理、汇总及响应读取。三组客户端起始时间跨度分别为 0.461 / 0.514 / 0.440 秒；每组全部准入，采样实际活动 B 峰值均为 8，八个处理区间重叠。

各组新启动测试进程，先完成全部实例加载、每实例一次 3 秒输入预热，再开始计时；预热取第一门课 60～63 秒，可含静音，仍实际执行网络，不作 VAD 语音验收样本。冷加载/预热时间已单列，不混入八个请求。原始逐请求 ID、UTC 开始/结束时间、单调时钟、耗时、RTF、段数、状态和服务端任务 ID 已导出 [full-audio-request-records.csv](/home/zhangs/workspace/asr_nvidia/test/tensorrt-backends-20261102/full-audio-request-records.csv)，完整 JSON 响应位于各后端目录。

按每组八个成功请求耗时排序为 x[0..7]，h=(N-1)×p，按 floor(h)/ceil(h) 线性插值；numpy.percentile 默认 linear。整批墙钟为最后结束时刻减最早起点；吞吐为总音频时长除以整批墙钟；RTF 为单请求耗时除以其原始音频时长。三组无请求失败、超时、准入拒绝，也未自动重试整音频隐藏失败。Qwen 内部既有复读修复/重试策略未改动。

## 4. 每份完整音频结果

| 编号 | 课程 | VAD 段数 | PyTorch 秒 / RTF | FP32 秒 / RTF | FP16 秒 / RTF |
| --- | --- | --- | --- | --- | --- |
| 1 | 通用学术英语Ⅱ | 196 | 178.651 / 0.06333 | 44.964 / 0.01594 | 44.386 / 0.01573 |
| 2 | 公共管理学 | 530 | 380.744 / 0.13489 | 90.912 / 0.03221 | 90.239 / 0.03197 |
| 3 | 多模态机器学习 | 640 | 422.143 / 0.12765 | 124.698 / 0.03771 | 107.851 / 0.03261 |
| 4 | 储能学术英语实践 | 753 | 415.164 / 0.13318 | 116.746 / 0.03745 | 97.843 / 0.03139 |
| 5 | 高等传热学 | 546 | 396.354 / 0.11981 | 114.897 / 0.03473 | 96.347 / 0.02912 |
| 6 | 综合岩石学 | 591 | 414.867 / 0.14670 | 109.039 / 0.03856 | 83.708 / 0.02960 |
| 7 | 科学的历程 | 811 | 431.762 / 0.13839 | 122.487 / 0.03926 | 106.039 / 0.03399 |
| 8 | 线性代数与解析几何II | 520 | 399.140 / 0.12788 | 117.738 / 0.03772 | 97.685 / 0.03130 |

VAD 段数按每段初始 Qwen 调用计数，并与 VAD 日志核对；三组均为 4587 段。输出段数会受 ASR 空文本/语气词筛选影响，不能以最终输出段数代替 VAD 段数。

| 编号 | 最终输出段数（PT / FP32 / FP16） | CAM 调用（各组相同） | CAM 跳过（各组相同） | emo 调用（各组相同） | emo 时长快捷分支（各组相同） | 最终 role/emotion null |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 174 / 175 / 175 | 170 | 26 | 170 | 26 | 0 / 0 |
| 2 | 519 / 520 / 520 | 478 | 52 | 478 | 52 | 0 / 0 |
| 3 | 618 / 617 / 619 | 563 | 77 | 563 | 77 | 0 / 0 |
| 4 | 733 / 734 / 735 | 556 | 197 | 556 | 197 | 0 / 0 |
| 5 | 494 / 495 / 495 | 467 | 79 | 467 | 79 | 0 / 0 |
| 6 | 577 / 577 / 577 | 547 | 44 | 547 | 44 | 0 / 0 |
| 7 | 802 / 803 / 802 | 697 | 114 | 697 | 114 | 0 / 0 |
| 8 | 514 / 513 / 514 | 494 | 26 | 494 | 26 | 0 / 0 |

各组均真实执行 CAM++ 3972 次、emotion2vec 3972 次；CAM++ 不足 1 秒跳过 615 次，emotion2vec 时长快捷分支 615 次。日志中的两模型推理失败均为 0，最终输出两个字段的 null 均为 0；短段角色继承和短段“平淡”不是模型调用。结束后音频 active/waiting、两个模型池 busy/waiting 均归零，容量保持 CAM++ 3、emotion2vec 1。

## 5. 等待、资源与单段推理

| 后端 | CAM 推理累计秒 | CAM 等待累计秒 | emo 推理累计秒 | emo 等待累计秒 | Qwen 累计秒 | Qwen 名额等待累计秒 | 音频准入等待累计秒 | 卡峰值 MiB | GPU 峰值 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch CUDA（原 FP16） | 1200.027 | 42004.545 | 361.285 | 5860.042 | 1662.852 | 0.026 | 0.000 | 24163 | 100 |
| TensorRT FP32 | 182.731 | 904.443 | 96.435 | 8542.496 | 5783.207 | 0.024 | 0.000 | 24145 | 100 |
| TensorRT FP16 | 180.410 | 351.268 | 83.699 | 7352.087 | 4483.569 | 0.027 | 0.000 | 23713 | 100 |

以上为并行段任务累计量，多个任务等待区间重叠，不能相加当墙钟。CAM++ 排队在 TensorRT 下明显缩短，单实例 emotion2vec 的等待成为更明显的剩余瓶颈；Qwen 并发请求更密集后，其累计服务耗时也会变化。不能把本地模型平均耗时倍数直接当作完整请求倍数。显存/GPU 每约 2 秒采样一次，峰值可能漏采；显存包含同卡 Qwen 和其他进程，不是单模型占用。

TensorRT 的原生分配不能由 torch.cuda.memory_allocated 完整统计，状态中相关池 allocated_mb=null，另列 PyTorch 分配差额和 context/engine/设备总占用口径，未用“0 MB”冒充无占用。

| 模型 | 后端 | 片段数 / 实测次数 | 单段 P50（毫秒） | P95（毫秒） |
| --- | --- | --- | --- | --- |
| speech_campplus | PyTorch CUDA（原 FP16） | 48 / 144 | 40.543 | 126.371 |
| speech_campplus | TensorRT FP32 | 48 / 144 | 5.940 | 7.901 |
| speech_campplus | TensorRT FP16 | 48 / 144 | 6.801 | 8.340 |
| emotion2vec | PyTorch CUDA（原 FP16） | 48 / 144 | 16.145 | 28.879 |
| emotion2vec | TensorRT FP32 | 48 / 144 | 4.104 | 8.463 |
| emotion2vec | TensorRT FP16 | 48 / 144 | 1.832 | 3.546 |

此表另用同一 48 个课堂裁剪串行测量，每个输入 shape 预热一次后测 3 次，N=144/模型/后端，含原始波形至 CPU 输出的特征、拷贝、网络与同步，排除模型池排队。仍为同一卡且 Qwen 驻留，但此微基准不额外发 Qwen 请求，**不冒充八路业务并发中的段延迟分位数**。同卡八路的实际模型推理及等待累计量见上表；完整请求指标见第 1 节。短段 CAM++ FP16 未必比 FP32 快。原始数据见 [single-segment-latency.json](/home/zhangs/workspace/asr_nvidia/test/tensorrt-backends-20261102/single-segment-latency.json)。

## 6. 默认 B=4 准入回归

| 后端 | 同时提交 | 成功 | 4017 拒绝 | 活动 B 峰值 | 成功请求 P50/P95（秒） |
| --- | --- | --- | --- | --- | --- |
| PyTorch CUDA（原 FP16） | 1 | 1 | 0 | 1 | 2.266 / 2.266 |
| PyTorch CUDA（原 FP16） | 2 | 2 | 0 | 2 | 1.473 / 1.483 |
| PyTorch CUDA（原 FP16） | 4 | 4 | 0 | 4 | 1.648 / 1.868 |
| PyTorch CUDA（原 FP16） | 8 | 4 | 4 | 4 | 1.967 / 2.315 |
| TensorRT FP32 | 1 | 1 | 0 | 1 | 1.278 / 1.278 |
| TensorRT FP32 | 2 | 2 | 0 | 2 | 1.437 / 1.438 |
| TensorRT FP32 | 4 | 4 | 0 | 4 | 1.406 / 1.407 |
| TensorRT FP32 | 8 | 4 | 4 | 4 | 1.451 / 1.452 |
| TensorRT FP16 | 1 | 1 | 0 | 1 | 1.385 / 1.385 |
| TensorRT FP16 | 2 | 2 | 0 | 2 | 1.427 / 1.427 |
| TensorRT FP16 | 4 | 4 | 0 | 4 | 1.471 / 1.472 |
| TensorRT FP16 | 8 | 4 | 4 | 4 | 1.409 / 1.410 |

此处为短课堂片段的准入/清理回归，两个能力均 true，总上限 8、B 上限 4、立即拒绝模式。八路提交得到 4 成功+4 个 4017，不能称八个同时活动 B；所有组结束计数归零。该表与 B=8 的完整课堂加速计算分开，原始响应/状态见 [admission/summary.json](/home/zhangs/workspace/asr_nvidia/test/tensorrt-backends-20261102/admission/summary.json)。

## 7. CAM++ 数值与完整课堂聚类

CAM++ fbank 与 FunASR extract_feature 逐位相同；输出 192 维且有限。profile T=50/200/3000 均可执行，界外输入拒绝。三个并行实例的 context、stream、输入缓冲地址各自独立，不同段长并行输出与顺序执行相同。两种 engine I/O 均为 FP32，FP16 仅指内部计算策略。

先以纯 PyTorch FP32 复核 33 个固定裁剪：TensorRT FP32 embedding 最大绝对差 8.58e-6、最小余弦 0.99999988；FP16 最大绝对差 0.0334633、最小余弦 0.99990165。后续与当前服务 PyTorch FP16 的对比如下，不能混用两种参照。原始 FP32 数值记录见 [runner-numerical.json](/home/zhangs/workspace/asr_nvidia/test/tensorrt-backends-20261102/runner-numerical.json)。

| 样本集 | 片段数 | FP32 对当前 PyTorch FP16 最小余弦 | FP16 最小余弦 |
| --- | --- | --- | --- |
| 原始固定裁剪 | 33 | 0.99998057 | 0.99984980 |
| 八门课扩展裁剪 | 48 | 0.99998838 | 0.99996203 |
| 情绪差异对应裁剪 | 11 | 0.99999350 | 0.99998075 |

**TensorRT FP32 整课共同输出片段**：

| 课程 | PT / TRT 输出段数 | 对齐段数 | PT / TRT 簇数 | 角色 ARI |
| --- | --- | --- | --- | --- |
| 通用学术英语Ⅱ | 174 / 175 | 174 | 9 / 9 | 1.00000000 |
| 公共管理学 | 519 / 520 | 519 | 2 / 2 | 1.00000000 |
| 多模态机器学习 | 618 / 617 | 617 | 7 / 7 | 1.00000000 |
| 储能学术英语实践 | 733 / 734 | 733 | 3 / 3 | 1.00000000 |
| 高等传热学 | 494 / 495 | 494 | 3 / 2 | 0.79837069 |
| 综合岩石学 | 577 / 577 | 576 | 3 / 3 | 1.00000000 |
| 科学的历程 | 802 / 803 | 802 | 1 / 1 | 1.00000000 |
| 线性代数与解析几何II | 514 / 513 | 513 | 3 / 3 | 1.00000000 |

**TensorRT FP16 整课共同输出片段**：

| 课程 | PT / TRT 输出段数 | 对齐段数 | PT / TRT 簇数 | 角色 ARI |
| --- | --- | --- | --- | --- |
| 通用学术英语Ⅱ | 174 / 175 | 174 | 9 / 9 | 1.00000000 |
| 公共管理学 | 519 / 520 | 519 | 2 / 2 | 1.00000000 |
| 多模态机器学习 | 618 / 619 | 618 | 7 / 7 | 1.00000000 |
| 储能学术英语实践 | 733 / 735 | 733 | 3 / 3 | 1.00000000 |
| 高等传热学 | 494 / 495 | 494 | 3 / 3 | 1.00000000 |
| 综合岩石学 | 577 / 577 | 576 | 3 / 3 | 1.00000000 |
| 科学的历程 | 802 / 802 | 802 | 1 / 1 | 1.00000000 |
| 线性代数与解析几何II | 514 / 514 | 514 | 3 / 3 | 1.00000000 |

对齐键为最终输出的 (bg, ed)，并非全部原始 VAD 段。ARI 比较分区、忽略角色名称的排列；没有人工说话人真值，ARI=1 只证明该集合上的一致性。Qwen 转写内容及筛选并非逐字稳定：FP32 / FP16 共同输出片段文本分别为 4085/4428、4093/4430 完全一致，因此不能把所有输出差异归因于本地后端。

高等传热学固定当前 PyTorch 的 494 个输出段，排除 Qwen 调用和筛选变化，保持聚类阈值 0.3：

| 运行器 | 簇数 | 对当前 PT FP16 ARI | 对纯 PT FP32 ARI |
| --- | --- | --- | --- |
| pytorch | 3 | 1.00000000 | 0.79837069 |
| pytorch_fp32 | 2 | 0.79837069 | 1.00000000 |
| fp32 | 2 | 0.79837069 | 1.00000000 |
| fp16 | 3 | 1.00000000 | 0.79837069 |

纯 PyTorch FP32 禁用 TF32，和 TensorRT FP32 得到同一划分；两种 FP16 同为 3 簇。FP16 engine 在 0.299 阈值变为 2 簇、0.2995～0.301 为 3 簇，说明样本靠近聚类边界。本次不改阈值、标签或精度使数字强行一致；相似度矩阵和逐段标签见 [cam-course-5-fixed-clusters.json](/home/zhangs/workspace/asr_nvidia/test/tensorrt-backends-20261102/cam-course-5-fixed-clusters.json)、[cam-course-5-similarity.npz](/home/zhangs/workspace/asr_nvidia/test/tensorrt-backends-20261102/cam-course-5-similarity.npz)。

## 8. emotion2vec 概率、标签与边界样本

完整网络包含归一化、特征提取、时间池化、9 类 head、去两个 unuse 类后的 7 类 Softmax；与真实 FunASR generate 返回的标签顺序/分数比较，未用中间 embedding 代替分类。

| 样本集 | N | FP32 同标签 | FP32 最大概率绝对差 | FP16 同标签 | FP16 最大概率绝对差 |
| --- | --- | --- | --- | --- | --- |
| 原始固定裁剪 | 27 | 27/27 | 0.00230223 | 27/27 | 0.01602605 |
| 八门课扩展裁剪 | 48 | 48/48 | 0.00483936 | 48/48 | 0.01992241 |
| 刻意选取的标签差异边界片段 | 11 | 8/11 | 0.00784156 | 2/11 | 0.03974044 |

11 个边界片段来自整课比较发现的差异，并非随机抽样；最小 top1-top2 概率间隔为 0.00067964（约 0.068 个百分点），本来就容易翻转。FP16 边界最大绝对概率差 0.03974044 约等于 3.974 个百分点，修正了仅以原 27 个裁剪得出“最大约 1.60 个百分点”的不完整结论。逐段分数差/原始类别/翻转标签见 [emotion-boundary-validation.json](/home/zhangs/workspace/asr_nvidia/test/tensorrt-backends-20261102/emotion-boundary-validation.json)，原始课堂与时间戳见 [emotion-boundary-samples.json](/home/zhangs/workspace/asr_nvidia/test/tensorrt-backends-20261102/emotion-boundary-samples.json)。

| 后端 | 共同输出段 | 业务情绪差异数 | 全部段一致率 | 1～10 秒子集段数 | 子集一致率 |
| --- | --- | --- | --- | --- | --- |
| TensorRT FP32 | 4428 | 3 | 99.93225% | 3948 | 99.92401% |
| TensorRT FP16 | 4430 | 9 | 99.79684% | 3948 | 99.77204% |

全部段含时长快捷分支，不应只报这个较高一致率。无人工情绪真值，这些是与现有系统的一致性，不是准确率。已保留每个差异的课堂时间戳和业务标签，不新增阈值、低置信度回退或静默 PyTorch 补算。

## 9. 回滚、故障、取消与测试

| 真实新进程用例 | 结果 | 首次请求错误码 |
| --- | --- | --- |
| both-trt | 通过 | 成功 |
| rollback-cam | 通过 | 成功 |
| rollback-emo | 通过 | 成功 |
| both-torch | 通过 | 成功 |
| speech_campplus-missing | 通过 | 4015 |
| speech_campplus-corrupt | 通过 | 4015 |
| speech_campplus-incompatible | 通过 | 4015 |
| emotion2vec-missing | 通过 | 4016 |
| emotion2vec-corrupt | 通过 | 4016 |
| emotion2vec-incompatible | 通过 | 4016 |

每个用例独立启动 main:app 临时服务，包含 CAM++ 3 实例和 emotion2vec 1 实例，使用已通过 VAD 的课堂短片段。故障仅修改临时模型目录中的 manifest/engine，原始权重和正式 engine 保持不变。六个故障用例随后都单独请求另一项能力并成功，状态故障实例不可用、无自动 PyTorch 回退。四个组合验证实际 effective_backend，默认两个开关 false 的完整课堂组另已完成。原始结果及每次重启配置/响应/状态见 [rollback/summary.json](/home/zhangs/workspace/asr_nvidia/test/tensorrt-backends-20261102/rollback/summary.json)。

初次 HTTP 冒烟误用了只适合网络预热的静音片段，得到明确 4008；随后改用已通过 VAD 的真实片段完成上述用例。初次记录保留于 [rollback-silence-attempt](/home/zhangs/workspace/asr_nvidia/test/tensorrt-backends-20261102/rollback-silence-attempt)，未当作成功证据。

相关单元/集成回归 **240 项通过，21.122 秒**，包括配置类型、四种后端开关组合、NPU 不查询 CUDA、依赖隔离、缺失/损坏/不兼容资产、部分实例失败、全部不可用、能力错误、A/B 准入、FIFO、取消、加载取消和关闭排空。真实 CUDA 延迟工作+两次取消验证：设备未完成前模型租约和请求不返回，完成后释放并关闭池。证据见 [unit-tests-final.log](/home/zhangs/workspace/asr_nvidia/test/tensorrt-backends-20261102/unit-tests-final.log)、[cuda-cancellation.json](/home/zhangs/workspace/asr_nvidia/test/tensorrt-backends-20261102/cuda-cancellation.json)。

最终资产完整性检查（默认关闭、FP32 开启、FP16 开启）和 OpenSpec 严格校验均通过，相关模块/工具语法编译与 git diff --check 通过。临时服务 18084/18085/18086 已退出；测试生成的 asr_stats.json 已在核对初始/HEAD 相同后恢复，仅保留测试统计副本。证据见 [delivery-check-final.json](/home/zhangs/workspace/asr_nvidia/test/tensorrt-backends-20261102/delivery-check-final.json)、[openspec-validation.log](/home/zhangs/workspace/asr_nvidia/test/tensorrt-backends-20261102/openspec-validation.log)。

原生网络 CPU 加载与 FunASR FP32 的最大差分别为 CAM++ 3.50e-6、emotion2vec 2.00e-5，验证了 NPU 运行器所需的权重映射和计算路径，不替代 NPU 算子支持或硬件验收。

## 10. 资产与复算

两个原始模型目录均保留权重及 tensorrt/{fp32,fp16}/engine.plan、manifest.json。四个候选 engine 沿用已验证字节，正式 manifest 补齐模型来源、导出源快照、SHA256、I/O/profile、时间和环境；原实验未记录精确构建起点，以旧 engine mtime 标为代理来源，未伪造构建时间。

| 模型 / 精度 | engine SHA256 |
| --- | --- |
| speech_campplus_sv_zh_en_16k-common_advanced / fp16 | `75cb3dc42def285fe132d6de21d5d55de354d9439f56aa962645bdc0e28a6a60` |
| speech_campplus_sv_zh_en_16k-common_advanced / fp32 | `282c67dfd6a4df77f3f7e02b7e3a71b310e358a97c3d54c6718fdf216bd5d7ab` |
| emotion2vec_plus_large / fp16 | `002ed8ffbe46e126c08532068c8752dea2e68806743d77c3feffa8304332e56b` |
| emotion2vec_plus_large / fp32 | `b2b22c67aa6c29ec2f327513b5c2fd910214a387ac55aa5ab8c71a0a1432b827` |

原始 engine/权重哈希须与最终 implementation-manifest 一致；正式 manifest 已有意更新，旧版保留 legacy-manifest.json。模型文件被仓库规则忽略，不会自动进入 Git；交付/部署须同步对应模型目录。加载严格匹配实际 GPU 架构及 TensorRT/CUDA 版本，迁移到不同环境应重建再验证。

以下命令在仓库根目录执行，需先准备文档所列 ASR / TensorRT 环境；完整测试会向同卡 Qwen 发送请求，应在受控时段运行。`--output` 使用不存在的新目录以保留旧结果。

```bash
python scripts/check_tensorrt_assets.py --config config.toml
python -m unittest discover -s tests
python scripts/bench_tensorrt_backends.py \
  --baseline test/tensorrt-backends-20261102/baseline.json \
  --output /tmp/asr-backends-full-new --port 18084
python scripts/validate_backend_admission.py \
  --full-run test/tensorrt-backends-20261102/full-audio \
  --baseline test/tensorrt-backends-20261102/baseline.json \
  --output /tmp/asr-backends-admission-new --port 18085
python scripts/summarize_tensorrt_backends.py --evidence test/tensorrt-backends-20261102
python scripts/validate_local_tensorrt.py \
  --samples test/tensorrt-backends-20261102/emotion-boundary-samples.json \
  --output /tmp/asr-emotion-boundary-new.json
python scripts/check_cam_cluster_boundary.py --evidence test/tensorrt-backends-20261102 --course 5
python scripts/bench_local_model_latency.py \
  --samples test/tensorrt-backends-20261102/classroom-samples.json \
  --output /tmp/asr-single-segment-new.json
openspec validate 2026-10-29-fireredvad-cam-emo-tensorrt-backends --strict
```

回滚工具 `scripts/validate_backend_rollback.py --evidence <证据根目录>` 使用该根目录的 admission/samples.json 和 config.toml 创建新 rollback/，已有目录时拒绝覆盖；另选证据副本或保留旧目录后重跑。正式开关、目标环境兼容范围及状态字段详见 [TensorRT 使用说明](TENSORRT.md)。本次未提交、推送、部署或归档 change。
