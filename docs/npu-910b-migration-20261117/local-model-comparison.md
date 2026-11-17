# CAM++ / emotion2vec / VAD 本地模型比较

正式主链继续采用 **CAM++ 与 emotion2vec 的 eager FP32**。本次两个来源、六个互不重叠片段的 CPU FP32 数值参考均通过；CAM BF16/FP16 和 emo BF16 未全部满足预登记门槛。emo FP16 六片数值通过，但本次稳态中位延迟为 20.399 ms，FP32 为 18.465 ms，没有观察到选择 FP16 的速度收益。TorchAir 六个编译候选均失败，未得到可选的整网图路径。

这是 2026-11-17 在 `xjtu910b7`、物理 `npu:7` 上的实机执行结果。**参考为原始 checkpoint 的 CPU FP32，没有 CUDA golden；不是 NVIDIA/NPU 性能对比，也不是人工说话人或情绪质量验收。** 本报告给出模型后端与精度的取舍依据。

## 原始证据、命令和统计口径

本轮 E/T 是串行的小模型候选实验，没有并发执行 Qwen BF16 from_pretrained。后续另行复现的进程全局 default-dtype 竞态，已通过共享加载 guard、native statecopy前FP32存储和实际参数/浮点缓冲校验修复；同一guard也保护CPU Kaldi fbank/Qwen processor与emo实例ALiBi factory/cache的默认dtype临时常量，保留原数学、输出device/dtype及全局锁外的设备forward。这不改本表已有串行原始记录，也不将它当作冷A+B并发加载/前端的通过证据。相关真实CPU数学与线程回归见覆盖矩阵。修复后冷并发另有实机`e2e-cold-mixed-i2-fixed`：A27.563秒、全增强B34.476秒，两请求成功并排空，CAM/emo实际FP32、HF32禁用，Qwen BF16，两NPU7 VAD及两CPU VAD全部正常exit0，parent exit0/端口释放且无强制回收。该轮临时Qwen双实例用于验收冷并发；最终宿主/容器配置均选择单Qwen副本、`segment_concurrency=request_concurrency=2`，不因该冷并发用例改为双副本。

后文 `E:L17` 表示 E 文件的第 17 个物理 JSONL 行，行号从 1 开始；区间包含端点。文件中的 raw embedding、七类概率、比较结果、时延、dtype/device 和共享卡快照均保留。各表是从这些记录派生，未修改失败数据。

| 代号 | 远程原始文件 | SHA256 | 大小 / 行数 |
| --- | --- | --- | --- |
| E | `/tmp/asr-apply-20261117-a99trpfv/model-validation-final-eager.jsonl` | `9b181283b6f7a9a1e54466b97cd30b03e1667b66e8067941e2c2ffb4572e1784` | 448445 bytes / 269 行 |
| T | `/tmp/asr-apply-20261117-a99trpfv/model-validation-torchair.jsonl` | `19288b46f0cb73efb1f674175fe591cd972b7b8e7a28279678dabb642dd0d618` | 338130 bytes / 179 行 |

E:L1 / T:L1 在执行前登记容差，E:L8 / T:L5 记录 `torch=2.5.1`、`torch_npu=2.5.1`，且 `conv_allow_hf32=false`、`matmul_allow_hf32=false`。EAGER 使用现有 `asr` 环境，TorchAir 使用独立临时前缀，没有原地替换生产环境。模型参数实际在 `npu:7`，精度是参数与输入显式转为对应 dtype，不能把本次全 BF16/FP16 候选的结果外推为所有混合精度策略的结论。emo 有 28 个实例 GELU(none) 经精确 GELU V2 适配，CPU 仍保持原 F.gelu(none)；CAM 使用项目内等价 segment pooling。实例与实际精度记录见 E:L16/L54/L92/L137/L174/L211。

原始已执行命令如下。复跑前须将 `--output` 和 stdout 日志目标改为新路径，保留本文 SHA 所标识的原件；以下保留原始路径用于追溯，不作为覆盖原件的操作指令。

```bash
cd /home/zhangs/asr_nvidia
set +u
. /usr/local/Ascend/ascend-toolkit/set_env.sh
/home/zhangs/conda-envs/asr/bin/python -u scripts/validate_npu_models.py \
  --sample-id audio-6156a4b38b576557c2be \
  --sample-id audio-cee89c0a4814d7e72f15 \
  --offset 60 --durations 1 3 5 --repeat 3 \
  --precisions fp32 bf16 fp16 --compilers eager \
  --output /tmp/asr-apply-20261117-a99trpfv/model-validation-final-eager.jsonl \
  > /tmp/asr-apply-20261117-a99trpfv/model-validation-final-eager.log 2>&1

/tmp/asr-rebuild-20261117-torchair/bin/python -u scripts/validate_npu_models.py \
  --sample-id audio-6156a4b38b576557c2be \
  --offset 60 --durations 1 3 5 --repeat 3 \
  --precisions fp32 bf16 fp16 --compilers eager torchair \
  --output /tmp/asr-apply-20261117-a99trpfv/model-validation-torchair.jsonl \
  > /tmp/asr-apply-20261117-a99trpfv/model-validation-torchair.log 2>&1
```

计时字段是 **CPU frontend + 传输 + 前向 + 显式 NPU 同步 + CPU 结果读取**，不是纯 kernel 时间，也不包含 `model_load`。每个原始片段做一次首推 `repeat=0`、三次后续测量 `repeat=1/2/3`；最后回访第一个片段，再做同样四次。稳态表只汇总六片的后续三次，排除首推与回访，因此每精度 18 个观测、每长度 6 个观测。数值 gate 按六个不同片段计通过数，不把重复或回访当独立样本。CPU reference 与 VAD 每片仅一次，不是相同重复轮次的稳态 benchmark。

`model_load` 是构建/加载/移动模型以及创建 torch.compile 包装的耗时；TorchAir 真正编译发生于首个 forward，因此不能把它当作冷编译耗时。Eager 首推也可能含底层算子冷启动/缓存成本。以下统计不把这些成本混入稳态。

E:L16 等快照记录卡 7 既有 `uvicorn` PID 1601397、约 11875 MB 的工具报告占用；设备聚合 HBM 还含运行时与其他分配。测试是共享卡顺序实验，不是独占卡或受控因果干扰对比。不能用卡级 HBM 差值冒充单模型 footprint、低精度节省量或峰值；本日志也没有单独归因的模型 allocated/reserved 峰值。

## 音频与预登记数值门槛

S1 是 `audio-6156a4b38b576557c2be`（多模态机器学习课堂），S2 是 `audio-cee89c0a4814d7e72f15`（金匮中医中文）。两者源 SHA 不同；每个来源使用 [60,61]、[62,65]、[66,71] 秒的互不重叠片段。路径使用原始 `path_hex` 只读打开，不改 GBK 文件名，不重采样、不补零截短。源 SHA 来自清单，日志中 `source_sha256_verified=null`，本次未重新完整 hash 源文件；PCM 片段各有独立 SHA。来源/参考没有人工标签。原始录音来源数是 2，六个片段及重复/回访不能算作更多独立课堂来源。

| 样本 | 源 SHA256（declared） | 原始记录 |
| --- | --- | --- |
| S1 | `00190f95f88dd6ecc8dde1af62c338a6c8753b9f5832e044fa9d23d57a6dd801` | E:L2–L4 |
| S2 | `b3f93e7aa4c7c11ae3353e8b3b8610468feea5cbde1f8d030d5ea09ccc1bad0b` | E:L5–L7 |

| 片段 | offset / duration（秒） | PCM SHA256 | 记录 |
| --- | --- | --- | --- |
| S1-1s | 60 / 1 | `deca744374d0344d521cf6bef6a403544c5cf659c0c496c820499852ad0d03b1` | E:L2 |
| S1-3s | 62 / 3 | `340e39f65835bb085f28ea2dc9db00c1ed8222bf56b5c871c05997207668290a` | E:L3 |
| S1-5s | 66 / 5 | `762c3451b37215286b00482a7016beb5aa2cb84bea545d304b94b4813ded5128` | E:L4 |
| S2-1s | 60 / 1 | `fc2bddc971858da8053f525ae7099b5c96a2683125599300ffe798e0355f105b` | E:L5 |
| S2-3s | 62 / 3 | `1889272c0ba512e4291820df477259dabf8d87c567fd7421539e045076c52190` | E:L6 |
| S2-5s | 66 / 5 | `26fa86404473de81dab0649863b6734e99061b3e8a8ca31d6aaaf94ac2e76356` | E:L7 |

| dtype | CAM 余弦下限 | CAM embedding 最大绝对误差上限 | emo 七类概率最大绝对误差上限 |
| --- | --- | --- | --- |
| FP32 | 0.9999 | 0.001 | 0.001 |
| BF16 | 0.999 | 0.03 | 0.02 |
| FP16 | 0.9999 | 0.01 | 0.01 |

此外 emo 必须七标签同序且 Top1 与 CPU 参考相同；CAM 需复验生产平均链接聚类、阈值 0.3 的 partition。上述条件取自 E:L1，不因本次失败而放宽。概率和会记录但不是另一个临时添加的 gate。

## Eager 数值与业务相关输出

| 模型 | dtype | 不同片段通过数 | 全部重复/回访中的最小 cosine | 最大绝对误差 | candidate_summary / 原始比较 |
| --- | --- | --- | --- | --- | --- |
| CAM++ | FP32 | 6/6 | 0.999999999997 | 5.36442e-06 | E:L53；E:L17–L50 |
| CAM++ | BF16 | 0/6 | 0.999113194576 | 0.0815699 | E:L91；E:L55–L88 |
| CAM++ | FP16 | 5/6 | 0.999991941296 | 0.0103995 | E:L129；E:L93–L126 |
| emo | FP32 | 6/6 | 0.999999999981 | 4.90993e-06 | E:L173；E:L138–L171 |
| emo | BF16 | 5/6 | 0.998292255047 | 0.0280496 | E:L210；E:L175–L208 |
| emo | FP16 | 6/6 | 0.999995954424 | 0.00148398 | E:L247；E:L212–L245 |

CAM 三个 dtype 都输出 192 维 embedding，六片在 0.3 下均得到与 CPU 相同的两个簇及相同 partition：S1 三片归一簇、S2 三片归另一簇（E:L52/L90/L128）。这只是固定片段集的参考一致性，没有人工“两个说话人”标签。BF16 的 cosine 均超过其下限，但最大绝对误差六片都超 0.03；FP16 只有 S1-1s 超门槛，`0.0103995204 > 0.01`（E:L93），其余五片通过。聚类碰巧一致不能抵消已失败的数值门槛。

emo 七标签顺序为 `生气/angry`、`恐惧/fearful`、`开心/happy`、`中立/neutral`、`其他/other`、`吃惊/surprised`、`<unk>`（E:L131）。所有 dtype 的七标签序列一致；BF16 的 S1-3s 概率 maxabs=0.028049618 超 0.02，并把参考 Top1 `<unk>` 改为 `中立/neutral`（E:L132/L180），因此失败。这里的 Top1 是原始分类头结果，不推断人工情绪或下游中文标签准确率。

FP16 emo 的六片明细如下；它通过的是自己的 0.01 门槛，不能冒充 FP32 的 0.001 误差等级。

| 片段 | FP16 概率 maxabs | 七标签同序 / Top1 一致 | gate | 原始记录 |
| --- | --- | --- | --- | --- |
| S1-1s | 8.71485e-05 | true / true | 通过 | E:L212 |
| S1-3s | 0.00148398 | true / true | 通过 | E:L217 |
| S1-5s | 0.000198543 | true / true | 通过 | E:L222 |
| S2-1s | 0.000107944 | true / true | 通过 | E:L227 |
| S2-3s | 5.17265e-18 | true / true | 通过 | E:L232 |
| S2-5s | 8.82895e-14 | true / true | 通过 | E:L237 |

FP32 七概率和范围约 [0.9999999683, 1.0000001004]，FP16 约 [0.9999051094, 1.0001175404]（E:L138–L171 / E:L212–L245）；没有为让候选通过而事后重归一化。S2 后两片 CPU 参考已很接近单类饱和，只能说明这些片段的数值一致，不能说明情绪分布覆盖充分。

E:L269 的 `baseline_passed=true` 表示必需 eager FP32 与本次 VAD 参考 gate 通过；**并不表示所有低精度候选通过**。失败候选的 false 状态原样保留。

## 首推、稳态与长度切换

单位均为 ms。“首推”是该模型/dtype 本轮第一次 forward；“稳态”是前述六片、每片三次后续测量的总体中位数，不能与首推混算。

| 模型 / dtype | model_load | 第一次 forward | 后续 18 次中位数 | 后续 min–max | 回访 S1-1s 首次 / 后续三次中位数 | 原始范围 |
| --- | --- | --- | --- | --- | --- | --- |
| CAM++ / FP32 | 862.660 | 4007.313 | 79.772 | 63.844–91.691 | 66.043 / 63.123 | load E:L16；forward E:L17–L50 |
| CAM++ / BF16 | 778.557 | 3019.428 | 80.028 | 61.747–88.134 | 61.235 / 63.811 | load E:L54；forward E:L55–L88 |
| CAM++ / FP16 | 769.086 | 6413.130 | 98.047 | 76.359–121.734 | 77.707 / 77.383 | load E:L92；forward E:L93–L126 |
| emo / FP32 | 2721.688 | 1372.497 | 18.465 | 17.477–22.357 | 18.330 / 18.515 | load E:L137；forward E:L138–L171 |
| emo / BF16 | 2979.273 | 1027.808 | 19.942 | 18.709–21.635 | 19.986 / 19.599 | load E:L174；forward E:L175–L208 |
| emo / FP16 | 2948.320 | 1814.960 | 20.399 | 19.211–27.289 | 21.086 / 21.143 | load E:L211；forward E:L212–L245 |

按长度汇总，每格为两个来源各三次后续测量的中位数（n=6），排除首推与回访；原始范围与上表一致。

| 模型 / dtype | 1s | 3s | 5s |
| --- | --- | --- | --- |
| CAM++ / FP32 | 65.019 | 83.606 | 79.772 |
| CAM++ / BF16 | 63.385 | 84.038 | 80.382 |
| CAM++ / FP16 | 77.859 | 98.047 | 106.290 |
| emo / FP32 | 18.932 | 18.098 | 18.271 |
| emo / BF16 | 20.125 | 19.829 | 19.459 |
| emo / FP16 | 21.146 | 19.940 | 20.375 |

每个 dtype 都实际走过 1→3→5 秒、第二来源 1→3→5 秒，最后回访第一个 1 秒片段。Eager 的 `new_graphs` 全为 0，回访没有重现数秒级首推开销；这证明当前 eager 路径的该次长度回访正常，不是 TorchAir 动态图/重编译成功证据，也不能排除 CANN 底层按形状/dtype 的算子缓存成本。

存在 NPU 首推慢于 CPU 单次参考的具体例子：S1-1s CAM CPU reference 为 878.435 ms（E:L10），NPU FP32 第一次 forward 为 4007.313 ms（E:L17），而相同 NPU 片段后续三次中位数为 64.632 ms（E:L18–L20）。CPU reference 只有一次且加载/缓存顺序不同，不能据此判 CPU 整体更快。六片 CAM CPU reference 范围 807.150–4147.955 ms、emo 为 2426.450–10123.761 ms（E:L10–L15 / E:L131–L136）；本轮暖 NPU 更短，但不以单次 CPU 参考给出稳定加速比。

**emo FP16 目前不值得替换默认 FP32**：六片数值合格，但总体稳态中位数 20.399 / 18.465 − 1 ≈ **+10.5%**，三个长度也都没有观测到降低延迟。实验只有两个来源、三次重复和共享卡背景，不能给出统计显著性或所有负载的永久排序；在没有已观察收益、误差更大且要维护额外精度路径的情况下，保留 eager FP32 更合理。CAM FP16 既有一片失败、稳态也未见收益；BF16 不能越过 gate 只凭吞吐或簇数被采用。

## 隔离 TorchAir 的具体结果

独立前缀 T 的 eager FP32 对照通过（CAM T:L32、emo T:L113，文件基线 T:L179 为 true），说明本次环境不是连 eager 正确性都未建立。TorchAir 使用 `torch.compile(..., dynamic=True, fullgraph=True)`，六个“模型×dtype”编译候选都在第一次 forward 失败，没有任何 compiled candidate 成功结果。

| 模型 / dtype | 包装/加载 ms | 首个失败记录 | 具体失败 |
| --- | --- | --- | --- |
| CAM++ / FP32 | 1434.648（T:L79） | T:L80 | `E40021 / E80028`；`BNInfer` 的 scale 通道维 `-2` 不等于 input_x 通道 `192` |
| CAM++ / BF16 | 767.104（T:L82） | T:L83 | `E40021 / E80028`；`BNInfer` 的 scale 通道维 `-2` 不等于 input_x 通道 `192` |
| CAM++ / FP16 | 751.908（T:L85） | T:L86 | `E40021 / E80028`；`BNInfer` 的 scale 通道维 `-2` 不等于 input_x 通道 `192` |
| emo / FP32 | 2511.440（T:L158） | T:L159 | `Unsupported`：`OmegaConf DictConfig.__getattr__` 位于 Dynamo skipfiles，首次在 `network.cfg.normalize` 读取处拒绝入图 |
| emo / BF16 | 2734.583（T:L161） | T:L162 | `Unsupported`：`OmegaConf DictConfig.__getattr__` 位于 Dynamo skipfiles，首次在 `network.cfg.normalize` 读取处拒绝入图 |
| emo / FP16 | 2636.937（T:L164） | T:L165 | `Unsupported`：`OmegaConf DictConfig.__getattr__` 位于 Dynamo skipfiles，首次在 `network.cfg.normalize` 读取处拒绝入图 |

CAM 错误发生于 CANN 8.1.RC1 `dynamic/bn_infer.py`，节点分别是 `BNInfer_121/_243/_365`，GE pre-compile 失败，最终含 `ERR03005 GRAPH internal error`。它反映本轮动态图通道/scale 形状链条未能编译，不能外推为“910B 永远不支持 CAM 图执行”。emo 则尚在 Python 配置对象 tracing 阶段，不能归因于已经证明某个 NPU attention/GELU 算子不可用。

虽然 T 计划同样包含 1/3/5 秒，compiled 变体在第一个 1 秒片段就抛错，后续 3/5 秒及回访没有运行，**动态长度、重编译、compiled 稳态时延均未验证**。`candidate_error` 没有失败 forward 的 elapsed 字段，因此不能从 JSONL 给出冷编译失败耗时；上表的 model_load 只是加载/包装，不能冒称冷编译成本。后续若继续该候选，应另记从第一次调用开始至异常的耗时、实际形状 guard 和图编译数，修配置 tracing/固定通道形状后重跑，不启用 suppress_errors 静默回退冒充图成功。本次选择不把失败的 TorchAir 或未验证 ONNX→ATC→OM 纳入正式执行链，也不永久排除将来有收益的格式转换方案。

## VAD 三路参考、耗时及设备配比建议

E 的五项 nested kwargs 相同（E:L248/L255/L262）：`max_single_segment_time=12000`、`max_end_silence_time=800`、`speech_noise_thres=0.8`、`decibel_thres=-40`、`snr_thres=18`。这保持既有 AutoModel 的实际构造/嵌套参数语义，不借迁移修阈值或 effective 段上限。六片 AutoModel CPU、native CPU、native NPU 的段数和边界完全一致：native CPU 对照容差为 0，NPU 预登记 20 ms，实际最大边界误差为 0（E:L249–L268）。

既有有效段上限仍来自权重`model_conf.max_single_segment_time=8000`，即8秒。服务声明的12000保持原nested位置，未在迁移中改成真正生效的12秒顶层覆盖；若需要12秒应另行评审分段行为变化。上述片段最长5秒，不能用这次零边界误差单独证明8秒截断边界的完整覆盖。

以下单位 ms，每片各一路只测一次，包括读取派生 WAV/前端/前向/后处理；不是 VAD worker 稳态多轮结果，也不含进程启动。

| 片段 | AutoModel CPU | native CPU | native NPU | 原始记录 |
| --- | --- | --- | --- | --- |
| S1-1s | 432.779 | 427.459 | 348.197 | E:L249 / E:L256 / E:L263 |
| S1-3s | 598.342 | 630.260 | 42.398 | E:L250 / E:L257 / E:L264 |
| S1-5s | 601.024 | 591.044 | 47.751 | E:L251 / E:L258 / E:L265 |
| S2-1s | 427.193 | 384.605 | 20.173 | E:L252 / E:L259 / E:L266 |
| S2-3s | 594.874 | 628.489 | 39.538 | E:L253 / E:L260 / E:L267 |
| S2-5s | 611.300 | 715.804 | 64.340 | E:L254 / E:L261 / E:L268 |

六次跨片中位数分别为 **596.608 / 609.767 / 45.075 ms**（Auto CPU / native CPU / native NPU）。它混合不同长度、含首片冷成本，只是便于索引的描述统计，不能当正式稳态加速比。隔离前缀 T 的三个 S1 片段也同方向：Auto CPU 为 432.963/643.069/636.972 ms（T:L168–L170），native CPU 为 461.080/562.896/579.701 ms（T:L172–L174），native NPU 为 366.705/48.248/56.186 ms（T:L176–L178），边界同样一致。

**本次 VAD 记录中没有 NPU 比两条 CPU 路径更慢的片段。** 首个 NPU 片段 348.197 ms 显示冷成本，后续为 20.173–64.340 ms；不能据“FSMN 小、CPU 本来快”的经验直接把设备配比改为偏 CPU。当前 2 CPU + 2 NPU 只是初始档位，也没有因此被证明最优。

值得追加完整录音 VAD-only 对照，再决定配比：

1. 对同一份完整原始录音、同参数与固定 CPU 线程数，分别单 worker 跑 AutoModel CPU、native CPU、native NPU；独立进程/明确预热，至少重复并保留逐次墙钟、RTF、完整 segment 列表/边界与错误，而不是只比较几秒片段。
2. 记录 CPU、RSS、进程 HBM 与 NPU 侧 allocated/reserved、共享卡占用，观察长音频前端/状态机和推理的比例；不将冷加载/进程启动混入暖推理。
3. 再在相同 Qwen 实例数/并发与样本下，比较 VAD 配比对完整请求吞吐、VAD 等待和 Qwen 等待的影响。NPU-only 的局部速度优势可能与同卡 Qwen 争用相抵；CPU-only 也可能因前端/线程争用更差，需要端到端证据。

这符合 design 的 Open Questions：VAD 的 CPU/NPU 配比须实测决定，可能多数 CPU + 少量 NPU；设备数组规格接受 `cpu` 与 `npu:N`，且要求配比有实机依据。任务 4.4 的 NPU 配置迁移意图不等于无条件将所有 VAD worker 放 NPU；经验证后显式配置 CPU 不是静默回退。若最后偏 CPU，应同步记录依据及任务/模板的准确表述。**本报告不改配置/源码，不把 2+2 或偏 CPU 预先宣布为最优。**

## 复算与结论范围

下列最小片段可复算主数值表与稳态统计；还应先以表头 SHA256 核对原件。回访只进入最坏误差统计，稳态时明确排除。

```python
import json, statistics
rows = [json.loads(line) for line in open("model-validation-final-eager.jsonl")]
for model in ("speech_campplus", "emotion2vec"):
    for dtype in ("fp32", "bf16", "fp16"):
        all_rows = [r for r in rows if r["type"] == "candidate"
                    and (r["model"], r["compiler"], r["precision"]) == (model, "eager", dtype)]
        originals = [r for r in all_rows if not r["revisited_first_shape"]]
        unique = [r for r in originals if r["repeat"] == 0]
        steady = [r["elapsed_ms"] for r in originals if r["repeat"] > 0]
        print(model, dtype, sum(r["comparison"]["passed"] for r in unique), len(unique),
              min(r["comparison"]["cosine"] for r in all_rows),
              max(r["comparison"]["max_abs"] for r in all_rows),
              statistics.median(steady))
```

结论仅支持：保持原始 checkpoint + torch_npu eager FP32，保留已验证的 CAM pooling、进程固定禁 HF32 和 emo 精确 GELU 语义适配；暂不启用本次失败或没有已观察收益的低精度/编译候选。它没有证明所有硬件/长度/多实例下的全局最优，也没有证明人工聚类或情绪质量。模型权重/revision 与执行时完整环境锁应结合迁移总报告；本 JSONL 未内嵌每个权重文件 SHA、执行源码 commit 或完整 TorchAir 包构建 hash，不能用本表虚构这些信息。后续长音频、同卡多模型干扰与实例数标定单独报告。
