# 完整课堂录音：阈值与输出复核

本次完整MP3返回**349个最终分段、12070个顶层文本字符**，请求耗时725.203秒、RTF=0.299364。时间边界检查通过，复读修复告警、修后超长段与文本形状异常均为0。当前证据支持保留说话人阈值0.3、复读保护40字符/秒作为迁移基线，**不足以证明阈值最优或识别/说话人/情绪准确率**。本报告没有人工标注，也没有CUDA golden。

前半部分是**一条补充完整录音**，后文另列八份独立完整录音g1；没有用这条MP3充作八份。g2同八源已完成并正常关闭，后文分别列明。输出/日志统计仅用Python标准库只读既有结果；为排查F3时间缺口，另有明确标注的纯CPU原始checkpoint与WAV参考。全程未使用NPU、未启动接口，未修改生产代码、阈值、原始音频或权重；自有临时输入副本已清理。

## 补充MP3：输入与执行范围

| 项目 | 实际记录 |
| --- | --- |
| 原始目录 | `/tmp/asr-apply-20261117-a99trpfv/e2e-long-mp3` |
| sample_id | `audio-37a1c322625a308c612b` |
| 原始音频SHA256 | `706d93e891b352d330b55cc7e55bf9f12c2673d59d5e26d73ff1d1316d17a894` |
| 完整音频时长 | 2422.476秒（40分22.476秒） |
| HTTP / 开关 | 200；`showSpk=true`、`showEmotion=true`；未请求teacher/student语义识别 |
| 请求耗时 / RTF | 725.2029859600589秒 / 0.29936436355202645 |
| 门控及设备 | 单Qwen副本，segment/request 2/2；物理npu:7共享卡；VAD为2CPU+2NPU |
| 实测并行峰值 | 此条B请求峰值1、音频总峰值1；不是八路容量证明 |

`summary.json`的单条P50/P95都等于本次耗时，只有一个观测值，不能解释成一般尾延迟。原始MP3完整上传，source SHA、完整时长与请求结果来自`full-samples.json`、`summary.json`和`request-full-0.json`；本次统计没有重新解码或改写音频。

## 补充MP3：最终输出及时间轴

| 指标 | 结果 |
| --- | --- |
| 最终响应segments | 349 |
| 顶层text字符数 / 去空白字符数 | 12070 / 11720 |
| 各段strip后字符合计 | 11722（与顶层统计口径不同；顶层还含段间分隔） |
| 段字符数min / P50 / P95 / max | 2 / 37 / 50 / 62 |
| language / language_ratio | Chinese / `{"zh":100.0}` |
| 起点最小值 / 终点最大值 | 0.38 / 2419.00秒；均在完整2422.476秒内 |
| 段时长min / P50 / P95 / max | 0.74 / 8.01 / 8.01 / 8.01秒 |
| 两位小数字符串、正时长、递增起点 | 349段均符合；非法/越界/倒序均0 |
| 相邻重叠 | 0 |
| 相邻间隔P50 / P95 / max | 0 / 1.918 / 17.83秒 |
| 输出时间区间并集 | 2270.74秒，约占源时长93.736%；这不是人工有效对话召回率 |

`application.log:L20`记录“有效段 完成=360/360 失败=0”。**360是处理阶段计数，349才是最终响应分段数**，二者不可互换。现有链路会在处理后过滤空文本、独立语气词及纯标点（`api/routes/asr_qwen_common.py:694–696、742–744`）；因此“处理完成”并不保证形成一个非空最终段。当前原始记录没有为这11个差额保留逐段过滤原因，不能凭最终payload断言各原因分别多少、也不能当作11次推理失败。

8.01秒的返回段长与原FSMN按帧计数、时间轴两位小数一致。权重`model_conf.max_single_segment_time=8000`的既有有效语义保持，服务声明的12000仍处于原nested kwargs位置；迁移没有顺手将其改成生效的12秒上限。没有人工参考时，未覆盖的151.736秒既可能是静音/过滤/空输出，也可能包含漏语音，不能把93.736%覆盖率与项目98%人工召回门槛直接比较。

## 补充MP3：说话人阈值0.3

349段均返回`spk_0`，累计分段时长2270.74秒、字符11722，只有一个返回簇、0次标签切换。该结果没有暴露标签碎片化，但**不证明录音中实际只有一人**；也不能排除阈值0.3将学生短发言合并到主讲人。短于1秒的段按原逻辑就近继承标签，不独立提取CAM embedding（`api/routes/asr_qwen_common.py:613–617、366–397`）。

算法为平均链接层次聚类，按簇间平均余弦相似度与0.3比较。此前两来源六片的严格FP32 CAM与CPU FP32参考在0.3下partition一致，支持数值迁移一致性；当前payload只保存最终标签，没有保存本条录音的embedding或人工说话人标注，不能用它重新sweep阈值、计算BCubed F1或判断漏分。

因此保留0.3。若要重新标定，应补本条或独立课堂中教师/学生切换的人工时间标注和相同分段embedding，再比较阈值及短段继承的影响；不能以“本次只输出一簇”作为调高阈值的充分依据，也不能加入只保留老师声音的规则。

## 补充MP3：情绪分布

| 返回业务标签 | 段数 | 占349段比例 |
| --- | --- | --- |
| 积极 | 13 | 3.72% |
| 平淡 | 326 | 93.41% |
| 强调 | 4 | 1.15% |
| 兴奋 | 5 | 1.43% |
| 疑问 | 1 | 0.29% |

所有最终段情绪非空。平淡占93.41%只是模型返回的分布；它不能证明该标签正确，也不足以据此重映射分类头或改阈值。课堂音质、语气、域偏差及原始七类到业务标签的映射都可能影响分布。本次没有人工情绪参考。

## 补充MP3：复读保护40字符/秒

| 观测 | 结果 |
| --- | --- |
| 最终段字符/秒P50 / P95 / P99 / max | 5.119 / 6.617 / 7.346 / 7.973 |
| 最终段超过`max(120,int(duration*40))` | 0 |
| 连续循环候选 / 相邻完全相同规范化文本 | 0 / 0 |
| 替换字符、异常控制字符、模型协议标记、空最终文本 | 0 |
| `application.log`复读修复告警 | 0 |
| 超长但未检出循环、保留原文的告警 | 0 |
| application错误签名（ERROR/Traceback/InstanceFailure/4011/4012/4015/4016） | 0 |

日志共30行，扫描了全部行；没有告警不是被省略的告警数。没有将service.log中第三方编译warning泛化成业务失败。字符统计是Python Unicode字符数，**不是token数、汉字数或词数**；基于输出`bg/ed`两位小数计算，临界门槛可能与修复时原始时长有整数舍入差异。本条离阈值较远，不受该临界问题影响。

实际门槛还有120字符下限：1秒段仍允许120字符，3秒为120，8秒为320。超门槛后需存在可折叠循环结构才修复；找不到循环保留原文并告警，不按长度直接截断。主循环规则为至少4字符/3次，折叠还检查2字符/4次、1字符/6次。统计脚本只审查候选并不改写文本，候选重复也不自动等于错误。

本条正常输出的最大7.973字符/秒低于40，但没有新出现的复读正例，**不能用零告警单独证明检测召回率或最优阈值**。现行源码注释引用68,018段旧样本、正常P99.9=20.3/峰值26.2；本轮未独立复核该历史原始数据，不把注释当作新NPU标定。建议保留40及120/4/3边界，继续保留修复前后长度和时长告警，再对真正触发的片段人工复听，防止把课堂正常重复改写掉。

## 补充MP3：资源采样与正常关闭

`status.jsonl`有104个周期快照，覆盖731.041秒。计划间隔约5秒，**实际间隔中位6.994秒、最大11.711秒**；采样开销/调度使间隔变长，离散最大值可能漏瞬时峰值。allocator只有102个可用观测，缺失值未填0。该历史记录没有`cpu_user_s/cpu_system_s`，不能补造主进程CPU均值或CPU峰值。

以下是MiB（字节/2²⁰）；三个中位数分别取时间顺序的前三分之一、中三分之一、后三分之一快照：

| 口径 | 首个可用 | 三段时间窗中位数 | 离散峰值 | 最后可用 |
| --- | --- | --- | --- | --- |
| 主进程RSS | 555.008 | 3226.602 / 3264.770 / 3295.104 | 3310.555 | 3057.055 |
| 本进程torch allocated | 26.833 | 4625.298 / 4626.310 / 4625.713 | 4648.907 | 4611.271 |
| 本进程torch reserved | 44.000 | 5294.000 / 5296.000 / 5296.000 | 5296.000 | 5296.000 |

torch累计peak allocated为4762.344MiB，累计peak reserved为5296MiB，与离散采样峰值口径不同。主进程线程的时间窗中位数均为251（峰值254，末值254），FD中位数均为133（峰值133，末值131）。allocator三窗中位数较平稳；RSS窗中位数仍逐步增加约68.5MiB，结束时回落。单条变长请求包含冷加载、算子缓存与不同形状，不能据此宣布无泄漏或直接认定泄漏，固定形状暖态增长门槛另看稳态验收记录。

另按npu-smi原报告MB口径：服务进程树HBM为250–5927MB，卡7设备聚合为15510–21137MB，AICore离散采样0–12%。既有PID `1601397`始终11875MB，单独保留，没有计成本项目内存。设备聚合、服务进程树Smi HBM、torch allocator及RSS是不同口径，不能混加；其差值也不能无证据分配给各模型。并发加载的before/after allocator delta会混入本进程其他模型分配，不能当作模型独占体积。

服务缓存最大1文件/77516844字节（约73.926MiB），末快照回到0文件/0字节；Qwen busy/waiting末值均0，VAD busy/waiting末值均0。状态采样末值仍允许模型加载常驻内存，不能把进程存活时的reserved不归零当作关闭失败。

`shutdown-after.json`记录parent exit0、无监听、无剩余后代，观察5秒，未超时/无forced child cleanup，application/worker shutdown_failed均false。四worker均exit0、forced=false、cleanup_complete=true：CPU graceful 1.759/1.841秒，NPU7为7.086/7.538秒；没有把内部超时后强制回收冒充通过。

## 补充MP3：原始证据SHA与复算

| 文件（均相对原始目录） | 字节数 | SHA256 |
| --- | --- | --- |
| `application.log` | 5742 | `ef6b3e5f4449bbaa318900a801029c4f8cf1896ec450021efe76736d5d29f2e7` |
| `asr_stats.json` | 131 | `c0c2eb70ab93818e467868a5c97fb14427a98707f6ecaaed2cf952edd31a5717` |
| `config.toml` | 4351 | `3e8c79b5f02340e415641666c500d29209e6901532b94cbdc832c99bf5cf3f80` |
| `full-samples.json` | 871 | `4ff5ed46e83aeaf5b50f3a42483e32af2ee225a32788e2179d76a321a13f0b0e` |
| `ownership.json` | 719 | `0246b77e7c77a5dd32e2fd4dd36ba37bad7ab96c5b6081f715cfdac6ceeb202d` |
| `request-full-0.json` | 141274 | `130b8c741d8842eebc2f87ac4abf6569fb5531cfca5cd05c0db8d78f3962ac49` |
| `service.log` | 35695 | `5144b242db282f253af90032ad25b34f2ccc8e5e4f23fe876cfcfbf00e659c36` |
| `shutdown-after.json` | 1896 | `4def202526ba52b5350c4f7aac146254428dbb8d4531bf661f6976c0ab0e37ba` |
| `shutdown-before.json` | 3028 | `765de7741a46b0558bb00935f4f6d7f8ab979f5d86b51757bcd18ca775acdcab` |
| `status-drained.json` | 19806 | `4a7816bffd78dcee3da556a838cd4a3f0c3a01d82f1be2a8c87b93c8b9486370` |
| `status-initial.json` | 17992 | `8d52138f49bd998c49c8b6307fd95ef3d2c9e21075e339ea13cf62fc15f72702` |
| `status.jsonl` | 696645 | `136c4bf241ab58ec20a80c7f0dac15aa6a9e55c600ae5b9ec9325f7aea9f2ca5` |
| `summary.json` | 593 | `d5dda601d5cd146a342119dfa1984bec30ec7b3a0176a7aecdd945acc89fbae5` |

原始分析JSON为`/tmp/asr-threshold-readonly-n9gyyacc/long-mp3-analysis.json`，SHA256=`75b869f3d8d57ce3808812e25fb9834cfa08f152c6a90d1fbc7750584b00a0d7`；它是原文件的派生统计，不能替代上表原件。标准库分析脚本SHA256=`8f735dda41857265a9b407f3c4315079d9a336adcd6a8d92ed49d56970fb5bb4`，实际执行副本与通用collector保留于该JSON所在目录。collector SHA256=`f09bc71f205e5002053c3b6fc154cef57256ffbffcb4f2fbe2fe5f2dd37e533b`。脚本没有读取原始音频或触发模型/NPU。

实际统计逻辑可在原服务器复算，`--output`需使用新文件名，不覆盖原件：

```bash
python3 /tmp/asr-threshold-readonly-n9gyyacc/analyze_classroom_thresholds.py \
  --directory /tmp/asr-apply-20261117-a99trpfv/e2e-long-mp3 \
  --collector /tmp/asr-threshold-readonly-n9gyyacc/collect_validation.py \
  --output /tmp/classroom-threshold-summary-recheck.json
```

资源曲线是同一`status.jsonl`按文件顺序取首/末/最值、再按快照数等分三窗取中位数；时间跨度与步长直接相减`sampled_at`。例如无需依赖第三方包即可复算RSS三窗与采样间隔：

```python
import json, statistics
from pathlib import Path
rows = [json.loads(x) for x in Path("/tmp/asr-apply-20261117-a99trpfv/e2e-long-mp3/status.jsonl").read_text().splitlines()]
times = [r["sampled_at"] for r in rows]
print(times[-1] - times[0], statistics.median(b-a for a,b in zip(times,times[1:])))
for j in range(3):
    vals = [r["owned_process"]["rss_bytes"] for i,r in enumerate(rows) if j*len(rows)/3 <= i < (j+1)*len(rows)/3]
    print(statistics.median(vals) / 2**20)
```

源码核查时`api/routes/asr_qwen_common.py` SHA256=`f3a33a0d5bc7f52e75aebd5de09f0a6e1120b3939856a976c8805f5fb5d0813b`，`utils/repetition.py`为`02be6e84270414d30be39f721876553577760c62c59aaa695bf0d9ef94c4f0ef`。本次保留原有过滤、聚类、复读和标签规则，没有据无标注统计修改业务逻辑。


## 独立八份完整录音：g1

`/tmp/asr-apply-20261117-a99trpfv/e2e-full-g1-final`为另八份原始WAV、八个不同SHA的完整输入，共22253.408秒；不是重复上传补充MP3或截取12秒窗口。g1实际配置为**单Qwen副本、segment/request 1/1**，音频准入8、B上限4；0.3/40及原VAD过滤参数保持。g2同八源2/2已完成，后文据两档实测选择1/1，不由单组外推全局最优。

八请求全部HTTP200并返回非空完整结构，未筛除失败；本批最终3109段、顶层文本合100071字符。P50=4104.338199秒、P95=4781.662453秒，实测活动峰8、B峰4。逐请求RTF包含共享后端排队，不能拿其中一个RTF当单文件独占速度。下面保留每一行：

| 编号 | 类型 | sample_id | 源时长s | 请求耗时s | RTF | 处理/最终段数 | 顶层字符 | language_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F0 | B | `audio-bbdf7c1a76f88262e22a` | 2615.808 | 3628.947 | 1.3873 | 326 / 311 | 11215 | `{"zh":100.0}` |
| F1 | B | `audio-ff1d4cae9950cf37211f` | 2702.016 | 3944.004 | 1.4597 | 360 / 351 | 12807 | `{"zh":99.8,"en":0.2}` |
| F2 | B | `audio-6fbfabcc5926e1583dab` | 2820.736 | 4264.672 | 1.5119 | 409 / 400 | 14363 | `{"zh":100.0,"yue":0.0}` |
| F3 | B | `audio-61d3d13113dbcf87833e` | 2821.056 | 2108.681 | 0.7475 | 189 / 166 | 7158 | `{"zh":28.8,"en":71.2}` |
| F4 | A | `audio-28a0c66565d3814b32af` | 2821.120 | 3873.073 | 1.3729 | 351 / 341 | 14170 | `{"en":36.3,"zh":63.7}` |
| F5 | A | `audio-b0b2abaab027fbd04b81` | 2822.560 | 4755.156 | 1.6847 | 524 / 516 | 13540 | `{"zh":100.0,"hi":0.0}` |
| F6 | A | `audio-36029767f173c95244c6` | 2823.040 | 4795.935 | 1.6989 | 549 / 535 | 13063 | `{"zh":99.9,"en":0.0,"hi":0.0}` |
| F7 | A | `audio-5d5328ae33020b071a83` | 2827.072 | 4685.418 | 1.6573 | 502 / 489 | 13755 | `{"zh":100.0}` |

处理阶段总计3210段完成、失败0，最终3109段；101段差额不能按推理失败计，既有空文本、独立语气词与纯标点处理可减少最终段数。未保留每一差额的逐项原因，不把它们强行分配成某类。language_ratio是按原文本非空白字符加权并保留一位小数，不是按时长或识别置信度；极小语种显示0.0、独立舍入后合计99.9均保持原契约，不在统计中删除字段。

### 角色与情绪：有实际输出，也有需要复听的线索

| 编号 | 返回簇数 | 各簇段数 | 标签切换/单段run | 情绪段数 |
| --- | --- | --- | --- | --- |
| F0 | 2 | spk_0=310、spk_1=1 | 2 / 1 | 强调=25、平淡=275、疑问=2、兴奋=3、积极=6 |
| F1 | 1 | spk_0=351 | 0 / 0 | 平淡=305、积极=14、强调=24、兴奋=7、疑问=1 |
| F2 | 7 | spk_0=221、spk_1=6、spk_2=7、spk_3=158、spk_4=1、spk_5=3、spk_6=4 | 17 / 4 | 兴奋=9、强调=111、平淡=195、积极=85 |
| F3 | 9 | spk_0=11、spk_1=1、spk_2=85、spk_3=16、spk_4=43、spk_5=1、spk_6=4、spk_7=4、spk_8=1 | 45 / 17 | 平淡=142、积极=21、疑问=1、强调=1、兴奋=1 |

F4–F7未请求增强，role/emotion各自全为null，这是A类契约，不能列为CAM/emo失败。不同文件的`spk_0`也不代表跨文件同一个人。

0.3下既有合并风险，也有碎片风险：F1只有一个簇，不能据此证明只有一个人；F0的唯一`spk_1`只持续1.02秒，靠近1秒embedding门槛，应优先复听。F2两大簇占379/400段，其余五簇共21段；F3为九簇、45次切换、17个单段run，既可能是真实多人问答，也可能有环境/音色变化造成拆分。没有人工说话人标注和本批完整embedding，不能把这些簇数当真值，更不能据此调高0.3。FP32参考一致性与说话人质量是两种证据。

### 时间边界与文本形状

全部3109段均为合法两位小数字符串、正时长、递增起点，越界/相邻重叠均0，各源最大段长均8.01秒。下表是输出覆盖，不是有效对话召回率：

| 编号 | 首bg–末ed秒 | 输出并集秒/源时长比例 | 最大相邻间隔s | CPS P95/max | 循环候选/相邻相同文本对 |
| --- | --- | --- | --- | --- | --- |
| F0 | 129.99–2615.79 | 1910.98 / 73.06% | 119.82 | 7.295 / 9.114 | 0 / 0 |
| F1 | 132.01–2702.00 | 2155.18 / 79.76% | 68.07 | 7.366 / 8.791 | 0 / 0 |
| F2 | 0.00–2763.92 | 2458.42 / 87.16% | 64.77 | 7.095 / 8.614 | 1 / 0 |
| F3 | 1207.77–2821.04 | 700.43 / 24.83% | 338.72 | 18.463 / 20.599 | 0 / 1 |
| F4 | 18.19–2193.85 | 2055.04 / 72.84% | 4.92 | 13.233 / 16.735 | 0 / 0 |
| F5 | 8.90–2822.54 | 2475.91 / 87.72% | 8.60 | 6.745 / 7.937 | 3 / 1 |
| F6 | 152.84–2823.02 | 2330.92 / 82.57% | 9.18 | 6.966 / 10.714 | 0 / 3 |
| F7 | 84.35–2794.31 | 2424.06 / 85.74% | 33.09 | 6.833 / 8.738 | 1 / 0 |

**优先检查过的F3大缺口另见CPU参考。** F0/F1也分别从129.99/132.01秒才有最终输出；其他长间隔同样需要结合原始输入或人工复听，而不能由“无技术异常”推出有效对话召回≥98%。

本批有5个正常长度范围内的连续重复候选，以及5对相邻规范化文本完全相同；均未超过长度门槛。它们可能是正常课堂强调，也可能是转写重复，不能无听辨标成错误。没有为清理统计而合并这些段，保持一段一响应的既有行为。文本形状异常（替换字符/异常控制符/模型协议标记/空最终文本）为0。

### 复读40：出现一个真实触发，仍不调整阈值

`application.log`共209行，L166实际记录一次复读修复：**2.0秒输入，506→6字符，长度门槛120**；告警预览为反复的短语气词，修前约253字符/秒，明显高于本批最终输出最高20.599。未检出循环而超长保留的告警0，ERROR/Traceback/InstanceFailure/4011/4012/4015/4016签名0。该告警没有唯一task_id，不能仅因相邻完成日志就将它准确归到某个文件/时间段。

这是长度保护确实触发的证据，但没有人工对齐音频，不能宣称折叠后的6字必然完整正确。所有最终输出都低于`max(120,int(40*duration))`，包括英语占比较高的F3（max20.599字符/秒）；短正常重复候选仍被保留。因此继续保留40及120字符下限较为合理，不能因为观察到5个低于门槛的候选就调低长度阈值或无条件折叠。

### F3：原始CPU VAD与一分钟能量参考

为排查首个输出1207.77秒、覆盖率24.83%的异常观感，另做了**同一完整WAV的纯CPU原始checkpoint参考**。进程使用4个intra-op线程/1个interop线程，`TORCH_DEVICE_BACKEND_AUTOLOAD=0`，模型全部参数为CPU FP32；未调用NPU、未启动VAD pool或服务。完整WAV以相同字节复制到唯一ASCII临时路径，模拟服务缓存路径，避免GBK文件名进入第三方音频loader；源SHA、大小及mtime校验未变，临时副本finally删除，PID1341402已退出。

| 层次 | 候选数 | 最早边界s | 起点<1200秒 |
| --- | --- | --- | --- |
| 原始AutoModel CPU结果 | 196 | 1207.77–1208.48 | 0 |
| 原短段过滤后 | 196 | 同上 | 0 |
| 复用生产范围/音量/SNR过滤后 | 189 | 同上 | 0 |
| g1最终响应 | 166 | 同上 | 0 |

CPU原始196、过滤189也与g1日志L57的计数相同，7个段级丢弃全部是SNR；前20分钟在原始VAD里就没有候选，不是ASR后处理把这些候选剔除了。g1最终166段的两位小数起止边界全部能在CPU过滤后189候选中找到，没有缺项。完整g1原始候选列表未保存，因此不把这项子集对照扩大为所有原始候选逐项相同。参考JSON保留全部原始结果、候选、过滤后候选及丢弃统计。`filter_stats`因跨asyncio上下文读取为空，本报告不将该空字段当作零过滤，采用实际候选数组和捕获的7条丢弃记录计数。

0–1200秒的20个一分钟窗：RMS范围−94.527至−93.946dBFS，peak最高−61.366dBFS；每窗全部10ms帧RMS都低于−60dBFS（亦低于−40），输入接近16位量化噪声。转折附近如下；完整48个一分钟窗留在原始JSON：

| 时段s | 整窗RMS dBFS | peak dBFS | 10ms帧RMS<−60dBFS |
| --- | --- | --- | --- |
| 1140–1200 | -94.099 | -76.330 | 100.00% |
| 1200–1260 | -31.966 | -0.018 | 77.47% |
| 1260–1320 | -63.271 | -35.055 | 89.75% |
| 1320–1380 | -63.688 | -29.787 | 97.70% |
| 1380–1440 | -63.365 | -29.754 | 96.15% |
| 1440–1500 | -56.688 | -21.491 | 95.52% |

这支持“输入前20分钟近静音，原CPU VAD也不切”的解释，没有观察到NPU新增的前20分钟丢失。**能量仍不是人工有效语音标签**，不能据此计算98%召回；F3后半段的较低能量、过滤和文本后处理仍需听辨。

逐参数区分声明nested值与FSMN有效选项。下表第三列均为**原权重config.yaml的model_conf明确值**，不是仅根据库默认值推断；与实际CPU实例读取值、NativeVad同YAML构造值逐一核对相等：

| 选项（原单位） | 服务传入nested值 | 权重model_conf | 原AutoModel CPU有效值 | NativeVad按原YAML构造值 |
| --- | --- | --- | --- | --- |
| max_single_segment_time（ms） | 12000 | 8000 | 8000 | 8000 |
| max_end_silence_time（ms） | 800 | 800 | 800 | 800 |
| speech_noise_thres | 0.8 | 0.6 | 0.6 | 0.6 |
| decibel_thres（FSMN帧级口径） | −40 | −100 | −100 | −100 |
| snr_thres（FSMN内部） | 18 | −100 | −100 | −100 |

`NativeVadRunner`执行`dict(cfg["model_conf"])`后`update(cfg)`，再构造原`FsmnVADStreaming`；原类用`VADXOptions(**kwargs)`建立选项。nested值没有被展开成这些构造覆盖项；即使max_end两边同为800，也不能据相等数值宣称nested已经生效。

**第二层服务段级过滤另行真实使用SNR18、RMS/低能量−40及开头/结尾各3分钟的范围规则**，不与FSMN内部−100混淆。本次实际7条SNR丢弃已经证明服务过滤调用发生；而第一层FSMN仍按其原权重选项切分。段级低能量阈值来自独立的`vad_segment_decibel_thres` getter（`core/config.py:379–380`，默认−40），由`core/vad.py:119`直接读取，**不由global `vad_decibel_thres`覆盖或fallback**。`core/vad_volume.py:165–168`的fallback只是低能量统计阈值→RMS阈值。

global `vad_decibel_thres`仍有明确读者：`core/vad_pool.py:233`和`scripts/bench_vad_cpu_vs_gpu.py:44`构造nested参数，`scripts/validate_npu_models.py:440`通过qwen兼容getter取值，另有兼容/配置测试；SNR配置还直接供主链段级过滤。因此保留这些key，不能因为当前FSMN不展开nested值就草率删除，也不能把两个独立的−40配置混成同一个生效来源。

完整原始参数证据为`/tmp/asr-b3-cpu-reference-3j24mrb9/configuration-provenance.json`，SHA256=`f7adeae9e66b242bb1fc171d3fff0ad87a68f3da6412c87d9ae2f5c86ecef830`；其中保存完整原YAML、库构造默认值及源行、CPU有效值、Native构造源码SHA/行、服务过滤源码与其他读者索引。该证据只读code/config/已完成CPU结果，没有新增模型或NPU执行。

NativeVad保留以上既有行为，迁移没有顺便改成“生效12秒/0.8”。后续若治理无效配置，应独立修改并补生效测试，不能把当前声明值误报成原VAD已经生效的标定值。

CPU加载0.056秒、完整参考VAD19.816秒；这些CPU参考耗时不用于比较NPU吞吐，也不计入g1原始请求耗时。原始参考`/tmp/asr-b3-cpu-reference-3j24mrb9/reference.json` SHA256=`6b3a5326d828b53de8831f39eb9d8fd2411762d713012a2056977e0ca1c72af2`；工装SHA=`282020ba2ff6020cb43c3348904131d4cce6694946d5955e0e274f403271f175`。权重`model.pt` SHA=`b3be75be477f0780277f3bae0fe489f48718f585f3a6e45d7dd1fbb1a4255fc5`，配置SHA=`899b215ee7d1733a6d076acb33944e72e9dc788d7bb39d4ab010a5255b9bda13`，均只读未改。

复算必须新建独立临时目录，避免覆盖已有证据：

```bash
set +u
. /usr/local/Ascend/ascend-toolkit/set_env.sh
reference_work=$(mktemp -d /tmp/asr-b3-cpu-recheck.XXXXXX)
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/zhangs/conda-envs/asr/bin/python \
  /tmp/asr-b3-cpu-reference-3j24mrb9/cpu_reference.py \
  --output "$reference_work/reference.json"
```

### g1资源曲线与正常关闭

682个快照覆盖4804.947秒，实际间隔中位7.006秒、最大12.312秒；allocator有681个有效值，缺失没有填0。以下字节口径换算为MiB，三窗按快照文件顺序等分取中位数：

| 口径 | 首值 | 三窗中位数 | 离散峰值 | 末值 |
| --- | --- | --- | --- | --- |
| 主进程RSS | 2897.301 | 5307.674 / 5171.812 / 4464.988 | 5407.438 | 3451.230 |
| 本进程torch allocated | 26.833 | 4624.303 / 4624.879 / 4624.583 | 4710.774 | 4610.112 |
| 本进程torch reserved | 44.000 | 5430.000 / 5446.000 / 5446.000 | 5446.000 | 5446.000 |

主进程CPU按单核100%为均值137.437%、最高采样区间162.264%，仅主进程，不包括四VAD worker或共享服务；不能解释成全宿主CPU占比。线程峰277/末273，FD峰147/末132。RSS与FD随请求排空回落；allocated三窗中位数接近，reserved保留缓存。没有将“关闭前模型仍加载”当作泄漏，也没有仅据这次变长工作负载宣称长期无增长。

服务进程树npu-smi HBM峰6298MB、物理卡7聚合峰21507MB、AICore离散峰15%；既有PID1601397恒11875MB，继续单列。缓存峰8文件/712109680字节，末值0文件/0字节；VAD busy/waiting峰4/4，末0/0，Qwen busy峰1、末0。不同资源口径不混加、不把其他PID归因给本项目。

g1父进程exit0、无剩余后代/无LISTEN、5秒观察，shutdown_timeout/forced_child_cleanup/application_shutdown_failed/worker_shutdown_failed均false。四VAD worker均exit0、forced=false、cleanup_complete=true，CPU graceful2.408/1.769秒，NPU7为6.864/6.296秒。g1运行在main关闭失败退出码修复之前；本批正常关闭的证据成立，但不能代替新main异常退出传播的CPU回归。随后main只改关闭结果传播，core/api/utils/entity的推理指纹另由快照核验，g2完整配对见下文。

### g1原始SHA与复算

| 源 | 原始完整WAV SHA256 |
| --- | --- |
| F0 | `25782e0ad459d6be270188951da7343a6951a78126bd549e9ee5c69a476a2f0e` |
| F1 | `99195ce63638c7a03c37b319054a655112325fe34c5b615e4e3151a88a6df8be` |
| F2 | `2e42db093a4149752ddeb1dd5d0c05efe7bec18a5f3147fe4e7cd947e3c8b6a1` |
| F3 | `5ae4039dea035b36b64c4dd4a0f1f774190a2080ec95d67aca91dceccf9db82a` |
| F4 | `5c663ae9217e2ec58d86459f115e11136d015671928b5cfceffe226d96b29a2e` |
| F5 | `0e9f4df99a73f4e5ea9b7415a9cac4db3a710dabe0beda5b722e7236ad36985e` |
| F6 | `f466ac0e160fb591b982e72a9f3e0faa99657abf5a807d4af2fcac7aa6526ffe` |
| F7 | `0d23b47226d120d621ac5c0a66ae71a03fe365964873ac304bc84ccb72b67212` |

以下文件均相对`e2e-full-g1-final`，全部8个请求结果保留，未复制完整课堂文本到Git：

| 文件 | 字节数 | SHA256 |
| --- | --- | --- |
| `application.log` | 37278 | `8d4cf1216a37e52926c72865ca1fd27c19bc6b0bf19ac4f8f92a8a743c8432e4` |
| `asr_stats.json` | 131 | `07e68f68a4b5852a4028b39b07cee208710e1c31028c4acf9a7cc96893a92161` |
| `config.toml` | 4356 | `32c03a26073dc1d6ddde86b260169fd4752b6254275630591ca06d852661a559` |
| `full-samples.json` | 8287 | `8c87f1280c9c30c443bef4f107ff51d0eb3d5aa90e2f50e5a0b82476a4bd40c8` |
| `ownership.json` | 723 | `a3c73e8032157f0270f4aae38be2edc9d0f732b24d3c9a07c9b86b3df58ab42f` |
| `request-full-0.json` | 126051 | `1d066e939755cc4bcf68a4c9b94d6bb232a28c9cc1f71a09c2ef9fdff03893d9` |
| `request-full-1.json` | 143579 | `e2ad47a1a41e9a5e4a9f1cde71dfb11d37ba034cb2635b23034f096106772f31` |
| `request-full-2.json` | 164679 | `f6cceff9973cfd906f520bccc056f85a59c0ea6879c751de31126e6e167f0936` |
| `request-full-3.json` | 55828 | `0bf9d9be95f91a5d8de331f5af621942f93f5a83646667d6f878c4c00a984919` |
| `request-full-4.json` | 125627 | `dd8eb807f6e2cbbe44da86d3e29132ff3642a1e8999f6b8c0bb9ee95b7a31d4b` |
| `request-full-5.json` | 178298 | `99e0fede758501f95b4ecce38047a3bd9de26954d209418b59544501d40205e4` |
| `request-full-6.json` | 179004 | `b0f113e07c64a617921b685580089397fd8fe1924ae50b5b776d824c213ac2b6` |
| `request-full-7.json` | 175215 | `c1fbc69d64e0af00bb2206cffc41009c2685979a5207e1ad3655cb33493a26cc` |
| `service.log` | 104004 | `0b133a090630091436e6c9604b286f8354271d76b1a9320d88d8d3ba55e41317` |
| `shutdown-after.json` | 1895 | `d69e3d878c76b9e4524154492a6d2593d2c612df54b92b9704927624d95a2708` |
| `shutdown-before.json` | 3029 | `55f95ce13371eeae979d9b517a4a902fae0799df4ec5afc5266d2314ce2584e7` |
| `status-drained.json` | 19843 | `a4eea065f7d3439ac73159fb0df863173486573db01f189707018872f6d5e954` |
| `status-initial.json` | 18003 | `9a2512961e396e28deb95b4faab4afee9c69e4d1e692f6420aae4aa988b260bd` |
| `status.jsonl` | 4676242 | `28040357c3983dc2724a5156b3c420b535fdad6b8a299366c08ea70b963bcebd` |
| `summary.json` | 2874 | `55cef5b4652a7393e048e033e75bf78b6d1eecbe072bc9d8a68410d06d85e685` |

标准库派生JSON为`/tmp/asr-threshold-readonly-xehkij_8/classroom-analysis.json`，SHA256=`29dee7f783c3ff55857c912146c71b6b0a206d1f059ab45913421167708b3852`。分析脚本与collector同在该目录；脚本SHA仍为`8f735dda41857265a9b407f3c4315079d9a336adcd6a8d92ed49d56970fb5bb4`，collector为`f09bc71f205e5002053c3b6fc154cef57256ffbffcb4f2fbe2fe5f2dd37e533b`。不读取音频的统计可复算为：

```bash
python3 /tmp/asr-threshold-readonly-xehkij_8/analyze_classroom_thresholds.py \
  --directory /tmp/asr-apply-20261117-a99trpfv/e2e-full-g1-final \
  --collector /tmp/asr-threshold-readonly-xehkij_8/collect_validation.py \
  --output /tmp/classroom-g1-summary-recheck.json
```

资源三窗方法与前文相同，仅替换`status.jsonl`目录。工程8/8成功、正常关闭、数值/文本形状观察与人工语音质量仍分别判定：本报告没有WER/CER、BCubed F1、情绪准确率或98%人工有效对话召回结论。

## 相同八源：g2与最终配对

完整g2目录为`/tmp/asr-apply-20261117-a99trpfv/e2e-full-g2-final`，配置为单Qwen实例、segment/request 2/2，4B+4A、同八个源SHA、同完整音频；8/8成功并已关闭。g1为1/1，比较维度是同一torch_npu后端的并发档位，**不是NVIDIA/NPU对比**。两组活动峰均为8/B4。

| 源 | g1耗时s | g2耗时s | g2相对变化 | g1/g2 RTF | g1/g2最终段数 | 分段业务字段差异数 |
| --- | --- | --- | --- | --- | --- | --- |
| F0 | 3628.947 | 3795.947 | 4.60% | 1.3873/1.4512 | 311/311 | 0 |
| F1 | 3944.004 | 4130.309 | 4.72% | 1.4597/1.5286 | 351/351 | 0 |
| F2 | 4264.672 | 4478.003 | 5.00% | 1.5119/1.5875 | 400/400 | 0 |
| F3 | 2108.681 | 2230.409 | 5.77% | 0.7475/0.7906 | 166/166 | 0 |
| F4 | 3873.073 | 4085.840 | 5.49% | 1.3729/1.4483 | 341/341 | 0 |
| F5 | 4755.156 | 4972.156 | 4.56% | 1.6847/1.7616 | 516/516 | 0 |
| F6 | 4795.935 | 5018.234 | 4.64% | 1.6989/1.7776 | 535/535 | 0 |
| F7 | 4685.418 | 4895.312 | 4.48% | 1.6573/1.7316 | 489/489 | 0 |

P50从4104.338199变为4304.155877秒（+4.87%），P95从4781.662453变为5002.106922秒（+4.61%）；本批八个请求均是2/2更慢。最终默认选择**单Qwen、1/1**，并同步host/docker配置、getter缺省和断言；先前12秒短测偏好2/2的原始指标保留作不同工作负载的历史，不改写成支持1/1的数据。每档只有一次完整批次、共享卡/宿主有背景工作；g2期间还做过约20秒四线程CPU参考（未用NPU）。本轮支持当前工作负载的保守选择，不构成无背景干扰或所有场景全局最优的因果证明。

### 业务输出逐项配对

独立只读比较全部8个raw请求，逐项核对source SHA、顶层parsed text、每段`segment_text/bg/ed/role/emotion`、language与language_ratio。**八源全部相等、3109对分段差异0**；角色/情绪分布及文本形状计数与g1相同，A类null保持禁用语义。计时字段自然不同，未把它们纳入质量一致性比较。没有为得到相等结果删除失败样本、归一化文本或拼接相邻段；也没有把一致性当作人工准确率。

无全文的配对JSON见[full-pair-output-comparison.json](full-pair-output-comparison.json)，SHA256=`1cd85cc86a38d87fd2d05f2be40bcb652eaded83009d03a0073bcdeb8213b6d7`。它只含计数、索引、输入/原始文件/输出哈希；完整课堂文本仍只留原始远程结果。F3的前20分钟解释复用已有原CPU参考，未再次构造模型。

### 午夜轮转日志必须一起检查

g2跨午夜，不能只读取当前`application.log`：当前文件35行、旧日`application.log.2026-11-17`173行，合计208行，八份完成记录全部存在。全批复读修复仍是**1次506→6字符/2.0秒/门槛120**（旧日文件L166），超长但无循环保留0、错误签名0。单看当前文件会错误得到“0次修复”；该不完整口径没有用于结论。最终文本相等也不意味着原始decoder从未复读。

### g2资源与关闭

713个快照覆盖5027.544秒，间隔中位7.011秒、最大11.674秒；allocator有712个有效值。MiB口径如下，三窗方法同前文：

| 口径 | 首值 | 三窗中位数 | 离散峰值 | 末值 |
| --- | --- | --- | --- | --- |
| 主进程RSS | 2891.383 | 5268.586 / 5164.848 / 4472.176 | 7914.422 | 3451.727 |
| torch allocated | 0.000 | 4631.311 / 4631.682 / 4631.558 | 4671.431 | 4616.990 |
| torch reserved | 0.000 | 5334.000 / 5342.000 / 5342.000 | 5342.000 | 5342.000 |

相较g1，g2主进程RSS离散峰从5407.438增到7914.422MiB，三窗中位数仍在接近量级并随排空回落；不能只凭单次瞬时峰判定泄漏。线程峰277/末273、FD峰147/末132；CPU主进程均值137.120%、区间峰170.764%（单核100%口径，不含worker）。服务进程树Smi峰6232MB、设备聚合21441MB、AICore峰33%，既有PID1601397仍11875MB。不同口径不混加，离散峰也不能证明持续利用率更高。缓存峰8文件/712109680字节，末0/0，Qwen和VAD busy/waiting均最终归零。

父进程exit0、四worker exit0/forced=false/cleanup_complete=true，5秒观察无LISTEN/后代，所有shutdown失败/超时/强制标志为false；CPU graceful1.741/1.845秒、NPU7为7.242/7.431秒。g1与g2的推理模块指纹一致，main变化只涉及关闭失败退出码；最终1/1新保护构建的独立稳态另见[compiled-final-g1-steady-evidence.json](compiled-final-g1-steady-evidence.json)，不把源码完整课堂直接替代编译产物验收。

### g2原始SHA与复算

全部文件相对`e2e-full-g2-final`，包括轮转日志：

| 文件 | 字节数 | SHA256 |
| --- | --- | --- |
| `application.log` | 8788 | `ffe5f76ec33a1336b4850813e4e05499bc66486ec8b791882af4a56d0e72eef7` |
| `application.log.2026-11-17` | 28476 | `f70d9ff81c3b73ae7913f0b52287bdc9be896caad4702e4edda4558cf2ccb1cf` |
| `asr_stats.json` | 131 | `07e68f68a4b5852a4028b39b07cee208710e1c31028c4acf9a7cc96893a92161` |
| `config.toml` | 4356 | `9015e5fb8c9484fa5b8a413de5a9a3f77d7b0d52a248e52c7d00173d0d558405` |
| `full-samples.json` | 8287 | `8c87f1280c9c30c443bef4f107ff51d0eb3d5aa90e2f50e5a0b82476a4bd40c8` |
| `ownership.json` | 721 | `b13bd634f01f76163f3c10bd079fbcbd80492bc34e600d2134d02e3f814fae47` |
| `request-full-0.json` | 126053 | `e44c1f3c071680213fa8688fe4e6be993a936f14445c72b2b12f44d17a8c23a2` |
| `request-full-1.json` | 143579 | `1e300005e290c7ecdf58f80423ed554f2ddff73cff6db79ef97d0bba6da78515` |
| `request-full-2.json` | 164678 | `b50d5e7f0a90a2ed2eafc3ca3098a7c7200106626821b5cba961da631e6ac624` |
| `request-full-3.json` | 55826 | `9ec28c7c2239b91a711922fa6dbca12317d15c6c5f45a7a198f8bd74034bc940` |
| `request-full-4.json` | 125627 | `4a5fc95b3cf114a1b49b800d25ca0c29bda4d8ffec15a7029760967c023ebab5` |
| `request-full-5.json` | 178297 | `5fea33a4ffab615a92778a7afe232e06d17ea3c4a84350ae6c302ea85de05b97` |
| `request-full-6.json` | 179004 | `4e962eb363055770a0ae18b7b74e3c57535622f8d1d339d3e7e715db3421d21e` |
| `request-full-7.json` | 175213 | `94ca7b5088120c1f59eb1ff6d30237b5936900496bd25b129d586a1f77938518` |
| `service.log` | 105894 | `680172c658416049cb9825eada409fd56485729a3aeb34b9fb39a40a42cb1bfa` |
| `shutdown-after.json` | 1891 | `dcb8f7a71a6523efcd5b9a2a43075bec7fe41ad5fdd14b9e3c75ecd0b3859bc5` |
| `shutdown-before.json` | 2977 | `ae2bcf1b5ea332c932fcd5df40878dd06a2a6e3b021c2bd8253ffd136fca8079` |
| `status-drained.json` | 19839 | `9db00ccb22691b16b10371b7568ddef56c18bc038fd6b472f31e91d8178e032c` |
| `status-initial.json` | 17990 | `acfada83c4e6220151091ef168c96fb1de20c5fbc6e5b595ecbd33130682ad58` |
| `status.jsonl` | 4886420 | `612dd26bfdec67b83e84ac8a854e177548a3239f2d9ebf2293b5ce507b96887b` |
| `summary.json` | 2871 | `d597cf5c04a07e6ebd668681ebc861c8d45fe461465d37e79d29e55dc4602d21` |

派生统计JSON `/tmp/asr-threshold-readonly-lez8n1r0/classroom-analysis.json` 的SHA256为`2e7329981c0d8510951844af8e39f24a1d40692cf145a28c964c88711940f2f3`。该工具对当前application.log的单文件计数由上述两文件完整扫描补足；原始日志均保留，未覆盖或删减。

可用同一标准库统计脚本将directory改为g2目录复算；配对逻辑也可不打印文本地复核：

```python
import json
from pathlib import Path
root=Path('/tmp/asr-apply-20261117-a99trpfv')
fields=('segment_text','bg','ed','role','emotion')
for i in range(8):
    records=[json.loads((root/f'e2e-full-g{g}-final'/f'request-full-{i}.json').read_text()) for g in (1,2)]
    a,b=[r['payload'] for r in records]
    business=lambda p:[{k:s.get(k) for k in fields} for s in p['segments']]
    print(i, records[0]['sha256']==records[1]['sha256'], a['text']==b['text'], business(a)==business(b), a['language_ratio']==b['language_ratio'])
```

两组的工程、关闭、输出一致性结论已有证据；没有新增人工标签，因此本报告仍不宣称WER/CER、BCubed F1、情绪准确率或98%有效对话召回达标。
