# SeaCraftASR

基于 FastAPI 的离线语音转写服务，面向教育课堂录音。服务本地用 FunASR FSMN-VAD 做语音切分，
在华为 Ascend 910B 上用本地 **Qwen3-ASR-1.7B** 原始权重按独立语音段转写（支持物理批执行），CAM++ 与 emotion2vec 使用原生 torch_npu。v3 请求、响应、错误码和后处理契约保持不变。

> `/v1.1.8/seacraft_asr`（SeACo-Paraformer + PUNC + Whisper 法语链路）已随本次变更下线，
> 服务只保留 `/v3/seacraft_asr`、`/get_status`、`/clear_tasks_list` 三个接口。

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v3/seacraft_asr` | 离线 ASR 转写（Qwen3-ASR 链路） |
| `GET` | `/get_status` | 服务状态、任务统计与本地模型显存明细 |
| `DELETE` | `/clear_tasks_list` | 清空当前实例的处理中任务列表 |

FastAPI 自动文档位于 `/docs`、`/redoc` 和 `/openapi.json`。

## 项目结构

```text
.
├── main.py                     # 应用装配 + 进程命名（asr-offline-model）
├── config.toml                 # 本地运行配置
├── docker/                     # 昇腾镜像、Compose 和部署说明
├── api/routes/
│   ├── asr_v3.py               # /v3/seacraft_asr
│   ├── asr_qwen_common.py      # v3 主链路：VAD → 并发转写 → 响应封装
│   ├── asr_common.py           # v3 共用：请求上下文、情绪标签映射、临时文件清理
│   └── status.py
├── core/
│   ├── config.py
│   ├── logging.py              # 日志按天 + 按大小双轮转
│   ├── models.py               # CAM++ / emotion2vec 启动预加载、共享有界预取与资源归属
│   ├── vad.py                  # v3 链路的本地 VAD 切分
│   ├── qwen_asr_client.py      # Qwen本地异步客户端（实例租约 / 批执行 / 重试）
│   ├── qwen_asr_runtime.py     # 独立音频列表、有效mask、逐成员结果与阶段计时
│   ├── qwen_batch_scheduler.py # 跨音频公平、长度分桶、有限批等待及帧/token预算
│   ├── segment_pipeline.py    # 有界段worker、PCM预算及所需分支安全汇合
│   ├── qwen_asr_queue.py       # v3 链路的音频排队与两级请求并发控制
│   └── status.py               # 状态接口响应组装
├── entity/data.py
├── model/                      # 模型权重，不进入 Git
└── utils/
    ├── audio_utils.py
    ├── asr_stats.py
    ├── feature_utils.py
    ├── itn.py                  # 文本规范化（WeTextProcessing）+ 标点卫生规则
    └── qwen_response.py        # Qwen 返回文本解析与语种占比统计
```

## 模型目录

模型需要完整放入 `model/` 下的三个目录：

```text
model/
├── README.md
├── speech_fsmn_vad_zh-cn-16k-common-pytorch/
├── speech_campplus_sv_zh_en_16k-common_advanced/
└── emotion2vec_plus_large/
```

| 模型 | 用途 |
|------|------|
| FSMN-VAD | 语音端点检测（切分转写单元） |
| CAM++ | 说话人嵌入与说话人聚类 |
| emotion2vec_plus_large | 情绪识别（原生 FP32 基线） |

模型权重被 `.gitignore` 忽略，但会进入 Docker 构建上下文并复制到镜像 `/app/model/`。

## 本地运行

开发主机 `xjtu910b7`，源码 `/home/zhangs/asr_nvidia`，仅使用宿主物理卡 `npu:7`。
卡 7 已有其他服务，测试结果属于共享卡测量。解释器固定为 `/home/zhangs/conda-envs/asr/bin/python`。

```bash
cd /home/zhangs/asr_nvidia
set +u
. /usr/local/Ascend/ascend-toolkit/set_env.sh
/home/zhangs/conda-envs/asr/bin/python scripts/check_npu_environment.py --config config.toml
CONFIG_PATH=./config.toml OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/zhangs/conda-envs/asr/bin/python main.py \
  --host 127.0.0.1 --port 8083 --workers 1
```

正式入口为 `python main.py`（支持 --host/--port/--workers 1）。关闭时向本次父进程发送 TERM，先完成 lifespan 再走 Python atexit；当前 NPU 运行时可能接管 INT，不能仅以 Ctrl+C 判断退出。等待 NPU 工作及 VAD worker 退出。测试必须记录自身 PID/进程树和端口，统一 finally 收尾并核对端口解除监听；不要按名称或端口批量终止既有服务。
启动预检校验依赖、设备及三个小模型资产，VAD 固定进程池预热后才就绪；Qwen 按自身 preload_on_startup 策略加载；已开启且设备数组非空的 CAM++、emotion2vec 在 HTTP 就绪前完成全部实例加载尝试并常驻复用。Qwen 目录默认 `/home/model/Qwen3-ASR-1.7B-hf`，不再依赖外部 HTTP 服务。

运行依赖统一为 `requirements.txt`，仅面向昇腾 910B；开发测试使用 `requirements-dev.txt`，独立构建工具使用 `requirements-build.txt`。完整 Conda/pip 参考锁位于 `environment/910b/`。环境重建使用 `scripts/rebuild_npu_env.sh` 的独立前缀，不原地升级当前 `asr`；av/pynini 仍由 Conda 提供。

### 模型内存压力诊断

`scripts/model_memory_pressure_diagnostics.py` 只附着显式 `--backend-pid` 对应的本项目实例，采集 NPU HBM、RSS/PSS、Swap、VAD worker、缓存与健康状态；它不拥有被测服务的关闭权。启动服务的测试程序负责收尾。
`/get_status` 的 `resourceDiagnostics` 保留既有业务结构，资源段改为 `npu`；采样失败用 `available=false` 和原因表达。`requestTelemetry` 有界保存阶段统计，不保存音频或正文。

音频递归清单与原始 GBK 路径映射由 `scripts/build_audio_manifest.py` 生成，见 `docs/npu-910b-migration-20261117/audio-manifest.json`；使用稳定 ASCII sample_id 上传，不重命名或覆盖原文件。

## ASR 请求

```bash
curl -X POST "http://localhost:8083/v3/seacraft_asr" \
  -F "audioFile=@sample.wav" \
  -F "language=auto" \
  -F "wordTimestamps=true" \
  -F "showSpk=true" \
  -F "showEmotion=true" \
  -F "showRoleIdentify=true"
```

主要表单参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `audioFile` | 必填 | 上传音频 |
| `language` | `auto` | 下发给 Qwen 的语言标记；`auto` 表示不指定，由模型自动识别 |
| `wordTimestamps` | `false` | 是否返回词级时间戳（v3 链路固定返回空数组） |
| `showSpk` | `false` | 是否启用说话人分离 |
| `showEmotion` | `false` | 是否返回情绪标签 |
| `showRoleIdentify` | `false` | 是否识别教师/学生角色 |

> `showSpk`、`showEmotion`、`showRoleIdentify` 的默认值均为 `false`。

响应包含 `language`、`segments`、`text`、`speed_info`、`load_audio_time_ms`、`gpu_time_ms` 与 `language_ratio`。
分段中的 `role`、`emotion` 和 `segment_words` 根据请求参数返回。`text` 为最终保留分段按空格拼接并按配置选择是否执行文本规范化（ITN）后的结果。纯「啊 / 哦 / 嗯 / 呃」分段会被移除；英文为有效文本占比最高时，边界明确的中文口气词会转换为 `ah / oh / um / uh`。

### v3 处理流程

1. 服务本地用 FunASR FSMN-VAD 把音频切成若干语音段；
2. 每个有效语音段建立独立Qwen逻辑任务；允许多个独立任务组成物理批，但**不拼接PCM或合并相邻段**，逐段保留语种提示和结果归属；
3. 按请求能力独立供给Qwen、CAM++和emo队列，Qwen完成即释放自身名额，不等慢增强；所需分支按请求ID/段序号汇合，响应`bg`/`ed`仍取原VAD边界；
4. 复读修复、单字语气词移除、语言占比判定、英文口气词替换、ITN 和标点清理依次执行，最终统计只使用最终保留分段。

### 语种占比 `language_ratio`

`language_ratio` 的键为语种**简写**（`zh` / `en` / `yue` / …），值为该语种文本字符占比（百分比，保留一位小数，合计 100.0）。
模型返回的语种名不在映射表内时原样透传。顶层 `language` 字段仍返回模型返回的语种全称（如 `Chinese` / `English`）。
完整映射表（22 个语种）见 [`utils/qwen_response.py`](utils/qwen_response.py) 的 `LANGUAGE_ABBREVIATIONS`：

| 语种 | 简写 | 语种 | 简写 | 语种 | 简写 |
|------|------|------|------|------|------|
| Chinese | `zh` | English | `en` | Cantonese | `yue` |
| German | `de` | Italian | `it` | Portuguese | `pt` |
| Spanish | `es` | French | `fr` | Thai | `th` |
| Malay | `ms` | Japanese | `ja` | Korean | `ko` |
| Russian | `ru` | Hindi | `hi` | Arabic | `ar` |
| Vietnamese | `vi` | Indonesian | `id` | Dutch | `nl` |
| Turkish | `tr` | Polish | `pl` | Greek | `el` |
| Hebrew | `he` | | | | |

### 文本后处理

上游拿到的文本已经过下列处理，与模型原始输出不完全一致，属于预期行为：

1. **复读修复**：模型偶尔在长片段上产生「同一短语无限重复」的循环，服务检测到后折叠重复部分。
2. **ITN 文本规范化**（默认关闭，由 `[itn] enabled` 控制，需规范化时显式设为 `true`）：WeTextProcessing 负责数字规范化（如「二零二五」→「2025」）与部分语气词剔除。ITN 之后还有两条与 ITN 开关**无关**、
   始终执行的标点卫生规则：
   - 整段文本只有标点符号时，该段从 `segments` 中移除；
   - 分段文本以标点开头时，移除开头**连续的所有标点**，直到落到有效字符。
3. **独立语气词剔除**：纯「啊 / 哦 / 嗯 / 呃」这类独立成段的口头禅会被移除，不计入 `segments` 与 `text`。

因此 `segments` 条数会少于音频中实际的人声段数，`text` 是最终保留分段的有序拼接。

### 错误码

| 错误码 | 含义 |
|--------|------|
| `4008` | 音频为空或未检测到任何人声 |
| `4009` | 提交的语种不被 Qwen 支持 |
| `4010` | 请求排队等待超时（`queue_wait_timeout_s > 0` 时可能触发） |
| `4011` | 本地 Qwen 目录/权重不可读、设备不可用或加载失败 |
| `4012` | 本地 Qwen 推理异常或超时（重试耗尽） |
| `4013` | 总音频处理容量满（消息包含总活动数/上限），或等待模式的共享等待队列满 |
| `4015` | 请求的 CAM++ 说话人/角色识别能力未启用或没有可用实例 |
| `4016` | 请求的 emotion2vec 情绪能力未启用或没有可用实例 |
| `4017` | 仅 B 类增强容量满，消息包含 `spk_emo` 和当前增强活动数/上限 |
| `4018` | 同时请求说话人与情绪能力，但两个设备数组均为空 |

> 默认 `queue_wait_timeout_s = 0`：总容量满立即返回 `4013`；总容量有余量但 B 容量满返回 `4017`。
> 两项容量同时满时优先返回 `4013`。能力开关校验早于容量检查。

## 配置

主要配置项：

| 配置 | 说明 |
|------|------|
| `instance_count` | `start.sh` 启动的 uvicorn 实例数，默认 `1` |
| `[log].log_path` | 日志文件路径，默认 `./asr_service.log`；修改后重启生效 |
| `[log]` | 日志保留策略：`max_days`（默认 7）、`max_size_mb`（默认 100） |
| `[model_paths]` | 三个模型目录 |
| `[speech_rate]` | 单句语速修正系数 |
| `[itn]` | 文本规范化开关与失败策略 |
| `[qwen_asr]` | v3 链路的 本地 Qwen 目录、设备、并发度和队列 |
| `[vad]` | FSMN-VAD worker、切分参数、全局 VAD 阈值和段级过滤参数 |
| `[model_device_pool]` | CAM++ 与 emotion2vec 的设备数组；每个元素对应一个独立模型实例和一个并发槽位 |

`speed_info` 按 1、5、10 分钟窗口统计，跨窗口句子按时间重叠比例分摊内容数量。单句语速使用 `[speech_rate].rate_factor` 修正，分时段语速不乘该系数。

### `[qwen_asr]` 配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enabled` | `true` | 是否开放 `/v3/seacraft_asr` |
| `model` | `""` | 本地模型显示名称；不发起模型发现请求 |
| `local_model_dir` | `/home/model/Qwen3-ASR-1.7B-hf` | 原始 HF 权重与 processor 目录 |
| `local_device` | `npu:7` | 宿主目标 NPU；容器仅映射物理 7 时使用逻辑 `npu:0` |
| `local_dtype` | `bfloat16` | 支持 bfloat16/float16/float32；默认采用已验证 BF16 |
| `generation_mode` | `continuous` | 正式多owner连续生成；旧静态批必须显式legacy |
| `preload_on_startup` | `true` | continuous就绪前顺序加载全部owner；legacy缺省false |
| `owner_threads` | `1` | 每owner CPU线程预算，父进程OMP/MKL另行配置 |
| `continuous_prefill_batch_size` | `min(16,S)` | 每owner每轮prefill上限，显式不得超过该派生值 |
| `local_instance_count` | `4` | continuous的owner数N，范围1～8；显式legacy缺省1 |
| `max_new_tokens` | `256` | 每段生成上限 |
| `segment_concurrency` | `16` | continuous每音频及其owner逻辑活跃窗口S，范围1～128；legacy缺省1 |
| `request_concurrency` | `N×S=64` | continuous由N×S派生，不在模板重复赋值；legacy缺省1 |
| `audio_concurrency` | `N=4` | continuous由N派生；legacy缺省8 |
| `audio_spk_emo_concurrency` | `4` | 请求任一增强能力的 B 类整音频上限，必须为正整数 |
| `queue_max_size` | `64` | A+B 共享等待队列容量 |
| `queue_wait_timeout_s` | `0` | `0` = 队列满立即拒绝；`>0` 才排队等待，超时返回 `4010` |
| `request_timeout_s` | `120` | 单次 Qwen 请求超时（秒） |
| `retry_count` | `2` | Qwen 请求失败重试次数 |
| `speaker_similarity_threshold` | `0.30` | CAM++ 说话人聚类的余弦相似度阈值（按实测 sweep 取值，阈值过高会严重过分割） |
| `execution_mode` | `process` | continuous必须process；显式legacy缺省thread |
| `process_shutdown_grace_s` | `120` | continuous模型关闭宽限；legacy缺省30，超期回收仍记关闭失败 |
| `max_inference_batch_size` | `S=16` | continuous每owner物理步上限，显式不得超过S；legacy缺省1 |
| `max_batch_audio_frames` | `14400` | continuous每owner帧预算，不随N放大；legacy缺省102400 |
| `max_batch_tokens` | `10000` | continuous每owner估算token预算，含生成预留；legacy缺省78336 |
| `pending_segment_limit` | `256` | 全局实际开工且尚未汇合的段预算，1～4096；同时作为每音频元任务worker上限，两个计数独立 |
| `pending_audio_bytes_limit` | `268435456` | 在途PCM字节预算（256MiB），范围1MiB～4GiB；达到预算则反压，单段超预算明确失败 |

新增字段未显式配置时使用上表缺省值；设备数组和原有字段表反映当前仓库`config.toml`，不是尚未完成的吞吐标定结论。全部调度/预算参数修改后重启生效；非法类型或范围在服务就绪前报错。

锁定5.13 processor的独立前端按16kHz单声道PCM16样本数估算：`valid_frames=floor(max(samples,8000)/160)`，完整single特征时间轴为`padded_frames=100*ceil(valid_frames/100)`。8000来自原processor最短0.5秒输入，100来自`2*n_window`（`n_window=50`）；例如0.3秒有效段仍保留原processor先补至0.5秒的行为，时间轴和最短段业务规则不变。帧预算为`max(padded_frames)×B`，token预算为`(ceil(max(padded_frames)/8)+max_new_tokens+256)×B`，256是调度器的保守提示预留。它们是当前锁定版本的调度估计，不是实际token数或NPU显存保证；更换processor或前端参数时必须复核，不套用旧公式。

批执行前逐成员使用FP32 processor准备输入，再显式collation：有效text tokens左padding，完整single features及原有效mask仅额外右padding；保留single原本已有的末尾上下文与100帧补齐，不在STFT前把不同PCM补成同长度。batch1直接保留原single输入；新增未知键、shape或不一致dtype明确报错。状态/批遥测记录`frontend_mode`、`single_processor_shapes`、最终`processor_shapes`及实发成员；这些CPU等价检查不代替NPU文本、吞吐或完整音频验收。

`batch_max_duration_ratio`限制有效STFT帧数比，而非原始录音秒数比。调度可以越过某请求不兼容队首取兼容尾成员，但公平锚点必须优先全局最老等待队首，避免持续补入的短段使旧长段饥饿；最终响应仍按原段序恢复。批等待到期、输入预算或时长比限制触发时按实际成员发出，拆批原因分别记录。

每音频只创建至多`min(有效段数, pending_segment_limit)`个轻量worker，worker逐段领取工作；缺省每音频至多256个，八条完整音频至多2048个元任务。它们可以在开工前等待全局预算，不能把2048个元任务或所有待处理段误写成已持有PCM的活动段。真正切片与分支分发前，另由全局`pending_segment_limit=256`和`pending_audio_bytes_limit=256MiB`共同约束；这两个预算在Qwen/CAM++/emo必要分支全部收尾后归还，不与Qwen逻辑窗口混为一项。最终文本和embedding结果用于响应汇合，不计作在途PCM缓存。

批遥测按每次物理`dispatch_attempts`记录其输入成员、实际有效成员和本次runtime包络；重试等待中取消的成员不会再次发出，非法语言/无效输入只隔离对应成员。批级summary表示最后一次attempt，不能将首轮成员索引套到重试后的幸存列表，也不能将多attempt包络重复算给每个成员。交给结果future或保存到汇合结果的异常会保留类型、code和message，但移除traceback/cause/context链，避免通过异常frame继续持有段PCM或正在收尾的worker。

单模型process候选保持原权重、HF引擎、Qwen BF16及增强FP32，不在父进程重复加载Qwen。状态的`model_workers`保留PID/创建时间、READY/实际模块来源、RPC及关闭证据；模型内存是worker口径，父进程allocator、worker allocator/RSS和IPC副本分别记录，256MiB仅是主流水线逻辑PCM预算。状态查询不启动worker或向忙worker发RPC。成员取消/请求timeout等待真实DONE或确认worker死亡后再释放；仅显式关闭启用grace，强制清理不作正常关闭。当前安全默认仍为thread，单process完整8B未通过后进入双process×64候选；完整音频验收通过后再推广默认值。

Qwen逻辑credit在PCM裁剪前按每音频/全局顺序取得，避免先完成VAD的文件占满全局PCM后让其余文件无法供给；转写分支完成即归还，裁剪失败、EOF短段与取消幂等兜底。credit已预留但还在等PCM不是实际模型执行，真实PCM预算耗尽时仍会反压。

仅扩大`request_concurrency`不会形成真实批计算。每音频16、全局128时，1/2/4/8条音频最多分别提供16/32/64/128个逻辑在途段，调度在等待截止内处理已有成员，不凑满128才开始。`local_instance_count`表示权重份数；`max_inference_batch_size`表示一次物理计算成员上限；二者都不改变总音频/B准入规则。

正式continuous模板不再显示 `batch_wait_timeout_ms`、`batch_max_duration_ratio`；二者只在显式legacy模式按原类型/范围及调度语义使用。两个增强模型实际计算固定batch1，模板省略 `speech_campplus_batch_size`、`emotion2vec_batch_size`；既有非1值拒绝约束保留。预取深度不改变实际batch。

### `[vad]` 配置项

下表列配置声明值。原FunASR通过nested kwargs传参，当前权重的FSMN实际选项来自原始model_conf：
单段8000ms、结束静音800ms、speech_noise=0.6、全局decibel=-100、全局snr=-100。
CPU原始AutoModel与原生NPU保持同一既有行为；服务段级过滤独立使用SNR18、segment_decibel=-40等参数。
不能将声明值、实际FSMN选项和段级过滤混为一层；本次不变更参数绑定或数值。
完整来源和原始选项见[课堂阈值评审](docs/npu-910b-migration-20261117/classroom-threshold-review.md)。

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `vad_device` | `["cpu", "cpu", "npu:7", "npu:7"]` | 每个元素启动一个固定 FSMN-VAD worker；数组长度决定完整音频 VAD 并发 |
| `min_segment_duration_s` | `0.3` | 过滤过短 VAD 段，单位：秒 |
| `vad_snr_thres` | `18.0` | 服务段级SNR真实取值；nested声明未覆盖FSMN全局-100 |
| `vad_decibel_thres_filter_enabled` | `true` | 段级低能量帧比例过滤开关 |
| `vad_segment_decibel_thres` | `-40.0` | 段级 dBFS 低能量帧阈值 |
| `vad_decibel_frame_sample_ratio` | `0.5` | 段内 decibel 抽帧比例，范围 `(0,1]` |
| `vad_decibel_low_frame_ratio_thres` | `0.8` | 抽样帧中低于段级阈值的比例达到该值时删除段 |
| `vad_snr_thres_filter_enabled` | `true` | 段级 SNR 过滤开关 |
| `vad_low_volume_rms_dbfs_thres` | `-40.0` | 段级 RMS dBFS 门限，低于该值时删除段 |
| `vad_low_volume_filter_enabled` | `true` | 段级 RMS 过滤开关 |
| `after_start` / `before_end` | `3` / `3` | 段级过滤作用于完整音频前后 N 分钟；`0` 表示整段，`-1` 表示该侧不限制 |

音频与段级并发参数的区别：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `audio_concurrency` | 同时处于处理中的 A+B **音频文件**总数 | continuous派生N=4，legacy缺省8 |
| `audio_spk_emo_concurrency` | 总音频中请求增强能力的 B 类音频数量 | 4 |
| `segment_concurrency` | **单个音频内**同时执行本地转写的 VAD 段任务数 | continuous缺省16，legacy缺省1 |
| `request_concurrency` | **所有音频合计**本地转写任务数上限 | continuous派生N×S=64，legacy缺省1 |

上述配置均在重启后生效，每个服务实例独立限流。旧配置缺少新增的
`audio_spk_emo_concurrency` 时同样采用 **4**，升级后增强音频并发会受此上限限制。
B 上限允许大于总上限，实际 B 并发至多为 `min(audio_concurrency, audio_spk_emo_concurrency)`；
配置值不会被静默截断。0 不能用于关闭增强能力；关闭CAM++或emotion2vec应把对应设备数组设为 `[]`。

三个参数 `showSpk`、`showEmotion`、`showRoleIdentify` 全为 false 的音频称为 A；
任一个为 true 称为 B。每个 B 同时占用一个总音频名额和一个增强名额，即使开启多项也只计一个 B。
不为 B 预留总名额，不改变 Qwen 段级信号量或模型设备数组决定的推理容量。

当前最终配置为总4/B4：4A、4B或合计4个混合请求均使总容量满，新A/B都返回4013；此配置不能单独触发4017。以下表格是显式总8/B4（或legacy）的通用示例，不是当前默认：

| 正在处理 | 新 A | 新 B |
|---|---|---|
| 3A + 4B | 正常处理 | `4017` |
| 5A + 3B | `4013` | `4013` |
| 8A | `4013` | `4013` |
| 4A + 4B | `4013` | `4013` |

`queue_wait_timeout_s > 0` 时，暂时满额的请求进入 A+B 共享等待队列，同类按到达顺序处理。
等待 B 不预占任何处理名额；B 因增强容量满而受阻时，A 可绕过它使用空闲总名额。
当较早 B 已满足两类容量条件时，它优先于后到 A。共享等待队列满返回 `4013`，
等待截止仍未取得完整名额返回 `4010`；`queue_max_size=0` 只允许立即获准的请求。

处理名额从准入成功持有至音频准备、VAD、转写、增强推理、后处理及临时文件清理结束。
成功、失败和取消都会回收；取消会等待已经启动且无法立即中断的线程或 VAD 工作必要收尾，
再清理文件和归还名额，因此取消完成可能晚于取消信号发出。

`GET /get_status` 用同一次快照返回音频准入状态，例如正在处理 `5A+3B`、等待 `1A+2B`：

```json
{
  "audioConcurrency": {"configured_total": 8, "active_total": 8, "waiting_total": 3},
  "spkEmoConcurrency": {"configured": 4, "active": 3, "waiting": 2}
}
```

A 的活动数和等待数分别用总数减去 B 数。这里的 B 等待按完整音频计数；
`localModels` 中的 `waiting` 则按等待 CAM++ 或 emotion2vec 的分段计数。
首次请求前活动数和等待数均为 0，查询不会加载模型，配置关闭而被拒绝的请求不增加计数。

新增 `[V3准入]` 日志记录 `request_id`、`category`（A/B）、`outcome`、`reason`、
`code` 和两组容量 `snapshot`。既有 `[V3 耗时]` 阶段日志保留，音频准入等待通过
遥测 `phases.audio_wait_ms` 记录，与模型池分段等待的 `spk_wait_ms` / `emo_wait_ms` 分开。
`admission.outcome` 为 `admitted`、`rejected`、`timeout` 或 `cancelled`；
未完成准入判定时为 `not_attempted`。准入后处理失败或取消仍为 `admitted`，
最终结果由请求 `status` 表达。快照在授予或拒绝时取得，等待取消时为回收后的状态。

`resourceDiagnostics.requestTelemetry` 增加 `admission.by_class`（A/B 请求数）和
`admission.rejections_by_reason`（按原因计数），并以 `window=retained_events` 标明统计窗口。
原因包括 `total_full`、`spk_emo_full`、`waiting_full`、`wait_timeout`、
`role_disabled`、`emotion_disabled`、`both_disabled`、`cancelled`、`qwen_disabled`。
拒绝汇总包含准入前的超时与取消；准入后的失败不计作拒绝。
遥测最多保留 4096 个请求事件，包含近期明细的诊断快照最多输出 20 条，
状态接口只输出汇总；这些是保留窗口统计，并非进程启动以来的累计值。
新增日志与遥测不保存音频、转写正文或模型对象。

### `[model_device_pool]` 配置项

控制 v3 链路本地模型副本和并发容量。数组长度是唯一槽位来源，不需要 `local_gpu_concurrency`、`cam_slots` 或 `emo_slots`：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `speech_campplus_devices` | [`"npu:7", "npu:7", "npu:7", "npu:7"`] | 每个元素加载一个 CAM++ 实例；长度决定 CAM++ 最大并发 |
| `emotion2vec_devices` | [`"npu:7"`] | 每个元素加载一个 emotion2vec 实例；长度决定 emotion2vec 最大并发 |
| `speech_campplus_execution_mode` | `"process"` | CAM原生执行：thread或process；process仅支持NPU设备数组，宿主npu:7/容器逻辑npu:0均可 |
| `speech_campplus_process_startup_timeout_s` | `60` | 每CAM worker的READY启动期限，有限正数≤300秒，拒绝bool/字符串 |
| `speech_campplus_process_shutdown_grace_s` | `30` | 仅显式关闭后的独立grace，有限正数≤300秒；不将普通取消改为kill |
| `speech_campplus_prefetch_queue_depth` | `0` | 每健康CAM++实例的预取等待深度，0～64，不含运行任务；0为无预取 |
| `emotion2vec_prefetch_queue_depth` | `0` | 每健康emo实例的预取等待深度，0～64，不含运行任务；0为无预取 |

当前最终选择 CAM++ 4 个process实例、emotion2vec 1 个实例；本轮宿主B4与增强驻留A4已完成4/4验收（200.10秒/183.47秒）：

```toml
[model_device_pool]
speech_campplus_execution_mode = "process"
speech_campplus_devices = ["npu:7", "npu:7", "npu:7", "npu:7"]
emotion2vec_devices = ["npu:7"]
```

预取使用每池共享FIFO任务源，未开始任务不永久绑定某个忙实例；空闲实例及时接下一段，无需等整音频响应。八个健康实例、深度5表示最多40个预取等待任务加最多8个运行任务，**不是batch5或40路实际计算**。深度0不建预取缓存，生产端公平等待空闲实例；生产端反压人数另计`backpressure_waiting`，不混入预取`waiting`。池还报告`waiting_capacity`、`oldest_wait_ms`、每实例`busy`、`handoff_count`及`last_handoff_ms`，交接间隙不把无任务时的空闲算作调度开销。

CAM的process模式每个设备数组元素对应一个owned spawn worker，在该worker中加载同原生FP32模型；emo仍使用父进程runner，Qwen模式独立配置，不新增HTTP服务或更改模型格式/依赖。返回保持CPU float32 Tensor[1,192]，原短段聚类与错误码不变。状态查询或未请求CAM不启动worker、不发送RPC。非法模式、CPU/process组合和非法期限在就绪前拒绝，空数组仍禁用能力。宿主与容器模板选择process、CAM++4/emo1；loader未显式设置CAM模式时仍保留thread兼容缺省。本轮B4与增强驻留A4及最终保护待验收，容器另待构建/验收，组件增益不能当成完整音频性能通过。

八CAM进程会带来额外冷启动、进程RSS和IPC副本。实验16段原型约272秒的launcher全程包含启动、权重加载、预热及关闭，不能与约2.72秒暖态推理混同；单worker约16秒启动是该机观测，60秒配置值是每worker的READY期限，不承诺整个八worker池在60秒内完成加载。普通请求取消等待已启动工作真实DONE；显式关闭先发布closing并启用独立grace，仅回收出生身份仍匹配的owned worker，确认死亡前不归还容量。强制清理、坏ACK或非零退出必须保留为关闭失败，重复关闭不能洗为正常。

两个模型池独立排队、互不借用容量；空数组表示禁用对应能力，请求会得到 CAM++ `4015` 或 emotion2vec `4016`。部分副本加载失败时，只有成功加载的实例计入容量，失败原因可从 `/get_status` 的 `localModels` 查询。该接口同时提供每个池的 `configured_instances`、`instances`、`available_instances`、`busy`、`waiting`、`failed_instances` 和等待摘要。

同一张 NPU 上增加数组元素会同时增加模型 HBM 和推理争用，可能降低单次推理速度或触发 OOM；应结合状态接口和 `npu-smi info` 的峰值 HBM、利用率及压测 P95 选择实例组合。VAD设备由`[vad].vad_device`独立控制；Qwen和CAM分别按执行模式在父进程或独立worker持有模型，共享目标NPU。

FSMN-VAD 的设备由 `[vad].vad_device` 数组独立控制，例如 `['cpu', 'npu:7']` 会启动两个固定 worker。旧 `[qwen_asr]` VAD 字段和旧动态池字段不再参与实例数计算，迁移时会记录告警。

### 细小声音阈值回归

参考音频的人工标注文件不是严格 JSON，使用以下可选入口解析并比较 `0.3`、`0.6`、`0.8` 三档：

```bash
python scripts/vad_reference.py --output /tmp/vad-reference.json
```

报告包含 46 个“细小声音”标注段的抑制率、未标注段保留率、VAD 段数、Qwen 请求数、耗时以及 RMS/峰值诊断。区间命中定义为 VAD 输出与标注区间重叠达到标注区间的 50%；响度只用于诊断，不作为唯一判据。该长测不会进入默认快速测试，也不修改原始音频和标注文件。

`vad_segment_decibel_thres` 使用段级 log 能量单位，不能直接填入 RMS dBFS；`vad_low_volume_rms_dbfs_thres` 才是面向采集音量的段级门限。历史试验最初默认关闭；当前宿主与容器模板启用三项段级过滤、窗口为开头/结尾各3分钟。新推荐值必须来自参考音频扫描并同时满足细小声音抑制率、标注区间无有效识别内容和正常语音保留率。扫描可用 `--target-suppression-rate`、`--target-retention-rate`、`--coarse-step`、`--fine-step` 和 `--output` 覆盖目标与输出路径。回滚时将开关改回 `false` 并重启，FSMN模型内部有效选项保持不变。现有 `min_segment_duration_s` 仍是通用兼容过滤，但不参与本次低音量第一阶段标定。

### role / emotion 的加载条件

v3 链路的真实 `role` 与 `emotion` 由本机模型给出，加载与计算条件是两回事：

- **加载**：CAM++ 与 emotion2vec **各自独立**，分别由 `speech_campplus_devices` 与
  `emotion2vec_devices` 数组是否非空决定启用；非空数组按元素数量在HTTP就绪前加载，空数组关闭，另一池独立运行。
  emotion2vec 使用禁用 HF32 的原生 FP32 正确性基线；低精度/编译候选独立测量后再决定是否采用。
- **计算**：`showSpk` 或 `showRoleIdentify` 请求说话人能力，`showEmotion` 请求情绪能力，
  三个参数默认均为 `false`。未请求的能力对应字段为 `null`，不执行其推理；非空数组的模型已按启动预加载契约驻留。
- **配置拒绝**：请求要求的能力设备数组为空时，在音频准入前返回 CAM++ `4015` 或
  emotion2vec `4016`；同时请求两项且两个数组都为空返回 `4018`，优先于总容量/B容量错误，拒绝不增加活动、等待或owner租约。
  请求整体失败，不开始转写，不以 `null` 字段成功返回；仅请求已开启的另一项能力可正常处理。
  两个设备数组均为空且三个请求参数均为 `false` 时，仍可普通转写。
- **并行**：role / emotion与Qwen各自独立排队、执行及释放阶段名额，再按原段汇合。增强等待在预算有余时不占住已完成Qwen的名额；段数/PCM预算耗尽时明确反压，不能无限堆积。
  已开启且有健康实例的模型在某个分段推理失败时，该段对应字段回退为 `null`，转写照常返回；
  设备数组非空但模型全部加载失败则返回对应的 `4015` / `4016`，不能当成数组关闭返回 `4018`。

### 短段与增强计算边界

正式内存裁剪先将起止秒端点分别`round(endpoint*sample_rate)`，再限制到源样本范围；毫秒接口与tensor/NumPy裁剪共用同一边界，EOF只裁短、不补零。`bg`/`ed`仍取原VAD时间。旧FFmpeg按秒转码兼容通道保留，压缩文件输出不承诺逐样本等长。

时长按裁剪后的实际有效样本数计算；段时间预检通过但EOF裁短至不足门限时，仍在三个模型分发前丢弃。不用响应中四舍五入后的`bg`/`ed`判断。最小有效时长0.3秒时：

| 有效段时长 | Qwen | 所请求CAM++ | 所请求emo |
| --- | --- | --- | --- |
| 不足0.3秒 | 分发前丢弃，不进入响应 | 不调用 | 不调用 |
| 0.3秒至不足1秒 | 正常转写 | 实际推理，成功embedding进入聚类，不再因不足1秒而丢弃/被就近标签覆盖 | 不推理，直接“平淡” |
| 1～10秒（含边界） | 正常转写 | 实际推理及聚类 | 实际分类及既有映射 |
| 超过10秒 | 按原转写规则 | 按既有输入支持/错误契约 | 不推理，直接“平淡” |

未请求能力对应字段仍为`null`。FP32策略、聚类阈值、失败分层与情绪标签不变；能力由设备数组派生；`showRoleIdentify=false`不关闭`showSpk=true`所需的CAM++/聚类。

### 进程与 HBM 定位

VAD每数组元素对应独立spawn worker；CAM++与Qwen各有thread/process模式，emo保持父进程独立模型池。`localModels` 与 `vadPool` 原键名保留，`qwenModel` 为同级新增字段，状态查询不触发加载。
`allocated_mb`/`memory_allocated_mb` 是 PyTorch 分配估计，reserved 是缓存分配器；`npu-smi info` 提供进程 HBM 与整卡 HBM，不可把共享卡全部占用归因本项目。未知值为 null 或 available=false，不能以零冒充无占用。

```bash
npu-smi info
curl http://127.0.0.1:8083/get_status
```

`instance_count` 默认 1；增加 uvicorn 进程或 `local_instance_count` 会复制本地 Qwen 权重，必须重新标定资源与并发。
`ASCEND_VISIBLE_DEVICES` 指定宿主物理卡，容器内设备编号从 0 重新映射；单独映射物理 7 的模板使用 `npu:0`。
旧 `base_url`/`connect_timeout_s` 仅兼容读取，不发 HTTP；四个旧 TensorRT 键被忽略并提示迁移，`cuda:N` 配置直接拒绝。

`bg`/`ed` 继续为两位小数字符串，`load_audio_time_ms`/`gpu_time_ms` 保持字符串。`gpu_time_ms` 保持原主链路墙钟计时范围，不表示纯 NPU kernel 时间。

## 吞吐调度实施与测试

当前OpenSpec change为[`2026-11-18-npu-asr-throughput-scheduling`](openspec/changes/2026-11-18-npu-asr-throughput-scheduling/proposal.md)，处于apply阶段；[设计](openspec/changes/2026-11-18-npu-asr-throughput-scheduling/design.md)、[测试计划](openspec/changes/2026-11-18-npu-asr-throughput-scheduling/test-plan.md)和tasks约束后续标定。实现能力不等于性能验收完成，未完成必需矩阵不得归档。

当前安全缺省为单Qwen、实际batch1、每音频/全局窗口1/1；仓库VAD设备及CAM三份/emo一份是前次已验收配置快照。独立候选profile为VAD八CPU、CAM八NPU:7、emo八NPU:7、单Qwen每音频16/全局128、物理批上限128、增强预取5。总音频上限8、B子上限默认4保持；纯B8容量验证另配子上限8。候选配置通过解析校验，不表示真实128批、八份健康池或完整音频性能已经通过。

本轮候选关闭`vad_snr_thres_filter_enabled`、`vad_decibel_thres_filter_enabled`、`vad_low_volume_filter_enabled`，`min_segment_duration_s=0.3`、`after_start=-1`、`before_end=-1`。三个false才关闭服务段级能量过滤；-1/-1仅表示不限制过滤位置，不能把它当开关。原FSMN检测和四层参数证据仍保留，新候选与基线须使用相同过滤/PCM清单，不与旧开启过滤的80分钟数据直接算加速比。

VAD按完整音频FIFO调度，每worker同时处理一条；八文件/四worker时，任一worker完成VAD即交给下一条，不等前四全部完成，也不等Qwen/CAM/emo。VAD租约归还不释放整音频准入租约；整音频仍持有至必要工作和finally清理结束。

测量采用同一组八条完整、源SHA256去重音频，在1、2、4、8完整音频并发各处理全部八文件，完成一条立即补下一条；A/B使用同源配对，`showRoleIdentify`省略或false。分别记录单请求P50/P95、整批墙钟/吞吐、各模型等待/执行和真实批分布；默认B4的混合八路与B8容量profile分列。冷启动、预热和稳态分开，不用短片重复或不同人声占比的文件代替同源配对。

所有性能测试使用**420秒诊断线**：完成样本P50/P95任一超过420秒，或任一尚未完成请求从调用起超过420秒，均在下一监测周期（默认1秒）停止追加负载并捕获准入、解码、VAD、Qwen、CAM、emo、汇合、后处理及资源卡点。阶段无进展提前告警；有界取证后安全结束本轮，记录`diagnostic_aborted`和全部成功/失败/未完成，不盲跑或仅用快请求子集宣布通过。诊断与关闭期限须有限，420秒不改业务HTTP超时或单Qwen调用超时。定位并小规模复验后才恢复完整测试。

每轮finally关闭自身登记PID/创建时间对应的服务、worker及端口，等待已启动设备工作与读取收尾，核对无孤儿和监听释放至少5秒；共享既有服务保留。收尾超时或强制回收不得改写为通过。批遥测中的CPU特征/传输/generate/读取/同步墙钟须区分，批包络按batch ID只算一次，不能按成员重复相加冒充NPU忙时；原`gpu_time_ms`字段及主链路口径不变。

真实128容量必须区分纯Qwen与八CAM/八emo已加载且实际工作的B环境；前者通过不能替代后者。最快批可能小于128，单份问题须先诊断，再按证据评估2/4/8副本及段级均衡，不以权重占用乘法证明运行容量。最终默认值按完整验收结果定案，当前不承诺性能最优、绝对准确率或已上线。

## Docker

本版源码容器交付与构建、启动命令见 [docker/README.md](docker/README.md)。`docker/`仅保留Dockerfile、docker-compose.yml、config.toml、README.md四个文件；单HTTP直接运行必要Python源码，不做Cython、不使用Nginx。默认Qwen4/CAM4/emo1，宿主物理卡7映射为容器逻辑npu:0。

底座为本机CANN8.2.RC1，容器独立使用torch2.5.1/torch-npu2.5.1.post1；宿主CANN8.1及asr环境不升级。[容器依赖材料](docs/container-cann82-environment.md)记录独立环境及最小上下文。配置必须从宿主只读挂载到/config.toml，修改后重建容器；三个小模型内置，Qwen HF权重独立只读挂载。镜像格式为`asr_offline_910b:v3_YYYYMMDD`。

本change的实际构建、完整B4和正常退出结果记录于[本版证据目录](docs/docker-source-cann82-20261125/)；在相关结果完成前不能宣称镜像可用或生产上线。失败保留独立构建缓存；成功验收后仅清理本轮中间镜像/cache及辅助资源，保留基础/最终镜像、原有服务和日志报告，禁止全局prune。

性能、精度、样本与执行证据见 [NPU 迁移报告](docs/npu-910b-migration-validation-20261117.md)。原 NVIDIA 报告仅作历史标定来源。

## License

MIT License


### 状态资源采样（吞吐变更v6）

`[status].resource_sample_interval_s = 1.0`控制按需资源重采样的最短间隔，接受0.1～60秒有限数，布尔/字符串/NaN/Inf就绪前拒绝，修改需重启。没有GET查询不永久轮询NPU；每个app只有一个在途collector。`/get_status`直接返回实时准入、模型队列及阶段进度，不等待npu-smi/procfs，资源部分使用最后完整缓存，首次或采集失败明确unknown。

`resourceDiagnostics.sampling`给出`has_snapshot`、`age_s`、`in_progress`、`last_error`、attempts及上一attempt的`last_started_monotonic_s`/`last_finished_monotonic_s`/`last_elapsed_s`。耗时含线程派发与collector，不是NPU核时间；失败attempt的耗时不重置旧资源快照age。关闭先阻止新采样，模型照常收尾，最后等待已启动collector实际完成，不能只取消future。

NPU资源行按采样时OS出生身份与主循环已缓存的模型worker PID/start_ticks/创建时间核对。`npu.backend_compute_apps`、`qwen_compute_apps`、`cam_compute_apps`、`other_compute_apps`及`unverified_compute_apps`分列，未知/重用身份不伪称本项目或外部；`qwen_execution`与`qwen_memory_process_scope`明确单模型process。父allocator、worker缓存allocator及设备采样HBM各有口径，不能拿父进程零值替代worker或相加不同时刻峰值。此扩展不改变v3请求/响应字段。


CAM进程缓存由`localModels.speech_campplus.model_workers`保留，包括尚在加载和已退休但仍需核对的runner；`instance_details[].memory`保留每worker的READY、allocator、last_done、IPC字节及关闭证据。`resourceDiagnostics.cam_model_workers`原样保留这些范围，`processes.cam_model_worker`区分已知存活、已回收与未知存活；未知count为null，不把父进程allocator=0当成worker=0。NPU进程行附OS采样的`process_identity`及`process_resource`（RSS/PSS），只在出生身份吻合且worker未确认退出时归属CAM；旧快照或PID重用归入未验证行。OS资源age与worker最后RPC/allocator时间各自保留，调用峰值、进程RSS和整卡HBM不相加成同时峰值；主流水线256MiB逻辑PCM预算不含IPC额外副本。


### 吞吐变更v16的状态历史窗口（apply中）

`/get_status`的Qwen `batch_scheduler.recent_batches`改为最近8条完整记录；保留全部成员/attempt/shape/计时，`history`明确可用/返回ID范围、截断和缺口提示，实时计数不截断。直接客户端/调度器`snapshot()`默认仍最多64条。HTTP直接渲染隔离的JSON数据，避免完整历史重复递归编码阻塞推理调度；这是有界快照，不是无损日志，审计必须去重并检测采样遗漏。CPU候选开销降低不等于完整音频性能已验收，仍须实机420秒闸门和最终矩阵；模型精度、生成上限及v3业务响应未改。


### 吞吐变更v17：内存WAV补位（apply中）

段级编码增加经原SoundFile逐字节验证的PCM16格点快速路径：仅原生一维native float32、有限[-1,1]且值落在16bit格点、采样率合法时直接序列化同一PCM16 WAV；其他输入保留原SF分支，不改变CAM/emo原PCM。128真实段及全部65536格点须源/实际so等价。CPU真实裁剪ABBA支持100ms组批等待候选，50ms仍碎；假模型结果不代表NPU速度，正式默认仍待完整NPU矩阵。NumPy/bytes临时工作buffer与pending PCM预算分开统计。模型仍原HF/BF16与原生FP32 torch_npu，不改模型格式/依赖或生成上限。


### 吞吐变更v18：双Qwen条件回退（尚待实机）

单份实际128容量已证，完整8B效率未通过，故试process2、每份物理批64、每音频16/全局128；一HTTP服务、VAD8CPU/CAM8进程/emo8线程、NPU7与精度不变。process可配置1～8不等于8份已通过HBM/性能；先2实测，4/8按需。首次发现全部健康才发布，成功发现后坏一份仍可用健康已加载副本，容量如实下降，不重启失败者。每physical attempt保留index/PID出生/launch/RPC归属，空闲FIFO接力、不绑定整音频。

正常drain后并发CLOSE，候选显式process shutdown grace120允许已测39秒长批排空，完成即返回；不改业务推理timeout120、测试420/无进展120、owned HTTP drain10/outer close240。缺省兼容值与待推广候选分开。双份暖身至多两个顺序真实clip、一次420总deadline，按两actor真实forward证据判定，不只看READY；单份仍一次。当前仍未完成最终source18矩阵或推广默认。


### 单Qwen启动预加载及128段测试配置（2026-11-19）

`[qwen_asr].preload_on_startup`为严格bool，缺省及根/容器配置均为false以保留原首次使用加载行为。true在VAD启动后、HTTP服务就绪前加载全部配置Qwen权重与processor并常驻；显示model名称不替代加载，失败/取消同样回收。该开关不执行语音generate，不承诺所有shape已暖身；不是在线换权重或免重启更新配置。预加载主要移除首请求的模型加载耗时，不能据此宣称暖态吞吐更快。

用户选择保持根配置，单份128独立候选在`test/config.qwen128.toml`（总8/B4）与`test/config.qwen128-b8.toml`（仅B8准入实验）。二者启用预加载、process1/BF16@npu7、每音频16／全局128逻辑段、物理批最多128、100ms有限等待、115200帧/80000估计token、256段/256MiB预算、显式关闭120秒；VAD八CPU、CAM八NPU process/emo八NPU thread、预取0且实际batch1。最短0.3秒及三过滤false/-100/-1/-1按本轮约定。仍未完成完整性能验收，不是新生产默认。

```bash
cd /home/zhangs/asr_nvidia
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
CONFIG_PATH=./test/config.qwen128.toml OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/zhangs/conda-envs/asr/bin/python main.py --host 127.0.0.1 --port 8083 --workers 1
```

`local_instance_count=1`是一份权重；`segment_concurrency=16`是一条完整音频的Qwen段窗口；`request_concurrency=128`是全部音频共享的逻辑在途段窗口；`max_inference_batch_size=128`是一次物理批上限，仍受就绪段数/长度/帧/token预算限制。完整HTTP音频并发为1/2/4/8，总容量仍8、B4或显式B8，不能用128并发上传完整音频代替段批处理。请求示例与客户端见test/中的使用说明。关闭使用本次父进程TERM并等待子进程/端口释放，勿接管已有服务。

纯Qwen历史固定同128段曲线中，batch64/128处理总量的三轮稳态中位约5.232/3.871秒，吞吐仍增加35.2%；128是最高已测档，不是已证最优。当前物理批校验上限128，未执行>128实验；不能推断拐点已在128之外，也不能拿该组件成绩替代完整A8/B8。


### 单Qwen八音频48／384段候选（2026-11-19）

参数已在`config.toml`与`docker/config.toml`显式列出，配置重启后生效。依用户此前选择，根配置保留1／1、物理batch1、预加载false及原小模型池行为；384只在独立候选`test/config.qwen384.toml`和`test/config.qwen384-b8.toml`启用。128测试配置仍保留。

| [qwen_asr]参数 | 384候选值 | 含义 |
| --- | ---: | --- |
| local_instance_count | 1 | 常驻Qwen权重副本数 |
| preload_on_startup | true | 权重/processor就绪后才开放HTTP，不做生成暖身 |
| segment_concurrency | 48 | 每条完整音频在途Qwen段窗口 |
| request_concurrency | 384 | 所有音频共享逻辑段窗口 |
| max_inference_batch_size | 384 | 每次物理批最多384，校验范围1～384；不开放512 |
| batch_wait_timeout_ms | 100 | 等待有限，到期发已有段，不强凑384 |
| batch_max_duration_ratio | 32 | 按最长/最短有效特征帧限制同批跨度 |
| max_batch_audio_frames | 345600 | 按每成员900padding帧×384估计 |
| max_batch_tokens | 240000 | 按(ceil(900/8)+256+256)×384估计 |
| pending_segment_limit | 512 | 独立活动PCM段数预算，旧256会先反压384供给 |
| pending_audio_bytes_limit | 268435456 | 256MiB原PCM预算，仍可能先反压；不含模型/前端/IPC临时内存 |
| audio_concurrency | 8 | A+B完整音频准入总上限 |
| audio_spk_emo_concurrency | 4或独立8 | B4默认保留；纯B八路使用-b8候选 |

八条各48为逻辑上限384；两条最多96、四条最多192，不等待凑满。B4纯B只准四条、最多192逻辑段，第5B仍4017；总8满时原4013优先级不变。四A四B也可供384，真实批仍取决于各音频VAD就绪、切片/WAV供给、长短/帧/token/PCM预算与增强汇合，不保证总是满384。VAD八CPU、CAM八process/emo八thread继续各自独立排队、batch1/预取0，不因Qwen384增加NN实例或跳过分支。

在远程项目根目录加载CANN后启动测试候选：

```bash
CONFIG_PATH=./test/config.qwen384.toml OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/zhangs/conda-envs/asr/bin/python main.py --host 127.0.0.1 --port 8083 --workers 1
# 八条纯B另用CONFIG_PATH=./test/config.qwen384-b8.toml；不会自动扩大根配置B4。
```

请求客户端仍用`test/v3_request.py --audio-dir test --limit 8 --mode A --concurrency 8`，其并发8是完整HTTP音频，不是384个HTTP请求。客户端不拥有服务；测试服务启动者须正常TERM并确认worker/端口关闭，不杀既有服务。

已有纯Qwen组件384是固定原128段各3份：中位7.415秒、同期128×3为11.389秒，峰allocated21.69GiB，27/384原始输出SHA不同，未验准确率。512首次暖身在Conv2D分配额外11.72GiB时OOM；该结论限当时共享卡/缓存条件。384在八CAM/八emo共驻和八完整音频上的稳定性、OOM、输出质量及吞吐尚未验收，不能以纯组件成绩推广正式默认。基于100ms/900帧的预算是供给/拆批门槛，不能当作HBM容量保证。

### 根配置同步384（2026-11-19）

按用户明确要求，根 `config.toml` 已原样同步 `test/config.qwen384.toml`：单 Qwen process、预加载 true、每音频48/全局384、物理批384、八CPU VAD、八CAM process/八emo thread、关闭三项能量过滤，总8/B4。修改需重启生效；此次未重启服务。此前“根配置保留1/1、batch1、预加载false”的描述仅代表同步前状态；容器模板未改。此配置选择不表示完整八音频/增强共驻性能已验收，现有验收缺口仍保留。

### 无效VAD配置清理（2026-11-19）

已删除 `vad_max_single_segment_time_ms`、`vad_max_end_silence_time_ms`、`vad_speech_noise_thres`、`vad_decibel_thres` 及对应Settings正名/别名和默认nested转发；它们不会覆盖FSMN真实构造选项。模型原始配置与有效选项不变，不将删除操作当作阈值调整。`vad_snr_thres`仍用于服务段级过滤，独立 `vad_segment_decibel_thres`、RMS、开关及范围参数均有效并保留。复读参数移至实际读取的 `[qwen_asr]`，此前误放 `[vad]` 的值恰与有效缺省一致，修正不改变当前有效值。历史章节中的旧参数仅代表清理前版本。当前服务未重启。

### 单Qwen持续生成候选（2026-11-20）

旧2026-11-20实验版本的`generation_mode`缺省legacy且限制continuous单实例；该限制已由2026-11-24正式多owner配置替代，见文末迁移说明。`online_repetition_enabled`为严格bool，缺省false，独立于生成后repetition_guard_enabled。候选配置`test/config.qwen-continuous128.toml`为音频8、每音频16、全局128；尚未完成真实8并发验收，不自动替换根48/384配置。配置变更需重启。

连续生成每段独立EOS和KV，底层仍按步批量计算。复读停止、未EOS达到长度上限及无效前缀按段恢复并返回显式警告，无法恢复段记录缺失并继续其它段；基础设施错误仍沿既有错误契约传播，取消不提前回收设备资源。只对明确未接收容量拒绝最多重试3次（总4次尝试）、首次拒绝后30秒内，未知执行状态不重发。状态摘要不记录正文。

`continuous_prefill_batch_size`范围1～128、缺省16，为每次接纳新段的prefill上限；prefill后推进已有decode，不等待凑满。该值仅为候选，需真实NPU标定。


生成中复读阈值均位于`[qwen_asr]`，使用严格正整数，拒绝bool、浮点和字符串；启用状态沿用`online_repetition_enabled`严格bool。以下默认值是保守候选，已完成构造文本保护与既有token离线回放，尚非真实复读召回标定或实测最优。配置修改需重启，与`repetition_guard_enabled`生成后修复独立。

| 字段 | 默认值 | 作用 |
| --- | ---: | --- |
| `online_repetition_min_generated_tokens` | 32 | 最少已生成token，未达到不检测 |
| `online_repetition_min_chars` | 120 | 正文异常长度绝对下限 |
| `online_repetition_max_chars_per_second` | 40 | 正文还须超过音频时长×该系数 |
| `online_repetition_min_repeats` | 8 | 尾部循环最少次数，必须≥3 |
| `online_repetition_min_repeat_chars` | 64 | 循环覆盖最少字符，必须≤tail_chars |
| `online_repetition_max_unit_chars` | 32 | 扫描单元最大字符，必须≤tail_chars |
| `online_repetition_tail_chars` | 512 | 有界检测尾窗大小 |
| `online_repetition_confirmations` | 2 | 连续正文增长确认次数，必须≥2 |

达到最少token及多次确认后，异常长度路径或强循环路径任一路径满足即停止；只检测`<asr_text>`后的正文，不通过全局禁止重复来修改正常讲授。`Settings.qwen_online_repetition_config`返回严格校验的`RepetitionGuardConfig`，无效组合即使保护关闭也拒绝启动，防止开关开启后才暴露配置错误。

2026-11-21：continuous段生成异常（复读、长度耗尽、无效前缀）经保守恢复后继续其它段。新响应`processing_status=complete|completed_with_warnings`及`segment_warnings`（原段index/bg/ed/reason/action/recovered）区分无异常与警告完成。无法恢复段保留警告，不伪装无人声；OOM/设备死亡/取消/timeout不被降级。legacy响应不变。

在线强循环判据新增`online_repetition_strong_tail_min_chars=96`、`online_repetition_strong_tail_min_ratio=95`（1～100百分比）、`online_repetition_strong_tail_repeats_per_second=4`，其余为严格正整数；与异常长度路径并列，仍需多次新增token确认。候选来源为两条真实循环轨迹和既有正常输出回放，不代表人工准确率或最优阈值。统计分列处理完成、严格无警告成功和可用文本，警告不能默算质量成功。

`continuous_admission_drain_ms`为0～5毫秒严格整数，缺省0保留即时poll旧行为；候选3ms只在本轮收到新submit后用于Pipe发送者交接的固定总预算，deadline不续期，prefill已满/容量满/closing立即停止，无新submit不增加等待。它不恢复整批generate屏障，也不重置原100ms组批等待。根配置保留0、独立候选3，需NPU实测。

### 正式多owner配置与迁移（2026-11-24）

当前loader及根/Docker模板采用 `generation_mode="continuous"`、`execution_mode="process"`、`preload_on_startup=true`、N=4、S=16；HTTP `instance_count=1` 不变。总音频容量派生为N=4，全局Qwen逻辑窗口派生为N×S=64。本轮最终选择Qwen4/CAM++4/emo1，宿主B4与增强驻留A4已完成4/4验收（200.10秒/183.47秒）。每条请求动态租用一个健康owner，到整音频全部分支安全收尾才归还；一条音频尾段不能借空闲owner扩大S。历史单实例/128/384章节只描述原版本与实验，不代表新默认或已通过本轮验收。

根/Docker模板不再写 `audio_concurrency`、`request_concurrency`、`max_inference_batch_size`、`continuous_prefill_batch_size`。continuous物理步缺省S，显式值只能1～S；prefill缺省min(16,S)，显式不能超过此值。旧audio/global显式值相等时给迁移提示，冲突、bool、字符串或越界在就绪前失败；应删除重复字段，或明确选择 `generation_mode="legacy"` 保留原独立容量配置。仅改N/S后重启即可更新派生配置，不需要跟着修改多份总量。legacy静态批行为仍使用其原独立参数和缺省，不暗中切换执行方式。

每owner帧/token预算为14400/10000，N变化不放大这两项。S或生成上限变化时校验最短单段估算可接纳性，长段和多成员超预算仍须拆分/反压并报告；这不是HBM保证。每owner CPU线程预算 `owner_threads=1`，父进程 `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4` 独立设置。全服务共享PCM预算为256段/256MiB，不复制N份。`audio_spk_emo_concurrency=4` 是B配置上限，有效B容量不超过健康Qwen数，健康下降不提前释放原请求。

CAM++和emotion2vec数组独立，用户最终选择CAM++4个process实例、emo1；本轮宿主B4及增强驻留A4已完成4/4验收。此前N6/CAM1/emo1的B4已4/4完成，但增强驻留后的A6真实OOM，历史结果保留。N变化不复制增强池。VAD保持八CPU。宿主只验证npu:7，Docker仅映射物理7时使用逻辑npu:0，按可见设备核对。配置修改后需重启；本轮未部署Docker。预加载仅顺序装载权重/processor，真实forward暖身与性能计时另行验收；此前N6源码/保护回归和正常关闭通过；纯A6墙钟186.32秒、B4墙钟302.60秒，但共驻验收失败，change保持开放，不能视为稳定部署配置。完整证据见[正式测试报告](docs/qwen-multi-instance-20261124/REPORT.md)。

continuous保留已批准并用于noHBM参考的在线复读保护 `online_repetition_enabled=true` 与 `continuous_admission_drain_ms=3`（收到提交后的有界IPC交接），不改变检测阈值或max_new_tokens；显式legacy缺省仍false/0。3ms不是重新等待凑整批或续期长等待，不影响无新段的decode周期。本次默认迁移不另行更换prefill/KV算法。

此前N6/N5成绩仅对应各自冻结配置；当前根/Docker已改为用户最终选择Qwen4/CAM++4/emo1，本轮宿主B4与增强驻留A4已完成验收。Docker CAM模式同步为process，仍保留其原有VAD过滤等环境差异，未构建/上线，不能套用宿主成绩。

### 2026-11-24用户授权N5补充验证

根配置副本仅N6→5，正式默认和业务代码保持不变。原完整a1～a5纯A5 5/5，P50/P95/墙钟173.70/183.18/183.83秒；A4 4/4墙钟195.44秒；双增强B4 4/4，284.91/296.24/296.96秒；增强驻留A5 5/5，173.32/203.23/206.12秒，严格200目标超约3.06%。本轮未复现OOM，最低HBM采样free5131原始MB（约5.01GiB），CAM排队/PCM反压及allocator缓存增长仍在。正常exit0、owned/端口/PCM回收、共享1601397/8007保留；N5不能替代原N6必需验收，change仍27/29/open。详见docs/qwen-multi-instance-20261124/N5-REPORT.md。

### 最终组合验收完成（2026-11-24）

当前正式默认Qwen4/CAM++4/emo1、总音频4/global64。真实完整B4 4/4、P50/P95/墙钟183.26/199.45/200.10秒；全部模型驻留后的纯A4 4/4、172.54/182.31/183.47秒，最低剩余HBM采样15406MB（约15.05GiB），无推理OOM，正常exit0/全部owned回收。B冷加载短clip暖身105.21秒单列，不把暖态B4当冷态完整课程成绩。完成含段级警告，B4严格无警告2/4，人工质量未认证；具体文本波动保留。剩余5.3/5.6按用户确认的最终范围验收完成，29/29；原N6 OOM及N5成绩保持历史。完整报告见[最终报告](docs/qwen-multi-instance-20261124/FINAL4-REPORT.md)，规格同步/归档身份见交付元数据。Docker未构建/上线。

测试控制器默认清单已移至稳定docs/qwen-multi-instance-20261124/test-inputs.json，归档后测试控制器仍可显式选择`--instance-count 4`，输出目录必须是新目录；仍需按独立端口/PID归属及故障停止流程执行，不自动重跑。

2026-11-24：`2026-11-23-qwen-multi-instance-serving`已29/29完成，六能力主规格已同步并严格校验，归档于`openspec/changes/archive/2026-11-23-qwen-multi-instance-serving`。当前正式默认Qwen4/CAM4/emo1；旧N6 OOM/N5结果和三个旧open change保留，不外推长期稳定/人工质量或Docker上线。所有测试服务已释放。


### 增强模型启动预加载（2026-11-24）

功能开启且设备数组非空的 CAM++、emotion2vec 在 HTTP 就绪前，按数组顺序完成全部实例的加载尝试；当前配置为 CAM4/emo1。关闭功能或空数组跳过，部分失败使用健康实例，某池全部失败仍允许其他能力服务，所需能力请求沿用原不可用错误。状态显示实际健康数及失败原因，GET 不触发加载；请求开关仅决定本次推理分支，后续请求复用已加载实例。启动耗时及权重内存占用提前发生，需重启服务生效；此步骤不执行语音 forward 暖身，首个推理仍可能有编译开销。无需新增配置或修改设备/实例数。历史懒加载描述代表此前版本。


### 2026-11-24 异常复读恢复修正

continuous已确认的异常尾部复读现在保留原前文，并把重复单元折叠为一次。例如`今天天气`后重复80次`很好`，以及整句`今天天气很好`连续循环，均保留`今天天气很好`；短英文/数字也不会仅因不足8汉字或4单词丢弃。保留首份原文空格与真实尾标点；循环在半个单元中停止时按字符周期及词界/分隔符对齐，避免截断正常前文。只处理已确认的异常尾部，不折叠前文合法强调。

折叠成功仍返回`processing_status=completed_with_warnings`，对应`segment_warnings.action=repetition_collapsed`、`recovered=true`及原段边界；不代表人工核验准确。纯填充词/空正文/无效前缀仍按原异常处理，非复读length_limit恢复策略、正常EOS、在线检测阈值和常规生成后`repetition_guard_enabled`作用不变。源码重启后生效，镜像/保护构建须在下一次构建纳入。本次不覆盖既有课程JSON，也不声称已找回历史2.02秒段。

实现/验证见`docs/qwen-repetition-recovery-20261124/REPORT.md`；当前根配置消费审计见同目录`CONFIG-AUDIT.md`。审计区分当前continuous不使用的legacy参数与仍有效的条件配置，本次配置保持原值。


### 增强能力设备数组统一配置（2026-11-24）

CAM++、emotion2vec启用状态现在只来自各自设备数组：非空数组开启，每元素一实例；`[]`关闭。数组缺省为空。`showSpk`/`showRoleIdentify`选择CAM++执行，`showEmotion`选择emo执行，均不能开启空数组能力。模型在HTTP就绪前按非空数组预加载，请求复用健康实例；A请求不执行增强，已加载权重继续驻留。预加载不执行语音forward暖身。

旧 `[features].open_v2_role`、`open_v2_emotion` 已删除，残留无论true/false都会在任何worker/模型创建前报迁移错误。迁移办法：原true删除旧键并保留设备数组；原false删除旧键，并把对应的 `speech_campplus_devices` 或 `emotion2vec_devices` 改为 `[]`。删除空 `[features]` 表。不允许只删原false键却保留非空数组，以免开启原已关闭的模型。

单项关闭请求仍在音频准入前返回4015/4016，双请求且两项均关闭返回4018，优先于容量错误，拒绝计数不变。非空数组全部加载失败是模型故障，不是配置关闭，不返回4018。数组改变需重启源码服务；保护构建和镜像在后续构建纳入。本次不构建上线；三个镜像内置小模型的必需资产预检保持，空数组不表示可以删除权重。

正式continuous根/Docker模板省略两项legacy组批参数和两个固定增强batch配置；legacy显式参数支持、实际batch1及非1校验保持。两个 `prefetch_queue_depth=0`、全部 `online_repetition_*` / `repetition_*`、`retry_count`、队列及VAD参数保持；预取0是在上游反压等待，不是无限预取。早先章节中涉及旧功能开关的已日期标记说明仅代表当时版本，以本节为当前规则。
