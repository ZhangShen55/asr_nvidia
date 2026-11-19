# AGENT.md

> 本文件面向在本仓库工作的 AI Agent，以及第一次接手这个项目的人。  
> 前半部分说明「这是什么系统、代码在哪」；后半部分说明「哪些事不能做、做完怎么算完成」。
>
> **冲突时的优先级**：实测数据 > `openspec/specs/` > `README.md` / `docs/` > 本文件。  
> 本文件若与代码现状不一致，以代码为准，并请顺手修正本文件。

---

## 0. 速查

| 项目      | 值                                                                     |
| ------- | --------------------------------------------------------------------- |
| 代码位置    | `xjtu910b7:/home/zhangs/asr_nvidia`   |
| 服务端口    | `8083`（容器内 `9000`，compose 映射 `${ASR_PORT:-9000}`）                     |
| 进程名     | 主服务 `asr-offline-model`，VAD 独立 `asr-offline-vad`                             |
| 启动      | 先 source CANN set_env.sh，再用 `/home/zhangs/conda-envs/asr/bin/python` |
| 单测      | `pytest tests/`                                                       |
| 规格目录    | `openspec/specs/`（当前基线）、`openspec/changes/archive/`（演进历史）             |
| 镜像构建    | `docker/build.sh`，单昇腾模板；镜像构建/上线后置                           |
| Qwen 依赖 | 本地原始权重，`[qwen_asr].local_model_dir` + `local_device`                      |

> 注意 `test/`（单数）是**实验与压测脚本**，`tests/`（复数）才是 pytest 单元测试。两者不要混。

---

## 1. 项目是什么

**SeaCraftASR**：面向课堂录音的离线语音转写服务（FastAPI）。

- 当前**只有 v3 一条链路**：`POST /v3/seacraft_asr`。v1.1.8（SeACo-Paraformer + PUNC + Whisper）和 v2 路由已下线，代码与模型均已清理，不要"顺手恢复"。
- FSMN-VAD是固定spawn进程池；CAM++懒加载池可选thread/process（安全默认thread），emotion2vec保持主进程runner。
- **Qwen3-ASR是本地懒加载原始权重**，BF16 / npu:7；支持独立段物理批执行，单实例同一时刻只持有一个物理执行。当前安全缺省仍batch1、段/全局1/1，候选16/128尚在本轮标定。
- 对外只有三个接口：`POST /v3/seacraft_asr`、`GET /get_status`、`DELETE /clear_tasks_list`。

### 业务目标（容易理解错，先看这段）

要的不是"把更多音频塞给 Qwen"，而是**让 Qwen 只处理真正该转写的内容**：

- **不该送进 Qwen 的**：课堂闲谈、咳嗽、翻页、敲击、椅子摩擦、教室环境底噪等非讲授语音
- **必须保住不丢的**：教师讲授、学生发言，**包括带方言口音的发言**（不因口音、低音量、语速慢而被当成噪声剔掉）
- 理想结果是两者同时成立：噪声几乎不进 Qwen，学生方言一句不丢

阈值取舍的原则由此确定：**降噪的收益是省一次推理，误删的代价是一段内容永久缺失**。
两者不对等，所以宁可放过少量噪声，也不能误杀有效发言。"放过噪声"是容错边界，**不是目标**。

> **当前能力边界（别误以为已经做到）**：现有三重过滤（低能量帧比例 / SNR / RMS dBFS）全部基于
> **能量与信噪比**，能拦的是咳嗽、翻页、敲击、椅子摩擦这类弱能量段；**课堂闲谈是正常人声，能量不低，
> 现有实现拦不住，仍然会送进 Qwen**。要真正区分"讲授 vs 闲谈"，属于
> `docs/课堂录音VAD细小声音过滤与说话人统计优化方案.md` 里的后续阶段（环境音分类模型 / 专用分类器），
> 不是调阈值能解决的。讨论这块需求时先讲清这条边界。

---

## 2. 一次请求怎么跑

```
上传音频
  ↓ 整段解码并使用服务专属缓存供VAD进程读取
VAD 固定进程池（vad_device 数组，每元素一个进程）
  ↓ FSMN-VAD 切分 → min_segment_duration_s 短段丢弃
  ↓ 过滤窗口判定：只有落在开头 after_start / 结尾 before_end 分钟内的段才进三重过滤，
    窗口外的中间段无条件保留（宁可放过噪声，也不误删讲授主体）
  ↓ 段级三重过滤（命中即丢，不发 Qwen）
      ① vad_decibel_low_frame_ratio_thres  低能量帧比例
      ② vad_snr_thres                      段级 SNR
      ③ vad_low_volume_rms_dbfs_thres      段级 RMS dBFS
  ↓
每段1个Qwen逻辑任务（内存WAV，单音频/全局窗口独立计数，多个独立段可组成物理批）
  ├─ 独立队列：CAM++声纹（从0.3秒实际推理）→ 全音频聚类；仅showRoleIdentify需要时作教师/学生判定
  └─ 独立队列：emotion2vec情绪（1～10s闭区间外记"平淡"）
Qwen完成即还自身名额，增强队列/汇合受段数和PCM字节预算反压；按原段汇合后继续后处理
  ↓
复读修复 → ITN → 标点卫生（纯标点段删除、去段首标点）→ 语速 / 语种占比
```

**核心不变量**（改代码时最容易踩坏的地方）：

1. **一个有效VAD段对应一个独立Qwen逻辑任务及至多一个有效响应分段**。允许物理批执行，不合并相邻段、不拼PCM；`bg`/`ed`取原VAD边界，既有空文本/标点过滤仍可能删除响应段。
2. **分段音频在内存流转**，不新增段级临时 WAV；整段上传沿用服务专属缓存供VAD worker读取，请求finally清理。
3. **VAD 池是静态的**：启动时一次性预热，任务 FIFO 分派，运行期不扩缩容。数组长度就是并发上限。
4. **NPU 加载失败禁止静默走 CPU**；预检/加载错误必须按阶段暴露。

---

## 3. 代码地图

### 入口与装配

| 文件                                                 | 职责                                   | 改动前必读                    |
| -------------------------------------------------- | ------------------------------------ | ------------------------ |
| `main.py`                                          | FastAPI 装配、lifespan、进程命名、NPU 环境/资产预检 | `core/deployment.py`     |
| `config.toml`                                      | 本地运行配置（唯一配置源）                        | `core/config.py` 的字段定义   |
| `core/config.py`                                   | 配置加载、校验、派生属性                         | 570 行，改字段要同步 README 配置章节 |
| `model_status_runtime.py` / `telemetry_runtime.py` | 运行时轻量封装                              | —                        |

### HTTP 层

| 文件                              | 职责                                  |
| ------------------------------- | ----------------------------------- |
| `api/routes/asr_v3.py`          | `/v3/seacraft_asr` 路由与参数校验          |
| `api/routes/asr_qwen_common.py` | v3主链路：VAD → 有界独立分支 → 原段汇合及响应 |
| `api/routes/asr_common.py`      | 请求上下文构建、情绪标签映射、临时文件清理               |
| `api/routes/status.py`          | `/get_status` 路由                    |
| `entity/data.py`                | `AsrRequestParams` 请求数据类            |

### 核心能力

| 文件                             | 职责                                                             |
| ------------------------------ | -------------------------------------------------------------- |
| `core/vad.py`                  | VAD 切分配置组装                                                     |
| `core/vad_pool.py`             | 固定spawn VAD池，完整音频FIFO、任一完成即补位与排队/worker包络计时 |
| `core/vad_volume.py`           | 段级声音特征（RMS / SNR / dBFS）与低音量过滤，**运行时与标定脚本共用同一套口径**             |
| `core/qwen_asr_queue.py`       | 三层并发闸门：音频准入分类 + 段级/全局请求信号量                                     |
| `core/qwen_asr_client.py`      | 本地Qwen实例租约、批执行/重试与批遥测 |
| `core/qwen_asr_runtime.py`     | 列表音频、逐成员语言/mask/输出及实际processor shape |
| `core/qwen_asr_process.py` | 单Qwen spawn模型进程、READY/RPC归属、独立资源与显式关闭grace；CPU状态不触发加载 |
| `core/qwen_batch_scheduler.py` | 公平共享队列、长度分桶、有限等待及估算帧/token预算 |
| `core/segment_pipeline.py`    | 每音频有限元worker、独立全局段数/PCM预算、取消安全汇合及异常链脱离 |
| `core/models.py`               | CAM++/emo懒加载、共享有界预取、FIFO反压、即时交接、HBM/忙闲状态 |
| `core/cam_process.py` | CAM原生FP32模型spawn worker、READY/RPC出生归属、Tensor[1,192]与显式关闭 |
| `core/local_inference.py`      | 设备 → 后端映射（`pytorch_cpu` / `torch_npu`） |
| `core/deployment.py`           | 昇腾 profile 启动前环境与资产校验                                 |
| `core/status.py`               | 状态接口响应组装                                                       |
| `core/logging.py`              | 日志按天 + 按大小双轮转                                                  |
| `core/request_telemetry.py`    | 有界请求级阶段遥测（只存统计，不存音频与正文）                                        |
| `core/progress.py`             | 可选的每音频处理进度显示                                                   |
| `core/async_utils.py`          | `await_completion`：取消时等待已启动工作收尾，避免提前归还资源                       |
| `core/resource_diagnostics.py` | 模型内存压力安全采样                                                     |

### 文本与工具

| 文件                          | 职责                                         |
| --------------------------- | ------------------------------------------ |
| `utils/itn.py`              | WeTextProcessing 逆文本规范化                    |
| `utils/repetition.py`       | 生成复读（幻觉循环）检测与折叠                            |
| `utils/text_postprocess.py` | 标点卫生、语气词处理等确定性规则                           |
| `utils/qwen_response.py`    | Qwen 返回解析：`language X<asr_text>` 前缀剥离、语种占比 |
| `utils/audio_utils.py`      | 音频解码、共用round端点裁剪/EOF clamp、内存WAV、服务缓存目录 |
| `utils/asr_stats.py`        | `asr_stats.json` 原子读写（带文件锁）                |
| `utils/feature_utils.py`    | 说话人角色判定（教师关键词等）                            |

### 周边目录

| 目录                                   | 内容                                                     | 是否入库                         |
| ------------------------------------ | ------------------------------------------------------ | ---------------------------- |
| `openspec/specs/`                    | 9 份能力契约（系统基线）                                          | 是                            |
| `openspec/changes/archive/`          | 19 个已归档变更，是**演进决策的唯一可信来源**                             | 是                            |
| `docs/`                              | 压测报告、方法说明、验证结论，命名带日期                                   | 是                            |
| `scripts/`                           | 压测 / 标定 / 导出 / 校验脚本（`bench_*`、`validate_*`、`export_*`） | 是                            |
| `tests/`                             | pytest 单元与集成测试                                         | 是                            |
| `test/`                              | 实验脚本与样本音频                                              | 脚本入库，`test/*/` 产物忽略          |
| `docker/`                            | 单昇腾 Dockerfile、`build.sh`、compose、部署说明                  | 是                            |
| `model/`                             | 三个模型权重                                                 | **否**（`.gitignore`，但会进构建上下文） |
| `.agents/skills/`                    | openspec 工作流 skill 定义                                  | 是                            |
| `asr_service.log` / `asr_stats.json` | 运行产物                                                   | **否**                        |

---

## 4. 三层并发与错误码

| 层       | 参数                                                | 语义                            | 当前值                 |
| ------- | ------------------------------------------------- | ----------------------------- | ------------------- |
| 整音频准入   | `audio_concurrency` / `audio_spk_emo_concurrency` | A（纯转写）+ B（要说话人/情绪）总数 / B 子集上限 | 8 / 4               |
| Qwen 请求 | `segment_concurrency` / `request_concurrency`     | 单音频内段并发 / 全局段并发               | 1 / 1（完整课堂对照；单Qwen实例）            |
| 本地模型池   | `speech_campplus_devices` / `emotion2vec_devices` | 数组长度 = 实例数 = 段级推理并发           | 3×npu:7 / 1×npu:7 |

音频准入、Qwen逻辑窗口、物理批大小与模型实例容量分别计数；增加等待任务不增加模型实际计算容量。表中当前值是前次配置快照，新增吞吐候选与实施状态见下一节；历史压测见`docs/cam*`、`docs/vad-pool*`。

**错误码契约不可随意改号**（客户端已按此判断）：完整定义分散在  
`core/qwen_asr_queue.py`、`core/qwen_asr_client.py`、`core/models.py`、`core/vad_pool.py` 的异常类 `code` 属性；  
对外说明在 `README.md` 错误码表。要点：

- `4008` 无人声 · `4011` Qwen 不可用 · `4012` Qwen 调用失败 · `4013` 总容量满 · `4017` 仅 B 类满 · `4010` 排队超时
- `4015` / `4016` / `4018` 能力未启用或无可用实例
- **能力开关校验早于容量检查**；两项容量同时满时优先返回 `4013`

---

## 4.1 当前吞吐调度变更（2026-11-18）

`2026-11-18-npu-asr-throughput-scheduling`处于apply，尚未完成性能验收或归档；前次`2026-11-17-npu-910b-local-qwen-asr`已完成并归档，历史实测结果不因本轮候选改写。以当前change的tasks、design和test-plan跟踪必需验收。

- 当前安全缺省：单Qwen、batch1、每音频/全局窗口1/1；现有`config.toml`的混合四VAD及CAM三/emo一属于前次交付配置。本轮独立候选profile为`/tmp/asr-throughput-20261118-xvior99k/candidate128.toml`，定义VAD八CPU、CAM/emo各八NPU:7、Qwen每音频16/全局128、批上限128、预取5、总8/B4；纯B8另配profile。配置校验不等于真实容量/吞吐通过，最终值须按实测更新本文件与README。
- 新`[qwen_asr]`字段：`max_inference_batch_size`缺省1（1～128）、`batch_wait_timeout_ms`缺省10（0～1000ms）、`max_batch_audio_frames`缺省102400、`max_batch_tokens`缺省78336、`pending_segment_limit`缺省256、`pending_audio_bytes_limit`缺省268435456（256MiB）。完整类型/范围见README和`core/config.py`，修改后重启。帧估计为最长100Hz帧数×B，token估计为`(ceil(max_frames/8)+max_new_tokens+256)×B`；processor实际shape另存，估计不保证显存容量。
- 新`[model_device_pool]`的`speech_campplus_prefetch_queue_depth`、`emotion2vec_prefetch_queue_depth`缺省0（0～64，0无预取），各`*_batch_size`仅支持1。数组长度=实例数；每健康实例等待深度=预取；batch=实际计算。八实例深度5为40等待+最多8运行，不是batch5。共享任务源不将未执行段锁死在忙副本，空闲即补位，反压等待另计。
- 每条完整音频最多占一个VAD worker。8条/4worker时任一FINISHED即分派下一条，不等最初四条或下游；VAD租约与整音频租约分开归还。Qwen完成即释放自己的段窗口，CAM/emo各自归还，最终仍按请求/段标识汇合；段数/PCM预算满时明确反压，不无限堆积future或PCM。
- 每音频元任务worker至多`min(段数,pending_segment_limit)`，缺省256；总八音频至多2048个元任务，不是全局future最多256。真正切片和分支开工另受全局256段/256MiB PCM预算限制，预算满则worker等待，必要分支全部收尾才归还；Qwen逻辑窗口独立释放。响应文本/embedding不计作在途PCM。
- 每次物理attempt独立记录输入成员、真实有效成员、状态及runtime包络；重试排除等待中已取消成员，批summary仅代表最后attempt，不能把上轮有效索引映射到新的幸存列表。传给结果future或汇合结果的异常脱离traceback/cause/context，保留类型/code/message，避免异常链滞留PCM与活动worker frame。
- 正式tensor/NumPy/毫秒裁剪共用起止端点分别round后的采样格并clamp至源范围，EOF不padding、不改原VAD bg/ed；实际裁剪不足0.3秒在三模型前丢弃，emo的1/10秒边界同样依据实际样本。旧FFmpeg秒级转码路径保持兼容，不为此逐段重读完整大音频。
- 本轮候选最短段0.3秒，三项服务能量过滤均false，`vad_decibel_thres=-100`、`after_start=-1`、`before_end=-1`；-1/-1仅不限位置，不能充当关闭开关。FSMN实际绑定/四层证据不变。不足0.3秒在三模型分发前移除；0.3至不足1秒CAM实际推理且embedding参与聚类，emo直接“平淡”；emo仅[1,10]秒推理，未请求能力仍null，roleIdentify本轮不传/false但保留原可选能力。
- 必测同组八条完整SHA去重音频的1/2/4/8并发和同源A/B，冷/暖态分列；默认B4与B8profile分列。单Qwen128必须区分纯Qwen与八CAM/八emo已加载且实际工作的容量，不能互相替代。单份有问题先诊断，再有条件试2/4/8副本与段级均衡。
- 完成P50/P95或任一在途完整请求墙钟超过420秒，下一监测周期（默认1秒）停止追加，捕获准入/VAD/三个模型/汇合/后处理及资源卡点，有界取证并关闭owned资源，记录diagnostic_aborted/未完成。不得原样无限重跑或只报成功快请求；不改变业务HTTP或单Qwen超时。阶段无进展提前告警，定位后先小规模复验。
- 每轮finally按PID和创建时间关闭本轮服务/worker，等待设备和读取完成，验证无孤儿和测试监听释放至少5秒；既有服务保留。内部切片/排队/批等待/CPU前端/传输/generate/同步分开，批包络只计一次，原`gpu_time_ms`不改义。真实Docker构建和上线仍后置。

### CAM-only模型进程候选（吞吐变更v9）

- `[model_device_pool].speech_campplus_execution_mode`支持thread/process，缺省和宿主/容器模板均保持thread；process仅支持合法NPU设备数组（含容器npu:0），数组长度仍是唯一实例数来源，空数组禁用。新增`speech_campplus_process_startup_timeout_s=60`、`speech_campplus_process_shutdown_grace_s=30`均要求真实有限数、0<值≤300，拒bool/字符串；就绪前配置校验不加载模型。
- 每CAM worker保持原torch_npu、FP32及CPU float32 Tensor[1,192]输出；emo留父进程、Qwen独立配置，不新开HTTP端口、不换格式/权重或升级依赖。尚未请求CAM和查询状态不spawn/RPC。普通取消等真实DONE，只有显式关闭启用grace并按PID/出生身份回收；强制、坏ACK或非0退出为失败，确认死亡前不还真实容量。
- 主循环读取ModelPool的`model_workers`缓存与每实例memory（含loading/retired），资源collector仅只读OS。父/Qwen/CAM/外部/未验证进程分开，CAM只按出生身份吻合且未退出归属；worker allocator/lastRPC峰值、OS RSS/PSS、IPC额外副本及设备HBM各自保留，不用父0代worker0、不加不同时刻峰值。保持状态singleflight/age语义。
- 原型组件结果不等于正式TorchTensor聚类、动态全语料或完整音频矩阵通过。每worker约16秒启动、八进程原型约272秒全程是冷启动/装载/执行/关闭成本，不能当暖态推理耗时。最终默认只有完整矩阵及源/保护版本验收后才推广，当前安全配置其他值不变。

### 单模型进程与多文件公平供给（吞吐变更v5）

- 新`execution_mode`支持thread/process，当前安全缺省thread；process支持经校验1～8配置实例（实际NPU合格数量另列），每子进程加载一份同HF Qwen，父进程不重复加载，不增加HTTP服务或更改引擎/精度。设备直接沿用宿主npu:7/容器逻辑npu:0。模式更改需重启，最终默认待整音频验收。
- 新`process_shutdown_grace_s`缺省30，有限正数≤300秒，仅显式关闭生效，不将单请求120秒timeout改为kill；超期按worker PID/创建时间有界终止/join且关闭失败。服务shutdown入口提前启动Qwen关闭看护，避免增强关闭先阻塞。状态查询不懒启动、不等待RPC；worker与父进程的allocator/峰值/RSS分开，IPC副本不伪称受主PCM256MiB覆盖。
- 已完成固定128段batch1/8/16/32/64/128每档三轮稳态及独立质量复算，文本差异不作准确率结论。单模型process在16/128段B配对比同进程更快，所有Qwen/CAM/emo对照及owned关闭通过。process预取0/1/5三档实测当前0中位最低；新候选`final-process-b4.toml`及`final-process-b8.toml`保留旧候选不覆盖，整音频矩阵仍待验收。证据见`docs/npu-throughput-20261118/qwen-process-paired-v5.json`、`process-prefetch-selection-v5.json`。
- 先获取每音频/全局Qwen credit再占PCM预算，保证八个VAD-ready文件可共同供给128。Qwen实际收尾即释放，剪切错误/EOF/取消幂等归还。等待PCM的预留credit不等于实际执行段，真实缓冲耗尽仍反压。

### Qwen独立前端与批预算（吞吐变更v2）

- 每成员在FP32 guard中独立调用processor，再把有效tokens左补齐、完整single features/mask额外右补齐；不得先把不同PCM补齐再STFT。保留single自带的无效尾部特征上下文，batch1不重排。未知processor键/shape/dtype明确失败，不改site-packages或权重。
- 当前锁定processor为`hop_length=160`、`min_length=8000`、`n_window=50`。调度估计`frames=floor(max(samples,8000)/160)`，完整特征补至`100*ceil(frames/100)`；帧预算必须计入这100帧补齐。token预算仍是8:1音频估计加生成上限和256提示预留，不当作真实token数/HBM保证。模型/processor版本改变先复验这些属性。
- 新`[qwen_asr].batch_max_duration_ratio`缺省4.0，接受1～64有限数，布尔/字符串/NaN/Inf就绪前拒绝。它控制有效STFT帧最长/最短比，含原0.5秒前端下限；32.0仅为候选。与实例数、单音频/全局窗口和批上限独立，修改后重启。
- 越过不兼容队首取尾成员需要防饥饿：每批公平锚点选全局最老等待队首，再在兼容集合中按请求轮转。不能仅凭成员轮转声称旧长段必有机会。
- CPU真实128段前端等价证据见`docs/npu-throughput-20261118/qwen-frontend-equivalence-v2.json`；原差异及HOL诊断见`qwen-quality-initial.json`。CPU等价及配置校验不表示NPU文本一致、完整并发矩阵或默认性能已经验收。

## 5. 配置体系

- 唯一源：`config.toml`（容器内通过 `CONFIG_PATH=/config.toml` 挂载覆盖，镜像内另有模板）
- 加载器：`core/config.py`，字段名与 TOML 键一一对应
- **配置错误必须在模型预热和对外就绪之前抛出**（`main.py` lifespan 第一段就是这个目的），不要改成懒校验
- 部分参数改动**需要重启生效**（设备数组、本地目录、`vad_speech_noise_thres` 等），改完必须在注释里标明

---

## 6. 构建与部署

交付入口是 `docker/build.sh` 与单份昇腾 Dockerfile，详见 `docker/README.md`。
基础镜像使用经核对的本地候选，环境通过独立 conda-pack 包或已核验前缀提供；不能依赖候选镜像里未知的 Python 包。运行、NPU、开发、构建依赖分开锁定。
`build_cython.sh` 只在有 `.asr-build-stage` 标记的 staging 中剥离源码；不要在真实仓库执行源码剥离。
镜像 tag 仍为 `v3_<YYYYMMDDHHMMSS>`。实际镜像构建与上线后置，不能把静态交付测试当作已部署。

宿主仅使用物理 `npu:7`，不调度其他卡。`ASCEND_VISIBLE_DEVICES` 指定宿主物理卡号，容器内设备编号从 0 重新映射；仅映射物理 7 时模板使用逻辑 `npu:0`。
NPU runtime/设备/IPC 参数放 Compose/宿主入口，`docker/start.sh` 只管理容器内环境、预检与进程。
`instance_count=1`；增加完整HTTP服务进程或Qwen实例会复制权重，重新标定后才调整。模型process执行是把每份副本放入独立worker，父进程不重复加载，不能与服务多进程混为一谈。
测试启动的服务、代理、候选引擎、worker 必须 finally 关闭，核对 PID 树退出和端口释放，保存证据；保留卡 7 既有服务。

## 7. openspec 工作流

本项目的规格与变更记录是**一等公民**，不是可选文档。

```
openspec/specs/<capability>/spec.md      当前系统「必须做到什么」（基线）
openspec/changes/archive/<change>/       历史上的每次变更（proposal / design / tasks / spec delta）
```

已归档的 19 个 change 构成完整演进史，遇到"为什么这么设计"的问题，先查 archive，别猜。

**写作约定**（`openspec/config.yaml`）：

- 正文中文；`Purpose`、`Requirement`、`Scenario`、`ADDED Requirements` 等结构关键字保持英文
- change 目录名 `YYYY-MM-DD-kebab-case`，日期取创建当天

**工作流**（`.agents/skills/` 里有对应 skill）：

1. **propose** — 只产出规划产物（proposal / design / tasks / spec delta），**不写业务代码**
2. **apply** — 按 tasks 实现，勾选任务
3. **archive** — 实现完成 + 验证通过后归档，**同时把 delta 同步进 `openspec/specs/`**

---

## 8. 项目开发约束（人 + Agent 共同遵守）

以下约束来自这个项目已经形成的做法，不是凭空规定。

### 8.0 一次改动的标准流程

不要从"打开文件开始改"起步，按这个顺序走：

1. **先同步**。`git status` / `git pull --rebase` 把本地拉到最新，确认在 `refactor/adapt-qwen-asr-2huawei910b` 分支上、没有别人刚推的改动。**不基于过期代码开工。**
2. **再规划**。业务改动先开 openspec change（propose）；纯文档/typo 可跳过。
3. **改代码**，同步改配置与文档（见 8.2）。
4. **验证**：`pytest tests/` + 相关压测，结论写进报告。
5. **提交并推送**：中文 Conventional Commits，代码 + 报告 + openspec 产物一起进。
6. **变更完成就归档**：openspec change 的 tasks 全勾选、验证通过后，`archive` 并同步 delta 到 `openspec/specs/`，不要让它长期挂在 open 状态。
7. **回写文档**：接口、配置项、默认值有变化时，同步 `README.md` 和本文件。

一句话：**改前先同步，改后必推送留档，change 完成即归档。**

### 8.1 绝对红线

1. **不碰 `model/` 下的权重文件**。它们不进 Git，重新获取成本极高。需要换模型/换 engine 时只改路径与配置。
2. **不执行 `docker builder prune` / `docker system prune`**。构建缓存是这套交付流程能跑通的前提，清掉等于每次全量重建。
3. **不擅自重启或 kill 正在跑的服务**。当前有真实请求在跑，`kill` 会丢任务；需要重启时先确认空闲或按 `docs/DEPLOYMENT.md` 走。
4. **不删除任何镜像 / 日志 / 报告**，除非明确授权。删除前先列出清单并确认无容器引用。
5. **不改错误码数值**，不改接口字段名。这些是对外契约。
6. **不在 `main.py` 之前做"临时验证"式的代码改动**去绕过启动预检。预检失败就是失败。

### 8.2 改代码

1. **先有 openspec change，再动业务代码**。纯 typo、注释、文档修正可以例外。
2. **参数取值必须来自实测标定**，不接受"看起来合理"。已有先例：
   - `speaker_similarity_threshold = 0.3` ← 来自 `scripts/speaker_threshold_sweep.py` 的 sweep
   - 复读阈值 40 字/秒 ← 来自 68,018 段真实样本 p99.9 = 20.3、峰值 26.2
   - VAD 三重过滤阈值 ← `docs/vad-segment-filter-range-validation-20261105.md` + 上线后实测；其中 `vad_snr_thres = 18.0` 是**当前实际承担主要过滤**的一档

   改这些值，必须附上新的统计依据。
3. **改配置项就要同步三处**：`core/config.py` 字段、`config.toml` 默认值与注释、`README.md` 配置章节。少一处就算没改完。
4. **保持 1 段 = 1 请求 = 1 响应分段**，不要为了"效率"合并相邻段。
5. **新增可选依赖必须延迟导入**（`importlib` 或函数内 import）。CPU 环境、无 NPU 依赖的 CPU 测试环境都要能启动。
6. **注释写"为什么"，不写"做了什么"**。现有代码的注释密度偏高，是刻意的，保持一致。
7. **关键逻辑处必须有中文注释**：阈值来源、单位、边界条件、为什么这样取舍（尤其是踩过坑的地方）。新增过滤/判定/并发逻辑时，注释要能让下一个人不看 commit 就看懂意图。
8. **可解耦的配置值一律释放到 `config.toml`**，不要硬编码在代码里。判断标准：这个值调大会变、不同机器/场景可能不同、或者需要在不改代码的情况下试。释放时配套三件事：
   - `core/config.py` 里给类型化字段 + 默认值
   - `config.toml` 里给注释说明取值范围与影响
   - `tests/test_config_contract.py` 覆盖新字段，避免默认值被悄悄改掉
   - 反过来，**纯结构常量（错误码、协议字段名、模型输入 shape）不要配置化**，那属于契约
9. 遵循现有代码风格：`from __future__ import annotations`、dataclass 配置对象、异常类携带 `code` + `message`。

### 8.3 验证与留档

1. **跑 `pytest tests/`**，改动涉及 VAD / 模型池 / 并发时必须跑相关子集并说明结果。
2. **性能与阈值类改动必须出报告**，放 `docs/`，文件名带日期（参考现有 40+ 份报告的命名）。
3. **上线门槛**：有效对话召回 ≥ 98%。低于此值的方案不予采纳，写进报告结论。
4. **不做"只保留老师声音"这类规则**。会误删学生发言，与业务目标直接冲突。
5. 报告要写清：测试条件（NPU、共享占用、并发、样本量）、结论、以及**不适用的场景**。

### 8.4 提交与留档

- **改前先同步、改后必推送**。改完不推送等于没做——这个项目靠 Git 历史 + openspec archive 追溯决策，留在本地工作区的东西下一个人看不到。
- 中文 Conventional Commits，scope 用中文或模块名：`feat(VAD): 增加段级过滤配置与范围控制`、`fix(状态): 显示处理中音频文件名`、`docs(部署): ...`
- **一次提交一件事**，代码 + 对应报告 + openspec 变更一起提交
- 归档 openspec 变更单独一条 `chore(归档): ...` 或 `chore(openspec): ...`
- **openspec change 一旦 tasks 全部完成且验证通过，当次就归档**，不要攒着。长期挂在 open 状态的 change 会失真。

---

## 9. Agent 行为约束

除了上面 8.x 的通用约束，Agent 在本项目还须遵守：

1. **先读后写**。改动前至少读过：目标文件、它调用的 `core/config.py` 字段、对应 `openspec/specs/` 条目。不确定就去读 archive。
2. **改动前说清影响面**：会动哪些文件、影响哪条链路、需不需要重启、要不要重建镜像。用户确认后再动手。
3. **区分"只读探查"和"修改"**。用户说"理解项目"就是只读，不要顺手改格式、补注释、删无用代码。
4. **不许在没有实际执行的情况下声称完成**。测试没过就说没过；构建被超时打断，要区分"构建失败"和"客户端断连但远端仍在跑"。
5. **长任务的正确姿势**：
   - `docker build` 单份 8~25 分钟，SSH 命令超时会断开，但远端进程仍存活
   - 先 `nohup`/后台方式起，再用 `ps` / `docker images` 轮询，**不要因为 SSH 退出码 137/143 就判定构建失败**
   - 判断"是否真在推进"看 IO 采样（`/proc/<pid>/io`）或 `iostat`，不看 CPU
6. **删除类操作三件套**：列出清单 → 确认无引用（`docker ps -a --filter ancestor=`）→ 用户确认 → 执行。绝不 `rm -rf`。
7. **不擅自升级依赖、不擅自改 `requirements*.txt`**，也不跑 `pip install -U`。
8. **不碰 git 历史**：不 `reset --hard`、不 `push -f`、不 `amend` 已推送的提交。
9. **发现规格与实现不一致时，先报告，不擅自"修正"**。这类偏差往往是有意的上线调参，写清楚差异让用户判断。
10. **输出用中文**，结论先行，给出可核对的证据（文件路径、行号、命令输出），不要只说"已完成"。
11. **能并行就并行，但要知道什么不能并行**。任务可拆成互不依赖的子任务时，用多智能体并行推进，不要串成一条长链：
    - **适合并行**：多个 `openspec/specs/` 条目的交叉比对、多份 `docs/` 报告的检索与归纳、互不相干的独立文件编写、互不争抢同卡的独立评审
    - **不该并行**：同一文件的改动（写冲突）、三份镜像构建（共享 build cache 与磁盘 IO，实测并行反而更慢）、需要前一步结论才能决定的验证
    - **并行前先划分资源**：NPU 卡、端口、临时目录、输出文件都要明确归属，不能两个 agent 抢同一张卡或同一个 `asr_service.log`
    - **汇总由主 agent 负责**：子 agent 的结论必须自己核对过（读文件、看命令输出）才能写进最终答复，不做纯转述

---

## 10. 常见坑


| 坑 | 说明 |
|---|---|
| CANN 环境脚本 | `set_env.sh` 部分变量未定义；在 `set +u` 下 source，随后显式使用 asr 解释器。 |
| CANN 编译器依赖 attrs | 包名是 `attrs`，导入模块是 `attr`；预检缺失必须明确报错。 |
| 容器内设备编号从 0 重新映射 | `ASCEND_VISIBLE_DEVICES=7` 是宿主物理 7，单卡容器内配置逻辑 `npu:0`。 |
| FP32不等于默认内部精度 | 原生FP32进程一次禁用conv/matmul HF32；参数dtype为FP32本身不足以保证数值参考一致。 |
| NPU context 默认卡 | 工作线程先 set_device 目标卡，再构造 device context，避免初始化/恢复到卡 0。 |
| VAD声明不等于实际选项 | nested传入值未覆盖当前权重model_conf；FSMN实际8000/800ms、noise0.6、global decibel/snr均-100。服务段级SNR18/-40独立生效；见课堂阈值评审，不借迁移改参数绑定。 |
| `test/` vs `tests/` | 前者是实验脚本，后者才是 pytest。 |
| `config.toml` 与文档默认值不一致 | 上线后经过实测调参，历史文档里留下的默认值（如 `after_start/before_end` 写过 3/3、6/6、0/0）都是调参过程的快照。**以 `config.toml` 为准**，同时记得回写 README。 |
| `vad_snr_thres` 的定位 | 当前取 `18.0`，**它是段级过滤的主要承担者**（历史统计里 665 次丢弃中 614 次是 snr 命中）。不要因为早期报告说"SNR 不承担主要过滤"就把它调回去。 |
| `after_start` / `before_end` 是**过滤窗口**，不是"处理范围" | 默认 **3/3** 的含义是：只有开头 3 分钟、结尾 3 分钟内的段才做三重过滤，**中间的段一律不过滤、直接保留**（`core/vad.py::_in_filter_scope`）。极易被误读成"只处理前 3 分钟"。出现过 6/6、0/0 都是试参数的过程，不是新基线；改动需重启生效。 |
| Qwen 原始输出 | 保留 `language X<asr_text>` 给既有解析器；浮点输入转模型 dtype，整数索引保持原 dtype。 |
| dBFS ≠ FunASR 的 log 能量 | `vad_decibel_thres` 是 FunASR 帧级 log 能量，不是 dBFS；段级 RMS 才是 dBFS。单位混淆会导致阈值差 40 个数量级。 |
| HBM 与实例数相关 | `speech_campplus_devices` 数组每多一个元素就多一份权重。`instance_count` 同理。 |
| 共享卡测量 | 卡 7 有既有 uvicorn；分开记录本项目进程 HBM 与设备聚合占用，不把既有服务计为本项目。 |

---

> **本文件的维护**：新增能力、修改接口字段、调整默认配置时，同步更新本文件对应章节。
> 发现本文件与代码不符，先改本文件再改代码注释——不要让第三个人再踩一次。


### v6状态资源采样约定（2026-11-18）

- 新`[status].resource_sample_interval_s`缺省1.0秒，严格0.1～60有限数（拒bool/string/NaN/Inf），修改需重启。按需TTL singleflight，不维护无人查询的永久NPU轮询；每app/loop独立拥有者。
- GET实时计数不等待重资源collector；缓存首次unknown，旧值带`resourceDiagnostics.sampling`的age/in_progress/last_error和last attempt起止/elapsed。elapsed含线程派发，非核时；cache age与失败attempt耗时分别记录。
- shutdown入口先发布sampler closing，与模型收尾并行，finally等待实际线程结束；重复取消不丢在途任务。采样后台线程只读OS，模型/准入可变状态仍主事件循环获取。
- Qwen独立worker的HBM按PID及出生身份核对，backend/Qwen/external/unverified分列。父进程0不是worker0；资源快照age与worker allocator的最后RPC口径不能混为实时值。
- 实机已发现旧GET约2秒且诊断读取5秒超时：两轮准入未通过均保留，新增本change任务2.8；v5源码695/保护696及NPU模型验证原证据不改写，v6状态改动须重新CPU/保护/实机及完整矩阵验收。


### 吞吐变更v16的状态历史窗口（apply中）

`/get_status`的Qwen `batch_scheduler.recent_batches`改为最近8条完整记录；保留全部成员/attempt/shape/计时，`history`明确可用/返回ID范围、截断和缺口提示，实时计数不截断。直接客户端/调度器`snapshot()`默认仍最多64条。HTTP直接渲染隔离的JSON数据，避免完整历史重复递归编码阻塞推理调度；这是有界快照，不是无损日志，审计必须去重并检测采样遗漏。CPU候选开销降低不等于完整音频性能已验收，仍须实机420秒闸门和最终矩阵；模型精度、生成上限及v3业务响应未改。


### 吞吐变更v17：内存WAV补位（apply中）

段级编码增加经原SoundFile逐字节验证的PCM16格点快速路径：仅原生一维native float32、有限[-1,1]且值落在16bit格点、采样率合法时直接序列化同一PCM16 WAV；其他输入保留原SF分支，不改变CAM/emo原PCM。128真实段及全部65536格点须源/实际so等价。CPU真实裁剪ABBA支持100ms组批等待候选，50ms仍碎；假模型结果不代表NPU速度，正式默认仍待完整NPU矩阵。NumPy/bytes临时工作buffer与pending PCM预算分开统计。模型仍原HF/BF16与原生FP32 torch_npu，不改模型格式/依赖或生成上限。


### v18 Qwen多owner条件验证（当前apply）

v17单份8B仍420诊断3/8且30秒显式关闭宽限触发inner abort/main1，不能以资源已回收判正常。当前进入process2×batch64、16/128逻辑窗口候选，保持一HTTP和原八VAD/八CAM/八emo/精度/源资产。process1～8配置范围不代表4/8实测；首次全体发现、发现与drain同锁、缓存后健康降级、全部坏4011、原drain后并发CLOSE与每attempt实例身份均须source/实际so测试。关闭候选grace120独立于业务timeout120/测试420，旧失败保留。双份顺序两个真实clip合用420总暖身，必须各index真实forward+PID出生/RPC覆盖；READY不算完成。先小规模同64参考、2×64真实并行/数值/HBM/取消关闭，再完整8B和最终18；正式默认未推广。

### 2026-11-19必要代码阶段提交

用户要求保存并推送当前必要实现。清单仍41/57，未归档、不推广候选默认；当前源码隔离完整CPU968项/281子测试通过、2保护专属skip、1硬件探针排除，237.17秒；源码/profile稳定、owned归零、共享cache写删尝试0。双份NPU及最终1/2/4/8完整音频矩阵待验收，v17的420秒/3of8及关闭main1失败保留。当前权威检查点为docs/npu-throughput-20261118/v18-precommit-checkpoint.json，历史章节只代表各冻结版本。
