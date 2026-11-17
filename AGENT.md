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
- FSMN-VAD 是固定 spawn 进程池；CAM++ / emotion2vec 是主进程内的懒加载模型池。
- **Qwen3-ASR 是本地懒加载原始权重**，BF16 / npu:7，单实例串行执行；正式后端选择以实测报告为准。
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
每段 1 个 Qwen 本地任务（内存 WAV 字节，segment_concurrency 控制单音频内并发）
  ├─ 并行：CAM++ 声纹 → 全音频层次聚类（阈值 speaker_similarity_threshold）→ teacher / studentN
  └─ 并行：emotion2vec 情绪（1~10s 之外记"平淡"）
  ↓
复读修复 → ITN → 标点卫生（纯标点段删除、去段首标点）→ 语速 / 语种占比
```

**核心不变量**（改代码时最容易踩坏的地方）：

1. **1 个 VAD 段 = 1 个 Qwen 请求 = 1 个响应分段**。绝不合并相邻段，`bg` / `ed` 直接取 VAD 边界。
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
| `api/routes/asr_qwen_common.py` | **v3 主链路**（807 行）：VAD → 并发转写 → 响应封装 |
| `api/routes/asr_common.py`      | 请求上下文构建、情绪标签映射、临时文件清理               |
| `api/routes/status.py`          | `/get_status` 路由                    |
| `entity/data.py`                | `AsrRequestParams` 请求数据类            |

### 核心能力

| 文件                             | 职责                                                             |
| ------------------------------ | -------------------------------------------------------------- |
| `core/vad.py`                  | VAD 切分配置组装                                                     |
| `core/vad_pool.py`             | 固定设备数组的 FSMN-VAD 进程池（564 行）                                    |
| `core/vad_volume.py`           | 段级声音特征（RMS / SNR / dBFS）与低音量过滤，**运行时与标定脚本共用同一套口径**             |
| `core/qwen_asr_queue.py`       | 三层并发闸门：音频准入分类 + 段级/全局请求信号量                                     |
| `core/qwen_asr_client.py`      | Qwen 异步客户端：本地推理、串行实例、重试                                          |
| `core/models.py`               | CAM++ / emotion2vec 懒加载、模型池调度、HBM 分配估计登记                             |
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
| `utils/audio_utils.py`      | 音频解码、内存 WAV、服务缓存目录                         |
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

音频准入、段任务闸门与模型实例容量分别限流；增加等待任务不会增加单实例执行并发。压测结论见 `docs/cam*`、`docs/vad-pool*`。

**错误码契约不可随意改号**（客户端已按此判断）：完整定义分散在  
`core/qwen_asr_queue.py`、`core/qwen_asr_client.py`、`core/models.py`、`core/vad_pool.py` 的异常类 `code` 属性；  
对外说明在 `README.md` 错误码表。要点：

- `4008` 无人声 · `4011` Qwen 不可用 · `4012` Qwen 调用失败 · `4013` 总容量满 · `4017` 仅 B 类满 · `4010` 排队超时
- `4015` / `4016` / `4018` 能力未启用或无可用实例
- **能力开关校验早于容量检查**；两项容量同时满时优先返回 `4013`

---

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
`instance_count=1`；增加进程或 Qwen 实例意味着复制权重，重新标定后才调整。
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
