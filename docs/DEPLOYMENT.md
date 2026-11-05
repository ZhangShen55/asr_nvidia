# 部署与运维说明

## 1. 部署

### 1.1 镜像

模型随镜像打包，运行期**不需要挂载模型目录**。构建时保留缓存（不使用 `--no-cache`）：

```bash
docker build -t jy-algorithm-app-asr-offline:v3.4 .
```

构建期会校验三个模型目录（FSMN-VAD、CAM++、emotion2vec）存在且非空，缺失时直接失败并指出目录名。

启动：

```bash
docker run -d \
  --name asr-offline \
  --gpus '"device=0"' \
  -p 9000:9000 \
  -e CONFIG_PATH=/config.toml \
  jy-algorithm-app-asr-offline:v3.4
```

容器内 Nginx 对外暴露 `9000`，内部按 `instance_count` 启动对应数量的 uvicorn 实例（默认 1）。

挂载配置只用于调整并发、设备、语速等参数，模型路径必须保持 `/app/model/`：

```bash
docker run -d \
  --name asr-offline \
  --gpus '"device=0"' \
  -p 9000:9000 \
  -v "$(pwd)/config.docker.toml:/config.toml:ro" \
  -e CONFIG_PATH=/config.toml \
  jy-algorithm-app-asr-offline:v3.4
```

### 1.2 Qwen3-ASR 容器（本服务的前置依赖）

Qwen3-ASR 不在本镜像内，由独立容器提供，通过「IP + 端口」访问，不需要 API Key：

```bash
docker run -d --name qwen3-asr-0.6b-8806 -p 8806:8000 qwenllm/qwen3-asr:latest
```

**吞吐扩容（方案 A）**：Qwen 服务侧起多个前端进程共享同一个 engine，可把 Qwen 层吞吐提升 3~7 倍
（GPU 利用率从 13~16% 提升到 82~92%）。推荐参数：

```bash
--api-server-count 4 \
--max-model-len 1024 \
--max-num-seqs 128 \
--max-num-batched-tokens 16384 \
--gpu-memory-utilization 0.45
```

启动后核对 vLLM 日志中的两行：

```text
GPU KV cache size: ...
Maximum concurrency for XX sequences per request: ...
```

扩容收益由 Qwen 服务侧承担，本服务侧的 `segment_concurrency` / `request_concurrency` / `audio_concurrency`
**默认值维持不变**（实测再放大本服务侧并发没有额外收益，反而会加剧公平性问题）。

注意 `--max-model-len 1024` 与 VAD 最长段参数耦合：`vad_max_single_segment_time_ms` 当前为 30000
（约 400 token），生成预算 624 token 足够；若调大 VAD 最长段，1024 以上的长段会被 vLLM 直接拒绝。

## 2. 运维

### 音频准入配置迁移

每个 uvicorn 实例独立执行 `[qwen_asr].audio_concurrency`（A+B 总音频数，默认 8）和
`audio_spk_emo_concurrency`（增强 B 类音频数，默认 4）两条限制，配置修改需重启。
旧配置缺少新增字段时也使用 4，升级前需核对客户端增强请求比例及所需上限。
新增字段只接受正整数，非法值在 VAD 预热和服务就绪前报错；关闭功能使用 `[features]`。
B 上限可以大于总上限，实际最多为 `min(audio_concurrency, audio_spk_emo_concurrency)`，
状态接口保留原配置值。多实例部署的限制按进程分别生效，不是整个部署的共享总额。

`showSpk`、`showEmotion`、`showRoleIdentify` 全 false 为 A，任一 true 为 B。
B 在同一个准入动作中占用总音频和增强音频名额，增强上限不调整模型池段级推理容量。
在默认不等待、总上限 8/B 上限 4 时：`3A+4B` 可接收 A，新 B 返回 `4017`；
`5A+3B`、`8A`、`4A+4B` 的新 A/B 均返回总容量满 `4013`，不为 B 预留总名额。
等待模式共享 `queue_max_size`，允许 A 绕过被 B 容量阻塞的 B；同类 FIFO，较早可处理请求优先。
共享等待满为 `4013`，等待超时为 `4010`，等待请求不占活动名额。

### 音频准入状态与诊断

使用 `GET /get_status` 查看两组准入容量。正在处理 `5A+3B`、等待 `1A+2B` 时：

```json
{
  "audioConcurrency": {"configured_total": 8, "active_total": 8, "waiting_total": 3},
  "spkEmoConcurrency": {"configured": 4, "active": 3, "waiting": 2}
}
```

两组字段来自同一次快照，A 的使用量等于总量减 B 量。首次请求前均为零，
能力关闭的请求在准入前被拒绝，查询状态不会触发模型加载。
`spkEmoConcurrency.waiting` 是等待整音频名额的 B 请求数，
`localModels.*.waiting` 是等待单模型实例的分段数，不能相加作为音频排队数。

`[V2准入]` 日志包含 `request_id`、`category`、`outcome`、`reason`、`code`、`snapshot`，
用于区分 `total_full`（4013）、`spk_emo_full`（4017）、`waiting_full`（4013）、
`wait_timeout`（4010）以及 `role_disabled` / `emotion_disabled` / `both_disabled`
（4015 / 4016 / 4018）。另有 `cancelled` 和 `qwen_disabled`。
准入时的快照用于复核容量，等待取消的快照在回收后取得；原 `[V2 耗时]` 字段保留。
准入后的处理失败、取消仍记录 `outcome=admitted`，最终请求状态分别为 `failed`、`cancelled`。

请求遥测新增 `admission` 对象（`category`、`outcome`、`reason`、`snapshot`），
`phases.audio_wait_ms` 表示整音频准入等待，独立于分段 `spk_wait_ms` / `emo_wait_ms`。
`resourceDiagnostics.requestTelemetry` 提供 `admission.by_class` 的 A/B 请求数、
`admission.rejections_by_reason` 的拒绝/超时/准入前取消计数及阶段耗时分位数。
所有汇总标记 `window=retained_events`，基于最近最多 4096 个事件；
含明细的诊断快照最多返回近期 20 条，状态接口仅提供汇总，不是历史累计计数。
新增记录不包含音频、转写正文或模型对象。

### 2.1 用 nvidia-smi 定位本服务的显存

主服务进程承载 CAM++ 与 emotion2vec，进程名为 `asr-offline-model`；VAD 由独立的 `asr-offline-vad` worker 进程承载：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

```text
pid, process_name, used_gpu_memory [MiB]
1352204, VLLM::EngineCore, 9998 MiB
1791632, asr-offline-model, 340 MiB
# vad_device = "cpu" 时不出现 asr-offline-vad GPU 行
```

VAD worker 默认在 CPU 上运行，不建立 CUDA context；只有把 `vad_device` 配置为 `cuda:N` 时，
才会在 GPU 进程列表中出现 `asr-offline-vad`，且显存随 worker 数量增长。

### 2.2 逐模型的显存明细

由于同进程承载，`nvidia-smi` 只能看到聚合显存。逐模型明细通过状态接口获取：

```bash
curl http://localhost:9000/get_status | jq .localModels
```

```json
{
  "speech_campplus": { "loaded": true, "device": "cuda:0", "devices": ["cuda:0"], "allocated_mb": 26.7, "instances": 1, "configured_instances": 1, "available_instances": 1, "failed_instances": 0, "busy": 0, "waiting": 0 },
  "emotion2vec": { "loaded": true, "device": "cuda:0", "devices": ["cuda:0"], "allocated_mb": 313.3, "instances": 1, "configured_instances": 1, "available_instances": 1, "failed_instances": 0, "busy": 0, "waiting": 0 }
}
```

`loaded` 为 `false` 表示该模型尚未被任何请求触发加载（懒加载），此时不占显存。

CAM++ 与 emotion2vec 的实例通过设备数组统一管理。数组长度就是实例数，每个元素决定对应实例的 CUDA 设备；数组可以为空，表示禁用对应能力：

```toml
[model_device_pool]
speech_campplus_devices = ["cuda:1", "cuda:2"]
emotion2vec_devices = ["cuda:1"]
```

当 `speech_campplus_devices = []` 时，`showSpk=true` 的 v2 请求返回 `4015`；当
`emotion2vec_devices = []` 时，`showEmotion=true` 的请求返回 `4016`。未请求对应能力时，
转写仍可正常执行。两种空数组都配置时，服务仍可启动，不会加载这两个模型。

功能开关优先于设备配置：`[features].open_v2_role=false` 时，`showSpk=true` 或
`showRoleIdentify=true` 返回 `4015`；`open_v2_emotion=false` 时，`showEmotion=true`
返回 `4016`。同时请求两项且两项都关闭返回 `4018`；同时请求两项但只有一项关闭时，
返回该关闭能力的单项错误。校验在音频准入前执行，非空设备数组不会触发关闭模型的加载或推理。
三个请求参数默认均为 false，只请求已开启能力可继续处理，全部 false 时普通转写照常。
配置关闭不会以 null 字段成功转写；null 适用于未请求的能力或已开启且有健康实例的模型段级推理失败。

同一模型可以在数组中显式指定不同 GPU，实例按数组位置加载，不会自动迁移。使用
`["cuda:0"]` 即可回到单副本；旧版 `speech_campplus = 2` / `emotion2vec = 2` 仍兼容读取，
但新配置应使用设备数组。`configured_instances`、`available_instances`、`failed_instances`、
`busy` 和 `waiting` 用于判断模型池的实际压力。

VAD 进程池运行信息也通过 `.vadPool` 暴露：

```json
{
  "loaded": true,
  "device": "cpu",
  "devices": ["cpu", "cpu", "cuda:0", "cuda:0"],
  "configured_instances": 4,
  "loaded_instances": 4,
  "available_instances": 4,
  "allocated_mb": 19.53,
  "failed_instances": 0,
  "failures": [],
  "busy": 0,
  "waiting": 0
}
```

### 2.3 实例数与显存的关系

`instance_count` 默认 `1`，只决定完整 uvicorn 服务进程数量。每个实例进程按
`[model_device_pool]` 各自加载模型副本；每个服务实例另外维护由 `vad_device` 数组决定的独立
VAD worker 池。VAD 默认在 CPU 上，因此默认不放大显存：

| `instance_count` | uvicorn 实例数 | `nvidia-smi` 中的进程行数 | 本地模型显存 |
|-----------------|---------------|------------------------|------------|
| 1（默认） | 1 | 1 个 `asr-offline-model` + `len(vad_device)` 个 VAD worker | 按模型池配置加载，VAD 显存随设备数组决定 |
| 1 + 模型池各 2 副本 | 1 | 1 个 `asr-offline-model` + `len(vad_device)` 个 VAD worker | CAM++/emotion2vec 各 2 份，显存按设备数组增长 |
| 2 + 模型池各 N 副本 | 2 | 2 个 `asr-offline-model` + 每实例 `len(vad_device)` 个 VAD worker | 每进程各 N 份，显存按进程和副本数乘积增长 |

多实例用于提升 HTTP 层吞吐，代价是本地模型显存等比放大，运维按需权衡。

### 2.4 VAD 进程池与 CPU 配额

VAD 配置位于 `[vad]` 的 `vad_device` 数组中，每个元素对应一个固定 worker：

| 配置项 | 默认值 | 说明 |
|---|---:|---|
| `vad_device` | `["cpu", ...]` | 数组长度决定固定 worker 数；元素为 `cpu` 或 `cuda:N` |

实测每个 CPU worker 约占 2.2~2.5 核，建议按以下配额配置容器：

| `vad_device` 中的 CPU worker 数 | 建议容器 CPU 配额 |
|---:|---:|
| 4 | 16 核 |
| 8 | 24~32 核 |
| 16 | 48 核；16 worker 后收益趋于饱和 |

VAD worker 只读取主进程已落盘的音频路径。段 clip 在转写返回后立即删除，异常路径由请求级 `finally` 兜底清理。

### 2.5 日志

日志按天 + 按大小双轮转，并按保留天数自动清理：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `log_path` | `./asr_service.log` | 当前日志文件 |
| `[log].max_size_mb` | `100` | 单个日志文件大小上限，超过即切出 `.1` / `.2` … |
| `[log].max_days` | `7` | 历史日志保留天数，超期自动删除 |

轮转产生的历史文件：

```text
asr_service.log              # 当前日志
asr_service.log.1            # 按大小切出的历史（序号递增，最旧的被覆盖）
asr_service.log.2026-10-23   # 按天切出的历史
```

### 2.6 复读后处理

Qwen3-ASR 偶尔会在某个 VAD 段上进入「同一短语无限重复」的生成循环，直到撞上 `max_model_len` 被截断（实测产出 1863 字符的复读串）。服务在转写结果落地时（**ITN 规范化之前**）做一次检测与折叠。

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `[qwen_asr].repetition_guard_enabled` | `true` | 是否启用复读检测与修复 |
| `[qwen_asr].repetition_max_chars_per_second` | `40` | 按段时长估算合理字符数的上限系数（字符/秒） |
| `[qwen_asr].repetition_min_chars` | `120` | 判定长度异常的绝对下限字符数 |
| `[qwen_asr].repetition_min_unit` | `4` | 判定为循环的重复子串最小长度 |
| `[qwen_asr].repetition_min_times` | `3` | 判定为循环的最小连续重复次数 |

**触发条件是双条件同时成立**：长度异常 **且** 检出连续重复结构。只满足长度异常而没有复读结构的文本会被**原样保留**，只在日志里留一条告警——宁可放过，也不可凭空截断有效内容。

**阈值是怎么定的**：取 40 字符/秒的依据是 68,018 段真实样本的统计——正常段 p99.9 为 20.3 字符/秒、观测到的最高值 26.2，而复读段是 282~313。40 约为最高正常值的 1.5 倍余量，同时远低于复读的量级。调整时建议先按业务语种的真实语速重算这两组数字。

**日志**：

```text
[复读修复] 长度 1863 -> 7（原上限 238，时长 6.0s）: '最小S零，这个这个这个这个，…'
[复读检测] 长度 300 超过上限 200（时长 5.0s），但未检出复读结构，保留原文
```

第一条表示发生了折叠，第二条表示长度异常但结构不明确、已原样保留。两者都只记录文本前缀，不记录全文。

**回滚方式**：把 `repetition_guard_enabled` 设为 `false` 并重启服务，响应即回到接入前的行为。

## 3. 发布说明

### 3.1 破坏性变更

**`POST /v1.1.8/seacraft_asr` 已下线**，请求该路径返回 `404`。

- 影响：仍在调用 v1.1.8 的上游需要改用 `POST /v2/seacraft_asr`；
- 响应结构兼容：v2 与 v1.1.8 的响应字段同构，差异点见下表；
- 语言能力：v1.1.8 的法语（Whisper）链路随之下线，镜像不再内置 `faster-whisper-large-v3`；
- 模型：镜像不再内置 SeACo-Paraformer 与 PUNC，只保留 VAD、CAM++、emotion2vec 三个模型。

v1.1.8 → v2 的响应差异：

| 字段 | v1.1.8 | v2 |
|------|--------|-----|
| `language_ratio` | 无 | 新增，键为语种简写（`zh` / `en`） |
| `segment_text` | 可能以标点开头、可能整段只有标点 | 段首标点已移除，纯标点段已丢弃 |
| `segment_words` | 词级时间戳 | 固定为空数组（Qwen 不返回时间戳） |
| `role` / `emotion` | v1.1.8 内置模型的推断值 | 由 CAM++ / emotion2vec 给出，字段含义不变 |

### 3.2 回滚

镜像回滚到 `v3.3` 即可恢复 v1.1.8：

```bash
docker run -d --name asr-offline --gpus '"device=0"' -p 9000:9000 \
  -v "$(pwd)/config.toml:/config.toml:ro" \
  jy-algorithm-app-asr-offline:v3.3
```

回滚时需要恢复**旧结构**的配置文件（`id_engine`、`version`、`device`、`ngpu`、`ncpu`、`concurrency`、
`hotword_path`、`[audio_chunk]`、`[features].open_spk` / `open_emotion` / `ban_hotword` / `open_mul_lang`、
`[model_paths]` 的 `asr_model_dir` / `punc_model_dir` / `whisper_model_dir`）。
`v3.3` 镜像仍需挂载 SeACo-Paraformer、PUNC 与 Whisper 模型目录。

Qwen 容器参数与本服务配置无关，无需回滚。
## 可选 TensorRT 后端

CAM++ / emotion2vec 的配置、依赖、模型资产、GPU/运行时兼容与独立回滚见
[TensorRT 部署说明](TENSORRT.md)。本地/容器模板默认两个开关均为 false；NPU 固定走 torch_npu。
Docker `ENABLE_TENSORRT=1` 只安装 NVIDIA 可选依赖，不自动打开能力或后端开关；实际 engine
仍须在目标环境校验。当前 CUDA 12.1.1 基础镜像未被本轮 CUDA 12.4 主机实验验证。
