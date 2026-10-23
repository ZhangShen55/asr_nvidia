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

### 2.1 用 nvidia-smi 定位本服务的显存

三个本地模型（VAD、CAM++、emotion2vec）由同一个服务进程承载，进程名统一为 `asr-offline-model`：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

```text
pid, process_name, used_gpu_memory [MiB]
1352204, VLLM::EngineCore, 9998 MiB
1791632, asr-offline-model, 420 MiB
```

为什么不再按模型拆分进程名（`vad` / `cam` / `emo`）：每个独立进程要各自创建 CUDA context，
实测仅 context 就占 386 MiB，从单进程拆三进程净增约 0.8 GB 显存；而且模型调用会从函数调用
退化为跨进程 RPC，异常与超时路径全部复杂化。业务侧明确不接受这份额外开销，因此改为统一进程名，
逐模型区分下放到状态接口。

### 2.2 逐模型的显存明细

由于同进程承载，`nvidia-smi` 只能看到聚合显存。逐模型明细通过状态接口获取：

```bash
curl http://localhost:9000/get_status | jq .localModels
```

```json
{
  "vad": { "loaded": true, "device": "cuda:0", "allocated_mb": 1.6, "instances": 4 },
  "speech_campplus": { "loaded": true, "device": "cuda:0", "allocated_mb": 26.7, "instances": 1 },
  "emotion2vec": { "loaded": true, "device": "cuda:0", "allocated_mb": 313.3, "instances": 1 }
}
```

`loaded` 为 `false` 表示该模型尚未被任何请求触发加载（懒加载），此时不占显存。

### 2.3 实例数与显存的关系

`instance_count` 默认 `1`。每个实例进程各自持有一份 VAD / CAM++ / emotion2vec 权重，
因此**实例数即显存倍数**：

| `instance_count` | uvicorn 实例数 | `nvidia-smi` 中的进程行数 | 本地模型显存 |
|-----------------|---------------|------------------------|------------|
| 1（默认） | 1 | 1 | 1 份 |
| 2 | 2 | 2 | 2 份 |

多实例用于提升 HTTP 层吞吐，代价是本地模型显存等比放大，运维按需权衡。

### 2.4 日志

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
