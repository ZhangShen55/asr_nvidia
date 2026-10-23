# Qwen3-ASR 吞吐优化方案（单卡多前端 / 多实例扩容）

本文记录 2026-10-23 对 `/v2/seacraft_asr` 链路的吞吐瓶颈定位与扩容方案讨论。
所有数据均为本机（`192.168.29.12`）实测，来源见每节的「数据来源」。

> **状态：方案讨论稿，尚未实施。** 按项目约定，未在 change 中明确要求前不构建镜像、不部署。

## 1. 背景

生产侧观察到 Qwen3-ASR 容器所在 GPU 的利用率长期处于低位，无法通过提高并发把
Qwen 容器的处理能力跑满。本文的目的是：

1. 定位「GPU 利用率上不去」的真实瓶颈；
2. 评估「单卡多前端进程」与「单卡多容器」两条扩容路线；
3. 给出参数取值、显存预算与分步验证方法。

## 2. 现状实测

### 2.1 纯转写链路的 GPU 利用率

输入 `zh600.wav`（10 分钟，118 段），`showSpk=false` + `showEmotion=false`，
`segment_concurrency=4`、`request_concurrency=32`，每档 1 轮：

| 并发 | 墙钟 | 吞吐 | GPU 0 平均利用率 | GPU 0 峰值 |
|------|------|------|------------------|-----------|
| 1 | 19.97 s | 5.91 段/s | **9.9%** | 13% |
| 2 | 21.79 s | 10.83 段/s | **8.5%** | 13% |
| 4 | 28.18 s | 16.75 段/s | **7.8%** | 14% |
| 8 | 45.54 s | 20.73 段/s | **8.4%** | 14% |

**结论：GPU 0 长期在 8%~10%，峰值 14%；吞吐从 1 路到 8 路只涨 3.5 倍就压平。**

### 2.2 瓶颈定位在前端 CPU，不在 GPU

8 路压测期间采样 Qwen 容器：

- `docker stats` 显示容器 CPU **110%~115%（约 1.1 核）**
- 引擎日志 `Running: 16~28 reqs, Waiting: 0` —— **engine 侧从未排队**

即瓶颈在 vLLM 的 API server 前端（HTTP + 音频预处理 + tokenize + detokenize，
单进程 Python，受 GIL 限制），既不是 GPU 算力也不是显存。

### 2.3 算力反推：GPU 侧的理论上界

```
每请求算力 ≈ 2 × 0.6e9 × (400 prompt + 90 decode) ≈ 590 GFLOP
当前 21 段/s → 12.4 TFLOPS → 约为 RTX 4090 bf16 稠密峰值(~165 TFLOPS)的 7.5%
```

与实测的 8%~14% 利用率高度吻合，说明算力模型可用。据此：

- **80% 利用率 ⇒ 约 220 段/s ⇒ 需要当前 10.5 倍吞吐**
- 纯算力上 80% 是够得着的（prefill 批量 128 时约 278 段/s；decode 走显存带宽，
  90 步 × 1.2 GB 权重，不是瓶颈）

**但 220 段/s 需要前端、本服务、本地模型三处的 CPU 同时到位，逐级见 §6。**

## 3. 三个必须先讲清的机制

### 3.1 KV cache 池按「可用显存」分配，与 `max_model_len` 无关

当前容器自身的数字就是反证：

```
gpu_memory_utilization 0.25 × 47.99 GiB = 12.00 GiB 预算
模型权重                              1.53 GiB
Available KV cache memory             6.66 GiB
→ GPU KV cache size        62,384 tokens
```

若按 `max_model_len × max_num_seqs` 预留，需要 `4096 × 32 = 131,072` tokens
（14.3 GiB），比实际给出的 6.66 GiB 还多一倍 —— **它没有这么预留**。
日志里的 `Maximum concurrency for 4,096 tokens per request: 15.23x`
是 `62,384 ÷ 4,096` 的**推导报告值**，不是预留量。

结论：

- 调低 `--max-model-len` **不会省显存**，只会改变报告里的并发上限
- 想省显存只能调 `--gpu-memory-utilization`
- 显存预算 = `util × 总显存 − 权重 − 激活/上下文开销`

### 3.2 GIL 是进程级的

- 4 个容器 = 4 个独立解释器 = 4 个独立事件循环，天然并行
- **`--api-server-count N` 同样是 spawn 独立进程**（vLLM 源码
  `vllm/entrypoints/cli/serve.py` → `run_multi_api_server` →
  `spawn_context.Process(target=target_server_fn, ...)`），所以它同样破 GIL
- 区别在于：多前端共享 1 个 engine / 1 份权重 / 1 个 KV 池；
  多容器则是多份，且多个 CUDA 上下文在同一张卡上互相争抢

### 3.3 `nvidia-smi` 的 `utilization.gpu` 不是算力饱和度

它的定义是「过去采样周期内**至少有一个 kernel 在运行**的时间占比」。

- 0.6B 小模型 + `--enforce-eager`（禁 CUDA graph）→ 每步几十个极小 kernel 串行发射
  → **util% 容易虚高，吞吐却没涨**
- 反之启用 CUDA graph → kernel 变少变大 → **util% 下降，吞吐上升**

**因此验收指标应当是「吞吐（段/秒）+ P95 端到端时延 + KV cache 使用率」，
`utilization.gpu` 只作为「GPU 有没有在动」的粗信号。**

## 4. token 账（实测）

### 4.1 分布

来自引擎 `/metrics`（累计 66,073 次请求）：

| 指标 | 数值 |
|------|------|
| prompt token 均值 | 62.8 |
| 生成 token 均值 | 18.8 |
| **每请求总 token 均值** | **81.6** |
| prompt ≤ 200 的请求占比 | 99.72% |
| prompt ≤ 500 的请求占比 | **100%** |
| 生成 ≤ 50 的请求占比 | 99.9% |
| 因撞上限被截断（`finished_reason=length`） | 7 次（0.011%） |

### 4.2 单段实测

用 `教师1.wav` 截 30 秒有语音片段，直接打 Qwen 容器并看 metrics 增量：

| 片段 | prompt token | 生成 token |
|------|-------------|-----------|
| 30 秒静音 | 约 20 + 音频 token | 0 |
| 30 秒语音（3 次采样） | **精确 400** | 平均 87，最长约 115（166 汉字） |

推导式：

```
prompt token ≈ 20（固定模板）+ 12.7 × 段长秒数
VAD 最长段 30 秒（vad_max_single_segment_time_ms=30000）→ 400 token
```

### 4.3 每 token KV 占用

```
2(K,V) × 28 层 × 8 个 KV 头 × 128 head_dim × 2 字节(bf16) = 114,688 B = 112.0 KiB/token
```

与引擎反推值（6.66 GiB ÷ 62,384 token = 114,632 B）吻合。
模型配置来自 `Qwen3-ASR-0.6B/config.json`：
`text_config.num_hidden_layers=28`、`num_key_value_heads=8`、`head_dim=128`。

## 5. `max-model-len` 的取值

### 5.1 256 不安全

30 秒段的 prompt 是 400 token，**超过 256 → vLLM 直接返回 HTTP 400**（拒绝，不是截断）。
客户端对 400 的处理是 `raise QwenAsrCallError` → 4012，而
`asr_qwen_common._process` 会把异常向上抛 → **一个段被拒，整个音频请求失败**。

### 5.2 档位对比

生成预算 = `max_model_len − prompt`：

| max_model_len | 30s 段 prompt 400 | 生成预算 | 结果 |
|----------------|-------------------|---------|------|
| 256 | ❌ 超限 | — | **HTTP 400 → 整个请求 4012** |
| 512 | ✅ | 112 | 最长的 30s 段（约 115 token）**会被截断** |
| 768 | ✅ | 368 | 安全 |
| **1024** | ✅ | **624** | **推荐**，最坏情况 5 倍余量 |
| 4096（现状） | ✅ | 3696 | 浪费；病态循环会一直烧到上限 |

### 5.3 附带收益

那 7 个病态循环请求（幻觉重复直到撞上限，占 0.011%）在 1024 就会被截断，
既省 KV 也省时间，比现在烧满 4096 更好。

## 6. 扩容方案对比

### 6.1 方案 A：单容器多前端进程（推荐）

```bash
--served-model-name Qwen3-ASR-0.6B --dtype bfloat16 \
--max-model-len 1024 \
--max-num-seqs 128 \
--max-num-batched-tokens 16384 \
--gpu-memory-utilization 0.45 \
--api-server-count 4
```

### 6.2 方案 B：4 容器，各自独立

```bash
# 每个容器
--max-model-len 1024 \
--max-num-seqs 32 \
--max-num-batched-tokens 16384 \
--gpu-memory-utilization 0.185
```

### 6.3 对比

| | 方案 A | 方案 B |
|---|---|---|
| 前端进程数 | 4 | 4 |
| → 前端 CPU 吞吐天花板 | ~80 段/s | ~80 段/s（**相同**） |
| engine 数 | 1 | 4 |
| 权重副本 | **1 份 (1.53 GiB)** | 4 份 (6.12 GiB) |
| CUDA 上下文 | 1 | 4 |
| 常驻显存 | **17.6 GiB**（util 0.45） | 25.6 GiB |
| decode 显存流量 | 227 GB → 0.227 s | 551 GB → **0.551 s** |
| 处理 128 个请求 | ~0.7 s | ~1.0 s |
| 客户端改动 | **不需要** | 多 base_url + 负载均衡 + 故障转移 |
| 再往上加并发 | `--api-server-count 16`，权重仍 1 份 | 加容器 = 加权重 + 加上下文 + 改客户端 |
| 故障隔离 | 单点 | 挂一个剩三个 |

**A 更优的三个决定性理由：**

1. **前端进程数相同 → 吞吐天花板相同**。实测 1 前端 ≈ 21 段/s（1.1 核），
   4 前端就是 ~80 段/s。B 花 4 份权重换来的前端并行度，A 一个参数就拿到了。
2. **B 的 decode 慢 2.4 倍**。decode 是带宽瓶颈，每步要读全量权重 1.2 GB；
   1 engine 批 128 → 权重读 1 遍；4 engine 各批 32 → 权重读 4 遍。
3. **A 能继续横向扩**。前端才是墙，加前端是改一个参数；B 加前端要加容器、
   加权重、加 CUDA 上下文、改客户端。

### 6.4 显存预算

固定开销 = 权重 1.53 GiB + 激活/CUDA 上下文。
`--max-num-batched-tokens` 从 8192 提到 16384 会让 profiling 峰值激活翻倍，
固定开销从 3.81 GiB 涨到约 5.3 GiB（已计入下表）。

**方案 A：**

| util | 预算 | KV 池 | 1024/请求并发上限 | 常驻 |
|------|------|-------|------------------|------|
| 0.40 | 19.2 GiB | 12.4 GiB / 115,674 token | 113x | 15.2 GiB |
| **0.45** | 21.6 GiB | 14.8 GiB / **138,138 token** | **134.9x** ✅ | 17.6 GiB |
| 0.74 | 35.5 GiB | 28.7 GiB / 268,429 token | 262x | 31.5 GiB |

128 路并发最坏情况需要 `128 × 1024 = 131,072` token = 14.0 GiB。
**util 0.45 刚好装下（134.9x > 128）；0.74 多给一倍、白占 13 GiB**，
且一旦 `qwen3-asr-stream-18806` 回到 GPU0（9.5 GiB），
31.5 + 9.5 = 41 GiB、启动峰值 45 GiB → 48 GiB 的卡会 OOM。

**方案 B：** 单实例 util 0.185 → 预算 8.88 GiB、KV 3.54 GiB（33,122 token）、
常驻 6.41 GiB；×4 = 预算 35.5 GiB、常驻 25.6 GiB。

### 6.5 达到 220 段/s 还差什么

按每请求 CPU 成本折算 220 段/s 的需求：

| 环节 | 每请求 CPU | 220 段/s 需要 | 现状 |
|------|-----------|--------------|------|
| Qwen 前端 | 52 ms | **11.5 核** | 4 前端 ≈ 4 核（~80 段/s）→ 需 `--api-server-count 12~16` |
| 本服务 | 17 ms | 3.7 核 | 容器内 `instance_count=4` → 4 个 GIL，够 |
| **本地模型（CAM++ 155 ms + emotion 64 ms 的并发累计，GIL 占用约 50~110 ms/段）** | ~50~110 ms | **11~24 个 GIL** | 4 个 GIL ❌ **过不去** |

**结论：**

- **纯转写链路**：方案 A + `--api-server-count 12~16` 有希望摸到 80% 利用率
- **开启 showSpk / showEmotion**：本地两个模型是纯 Python/GIL 活儿，
  220 段/s 需要 11~24 个本服务进程，不是加卡能解决的。
  要解决只能把本地模型推理拆到独立进程（各自独立 GIL），
  或采用批量推理 —— 但 CAM++ 批量已验证**不等价**（见 `qwen-asr-v2-test-report.md` §8.3），
  不可用

## 7. 配套配置

本服务侧要能喂满 128 路在飞请求，配置需同步放大：

```toml
[qwen_asr]
request_concurrency = 128     # 全局总闸门，从 32 放大
segment_concurrency = 16      # 单音频段并发
audio_concurrency = 8         # 8 音频 × 16 段 = 128
```

## 8. 分步验证计划

一次只动一个变量，每步看两个数：**吞吐（段/秒）** 与 **GPU util%**。

| 步骤 | 动作 | 预期 | 判读 |
|------|------|------|------|
| 1 | 现有容器加 `--api-server-count 4` | 吞吐 21 → ~80 段/s | 若 util% 也涨到 80%，多半是 `--enforce-eager` 在刷指标，别被骗 |
| 2 | 加 `--max-model-len 1024` + `--gpu-memory-utilization 0.45` | 吞吐不变，显存从 9.53 → 17.6 GiB | 启动日志核对 `GPU KV cache size` ≥ 131,072 |
| 3 | `--api-server-count 12~16` | 吞吐 → ~160~220 段/s | 此时 util% 才值得看 |
| 4 | （可选）去掉 `--enforce-eager` 启用 CUDA graph | 吞吐上升、util% 可能下降 | 印证 §3.3 的指标判断 |

**每步必看启动日志这两行**（权威值，优先于任何推算）：

```
GPU KV cache size: N tokens
Maximum concurrency for 1,024 tokens per request: X
```

## 9. 风险与注意事项

1. **单点故障**：方案 A 的容器挂了全断。缓解：GPU1/GPU2 留 standby 容器，
   后续做多后端故障转移（需改客户端，走 openspec change）。
2. **启动瞬时峰值**：4 个容器同时起会顶到 35.5 GiB，需错峰启动。
3. **显存留白**：`qwen3-asr-stream-18806` 可能随时回到 GPU0，util 不要超过 0.5。
4. **不要用 GPU-Util 作为验收指标**，理由见 §3.3。
5. **本方案不涉及代码改动**（方案 A）；方案 B 需要改 `core/qwen_asr_client.py`
   支持多后端负载均衡，并同步 `openspec/specs/qwen-asr-v2-adapter/spec.md`
   中 `segment_concurrency` / `request_concurrency` 的默认值描述。

## 10. 数据来源

| 内容 | 来源 |
|------|------|
| 容器启动参数 | `docker inspect qwen3-asr-0.6b-8806` |
| 权重/KV/并发上限 | 容器启动日志（`gpu_worker.py`、`kv_cache_utils.py`） |
| token 分布 | `http://127.0.0.1:8806/metrics`（累计值 + 直方图） |
| 单段 token 实测 | 向 8806 直发 5s / 30s 片段，比对 metrics 增量 |
| GPU 利用率 | `nvidia-smi --query-gpu=utilization.gpu`（200 ms 采样） |
| 前端 CPU | `docker stats`（1 s 采样） |
| 模型结构 | `Qwen3-ASR-0.6B/config.json` |
| vLLM 参数支持 | 容器内 `vllm/entrypoints/cli/serve.py`、`openai/cli_args.py` |

