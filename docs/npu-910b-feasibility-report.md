# NPU（昇腾 910B）可行性调研报告

> 调研对象：`asr_offline` 服务在 v2 链路中使用的三个**本地**模型 —— FSMN-VAD、CAM++（说话人嵌入）、
> emotion2vec_plus_large（情绪）。
> 调研范围：这三个模型在华为昇腾 910B + `torch_npu` 上能否跑通，以及本仓库需要做哪些改造。
> **不在范围**：Qwen3-ASR 的 NPU 部署。本服务通过 HTTP 调用 Qwen 容器，Qwen 是否部署在 NPU 上与本服务解耦，
> 只需保证本服务的 HTTP 客户端契约不变即可。
> 报告日期：2026-10-23

---

## 一、结论摘要

| 模型 | 910B 官方支持 | 结论 | 主要风险 |
|---|---|---|---|
| FSMN-VAD（`speech_fsmn_vad_zh-cn-16k-common-pytorch`） | ✅ 昇腾官方 FunASR 文档在 Atlas 900 A2 PODc（910B4）上验证通过 | **可跑通** | 社区有「910B 上 VAD 显著变慢」的反馈，需实测（我们的用法是离线批量，非流式） |
| CAM++（`speech_campplus_sv_zh_en_16k-common_advanced`） | ⚠️ 官方文档给出 NPU 指南与补丁，但**需打补丁** | **可跑通，需改造** | `seg_pooling` 用了 NPU 不支持的 `AvgPoolV2` 大 stride，必须替换实现 |
| emotion2vec_plus_large | ✅ 昇腾官方文档明确给出 `device="npu:0"` 的调用示例与输出格式 | **可跑通** | 需把 `torch.cuda.amp.autocast` 换成 `torch.npu.amp.autocast` |

**总体结论：架构上可以在 NPU 与 GPU 上双跑通。** 三个模型都在 FunASR/昇腾的官方支持矩阵内，本仓库的改造量集中在
「设备抽象层 + funasr 的 CUDA 硬编码绕过 + CAM++ 算子补丁 + 独立的 NPU 镜像流水线」四处，属于**中等规模工程改造**，
不是推倒重来。但有两个前提需要业务侧确认：

1. **Qwen-ASR 若也迁到 NPU**，则整条链路是「NPU 本地模型 + NPU Qwen」的同构部署，本服务侧无需额外改造；
   若 Qwen 留在 GPU，本服务跑 NPU 也完全成立（两者只是 HTTP 调用关系）。
2. **910B 机器与 CANN 环境的可获得性**：本报告基于官方文档与社区先例，**未在真实 910B 硬件上实测**，
   建议先在开发机上做一轮小规模验证（见第七节）再决定是否立项。

---

## 二、官方支持矩阵与已验证环境

### 2.1 昇腾官方 FunASR 文档（权威）

昇腾官方维护的 FunASR NPU 指南明确给出在 **Atlas 900 A2 PODc（Ascend 910B4，64 GB HBM）** 上的验证结果：

| 项 | 取值 |
|---|---|
| 系统 | Ubuntu 22.04 |
| CANN 镜像 | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12` |
| torch | `2.9.0+cpu`（**CPU wheel**，与 torch_npu 严格配对） |
| torch_npu | `2.9.0.post2` |
| torchaudio | `2.9.0` |
| Python | 3.12 |

文档中的三个示例恰好覆盖了我们的三个模型：

- **FSMN-VAD**：`AutoModel(model="fsmn-vad", device="npu:0")`，与 Paraformer / ct-punc 组成 pipeline 时也在 `npu:0` 上跑通；
- **CAM++**：以 `spk_model="cam++"` 挂在 pipeline 上（`device="npu:0"`），输出 `sentence_info` 中含 `spk` 字段；
- **emotion2vec_plus_large**：`AutoModel(model="emotion2vec_plus_large", device="npu:0", granularity="utterance")`，
  输出 `labels` / `scores` / `top_emotion`，共 **9 类情绪**，其中含「中立 / neutral」——
  与本项目当前使用的 `emotion2vec_plus_large` 完全对应。

另外文档也给出了 CLI 侧 `--device npu:0` 的用法，说明 funasr 的设备串体系原生支持 `npu:<index>`。

### 2.2 FunASR 官方 NPU 指南（PR #3708）

FunASR 仓库在 2026-09 合并了《在昇腾 NPU 上运行 FunASR》指南（`modelscope/FunASR#3708`），其中的关键结论与本项目直接相关：

> **CAM++ 在 NPU 上会直接失败**：`CAMLayer.seg_pooling` 原本用 `F.avg_pool1d(kernel_size=100, stride=100, ceil_mode=True)`，
> 在 NPU 上会被融合为 `AvgPoolV2` 算子，而该算子**只支持 stride ∈ [1, 63]**，stride=100 会导致图编译失败 / 运行异常。
>
> 官方给出的修复方式：用「reshape → sum / max → expand → slice」等价替换池化。官方在 CPU 上做了等价性回归
> （11 种输入长度 × {avg, max} × {全 1、负数、随机} 共 66 组，`allclose(rtol=1e-4, atol=1e-6)` 全部通过，
> 最大残差约 `1.2e-7`）。

该指南还给出两条重要的运维结论：

- **首次推理会包含 NPU 图编译开销**（数十秒级），属正常现象，第二次起才是稳态耗时；
- **NPU 算子默认异步执行**，异常堆栈常指向「后一个」模型，调试时设 `ASCEND_LAUNCH_BLOCKING=1` 强制同步。

### 2.3 910B 上的同系列先例

昇腾社区已有 **`emotion2vec_plus_seed`**（与本项目的 `emotion2vec_plus_large` 同系列）在 910B（Ascend A2）跑通的实测记录：

- 容器：`swr.cn-north-4.myhuaweicloud.com/ascend-sact/ascend-910b-ubuntu:latest`
- torch `2.8.0+cpu` + torch_npu `2.8.0`、Python 3.11.14、CANN 8.3.RC2
- 输出同样是 9 类情绪
- 社区教程用 `from torch_npu.contrib import transfer_to_npu` 把 CUDA API 自动迁移到 NPU

注意：该先例是 `plus_seed`（种子版），不是 `plus_large` 本身；`plus_large` 的官方验证以 2.1 节的昇腾官方文档为准。

### 2.4 依赖版本配对规则（踩坑高发区）

- torch_npu 要求 **CPU 版 torch** 与之**严格配对**（如 `torch 2.9.0+cpu` ↔ `torch_npu 2.9.0.post2`）；
- `funasr` 自身不声明 torch 依赖，`pip install funasr` 会顺带装上最新的 CUDA 版 torch，**把 torch_npu 顶掉**，
  社区已有对应 issue（modelscope/FunASR#3707）。规避方式：constraints 文件锁版本，或 `--no-deps` 单独装 funasr。
- CANN 版本与 torch_npu 版本要匹配（9.1.0 ↔ 2.9.x、8.3.RC2 ↔ 2.8.x）。

---

## 三、本仓库的 NPU 改造点（逐处，带文件与行号）

以下均基于当前分支 `refactor/adapt-qwen-asr` 的代码实测核对。

### 3.1 设备解析层：只认 CUDA

`core/models.py::_resolve_device()`（第 92~99 行）只处理 `cuda:<index>`，并依赖 `torch.cuda.is_available()` /
`torch.cuda.device_count()`：

```python
def _resolve_device(index: int) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("v2 链路要求使用 GPU，但 CUDA 不可用")
    visible = torch.cuda.device_count()
    ...
    return f"cuda:{index}"
```

`core/vad.py::_create_vad_model()`（第 89~102 行）用它构造 FSMN-VAD 的 `AutoModel(device=...)`。

**改造**：抽象一层 `resolve_device(index) -> str`，按 `torch.npu.is_available()` / `torch.cuda.is_available()` 分派，
返回 `npu:<index>` 或 `cuda:<index>`。同时 `[model_device]` 的配置语义要明确「索引在 NPU 上指 NPU 卡序号」。

### 3.2 funasr 1.2.6 的 CUDA 硬编码（**最关键的一处**）

funasr 1.2.6 的 `auto/auto_model.py` 第 184~190 行：

```python
device = kwargs.get("device", "cuda")
if not torch.cuda.is_available() or kwargs.get("ngpu", 1) == 0:
    device = "cpu"
    kwargs["batch_size"] = 1
```

在**纯 NPU 机器**上 `torch.cuda.is_available()` 返回 False，即使我们显式传 `device="npu:0"`，也会被 funasr
**强制改写成 `cpu`**——模型能跑，但全部落到 CPU 上，性能会崩。

**改造**：需要绕过这段判定。可选做法：

- 启动时 monkeypatch `torch.cuda.is_available`（社区常用的 `transfer_to_npu` 即属此类思路），风险是影响全局语义；
- 或者升级到带 `npu_compat` 的 funasr 版本（该模块提供 `ensure_npu_runtime(device)`），
  但需评估从 1.2.6 升级到 1.3.x 的兼容性影响；
- 或者给 funasr 提 PR，把判定改为「cuda 不可用时再看 npu」。

> 建议做法：先在 `core/models.py` 里加一个集中的 `ensure_device_runtime(device)` 适配层，
> 用最小侵入的 monkeypatch 绕过，并在启动时打印最终生效的 device 供验证。

### 3.3 CAM++ 的 `seg_pooling` 算子补丁

已核对本仓库实际使用的 funasr 1.2.6，`funasr/models/campplus/components.py` 第 172~177 行：

```python
def seg_pooling(self, x, seg_len=100, stype="avg"):
    if stype == "avg":
        seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
    else:
        seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
```

与 2.2 节描述完全一致 —— 这正是 NPU 上会失败的那一处。

**改造**：按官方指南等价替换 `seg_pooling`（reshape → sum/max → expand → slice），
并在我们的代码里做 monkeypatch（`funasr.models.campplus.components.CAMLayer.seg_pooling`）。
补丁必须**在模型构造前生效**，且需在 NPU 上重新做一次等价性回归（官方只验证了 CPU 侧）。

好消息：本项目**单独加载 CAM++ 模型**（`speech_campplus_sv_zh_en_16k-common_advanced`），
不走 funasr 的 `spk_model="cam++"` pipeline 挂载方式，补丁作用点更简单（只需改一个类的方法）。

### 3.4 AMP 混合精度调用

`api/routes/asr_qwen_common.py`：

- 第 153~154 行：`with torch.cuda.device(settings.model_device_emotion2vec)` +
  `with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)`（emotion2vec fp16）；
- 第 119 行：`with torch.cuda.device(settings.model_device_speech_campplus)`（CAM++）。

**改造**：抽出 `device_context(index)` 与 `amp_autocast()` 两个辅助函数，GPU 用 `torch.cuda.*`，
NPU 用 `torch.npu.device(...)` / `torch.npu.amp.autocast(...)`。

补充：funasr 的 `models/campplus/model.py:29` 与 `models/emotion2vec/model.py:26` 都在模块导入时
`from torch.cuda.amp import autocast`。在 torch_npu 环境里 `torch.cuda.amp` 模块本身可以导入成功（torch_npu 依赖 CPU 版 torch），
但**一旦调用就会因无 CUDA 而报错**。需确认这两处在推理路径上是否真被调用（当前 grep 显示只有 import，未见调用），
若被调用则需要在导入层面一并处理。

### 3.5 监控命令

GPU 用 `nvidia-smi`；NPU 用 **`npu-smi`**。本 change 落地的进程命名方案（`setproctitle` → `asr-offline-model`）
在 NPU 侧同样适用，`npu-smi info` 的进程列表同样读 `/proc/<pid>/cmdline`。

### 3.6 ITN（WeTextProcessing / pynini）

纯 CPU 实现，**与 NPU 无关**，无需改动。

### 3.7 脚本文件

`scripts/speaker_threshold_sweep.py:59`、`scripts/profile_role_emotion.py:67/112/124/219` 有 `torch.cuda` 硬编码，
属于离线分析脚本，NPU 适配时一并改或明确标注仅支持 GPU。

### 3.8 改造点汇总

| # | 位置 | 类型 | 难度 |
|---|---|---|---|
| 1 | `core/models.py::_resolve_device` | 设备抽象层 | 低 |
| 2 | funasr `auto_model` 的 `cuda.is_available` 硬编码 | 绕过 / 升级 funasr | 中 |
| 3 | `funasr/models/campplus/components.py::seg_pooling` | 算子等价替换 | 中 |
| 4 | `api/routes/asr_qwen_common.py` 的 `torch.cuda.*` | 设备与 AMP 上下文 | 低 |
| 5 | `scripts/*.py` 的 `torch.cuda` | 离线脚本 | 低 |
| 6 | Dockerfile / 依赖 | NPU 镜像与依赖矩阵 | 中 |

---

## 四、性能与风险评估

### 4.1 VAD 性能：社区已知的慢点

昇腾社区有反馈：「910b 上运行 Funasr 实时转录的例子，加上 VAD 静音识别时会变得特别慢」，关闭 `async_vad` 后恢复正常。

**但我们的用法不同**：该反馈来自 `runtime/python/websocket/funasr_wss_server.py` 的**流式**链路，
逐帧调用 `async_vad` 会引入大量 host-device 同步；而本项目的 `core/vad.py` 用的是**离线批量**接口
`model.generate(input=audio_path, cache={})`，一次算完整段音频，host-device 交互次数少一个数量级。

**判断：风险中等偏低，但必须实测。** 验证方法见第七节。

### 4.2 首帧 / 首次推理开销

NPU 侧首次推理包含图编译开销（数十秒）。对本项目的影响：

- **服务启动不受影响** —— 本项目是懒加载，首次请求才加载模型；
- **首个 `showSpk` / `showEmotion` 请求会明显变慢**（几十秒），之后稳态；
- 需要在超时 / 排队策略上留出余量，或在健康检查阶段预热模型。

### 4.3 精度

三个模型的官方文档输出格式与本项目当前用法一致（emotion2vec 的 `labels`/`scores`、CAM++ 的 `spk_embedding`、
VAD 的分段列表），**接口契约不变**。但：

- fp16 autocast 在 NPU 上的数值行为与 CUDA 不完全一致，情绪分数可能有细微偏差（业务上「有结果即可」，不敏感）；
- CAM++ 算子替换后说话人嵌入的数值一致性**必须在 NPU 上重新回归**，不能只信官方的 CPU 侧等价性结论。

### 4.4 生态与工程风险

| 风险 | 等级 | 说明 |
|---|---|---|
| funasr 1.2.6 无 `npu_compat` | 中 | 需要 monkeypatch 或升级 funasr，升级涉及 1.2.6 → 1.3.x 的行为差异 |
| 依赖版本配对（torch / torch_npu / CANN / Python） | 中 | pip 解析会把 CPU torch 顶成 CUDA 版；需 constraints 文件 |
| 镜像需换架构（x86_64 → aarch64） | 中 | 现 Dockerfile 基于 `nvcr.io/nvidia/cuda:12.1.1-cudnn8-runtime-centos7`（CentOS 7, glibc 2.17） |
| Python 版本变化 | 低 | NPU 侧推荐 3.11/3.12，本项目现为 3.11（conda `asr` 环境），兼容面尚可 |
| NPU 硬件可得性 | — | 本报告未在真实 910B 上验证 |

---

## 五、镜像与部署改造（NPU 侧独立流水线）

GPU 与 NPU 是**两条独立的镜像流水线**，不是同一个 Dockerfile 加参数：

| 维度 | GPU（现状） | NPU（需新建） |
|---|---|---|
| 基础镜像 | `nvcr.io/nvidia/cuda:12.1.1-cudnn8-runtime-centos7` | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`（或 `ascend-sact/ascend-910b-ubuntu:latest`） |
| 架构 | x86_64 | aarch64 |
| 运行时 | CUDA + cuDNN | CANN + Ascend 驱动固件 |
| torch | `torch>=1.13`（pip 默认装 CUDA 版） | `torch==2.9.0+cpu`（**必须 CPU wheel**）+ `torch_npu==2.9.0.post2` |
| 系统依赖 | ffmpeg / libsndfile 等 | 同左，另需 CANN 相关包 |
| 模型目录 | `/app/model` | 同左（三个模型目录一致，无需改动） |
| 监控 | `nvidia-smi` | `npu-smi` |

模型文件本身（FSMN-VAD / CAM++ / emotion2vec_plus_large）在两个平台上**完全复用**，无需重新导出。

---

## 六、GPU 与 NPU 并存的部署形态

由于 `[model_device]` 已经是配置项（vad / emotion2vec / speech_campplus 各自落卡），
理论上可以做到「VAD 跑 NPU、CAM++ 跑 GPU」这类混合部署。但需要注意：

- funasr 的 `AutoModel` 是按实例绑定的单一 device 串，混合部署只需为不同模型传不同的 `device`；
- 3.2 节的 `torch.cuda.is_available()` 绕过补丁在混合场景下要写成「设备类型感知的」版本，
  不能简单 monkeypatch 成恒 True（否则 NPU 机器上的 CPU 回退逻辑会失效）；
- 跨卡的音频搬运只发生在我们自己的 numpy 数组层面（本来就是 CPU 内存），无额外拷贝开销；
- **不建议一上来就做混合部署**：先把「全 NPU」跑通（环境、依赖、算子补丁都是新变量），再考虑拆分。

---

## 七、建议的验证计划（若拿到 910B 机器）

建议按下列顺序做**最小验证**，每一步都能独立判定通过与否：

| 步骤 | 验证内容 | 通过判据 |
|---|---|---|
| 1 | 环境：torch CPU wheel + torch_npu 版本配对、`import torch_npu` 成功 | `torch.npu.is_available()` 为 True，`npu-smi info` 正常 |
| 2 | 绕过 funasr 的 CUDA 硬编码，确认 `AutoModel(device="npu:0")` 未被改写成 `cpu` | 打印的生效 device 为 `npu:0`，模型参数落在 NPU 内存 |
| 3 | FSMN-VAD 单文件切段 | 分段边界与 GPU 侧逐段对齐（允许 1~2 帧误差） |
| 4 | CAM++ 算子补丁 + 说话人嵌入回归 | NPU 嵌入与 GPU 嵌入 `allclose(rtol=1e-3)`，说话人聚类结果一致 |
| 5 | emotion2vec_plus_large（含 `torch.npu.amp` fp16） | 9 类标签与 GPU 侧一致率 ≥ 95%，分数偏差可接受 |
| 6 | VAD 性能实测（本项目离线批量用法） | 单个 10 分钟音频的 VAD 耗时与 GPU 侧同量级（不出现数量级劣化） |
| 7 | 端到端 `/v2/seacraft_asr` | 转写结果、`language_ratio`、role / emotion 字段结构与 GPU 侧一致 |
| 8 | 并发压测（1/2/4/8） | 记录 `npu-smi` 的设备利用率与显存占用，摸清瓶颈 |

其中第 4、6 步是**风险最高**的两步，建议优先安排。

---

## 八、最终结论

1. **三个模型在华为 910B 上都存在官方支持的运行路径**（昇腾官方 FunASR 文档在 910B4 上验证了 FSMN-VAD、
   CAM++、emotion2vec_plus_large），本项目使用的模型权重**无需转换或重新导出**。
2. **CAM++ 是唯一必须打补丁的模型**：其 `seg_pooling` 依赖 NPU 不支持的 `AvgPoolV2` 大 stride，
   官方已给出等价实现与回归方法，改造量约 20 行。
3. **本项目需要改造的核心是设备抽象层**：把 `torch.cuda.*` 与 funasr 的 CUDA 硬编码收敛成一层可分派的
   设备/AMP 上下文，预计改动集中在 4 个文件（`core/models.py`、`core/vad.py`、`api/routes/asr_qwen_common.py`、脚本）。
4. **架构上可以在 NPU 与 GPU 上都跑通**，且两者可以并存（模型目录、接口契约、响应结构完全一致），
   但需要**独立的 NPU 镜像流水线**（aarch64 + CANN + CPU 版 torch + torch_npu）。
5. **本报告未在真实 910B 硬件上实测**，属于「官方文档 + 社区先例 + 代码静态核对」的结论；
   立项前建议按第七节跑一次最小验证，尤其是 CAM++ 数值等价性与 VAD 性能这两项。
6. **Qwen-ASR 不影响本结论**：本服务与 Qwen 之间只有 HTTP 契约，Qwen 是否上 NPU 与本服务解耦。

---

## 参考资料

- 昇腾官方 FunASR NPU 指南（Atlas 900 A2 PODc / 910B4 验证环境与三个模型示例）：
  <https://ascend.github.io/docs/sources/funasr/index.html>
- FunASR 官方 PR：《在昇腾 NPU 上运行 FunASR》（CAM++ `seg_pooling` 算子问题与等价补丁、依赖配对注意事项）：
  <https://github.com/modelscope/FunASR/pull/3708>
- FunASR torch_npu 被 pip 解析顶掉的 issue（需 constraints / `--no-deps`）：
  <https://github.com/modelscope/FunASR/issues/3707>
- 昇腾社区：emotion2vec_plus_seed 在 910B（Ascend A2）跑通记录（torch 2.8.0+cpu / torch_npu 2.8.0 / CANN 8.3.RC2）：
  <https://bbs.huaweicloud.com/ascend/forum/forum/thread-02144209204251079316-1-1.html>
- 昇腾社区：910B 上 FunASR 实时转录叠加 VAD 后变慢的反馈（流式 `async_vad` 场景）：
  <https://www.hiascend.com/dev/forum/thread-02144209204251079316-1-1.html>
- torch_npu 与 CANN 配套矩阵：<https://github.com/Ascend/pytorch>

## 附录：本仓库当前环境基线（用于对比）

| 项 | 当前（GPU） |
|---|---|
| conda 环境 | `/root/anaconda3/envs/asr`，Python 3.11.16 |
| torch | 2.6.0+cu124 |
| funasr | 1.2.6 |
| 硬件 | 3 × RTX 4090（Qwen 容器占 GPU0） |
| 目标 NPU 环境（参考官方验证组合） | Python 3.12 / torch 2.9.0+cpu / torch_npu 2.9.0.post2 / CANN 9.1.0 |
