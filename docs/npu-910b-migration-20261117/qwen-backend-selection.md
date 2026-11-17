# Qwen 后端候选评估记录（2026-11-17）

本记录区分环境/源码核查、CPU 工具测试与 NPU 实测。原生 Transformers + torch_npu eager 是当前交付基线；不能据此宣称性能最佳。候选通过后也不会自动替换生产主链，替换前须同步 OpenSpec 和运行依赖边界。

## 当前主环境与证据等级

环境/源码核查对象为 `xjtu910b7`、解释器 `/home/zhangs/conda-envs/asr/bin/python`；本节只记录版本和源码事实。NPU实测使用宿主 `npu:7`，结果与日志见后节，不能用环境元数据替代真实前向。

| 项目 | 实际读取值 / 证据 |
| --- | --- |
| Python 包元数据 | torch 2.5.1，torch-npu 2.5.1，transformers 5.13.0，setuptools 84.0.0，protobuf 7.36.2 |
| torch_npu 源码 revision | `torch_npu/version.py`: `git_version = 9e105eaa2a2efb6f70c71f9ccab318ddcea008ce` |
| CANN | `/usr/local/Ascend/ascend-toolkit/latest/opp/version.info`: `Version=7.7.0.1.238`、`version_dir=8.1.RC1`、`timestamp=20250428_203017761` |
| 顶层可选包元数据 | `torchair`、`vllm`、`vllm-ascend` 均无独立 distribution。此结论不等于没有内置 TorchAir |
| 内置 TorchAir | `torch_npu/dynamo/torchair/__init__.py` 存在，SHA256 `bf102ed6e3fec2ede70252e0a9c75d312be105cf676a231f563b2ae6984879c4`；`core/_torchair.so` 851921 字节，SHA256 `8910be0305c5563b4950ff521b0afaf2a5d6acb2eadd928aa364eefe44350dec` |
| 当前 Qwen config | `/home/model/Qwen3-ASR-1.7B-hf/config.json` 2399 字节，SHA256 `117ac8e63e2af7cae3665e5a632d6eb03f5f384915519ceb6403c15ec6533f63`；`model_type=qwen3_asr`，`architectures=[Qwen3ASRForConditionalGeneration]`，`transformers_version=5.13.0.dev0`，未包含可追溯 `_commit_hash` |

config哈希不是权重哈希，也不能冒称上游模型revision。`model-assets.json`已登记本地`model.safetensors`为4,076,193,080字节、SHA256 `2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1`，tokenizer_config SHA为`945e980986de2ca7768f3326bfdbb4fbea3406f972b8ae0be233089f2b253c11`；这是资产清单所记录的文件标识，最终asset-audit另行刷新，不表示每次读取包metadata时都重新hash大权重。上游模型revision仍未知，不能由目录名/config hash或本地权重hash猜测。

## eager 与长度分桶原生 batch

`transformers/models/qwen3_asr/processing_qwen3_asr.py` 的已装 5.13.0 源码确认：`apply_transcription_request` 接受独立音频数组列表和一一对应语种列表；全名/ISO 码由 `resolve_language` 验证，具体语种作为 system 消息。当前版本不是 assistant 预填语种方案。默认保留 token `attention_mask` 与音频 `input_features_mask`，音频不会被拼接。

语言契约：未提交、`None`、空串和 `auto` 均归一为自动识别；具体合法全名/ISO 码传入 processor；非法语种才映射 `4009`。替身注入 `QwenAsrUnsupportedLanguageError` 只证明错误 payload 映射，不能作为空串/auto 被拒绝的证据。真实语言与输出验证以NPU/HTTP验收日志为准。

新增 `scripts/benchmark_qwen_candidates.py`：

- 每进程只运行一个 eager 或 batch 候选，独立记录加载时间。默认 BF16、宿主 `npu:7`、256 输出 token；拒绝其他设备及 `ASCEND_VISIBLE_DEVICES` / `ASCEND_RT_VISIBLE_DEVICES` 重映射，避免误把逻辑卡号当物理卡 7。
- JSON manifest 支持稳定 ID、原路径或 `path_bytes_b64`、起止秒和逐段语言；只读来源 WAV，在内存提取 16k/单声道/PCM16 独立片段，登记来源 SHA256、clip WAV SHA256 和精确帧边界。
- eager 按样本逐段执行；batch 按时长桶排序，保留原 index/ID、尾批和每条语种。直接调用 processor 原生列表输入，检查 token/音频 mask 和有效长度，保留整数/布尔 dtype，不使用无 mask 的补零均值。
- 每个预热及稳态批次保存原始耗时、raw/parsed 输出、mask 长度、失败及结果归属；缺语种前缀的非空输出记协议失败，不能伪装无人声。batch raw 保留 decoder 结束控制 token；eager raw 是生产 runtime 已剥离结束控制 token 的返回值，报告有 `raw_output_stage` 标记，质量比较应使用 parsed 字段。
- P50/P95 是批次耗时；批大小不同不能把它当相同含义的单段延迟。另给成功批音频吞吐/RTF并保留失败数；有失败时不据成功子集认定候选通过。包含预处理、传输、generate、解码与同步，不是纯 kernel 时间，也不替代 v3 `gpu_time_ms` 原主链计时。
- `finally` 关闭本工具持有模型；SIGINT/SIGTERM 仅请求停止，等待当前同步设备工作、关闭后退出，重复信号不会跳过关闭。无网络服务、端口或共享进程处置。输出路径必须新建，逐次原子保存证据。

独立候选运行示例（实际eager/batch结果见后节）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
/home/zhangs/conda-envs/asr/bin/python scripts/benchmark_qwen_candidates.py \
  --manifest /tmp/qwen-candidate-clips.json --candidate eager \
  --warmup 2 --repeats 5 --output /tmp/qwen-eager-run.json
/home/zhangs/conda-envs/asr/bin/python scripts/benchmark_qwen_candidates.py \
  --manifest /tmp/qwen-candidate-clips.json --candidate batch \
  --batch-size 2 --bucket-seconds 2 --warmup 2 --repeats 5 \
  --output /tmp/qwen-batch-run.json
```

这是离线候选工具，不是生产微批调度器。当前没有启用生产微批，因此生产批等待上限/容量计数/取消归属变更不适用；若后续采用，须补设计与 task 10.6 / 11.3 的调度、尾批和取消验收。无人工参考时只能报告输出差异，不能宣称 WER/CER 达标。

## TorchAir：内置但当前主环境存在导入阻点

`torch_npu/dynamo/__init__.py` 确认内置 TorchAir 通过 `sys.modules['torchair']` 懒暴露，并注册 Dynamo 的 `npu` backend。官方配套表列 TorchAir 7.0.0 / PyTorch 2.5.1 / CANN 8.1.RC1，说明当前底座有匹配图模式路线；这不是当前 wheel 的独立 TorchAir revision 证明，更不是 Qwen 算子支持/性能证明。[官方 TorchAir 版本表与入口](https://github.com/Ascend/torchair/blob/master/README.md#版本配套表)

已运行的只读导入探针：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
/home/zhangs/conda-envs/asr/bin/python - <<'PY'
import traceback
import torch, torch_npu
try:
    import torchair
    print(torchair.get_npu_backend(compiler_config=torchair.CompilerConfig()))
except BaseException:
    traceback.print_exc()
print('NPU_INITIALIZED', torch.npu.is_initialized())
PY
```

实际失败链：`torch_npu/dynamo/torchair/__init__.py:8 import pkg_resources` → `ModuleNotFoundError: No module named 'pkg_resources'` → `torch_npu.dynamo._TorchairImportError`；`torch.npu.is_initialized()` 前后均为 `False`。因此当前主环境无法直接启动图候选，不能写“没有 TorchAir 包”或“910B 不支持 TorchAir”。Setuptools 官方记录已删除 `pkg_resources`，当前 84.0.0 与老 TorchAir 的该依赖冲突。[Setuptools 删除记录](https://setuptools.pypa.io/en/latest/history.html#v82-0-0)

最小候选隔离方案：按 conda explicit + 完整 pip 锁在独立临时前缀重建当前运行环境，在该前缀固定 `setuptools==80.9.0`，保留 torch/torch-npu/CANN 版本；先验证 `import pkg_resources`、TorchAir backend 构造与 `pip check`，再跑同样本 eager 参考及 `--compile torchair`。这仅消除已知首个导入阻点，未验证的后续 ABI/protobuf/算子问题仍须记录，不承诺一次安装即可执行；不修改主 `asr` 前缀或 site-packages。环境重建实测发现 `conda create --clone asr` 可能重新链接 conda 原 NumPy2，覆盖主环境 pip 的 NumPy1.26.4；因此不能把 clone 成功当一致性证据，必须实际导入核对 NumPy版本/路径、逐项比对锁，修复只发生在候选前缀。

工具 `--compile torchair` 使用匹配 wheel 的 `torchair.get_npu_backend(CompilerConfig())` 与 `torch.compile(model.forward, dynamic=True)`，保留 Python generate 循环。冷调用/编译耗时在预热记录中，必须同时保存 graph break/recompile 日志，不能只看编译包装函数返回或把 eager 回退称为图优化成功。初期探针阶段尚未获得完整 NPU 编译/稳态结果；后续隔离前缀的两组42次失败实测见后节，不将这段历史状态作为最终结论。

## vLLM-Ascend：锁定历史候选，当前环境不适用

| 路线 | 官方源码/矩阵事实 | 对当前项目的判断 |
| --- | --- | --- |
| v0.7.3rc2 一代 | 要求 torch ≥2.5.1、torch-npu ≥2.5.1.dev20250320、CANN ≥8.0；对应上游 v0.7.3 模型 registry 没有 Qwen3-ASR | 底座接近当前不代表存在该模型适配，不能仅降 vLLM 版本就算可用 |
| v0.14.0rc1 | 配套 CANN 8.5.0、torch/torch-npu 2.9.0，模型矩阵未列 Qwen3-ASR | 与当前底座不匹配，缺少该 tag 下的模型验收证据 |
| 可复查候选 v0.23.0rc1 | 插件 commit `f4a08bddd0cc65a0bd8c3d377b158ae5ca7527db`，配套上游 vLLM 0.23.0，CANN 9.0.1、torch 2.10.0、torch-npu 2.10.0.post2、Triton-Ascend 3.2.1；requirements 另锁 transformers 5.5.4。该 tag 提供单卡 64GB 910B 的 Qwen3-ASR BF16/eager 教程 | 存在可评估实现，不能宣称永久不可用；但不能装入当前 torch2.5.1/CANN8.1RC1/transformers5.13 环境，也不能只建新 conda 就忽略 CANN/宿主驱动配套 |

依据：[v0.7.3rc2 安装源码](https://github.com/vllm-project/vllm-ascend/blob/v0.7.3rc2/docs/source/installation.md)、[vLLM v0.7.3 registry](https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/models/registry.py)、[v0.14.0rc1 安装](https://github.com/vllm-project/vllm-ascend/blob/v0.14.0rc1/docs/source/installation.md)、[v0.14.0rc1 模型矩阵](https://github.com/vllm-project/vllm-ascend/blob/v0.14.0rc1/docs/source/user_guide/support_matrix/supported_models.md)、[v0.23.0rc1 发布配套](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.23.0rc1)、[同 tag requirements](https://github.com/vllm-project/vllm-ascend/blob/v0.23.0rc1/requirements.txt)、[同 tag 910B 模型教程](https://github.com/vllm-project/vllm-ascend/blob/v0.23.0rc1/docs/source/tutorials/models/Qwen3-ASR-1.7B.md)。在线文档和部分 tag 的 `conf.py` 替换变量存在滞后，最终锁定实际 commit、wheel/image digest 及 requirements，不能只记页面标签。

维护状态补充：当前官方main的Qwen3-ASR教程已声明从vLLM-Ascend0.30.0起停止该模型的后续支持与修复，并将部署指向历史文档。因此此处锁定0.23.0rc1是历史候选评估，不能假定跟随最新main即可持续获得模型维护；这也不能反推历史配套永远无法部署。若后续选入生产，须明确固定版本、漏洞/兼容补丁与长期维护责任。[官方当前维护声明](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/Qwen3-ASR-1.7B.md)

还存在权重格式与语言契约差异。已读 vLLM v0.23.0 的 Qwen3-ASR 实现访问 `hf_config.thinker_config`，权重 mapper 处理 `thinker.*`；当前 5.13 原生 `-hf` 目录只有扁平 `audio_config`/`text_config`，不含 `thinker_config`。其 config 在字段缺失时会构造默认 thinker，不能把“config 对象构造成功”当权重匹配。历史候选应使用独立原版 Qwen checkpoint及 revision，或先验证项目自己的权重映射，禁止覆盖现有模型目录。[vLLM v0.23.0 模型加载与提示](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/model_executor/models/qwen3_asr.py)、[其嵌套 config](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/transformers_utils/configs/qwen3_asr.py)

该历史 vLLM 原生转写接口对指定语种使用 assistant `language …<asr_text>` 预填，并在后处理中剥离语种前缀，与当前 processor 的 system hint 和项目 raw 解析契约不同。单纯看到转写文本不能证明 `4009`、自动语种和 raw 前缀等价，需逐项适配与质量回归；不应直接替换现有客户端。

最小隔离评估方案（本轮只记录，未安装或启动）：

1. 选官方 `quay.io/ascend/vllm-ascend:v0.23.0rc1` 历史镜像或该 commit 构建产物，并固定实际镜像 digest；使用其 CANN9.0.1/NNAL 与 Python3.10–3.12 配套，核对宿主驱动兼容，不升级共享宿主。新 conda 前缀不能隔离不匹配的系统库。
2. 只暴露宿主物理卡7，容器内按实际重编号使用逻辑卡0；只读挂载独立原版 checkpoint 与样本。先用离线 `LLM.chat/generate`，无端口；若需HTTP测试，另行登记唯一端口和 owned 进程树、统一 teardown。
3. 固定 vLLM0.23.0 / vLLM-Ascend0.23.0rc1 / torch2.10.0 / torch-npu2.10.0.post2 / transformers5.5.4 / triton-ascend3.2.1，保存 `pip check`、完整锁、源码commit和补丁；不与主 asr 环境混装，也不把元数据通过算前向通过。
4. 同一音频和上游权重比较 eager基线、语言矩阵、raw/parsed、质量差异、加载/编译/稳态耗时、HBM及同卡干扰；结束关闭引擎和所有 owned worker。没有实测证据前标记“当前栈不适用/隔离候选未执行”，不能声称已比较并优选。

## CPU 工具验证与剩余验收

已运行：

```bash
cd /home/zhangs/asr_nvidia
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /home/zhangs/conda-envs/asr/bin/python -m pytest \
  tests/test_benchmark_qwen_candidates.py tests/test_memory_clip_pipeline.py -q --disable-warnings
```

候选工具与内存音频测试先通过 `21 passed in 2.88s`；补充 CLI 重复停止信号和 HTTP 空语言语义后，三份文件子集通过 `27 passed, 11 subtests passed, 1 warning in 4.14s`。覆盖 PCM 只读提取与哈希、起止界限/重复 ID、时长桶/尾批/原序号、混合语言独立数组、token及音频 mask保留/有效长度、prompt切片、raw结束标记和parsed、逐次预热/稳态计数与RTF、失败归属、缺前缀错误、加载/执行中断/关闭异常收尾、停止后不启动新工作、宿主卡7约束。全部为 CPU 替身，不含 NPU 加载/编译/性能或质量验收。

同回归样本eager与原生分桶batch已完成实测，输出差异/速度在后节复核；TorchAir 独立兼容前缀及 vLLM 隔离候选条件另记。当前结论是保留可验证的原生主链并保留优化评估入口，不宣称 eager 性能最优。

进程级FP32辅助算子精度统一如下：`core.local_inference.configure_npu_precision` 在首次 NPU 初始化时固定禁用 conv/matmul HF32，并由 Qwen runtime 首次加载前调用，状态记录 `precision_policy`。这不改变 BF16 权重 dtype；候选比较必须使用同一数学模式，已有旧模式测量不得混作同一性能/质量基线。该helper的实机数值验证见增强模型数值验收报告，本记录不将 CPU 配置测试冒称数值验收。

## 已完成的 eager / batch2 NPU 数据复核与本轮选择

以下数据来自宿主 `npu:7` 串行运行候选的原始JSON。两次均使用同一 6 条片段 manifest、同一 `-hf` 权重目录、BF16、`max_new_tokens=256`、两轮预热、五轮稳态、关闭模型成功；两份 runtime.precision_policy 均为 `fp32_ops_hf32_disabled`，conv/matmul HF32 均 False。模型上游 revision 仍未知，不以 config 哈希冒充。

| 原始证据 | SHA256 |
| --- | --- |
| `/tmp/asr-apply-20261117-a99trpfv/qwen-eager.json` | `c6896773333534f977533282a6d37509bb81ce4961f9e8c7e3870dda06238b4a` |
| `/tmp/asr-apply-20261117-a99trpfv/qwen-batch.json` | `b6a3cd01d06b607b2d1527aafbafa956f55d838f9f9c8525174a074386de69dd` |
| manifest | `/tmp/asr-apply-20261117-a99trpfv/asr-qwen-bench-manifest.json`；每份 JSON 内附完整 samples、来源路径原始字节、来源 SHA、PCM帧边界和 clip SHA |

### 性能：存在值得继续评估的收益

| 稳态指标 | eager（单段） | 原生 batch2 |
| --- | ---: | ---: |
| 首次模型加载 ms | 18247.997510246933 | 18048.324500210583 |
| 成功批次 / 失败批次 | 30 / 0 | 15 / 0 |
| 稳态输出段数 | 30 | 30 |
| 稳态音频秒数（含重复测量） | 138.0 | 138.0 |
| 稳态总同步执行耗时 ms | 21629.626262001693 | 13827.879809774458 |
| 批次 P50 ms | 736.6348705254495 | 958.5270397365093 |
| 批次 P95 ms | 1021.9531105831265 | 1201.1086698621511 |
| RTF（执行秒 / 音频秒） | 0.15673642218841816 | 0.10020202760706128 |
| 吞吐（音频秒 / 执行秒） | 6.38013797965776 | 9.979837972156261 |
| 自有模型关闭 ms / 结果 | 409.2683298513293 / ok | 392.51079969108105 / ok |

该样本组合吞吐提高约 56.4%，RTF 降低约 36.1%。P50/P95 的单位仍是批：batch2 处理两个段，不能把 batch 耗时直接当作与单段同口径的延迟，也不能据此推断端到端八路业务吞吐。这是共享卡串行测量，未隔离同卡服务影响。完整逐次预热/稳态耗时保留在原始 JSON，未混入纯 kernel 计时。两次 processor 源码 SHA256 同为 `c8aac6dc0ae9e8818a1f2dc7353a4f90e9f9f727888111adb5dee1cf5c16895d`。

预热原始耗时（ms，按记录顺序；eager每轮6段，batch每轮3批）：eager `[2716.225239,1021.482680,565.437309,889.664110,390.745500,379.683990,871.893180,1004.923350,564.962720,930.160420,392.104981,397.029460]`；batch `[2648.500680,1241.819439,590.773550,887.997900,1088.967180,597.385710]`。首个前向开销保持在预热数据中，不混入稳态统计。

### 样本身份、结果和 mask 核查

两份 JSON 的 samples 列表逐字段完全相同。全部片段从各来源第 60 秒开始：`start_frame=960000`；4.1秒片段 `end_frame=1025600 / frames=65600`；5.1秒片段 `end_frame=1041600 / frames=81600`；均为16kHz、单声道PCM16、自动语种。每来源两个长度片段互相重叠，不能当独立质量验收样本。

| 来源短 ID | 来源完整 SHA256 |
| --- | --- |
| `6156a4b38b576557c2be` | `00190f95f88dd6ecc8dde1af62c338a6c8753b9f5832e044fa9d23d57a6dd801` |
| `96862e257ed1629907b8` | `26700c75b989fcb8623e5858daa2c24d23452ae1a15b283ef9c0e2e7e87b56ac` |
| `4b897698e079522a81b8` | `f5adfe2b880e2064e4990ac87d8144bdd159faacb0deb4ee3829fdf45f553b4e` |

以下 ID 均带 `audio-` 前缀，表内缩写仅用于阅读；完整 ID 与原路径在 JSON 中。两次各有42个输出（6样本×2轮预热+5轮稳态）。每个后端内部，同一样本七次 parsed 结果均稳定。

| 原 index / 样本 | clip WAV SHA256 | eager parsed | batch2 parsed | 比对 |
| --- | --- | --- | --- | --- |
| 0 / `6156a4b38b576557c2be-4.1` | `b610f02665da51ad38a67e8301cc6d25abb4da7ff3690706f6547adf7561df8d` | English / `How to remove the last blue in the.` | 同 eager | 一致 |
| 1 / `6156a4b38b576557c2be-5.1` | `3c4fadf973bf2927dc9b3eee232a61a280726315e3872fe30f8a21723444418b` | English / `How to remove the last blue in the second row?` | 同 eager | 一致 |
| 2 / `96862e257ed1629907b8-4.1` | `d968123395602fc11e33515c8c074550c89fc268a5f1dbd98f8c671b0bcc2a36` | Chinese / `发布一个。` | 同 eager | 一致 |
| 3 / `96862e257ed1629907b8-5.1` | `a61b989a952256544f7476a8832e790801b77f2ba13334f5613418b164345559` | Chinese / `我还没发布呢，我还没发布呢。` | Chinese / `发布一个。` | 七轮均存在实质文本差异 |
| 4 / `4b897698e079522a81b8-4.1` | `bae944573dc4ea548a93be6b26d11cdbe72a6ff0feaf9d52ab8945bba0348043` | language=None / text空 / has_speech=False | 同 eager | 一致 |
| 5 / `4b897698e079522a81b8-5.1` | `0d53c7aadf1d635cf72e288dd69a7fe3d8f715fe57c39984084bcdd817cd9634` | language=None / text空 / has_speech=False | 同 eager | 一致 |

所有 batch 记录的 `sample_id`、`outputs.original_index`、`record.original_indices` 和 `record.sample_ids` 逐条对应，未发现结果错位。长度排序后每轮批次如下，原 index 足以恢复请求顺序：

| 每轮批次 | 原 index | 实际PCM帧数 | token mask shape / 有效长度 | 音频 mask shape / 有效长度 |
| --- | --- | --- | --- | --- |
| 0 | `[0,2]` | `[65600,65600]` | `[2,69] / [69,69]` | `[2,500] / [410,410]` |
| 1 | `[4,1]` | `[65600,81600]` | `[2,82] / [69,82]` | `[2,600] / [410,510]` |
| 2 | `[3,5]` | `[81600,81600]` | `[2,82] / [82,82]` | `[2,600] / [510,510]` |

声学有效帧数410/510分别等于PCM样本数65600/81600除以processor的160采样步长；异长批保留了有效长度与 token mask。未发现缺 mask、帧长或顺序错误，但这不能独立证明模型内部数值/注意力与逐段严格等价。

raw 的结束控制 token 差异已与文本差异分开：eager 返回值已剥离结束控制 token；batch decoder 保留 `<|im_end|>`，本组 raw 的结束 token 数为1/4/7/12，较短输出会以该结束/padding token 补齐，parsed 正确剥离。Chinese5.1 的不同存在于剥离后的文本，不能用 EOS 补齐解释。两条无有效语音输出均保持 `language None<asr_text>` 的既有解析语义。没有人工参考，当前不判定哪份文本正确，也不宣称 batch 质量合格或不合格。

### 本轮生产选择与后续最小边界

本轮选择保留 eager 主链：候选有明确吞吐潜力，但只覆盖3个来源的重叠短片段，其中2条无有效语音，且1/6样本存在七次稳定的实质文本差异；缺少人工参考，不能确认质量满足同一准入标准。此决定是对现有证据范围的取舍，不是“eager 已证明最快”或“微批永远不可用”。batch工具保留为离线评估工装，当前没有第二条正式生产执行路径。

若后续决定生产微批，最小变更与门槛为：

1. runtime 提供原生独立音频数组/语种列表批接口，保留浮点dtype、整数索引、token/声学mask、raw前缀和逐段结果恢复；不改VAD分段或时间轴，不拼音频。
2. client 增加有界待处理段队列、长度桶、初始最多2段及有限最大等待/尾批策略；不能无限等满批。`busy`按实际模型执行批数，`waiting`按待执行段数，分别报告，不把段/批混算。
3. 调整全局闸门计数：生产微批必须区分待凑批段与实际设备批的容量；若门限不足一个批次，会阻止后续成员入队，不能沿用单段闸门制造等满批死锁。具体容量须按最终配置重新验收，配置/状态和测试同步修改，v3 public schema可以保持不变。
4. 入批前逐段验证具体语种，避免一个非法语种把健康段一起误报4009。等待中取消只移除未启动段；批已执行则仍持有独立worker至设备同步结束。单段取消/重试/错误、整体批错误和实例失败的归属须明确，不能提前归还设备实例或音频租约。
5. 关闭停止收新段，排空已执行批并处理尾批，重复取消仍等待 owned 工作及关闭完成。补混合长度/语言、尾批、失败、单成员/全部取消、超时和关闭中取消的CPU及安全实机用例。
6. 先针对Chinese5.1差异保留音频/参考并定位processor/模型数值来源；扩充人工核对的课堂、低音量、静音、中英混合及长短段质量集，再与逐段参考按相同质量标准比较。不能事后为候选调整准入门槛。
7. 通过同卡共享条件下完整业务/8路并发的吞吐、P50/P95、RTF、HBM及稳定性验收，且收益足以覆盖队列维护复杂度后，先更新 proposal/design/specs/tasks 与依赖边界，再改变生产路径。

上述后续范围主要是 `core/qwen_asr_runtime.py`、`core/qwen_asr_client.py`、`core/qwen_asr_queue.py`、配置与测试；当前未实施生产微批、未改生产模型配置或功能。TorchAir隔离候选结果见后节，未通过不改变本轮eager生产选择。

## TorchAir 隔离前缀的实际编译失败与下一候选

TorchAir真实NPU候选使用 `/tmp/asr-rebuild-20261117-torchair` 独立前缀。该前缀保留目标运行栈、仅将候选环境 setuptools 固定为80.9.0，解决前述 pkg_resources 导入阻点；主 `asr` 环境不变。下表依据原始报告。

| 证据 | 路径 / 值 |
| --- | --- |
| 原始 JSON | `/tmp/asr-apply-20261117-a99trpfv/qwen-torchair.json` |
| JSON SHA256 | `7f5a788bdacd7cd22edd794db18709803c77fb53c22ce4c28633ff2778330148` |
| 原始日志 | `/tmp/asr-apply-20261117-a99trpfv/qwen-torchair.log` |
| 日志 SHA256 | `e2543dda88b303127400709377b2b445515ccf9f55269405fdeb7ef879ae8d0a` |
| 运行配置 | eager单段候选 + torchair动态forward编译；6样本×2轮预热+5轮稳态；BF16/npu:7/256 token，与前节同manifest |
| 模型加载 / 编译包装设置 ms | `18049.746019765735 / 20.236419513821602`；后者仅 torch.compile 包装设置，绝非编译成功耗时 |
| 首个前向尝试 ms | `36669.059690088034`，包含冷编译尝试并最终失败，不能作成功编译或稳态性能数字 |
| 成功 / 失败 | 0 / 42；预热12次、稳态30次全部 `BackendCompilerFailed`，没有有效 raw/parsed 转写输出 |
| 关闭 | `ok / 709.8248396068811 ms` |

全部42个错误字符串均包含 `Unsupported torch op aten._safe_softmax.default by ge`。调用来源为已装 `transformers/integrations/sdpa_attention.py:93` 的 `sdpa_attention_forward` → `torch.nn.functional.scaled_dot_product_attention`，在 TorchAir FX→GE 转换阶段失败。报告未产生可比较稳态RTF/P50/P95（字段为null或0项统计），不能把失败执行的耗时当作候选速度，也不启用 `torch._dynamo.config.suppress_errors` 把静默回退误报为图优化成功。

这个结果只说明当前匹配 TorchAir 与此 SDPA 算子组合未通过，不能推出910B永久不支持图模式。下一次只改变离线候选的 attention：新增 `--attention default|eager`，默认 `default` 保持现有选择；指定 `eager` 时在 `ensure_loaded` 之后、编译之前调用 `model.set_attn_implementation('eager')`。生产 runtime、权重文件和生产配置均不修改。

已读本机 Transformers5.13源码确认 `PreTrainedModel.set_attn_implementation` 支持字符串并向所有适用submodel及sub_config传播；`Qwen3ASRPreTrainedModel` 声明 `_supports_attention_backend=True`，音频attention通过 `ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)` 选择实现。候选工具同时记录调用前后真实 config 对象、audio/text子配置及其module路径的 `_attn_implementation`，并在指定eager但任一配置未传播时明确失败，不能只记用户传入参数冒充实际实现。

eager-attention候选以新进程运行相同样本的 `--compile torchair --attention eager`，记录真实子配置、图转换/重编译、raw/parsed差异、冷尝试和稳态耗时、资源与关闭。它须与相同attention的未编译参考区分：attention改动和图编译可能分别改变输出/耗时，不能将二者混作纯图加速收益。真实结果见后节。

## eager attention 与 TorchAir eager-attention 实测复核；本轮最终后端选择

后续两组相同manifest的独立进程测量已完成，以下依据JSON与日志复核。两份 samples 与前面默认 attention 单段/batch2 的身份、来源/clip SHA、帧边界逐字段完全相同；未改权重或生产 runtime。

| 新证据 | SHA256 |
| --- | --- |
| `/tmp/asr-apply-20261117-a99trpfv/qwen-eager-attention.json` | `3eb8618adc1c9cd7328088bbd5ac818cf954b691bd3a6406b12923fb95d76047` |
| 同名 `.log` | `3fff4df8c075a94cad9400372faf954ab5d6d5c0d9689e16dff37e61d8d3917c` |
| `/tmp/asr-apply-20261117-a99trpfv/qwen-torchair-eager-attention.json` | `be3915157b5f7cc7d3da834b0fa873d1facba8523216d61034ba8bdd4fb2f79e` |
| 同名 `.log` | `e49cc1e3807491da062c68d48126748d28da72c77a77bedae481b6c80b5de18b` |

### attention 真实传播核验

两组新报告的 `model.attention.before/after` 与 `actual_attention_configurations` 一致确认以下三个真实 config 对象从 `sdpa` 变为 `eager`，不是仅保存CLI参数。配置对象关联的全部module路径均记录在JSON中：

| 实际 config | before | after | 关联module路径数 / 代表路径 |
| --- | --- | --- | --- |
| `Qwen3ASRConfig`（config） | sdpa | eager | 2；`<root>`、`model` |
| `Qwen3ASREncoderConfig`（config.audio_config） | sdpa | eager | 25；`model.audio_tower`及24个encoder self_attn |
| `Qwen3Config`（config.text_config） | sdpa | eager | 58；`model.language_model`、28个self_attn/MLP及rotary_emb |

先前未加attention元数据的默认基线保留原加载行为；相同权重/5.13加载的这两次 `before=sdpa`，加上默认TorchAir日志的SDPA调用路径，支持本轮锁定栈默认使用SDPA的结论。此处的 **eager执行** 指不编译Python/PyTorch前向；**eager attention** 指手写attention实现，两者是不同维度，报告不能混称。

### 未编译 eager-attention：小幅吞吐变化及一条文本变化

| 指标 | 默认SDPA + eager执行（已有基线） | eager-attention + eager执行 |
| --- | ---: | ---: |
| 稳态成功 / 失败 | 30 / 0 | 30 / 0 |
| 稳态音频秒 | 138.0 | 138.0 |
| 模型加载 ms | 18247.997510246933 | 17244.509000331163 |
| 稳态总同步执行 ms | 21629.626262001693 | 21345.447765663266 |
| 批次（单段）P50 ms | 736.6348705254495 | 669.2141746170819 |
| 批次（单段）P95 ms | 1021.9531105831265 | 1075.4835698753595 |
| RTF | 0.15673642218841816 | 0.15467715772219767 |
| 吞吐（音频秒/执行秒） | 6.38013797965776 | 6.465078714440913 |
| 自有模型关闭 ms / 状态 | 409.2683298513293 / ok | 391.35672990232706 / ok |

新未编译候选42个输出（2轮预热+5轮稳态）全部成功，每样本七次parsed结果稳定。吞吐约提高1.33%，但P95约增加5.24%；共享卡、短测及小样本下，这不是稳定加速结论，不能仅根据P50或微小总吞吐差异替换生产attention。

三份未编译报告中，其余5/6样本的语言、文本和has_speech逐样本等价，唯一不同仍为 `audio-96862e257ed1629907b8-5.1`（81,600帧，clip SHA见前表）：

| 候选 | 该样本七次稳定parsed输出 |
| --- | --- |
| 默认SDPA + 单段eager执行 | Chinese / `我还没发布呢，我还没发布呢。` / has_speech=True |
| eager-attention + 单段eager执行 | Chinese / `我还没发布呢。` / has_speech=True |
| 默认attention + batch2 eager执行 | Chinese / `发布一个。` / has_speech=True |

第二行少一遍重复短语，第三行文本有实质区别。没有人工参考，不能将少复读自动判为准确率提升，也不能判定三者哪份正确；此次只报告差异。两条无语音片段三路线均保持language=None/text空/has_speech=False。没有发现样本身份、帧界或结果索引错位的证据。

### TorchAir + eager-attention：传播生效但42次前向仍失败

| 项目 | 实测值 |
| --- | --- |
| 配置 | `candidate=eager / compile=torchair / attention=eager`，同6样本/BF16/npu:7/256token |
| 模型加载 ms | `18724.624060094357` |
| torch.compile包装设置 ms | `22.155740298330784`，不是成功编译耗时 |
| 首个冷前向/编译尝试 ms | `31653.850429691374`，最终失败 |
| 成功 / 失败 | 0 / 42（预热12、稳态30均失败） |
| 错误 | 全部 `TypeError: linear(): argument 'input' (position 1) must be Tensor, not NoneType` |
| raw/parsed与稳态P50/P95/RTF | 没有成功输出；稳态性能字段不可用，不能把约0.2秒的后续失败快速返回当加速 |
| 关闭 | `ok / 505.90927992016077 ms` |

原始日志的可核对定位（位于隔离前缀的5.13源码）：

- `.log:16–31` 显示 `modeling_qwen3_asr.py:649 forward → :549 get_audio_features → :494 audio_tower → :395 get_audio_cu_seqlens → :303 feature_lens_after_cnn.max().item()` 的Dynamo图中断路径。
- `.log:34–45` 显示 fake tensor compiler exception / Adding a graph break，定位 `modeling_qwen3_asr.py:193 torch_dynamo_resume_in_forward_at_146` 返回attention输出，以及`:171`的动态split列表推导。
- 该源码`:146–147`按配置选attention，`:171–188`按`cu_seqlens`有效长度split/chunk后拼回，`:190–193` reshape→out_proj→返回。它与日志一起将编译不兼容范围指向动态音频attention/分段图路径。

JSON保存最终linear的NoneType错误；当前日志未包含该TypeError的完整Python调用栈。因此不能精确断言是某个特定linear算子的哪个输入首先变None，也不能把“动态attention图路径相关”这一推断写成已证实根因。可以确定的是：attention切换已生效，原SDPA `_safe_softmax` 路径的首个阻点不再是本次最终错误，但现有图组合依然没有得到任何有效转写。未打TorchAir/Transformers补丁、未改site-packages、未以静默eager回退伪造成功。

### 本轮正式选择

本轮正式主链保持 **Transformers5.13默认SDPA attention + 原生eager执行 + BF16权重 + torch_npu**。batch2虽有约56.4%的短样本吞吐潜力，但质量差异尚缺人工参考；eager-attention未显示足够稳健的整体收益且有输出变化；两种TorchAir图候选在当前匹配栈均未产生成功转写，vLLM历史候选需另一套CANN/驱动/原版checkpoint隔离条件。因此本轮不把batch、eager-attention或图编译加入正式生产执行路径，也不宣称这些路线永久不可用或默认主链已证明全局最优。

native 实例1与2的标定只调整同一原生后端的权重副本/调度容量，不是切换backend；须另核对逐请求输出、P50/P95、吞吐、HBM和同卡干扰后选择，不能从本节单实例数据预断instance_count=2必然更快。正式配置保留1实例，具体标定数据和并发档位见独立容量标定报告；后节双实例冷启动属于边界回归，不表示正式配置改为2实例。

## 初始化默认dtype竞态：CPU实证与共享guard修复

后续审计发现另一项与后端选型无关、但会影响首次并发正确性的P1问题。已装Transformers5.13 `modeling_utils.py:239–260` 的 `local_torch_dtype` 在进入时保存 `torch.get_default_dtype()`，临时调用进程全局 `torch.set_default_dtype(dtype)`，最后恢复原值，没有线程锁；`get_init_context:3788`及`from_pretrained:4320–4324`在模型构造时使用该上下文。

生产原native路径构造CAM++/emotion2vec后直接 `load_state_dict`，没有显式先转换FP32；`TorchNpuRunner.actual_precision`却固定报告FP32。单个B请求会先await增强模型加载，再触发Qwen，因此该单请求自身不会交叠这两个阶段；并发冷A/B请求或两个Qwen实例的首次构造会存在全局dtype窗口。先预热Qwen可规避某次测量，但不能作为首次并发正确性的验收。

实际CPU复现证据为 `/tmp/asr-hf-dtype-race-20261117-wb6h6e2z/result.json`，SHA256 `a78675e8588416a1c160c3ec96cb23707aa992a44e00e468215e1af0ad4a450b`，同目录 `probe.py/stdout.log/stderr.log` 可复算。使用真实HF `from_pretrained` 和独立临时2×2线性层CPU checkpoint，只将Transformers的NPU availability探测置False以避免导入后端，dtype/构造/加载路径未替换；没有加载用户权重、导入torch_npu、创建NPU上下文或启动服务。

- 一个HF BF16构造窗口内，另线程默认native网络参数被构造成BF16；把FP32 checkpoint复制进去后，参数仍为BF16。构造后、复制checkpoint前明确 `.float()` 才能保持FP32；复制后再float不能找回已舍入的精度。
- 两个实际HF BF16构造交错：A先退出恢复FP32，B后续参数构造成FP32，B最后又把全局默认恢复成其入场观察到的BF16；最终global默认被污染为BF16。这既可能污染native构造，也可能使Qwen实例自身混合精度偏离请求。

项目修复采用 `core/model_loading.py:model_loading_guard(dtype=None)` 的进程共享 `threading.RLock`。必须先获得锁，再保存global default dtype；等待锁前读取原值会把另一个HF临时BF16当作需要恢复的值。guard在正常/异常/嵌套退出时恢复进入前dtype，模块导入不导入torch或初始化设备。Qwen以 `guard()` 覆盖processor与完整HF构造/权重加载，避免主动把整个加载阶段都设为BF16；HF仍只在自身构造窗口临时设置。设备传输可在锁外。Qwen加载后验证全部浮点权重参数符合请求dtype，保留库有意使用的FP32缓冲区，不能把全部buffers盲转BF16。

native与CPU增强模型接入同一guard的显式FP32构造，并在 `load_state_dict` 前 `.float()`、核对所有浮点参数和buffer后才能声称FP32。此修复不改变正式后端/attention选择。

新增 `tests/test_model_loading.py` 的6项CPU回归覆盖：锁等待者未在锁外捕获临时dtype、异常/嵌套恢复、全部参数dtype验证/保留FP32buffers、真实HF BF16与native FP32构造互斥且checkpoint小数精确复制、两次真实HF BF16加载后所有权重仍BF16/global恢复FP32、真实HF构造异常恢复。连同 `test_local_qwen_runtime.py`、`test_qwen_cancellation_tool.py` 共 `29 passed in 8.06s`。这不是NPU模型质量验收。

状态实时allocator metadata新增 `torch_default_dtype`：仅查询已在sys.modules中的torch，无方法的CPU替身返回null，NPU未初始化也不为该元数据创建上下文。`test_npu_resource_status.py`（16项）与`test_status_sampling.py`（4项）共 `20 passed in 3.23s`。真实冷A/B及两实例回归应在排空后检查default dtype恢复 `torch.float32`；初始化期间短暂观测到HF BF16允许。源码冷并发验收见末节，不能以CPU回归或提前预热替代。

### 冷并发验收发现的懒导入与热前端边界补充

修复前首次冷A+B、双Qwen实例实机请求未通过：服务日志报 `ImportError: cannot import name AutoProcessor from transformers`，请求返回4011；失败服务已正常关闭并释放端口。先前共享guard只包住 `from_pretrained`，Qwen的Transformers懒属性解析仍在锁外，而FunASR导入在同一guard内，因此不能将该首次并发作为修复完成。现把 `from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration` 一并移入共享guard，完整覆盖懒导入、processor加载、HF模型构造和权重加载。该修复不以预热绕过首次并发。

另查已装5.13真实Qwen前端 `models/qwen3_asr/feature_extraction_qwen3_asr.py:106–107`：waveform显式转FP32，但 `torch.hann_window(self.n_fft, device=device)` 没有指定dtype，会继承另一模型首次加载时的进程默认BF16。输出特征即使仍为FP32，也不能证明Hann计算未被污染。Whisper基类同类CPU复现证据 `/tmp/asr-whisper-dtype-probe-20261117-dvk88u9r/result.json`（SHA256 `dc8252fad807f5ff8dc4be8bdc883fc64cf54374de297855ec8cdaa28826efc1`）在FP32输入下仅改变默认dtype便出现最大绝对特征差 `0.6704812049865723`；这项证据用于说明风险机制，不冒称Qwen最终文本或NPU质量结果。

生产现在仅对 `processor.apply_transcription_request` 完整CPU特征/token构造使用同一 `guard(dtype=torch.float32)`，结束后再按模型dtype明确移动浮点特征，整数token与mask保持原dtype。模型generate、结果decode与设备同步均不持有这把进程级锁，未把不同模型的NPU推理串行为全局大锁。

补充CPU回归：

- 人工延迟Transformers模块解析器验证，另一个初始化持锁期间Qwen连懒属性查询都不能执行；释放后正常加载并验证全部BF16权重。
- 已加载Qwen遇另一个模型BF16窗口时等待CPUprocessor，processor进入时为FP32；其后的fake generate允许另线程获得共享guard，明确证明forward在锁外。
- 使用真实 `Qwen3ASRFeatureExtractor`、65,600个PCM采样点（4.1秒）FP32波形与真实HF tiny-checkpoint BF16构造窗口，两线程控制验证前端等待、Hann为FP32，得到的FP32特征和原int32 attention mask与正常FP32参考逐元素完全一致。随后只将浮点特征明确转为BF16，mask仍int32且逐元素不变。额外保留实际Whisper前端相同guard的精确回归。

最终本轮命令为 `TORCH_DEVICE_BACKEND_AUTOLOAD=0 /home/zhangs/conda-envs/asr/bin/python -m pytest tests/test_model_loading.py tests/test_local_qwen_runtime.py tests/test_qwen_cancellation_tool.py -q --disable-warnings`，结果 `33 passed in 8.34s`（其中model-loading 10项）；只使用CPU/替身和临时tiny checkpoint，无用户模型/NPU/服务。这些CPU结果不替代硬件验收，源码冷A+B重验结果见末节；早先失败记录保留。

本次同时只读核对正式单设备greedy热路径：`processing_qwen3_asr.py`无未指定dtype的torch浮点factory；ASR模型258/259/315/419行的arange/tensor输入是整数位置或长度，Qwen3的56行ones只在受guard保护的RMSNorm构造中执行，399行position arange为整数；cache/mask/rope正常浮点分配显式使用输入dtype。`generation/utils.py:2665`无dtype的浮点scalar仅在 `synced_gpus` 分布式路径，本项目没有启用。该审计仅覆盖本轮锁定栈与实际单设备greedy路径，不声称可推论所有可选解码/分布式模式永久安全。

## 冻结源代码的冷并发验收与接口边界

冻结源代码已在宿主物理 `npu:7` 完成修复后的独立全冷A+B、双Qwen实例边界回归，原始目录为 `/tmp/asr-apply-20261117-a99trpfv/e2e-cold-mixed-i2-fixed`。关键文件为 `summary.json`、`request-cold-A.json`、`request-cold-B.json`、`status-drained.json`、`ownership.json`、`shutdown-after.json`。summary、排空状态和shutdown记录均经执行者核验。此处为源码实机冷启动证据；编译stage另验，原始文件归档及SHA以交付证据清单为准。

| 冷并发项 | 源码实机核验结果 |
| --- | --- |
| 冷A请求耗时 | `27.56326445 s`，成功 |
| 冷B请求耗时 | `34.47614458 s`，成功；2段中文，角色/情绪均非空（teacher / 平淡） |
| 排空后的进程默认dtype | `torch.float32` |
| NN失败计数 | `0` |
| 主进程退出 | `0` |
| VAD自有worker退出 | 4个全部退出0，无强制结束 |
| 端口与自有进程树 | 无监听端口、无剩余children |

这次是两个冷请求和增强模型首次加载的真实交叠验收，替代前节“仍须重新验收”的待办状态；早先AutoProcessor导入失败记录仍保留，不能删去历史失败。双实例用于检验共享初始化与冷并发边界，正式生产仍保留1个Qwen实例；单次成功不构成长期稳定性、性能最优、文本准确率或全部故障注入路径均已验收的结论。本文的候选速度数据来自此前注明配置的专用工具，不把此次冷请求耗时混作稳态P50/P95。新编译交付版本须另核对源文件指纹和实机回归，不能自动继承旧stage或此前源代码版本的通过记录。

v3接口保持的依据与数值质量评估分别记录：`tests/fixtures/v3_contract.json` 冻结原版本 `79c7c1bbc94ec1824281c8e06c26e4f2a3ae5ff9` 的请求默认值、代表性响应及错误payload，并曾在该HEAD隔离源码上重放通过（5项测试、11个subtests，日志 `/tmp/asr-v3-contract-reference-ul80rptm/contract-baseline.log`）。迁移后的确定性合同测试保留8种增强组合、两位小数字符串bg/ed、字符串load_audio_time_ms/gpu_time_ms及主链时间语义、错误4008/4009/4011/4012、无人声与标点清理边界；v2仍为404，不恢复旧接口。空/省略/auto语言均为自动识别，非法language才对应4009。合同替身验证字段/流程，不能替代真实模型准确率；真实HTTP矩阵、长音频、取消和清理记录见总体验收报告。

最终选择仍是 **默认SDPA attention + 原生eager执行 + BF16 Qwen权重 + torch_npu，原生增强模型严格FP32并固定禁用HF32**。共享初始化/CPU前端guard修复首次并发正确性，不改变公开接口或把全NPU forward锁成串行。batch2的吞吐潜力、attention候选的文本变化、TorchAir两组全失败和vLLM隔离环境条件均继续保留为待评估证据，不能改写为“已证明当前后端在910B性能最佳”。


## 完整 WAV 验收前的实现核查（历史配置快照）

恢复验收后的只读核查再次确认：主`asr`包metadata仍为torch/torch-npu/torchaudio2.5.1、transformers5.13.0、numpy1.26.4、funasr1.2.6、av14.2.0、attrs26.1.0、setuptools84.0.0，与运行依赖锁吻合；没有为候选原地更换主环境。HF config的SHA仍为表头`117ac8...`，实际model_type/architecture为`qwen3_asr`/`Qwen3ASRForConditionalGeneration`；config自记5.13.0.dev0与实际安装包5.13.0分别记录，不混称。该核查仅读取文件和包metadata，未加载权重、初始化NPU或启动服务。

该次核查的`config.toml`选择`local_dtype=bfloat16`、`local_instance_count=1`、`local_device=npu:7`、segment/request=2/2；这是完整WAV验收前的短片暖标定阶段快照，后续完整八WAV实测后的正式并发默认改选1/1，依据见下节。`core/qwen_asr_runtime.py`仍以本地`Qwen3ASRForConditionalGeneration.from_pretrained(dtype=...)`执行；生产不调用`torch.compile`、不调用`set_attn_implementation`切换attention、不引入批执行/vLLM，也不依赖外部转写HTTP。默认SDPA的真实配置证据来自前述相同安装栈/资产候选的before metadata。当前runtime源码SHA256为`cd643eeec82961c9ec724aefc70cd04b6c664da23123a9967ec90c7d3f0d23bc`，懒导入/完整加载在共享guard内，CPU processor在FP32 guard内，move_inputs与generate在guard外；参数dtype/整数mask的约束与前节回归一致。

微批、TorchAir与CAM/emo BF16/FP16额外优化均未选入生产，不能把其混合长度/尾批等专项验证写为“生产已通过”。单段Qwen BF16是本轮原生转写基线，仍适用已登记的dtype、语言、raw解析、错误/取消归属和v3业务参考；它没有建立FP32 Qwen或人工转写准确率golden。未采用的优化路径按test-plan记有依据的不适用，保留失败/差异和后续准入门槛；原生路径最终全套、长音频、并发及新编译版稳态仍以总体验收报告中的实际结果为准。本文不提前完成这些待验收门槛。


## 完整八条 WAV 验收后的最终并发选择（2026-11-18）

最终选择为 **单Qwen副本、segment/request=1/1、默认SDPA attention、原生eager、Qwen BF16、CAM++/emotion2vec严格FP32**。总音频/B上限仍为8/4，CAM++/emo仍为三/一实例；Qwen逻辑闸门不等于这些模型池或整音频容量。此次只收敛同一后端的并发默认，没有采用生产微批、图编译或第二套后端。

早期暖标定每档16次timed请求，使用同一12秒片段的两轮4B+4A；当时g2单副本P50较g1低约29%，因此曾选择2/2继续验证，增加第二Qwen副本未显示足够收益。该阶段不是八条独立完整音频，不能据此承诺课堂长音频加速。原始目录位于 `/tmp/asr-apply-20261117-a99trpfv/`：

| 历史暖标定目录 | 段/全局闸门；Qwen副本 | P50 s | P95 s |
| --- | --- | ---: | ---: |
| calibration-g1-i1 | 1/1；1 | 20.02011025 | 23.56287287 |
| calibration-g2-i1 | 2/2；1 | 14.14107574 | 23.72603279 |
| calibration-g2-i2 | 2/2；2 | 14.45330876 | 22.89813554 |

暖态数据保留其实际版本范围，不追改为最终完整WAV版本的数值；双实例冷A+B是初始化/隔离边界回归，也不能充作正式双副本收益证明。

最终两个源码批次均使用同一完整源SHA256去重的八WAV清单、4B+4A、宿主物理npu:7、单Qwen与三CAM++/一emo，实际总/B活动峰值均8/4，全部8/8成功。每批新进程先完成VAD启动预热，再发完整请求；Qwen首次加载计入客户端从上传/HTTP调用到完整读取响应的耗时，不称暖态稳态时延。P50/P95采用NumPy默认linear插值，每档N=8、只一批，不将两档合并或将单段计时替代端到端。

| 完整批次 | 段/全局闸门 | 成功 | P50 s | P95 s |
| --- | --- | ---: | ---: | ---: |
| e2e-full-g1-final | 1/1 | 8/8 | 4104.338199445512 | 4781.66245310544 |
| e2e-full-g2-final | 2/2 | 8/8 | 4304.155876739882 | 5002.106922458205 |

本次g2相对g1的P50/P95分别高4.86845%/4.61021%，没有显示长音频端到端收益，因此选择更保守的1/1作为课堂默认。两个批次在共享卡上顺序执行，既有PID1601397始终保留；外负载、主机竞争与小样本不能控制，结果不证明闸门是全部差异的单一因果，也不证明1/1是所有负载下的全局最优。

两份summary保留逐请求elapsed_s、音频时长/RTF、稳定ID和完整源SHA256，可独立复算分位数。独立请求start/end时间戳与精确整批墙钟没有记录，必须明确缺失，不能从最大请求耗时伪造精确批吞吐。ownership的source_snapshot_sha256为null，不能事后把复核指纹冒充工装当时的源码快照。八条的分段数与文本字符统计在两档相同，仅能确认这些汇总一致；原始payload含计时差异，且无人工转写/角色/情绪标注或CUDA golden，不由工程成功和汇总一致推出准确率、召回率或标签正确性。

两个批次shutdown均主进程退出0、四个VAD worker全部退出0且无强制关闭、failed=false、无剩余children或监听端口；完整请求/资源/关闭证据分别在 `/tmp/asr-apply-20261117-a99trpfv/e2e-full-g1-final` 和 `e2e-full-g2-final`。这些是源码完整音频批次，最终编译stage与变更后的配置仍由总体验收报告单独登记，不能自动继承旧stage通过记录。

| 仓库证据 | SHA256 |
| --- | --- |
| `docs/npu-910b-migration-20261117/full-pair-capacity-decision.json` | `e9a5574b8f63fed06cfe084ee3ab4e53097bf7b1e84271490ca71ee11e9f61e6` |
| `docs/npu-910b-migration-20261117/full-pair-final-evidence.json` | `8300649dcbae86730981355c7e762718d5c78653877032320839dca384738256` |

上述JSON固定原始summary、shutdown、samples和ownership的SHA，第二份还登记逐请求/采样/日志文件摘要及统计口径。复算示例（只读取JSON，不运行模型）：

```bash
/home/zhangs/conda-envs/asr/bin/python - <<'PY'
import json, numpy as np
p = 'docs/npu-910b-migration-20261117/full-pair-capacity-decision.json'
d = json.load(open(p))
for gate, phase in d['phases'].items():
    elapsed = [m['elapsed_s'] for m in phase['summary']['measurements']]
    print(gate, len(elapsed), np.percentile(elapsed, [50, 95], method='linear'))
PY
```
