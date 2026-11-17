# 910B 迁移补充审计与实施边界

日期：2026-11-17。

本文件是既有 change `2026-11-17-npu-910b-local-qwen-asr` 的补充审计产物，用于修订后续 design、tasks 与 spec delta。记录依据为本次已经完成的远程只读检查及官方资料核查。本文与 proposal、design、tasks 和 change 内 spec delta 一并同步；`openspec/specs/` 基线在验收归档前保持不变。本轮未实施业务代码、依赖安装、模型推理、镜像构建或上线。

## 1. 已确认的开发边界与证据等级

- 远程项目：`xjtu910b7:/home/zhangs/asr_nvidia`；检查时分支为 `refactor/adapt-qwen-asr-2huawei910b`。
- 本轮开发限定宿主设备 `npu:7`，不调度其他卡，不停止已有服务。服务器为 aarch64、8 张 910B3、每卡 64 GiB HBM；检查时 7 号卡已有 uvicorn 进程，不能视为独占空闲卡。
- 环境使用 conda `asr`，解释器 `/home/zhangs/conda-envs/asr/bin/python`。后续仅映射物理 7 号卡的容器，应核对逻辑编号，不能直接照搬宿主 `npu:7`。
- 当前远程业务代码仍为外部 Qwen HTTP 服务与 CUDA/TensorRT 主链路；已有 `TorchNpuRunner` 只是迁移基础，不代表完整链路可用。
- 资产状态更新（2026-11-17）：用户已补充 `model/emotion2vec_plus_large`、`model/speech_fsmn_vad_zh-cn-16k-common-pytorch`、`model/speech_campplus_sv_zh_en_16k-common_advanced`。此前“model/ 仅见 README”是首次检查的历史状态，不再作为当前缺失结论。已通过 SSH 现场核验三个目录及关键权重/配置文件存在且非空：emo model.pt 1,945,790,254 字节，VAD model.pt 1,721,366 字节，CAM++ 权重 28,044,640 字节，emo tokens.txt 108 字节；tokens.txt 已读到 9 个标签、unuse 索引为 [1, 6]，与预期标签结构一致；checkpoint 匹配和 NPU 加载仍待验收；Qwen 权重另行核对。用户确认文件已补充不等于加载或前向验证通过。
- 已核查音频样本为 52 个 WAV、2 个 MP3，合计 5,144,064,354 字节；其中 52 个 WAV 的文件头均为 16 kHz/单声道/16 位，WAV 合计 43.6597 小时，单文件 2422.4～3308.672 秒。存在 GBK 编码目录名。该清单不等于已成功解码或完成全部转写。样本清点和可读性测试须保留原始路径字节，避免为方便而批量重命名。
- 安装元数据、依赖一致性检查、导入成功、NPU 单模型前向、v3 端到端回归、稳定性能验收分别属于不同证据等级。任何前一级通过均不能替代后一级。

## 2. Qwen 后端选型修订

### 2.1 正确性基线和生产路线分开验收

Transformers + torch_npu eager 是可验证迁移基线，不预先认定为最佳性能方案。候选包括原生 eager、目标运行时支持的图编译、保持段独立性的微批处理，以及锁定历史版本的 vLLM-Ascend。以质量、稳态吞吐、P50/P95、资源占用和维护成本共同决定生产资格。

已有设计记录的 4.1 秒音频推理 13.11 秒、约 3.80 GiB 分配属于历史单段验证记录，本轮未重跑。它不证明完整服务可用，也不代表经过 warmup、批处理优化后的性能上限。

### 2.2 修正排除 vLLM 的论证

- 官方当前 main 明确 Qwen3-ASR 自 v0.30.0 停止维护；但历史 v0.23.0 教程同时给出单张 910B 64GB BF16 和 A2 镜像部署。因此应表述为旧版可验证候选与维护风险，不宜写成完全不存在 NPU 部署方案。
- 当前 vLLM 源码已提供 HF 结构的 `model.audio_tower`、`model.language_model` 和多模态 projector 映射。旧版加载失败只能证明该特定组合需要适配，不能证明整个路线不可行。不能据此反推最新 Ascend 版本已支持。
- `/audio/transcriptions` 后处理剥离前缀并非所有 vLLM 调用形式的必然限制。官方 SDK 的 vLLM 分支直接读取 `LLM.generate()` 原始输出，可作为语言前缀兼容的验证方向。
- vLLM 候选必须在独立环境或镜像验证，不能原地升级、降级或污染 conda `asr`。锁定 vLLM、vLLM-Ascend、torch、torch-npu、CANN、驱动、模型 revision 与补丁，并记录后续维护责任。

### 2.3 必须明确的语种和并发契约

最新官方 HF 模型卡描述的 `language=` 使用 assistant 预填前缀强制语种；切除 prompt 后，生成部分可能仅含正文。但本机实际安装的 transformers 5.13.0 `processing_qwen3_asr.py:495–540` 把 language 放在 system hint 中，行为与 latest 文档不同。必须锁定实际处理器源码和模型 revision，分别测试 auto、具体语种和非法语种，不能仅凭 latest 文档认定本机有前缀缺失问题，也不能把人为预填语言当作模型独立识别结果。内部适配器若补回 raw prefix，应记录来源。保持 v3 对外字段不变，以现有业务实现和测试冻结语言契约后再实施。

保留 `QwenAsrClient.transcribe_bytes()` 与错误类型契约，替换内部传输层；`base_url` 可按现有规划作为忽略并告警的迁移字段，不默认维持两套生产后端。移除 httpx 前仍需检查保留脚本和测试用途。

新增懒加载初始化锁、实例所有权保护、设备同步和取消收尾。Python 超时不等于底层 NPU 工作已停止，不能提前归还仍忙实例。多 uvicorn 进程会复制 Qwen 权重，必须纳入容量计算。

若评估微批，按时长或输入长度分桶，配置批大小和等待上限；每段独立输入、独立结果及 VAD 时间轴，不拼接相邻段。padding 只能在有效 mask 与数值回归验证后使用；涉及原 spec 的一次单段调用要求时应显式修订。编译路径需覆盖动态长度、首次编译和重编译成本，不能把通用 torch.compile 示例当作当前 CANN 组合成功证据。

## 3. CAM++、emotion2vec 和 VAD

- CAM++、emotion2vec 的首选迁移基线仍是原始 checkpoint + torch_npu，不预设 ONNX/OM 转换。保留权重，不覆盖或删除 `model/` 资产。
- CAM++ 保留 80 维按段去均值 fbank 和 192 维 embedding；适配 `seg_pooling` 时保持边界、尾段和数值语义，不能仅验证维度。基准样本、余弦与数值容差、0.3 阈值聚类一致性需事先明确。
- emotion2vec 保留最终分类头及有效 7 类映射，校验 9 类中 unuse 位于 1、6；1～10 秒外按现有“平淡”规则，不能用中间特征替代分类输出。
- 原生 FP32 作为数值参考；混合精度、编译、批处理和格式转换属于测量后选择，不预先宣称哪个最快。格式转换若成为候选，必须补导出、动态输入、算子兼容、精度、交付和重建验证。
- `core/vad_pool.py` 实际为固定独立 worker 进程池，并非 VAD 与两个增强模型同一进程。相关 spec delta 应修正，除非另立有依据的架构变更；启动预热、FIFO、数组长度决定 worker 数的语义继续保留。
- `core/config.py:385–399` 的 VAD 设备校验仍仅 CPU/CUDA，`core/vad_pool.py:64` 仍交给 FunASR AutoModel，`:100` 内存统计仅 CUDA。NPU VAD 需要专门验证加载与资源统计，不能只把配置字符串换成 npu。
- 小模型设备配比由实测决定；本轮保持 VAD 切分、低音量/SNR/RMS 阈值与过滤窗口业务语义，不借迁移擅自调参。

## 4. 修改、保留和删除清单

| 范围 | 应修改／补充 | 应保留 |
| --- | --- | --- |
| Qwen | `core/qwen_asr_client.py` 本地加载与推理、实例调度、生命周期、错误映射 | 类名、调用方法、语种规范化、4011/4012/4009 契约；失败不能当4008 |
| 本地模型 | `core/local_inference.py`、`core/models.py` NPU执行、算子、同步、状态与HBM统计 | 独立池、按设备数组建实例、能力开关、健康容量和取消收尾 |
| 配置 | `core/config.py`、`config.toml`、`docker/config.toml`、README同步新增模型目录/设备与NPU校验 | 现有v3业务默认值及阈值；并发参数改动必须有报告 |
| 启动 | `main.py`、`core/deployment.py` 去CUDA/TRT预检，补NPU适用检查 | 就绪前校验、VAD预热、明确失败、退出收尾 |
| 状态诊断 | `core/status.py`、`core/resource_diagnostics.py`及内存采样改NPU口径 | 既有监控键与准入语义；未测资源不能报已测为零 |
| HTTP业务 | 路由内部适配调用及相关测试替身 | `/v3/seacraft_asr` 参数、字段、类型、顺序、分段与时间轴 |
| 文本处理 | 仅在必要兼容点适配，增加回归 | ITN、复读、标点、语速、语言统计与有效发言保护 |
| Cython | 构建版本、NPU元数据、aarch64/Python ABI验证 | 源码保护编译、扩展完整性检查、运行层导入 smoke test |
| 文档 | AGENT、README、部署说明、带日期可复算报告 | 18个历史归档change、决策追溯与现有报告证据 |

拟删除范围须逐项核对引用和用户授权后执行，本轮没有删除：TensorRT运行器和资产校验路径、四个TensorRT配置、NVIDIA profile Dockerfile与engine打包逻辑、`requirements-tensorrt*.txt`、仅用于TensorRT的导出/验证/压测脚本和测试断言。非NVIDIA专属业务测试不得因名字含v2或CUDA就整份删除；应保留并替换后端替身。历史报告先列清单、确认结论承接，不能用全局关键词删除。

## 5. 依赖分层与真实安装版本

下表来自 conda `asr` 的 importlib.metadata 查询，无模型或NPU初始化：

| 包 | 已安装版本 |
| --- | --- |
| torch | 2.5.1 |
| torch-npu | 2.5.1 |
| torchaudio | 2.5.1 |
| transformers | 5.13.0 |
| numpy | 1.26.4 |
| funasr | 1.2.6 |
| av | 14.2.0 |
| torchvision | 0.20.1 |
| attrs | 26.1.0 |
| WeTextProcessing | 1.2.0 |
| pynini | 2.1.7 |
| scipy | 1.15.3 |
| scikit-learn | 1.7.2 |

推荐分通用业务依赖、CPU/模型公共依赖、NPU运行依赖、构建测试工具和独立后端候选五层；锁定有效组合、安装来源与平台，不把整环境所有包不加判断地写进运行清单。

已核实的清单问题：

- `requirements.txt:3` 的 av14.4.0 与实际14.2.0不一致；`:12–13` torch范围过宽且torchaudio未锁；缺少本地Qwen/NPU必要声明。
- `requirements-pip.txt:11–12` 同样需锁配套torch/torchaudio；`:1` 已说明av由conda提供，应与重建脚本一致。
- `requirements-pip-runtime.txt:2` 固定httpx用于旧Qwen；核心目录搜索只在 `core/qwen_asr_client.py:11` 直接导入，但还需核对测试和脚本再删包。
- requests 被多个压测、阈值扫描和诊断脚本使用，不能因转写不走HTTP而全删。
- torchaudio在 `core/local_inference.py:53` 生成fbank，在 `utils/audio_utils.py:24、120` 也使用，必须保留匹配版本。
- av、torchvision未在所查核心代码中发现直接导入，不代表没有传递依赖，不能据此直接移除。
- ITN所需WeTextProcessing/pynini及其安装来源需保留；NPU迁移与删除ITN无关。

## 6. Docker 与 Cython 交付修正

- `docker/Dockerfile:2` 是CUDA底座；`:16–25`安装cu124 torch2.6/torchaudio/TensorRT；`:29–31`校验engine；`:45`设置CUDA profile，均需改为NPU环境及适用预检。
- `docker/build.sh:7–21`为NVIDIA profiles；`:51、55`复制TRT检查和依赖；`:80–113`强制fp16 engine。移除这些逻辑后，仍应明确检查三个本地模型原始权重与配置文件。
- `docker/start.sh` 是容器内entrypoint。`:5、10`的CUDA变量应清理，`:15–16`解释器需匹配目标环境；`:63–67`启动多个uvicorn导致模型复制，须核算。`--privileged --runtime=ascend --ipc=host`、物理卡可见性应在compose或外部容器启动入口配置，不能加在容器内脚本中冒充设备绑定。
- `setup_cython.py:14、32`自动覆盖core/api/utils/entity，新模块可自动纳入；`scripts/build_cython.sh:8`构建工具未锁版本，`:44`元数据仍记TensorRT，应更新为NPU及Qwen依赖。保留构建后无源码环境导入检查。
- 交付文件准备纳入迁移任务；依赖安装及环境验证属于后续实施阶段的先行工作，实际镜像构建与上线安排在服务稳定后。不能把文件改好描述为已安装或已部署。

## 7. 验收与任务修订清单

- [ ] 核验用户已补充的三个模型目录及关键文件，补充资产来源/revision；另行核对 Qwen 权重，建立音频 manifest，验证原始编码路径可读、WAV/MP3可解码，不改原始样本。
- [ ] 修正VAD同进程表述、delta中v2接口残留、16/128默认值与实测标定要求的冲突；当前config实际为15/120，不能混用文档快照。
- [ ] 修正清理任务“openspec list为空”：当前change归档前应保留；只排查其他明确废弃的未完成change。
- [ ] 固化环境版本与来源并分级验证；安装metadata、依赖检查、导入、单模型前向、业务回归分别记录。
- [ ] 保留 `/v3/seacraft_asr` 契约测试：字段/类型/默认值、分段顺序与边界、能力开关、A/B准入、排队、错误码、文本处理与语种路径。
- [ ] 保留 `gpu_time_ms` 原字段名、原计时边界及统计语义，不因设备改成NPU改名或重新解释。先审计实际计时实现，再加契约断言；新内部设备/编译/排队遥测单独记录。
- [ ] 以实际同步完成为准验证取消、重复取消、超时与关机，禁止提前归还实例或音频租约。
- [ ] 将HTTP、TensorRT测试替身改为本地NPU接口替身，保留有效业务断言；纯桩测试和真实硬件测试分别报告。
- [ ] 对Qwen eager、可用编译、微批及隔离vLLM候选进行同条件比较；不能仅依赖4.1秒样本或单次冷推理。
- [ ] CAM++与emotion2vec完成前述数值及完整业务回归后，方可选择精度、并发档位与后端。
- [ ] 真实课堂端到端及八路并发报告记录样本SHA256、生成策略、warm/cold、有效音频时长、RTF、逐请求测量、P50/P95、活动B峰值、HBM及已有uvicorn干扰；验收不等于仅看AICore瞬时利用率。
- [ ] 依赖/脚本/文档清理逐项交叉检查引用；模型权重、已有服务、镜像和历史证据不得顺手删除。
- [ ] 验证完成后再勾选任务、同步spec delta并归档；本补充文件不代替实施结果或验收报告。

## 8. 官方资料

- [vLLM-Ascend 历史v0.23.0 Qwen3-ASR 910B部署教程](https://docs.vllm.ai/projects/ascend/en/v0.23.0/tutorials/models/Qwen3-ASR-1.7B.html)：单张910B、BF16、A2候选镜像；并非本机已验证成功。
- [vLLM-Ascend 当前维护状态](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Qwen3-ASR-1.7B.html)：v0.30.0起停止维护，旧版候选需承担维护成本。
- [vLLM 当前Qwen3-ASR实现](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen3_asr/)：已含HF权重mapper；不能据此宣称当前Ascend可直接使用。
- [Qwen官方HF模型卡](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)：原生Transformers、强制语种、batch与compile示例，需在目标NPU组合实测。
- [Qwen官方推理SDK](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py)：vLLM.generate原始输出与forced-language解析路径。
