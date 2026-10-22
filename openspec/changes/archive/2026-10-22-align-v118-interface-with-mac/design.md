# Design

## Context

服务器仓库与 Mac 仓库 `asr_offline` 是同一套离线 ASR 服务的两个部署实例。上一轮精简在服务器侧删除了 `/v1.1.7`、Whisper、Pyannote、BERT 与 ITN，Mac 侧则保留了 `v1.1.8` 的法语 Whisper 分流与 WeTextProcessing 规范化。逐文件比对两侧 `api/routes/asr_common.py`、`core/config.py`、`core/models.py` 与 `utils/feature_utils.py` 后确认，服务器侧缺失的能力共五项：语言分流与 `4009` 语义、Whisper 转写函数、ITN 规范化、`text` 字段口径、Unicode 语速计数。

约束：模型不进 Git 但要随镜像交付；镜像基于 CentOS 7 + glibc 2.17，Python 3.10 conda 环境；`asr` conda 环境已有 `faster-whisper 1.1.1`、`ctranslate2 4.4.0`、`pynini 2.1.7`、`WeTextProcessing 1.2.0`。

## Goals / Non-Goals

**Goals:**

- 两侧 `v1.1.8` 的请求契约、语言分流、错误码与响应字段完全一致。
- 保持一项有意的部署差异：Mac 的 `showSpk`、`showEmotion` 默认 `true`，12 服务器默认 `false`。
- 法语能力与 ITN 均为可开关能力，缺失模型或依赖时降级而不影响中英文转写。
- 模型随镜像交付，构建期能对必需模型给出明确失败信息。

**Non-Goals:**

- 不对齐 Mac 的算子注册、GPU 租约与日志体系（`packages.operator_registry_client` 等调度侧设施不属于本接口契约）。
- 不恢复 Pyannote 说话人分离与 BERT 五何分类。
- 不改动临时文件命名策略等与响应契约无关的健壮性改进（如 Mac 侧的安全随机临时文件名）。

## Decisions

### 1. 语言分流放在路由层，并在读取音频前完成校验

在 `api/routes/asr_v18.py` 内按 `PARAFORMER_LANGUAGES = {"auto", "zh", "en"}`、`WHISPER_LANGUAGES = {"fr"}` 分流，未命中语言直接返回 `4009`。

选择理由：Mac 侧已是该结构，直接对齐可保证错误码、错误文案与调用顺序（先校验语言、再读音频）完全一致；把校验前置还能避免无效请求占用 GPU 队列与临时文件。

备选方案：在 `prepare_asr_context` 内统一分流。放弃原因是语言错误将晚于音频读取发生，与 Mac 行为不一致。

### 2. 显式空语言通过表单检查区分，而非依赖框架默认行为

`get_asr_params` 增加 `Request` 依赖并 `await request.form()`，仅当表单确实包含 `language` 字段时采用其原始值，从而把“未提交”（`auto`）与“显式提交空串”（不支持语言）与 Mac 区分开。

选择理由：Mac 通过该方式固化了这一语义（提交 `9d8d89a`），沿用同一实现可避免两侧在空串场景下分叉；显式检查也不依赖框架版本对空表单值的默认处理。

### 3. Whisper 作为可开关能力懒加载

`core/models.py` 增加 `_model_whisper` 单例与 `get_whisper_model()`，仅当 `settings.open_mul_lang` 为真时加载；`compute_type` 固定 `int8`，设备与索引由 `settings.device`（如 `cuda:1`）解析，与 Mac 固定 int8 的决策一致。

选择理由：`open_mul_lang` 默认 `false`（与 Mac 的 `config.toml` 一致），保证未提供 Whisper 模型的部署仍能正常启动并处理中英文；固定 int8 避免为不同部署引入新的可调参数。

备选方案：把 `compute_type` 暴露为配置项。放弃原因是会引入两侧不一致的配置面，且 int8 已是两侧既定取值。

### 4. ITN 与 `text` 口径以 Mac 实现为准

新增 `utils/itn.py`（与 Mac 同源），在分段组装后统一调用；`text` 字段改为规范化后的分段文本按空格拼接。`zh`/`en` 走规则包，`fr` 等其他语言跳过规范化；依赖缺失或初始化失败按 `failure_mode = "passthrough"` 返回原始文本。

选择理由：Mac 的 `text` 已是规范化拼接结果，服务器保留原始拼接会让同一音频在两侧得到不同文本；透传策略保证依赖缺失时请求仍然成功。

### 5. 语速内容计数改为 Unicode 感知实现

移植 Mac 的 `count_content_words`：以 Unicode 类别判定字母与数字，连字符与引号仅在词内保留。

选择理由：服务器现实现用 `[a-zA-Z']` 匹配，带重音的法语词汇会被计为 0 词，导致法语语速恒为 0，而 Mac 侧会计入实际词数。

### 6. 镜像依赖与模型校验按能力开关条件化

`requirements-pip.txt` 增加 `faster-whisper==1.1.1`、`ctranslate2==4.4.0`、`WeTextProcessing==1.2.0`；`pynini` 与 `av` 一样通过 conda-forge 安装，避免源码编译。Dockerfile 把 `COPY config.docker.toml /config.toml` 提前到模型校验之前，据其 `open_mul_lang` 取值决定是否强制校验 `faster-whisper-large-v3` 目录。

选择理由：默认配置不开启法语，构建上下文无需携带 5.8 GB 权重；一旦开启，缺失模型必须在构建期暴露，而不是运行期才发现。

### 7. 新增仓库内接口契约测试

在 `tests/` 下以 `unittest` 编写参数默认值、语言分流与 Whisper 响应结构测试，通过 `python -m unittest discover -s tests` 执行，不引入新的测试框架依赖。

选择理由：服务器仓库当前没有测试目录，Mac 侧同类测试使用 `unittest`，沿用同一风格可让两侧测试用例可直接对照。

## Risks / Trade-offs

- [Whisper 依赖树在 CentOS 7 上缺少预编译 wheel] → 构建阶段实际验证；`faster_whisper` 改为按需导入，未开启能力时不影响服务启动。
- [`pynini` 通过 conda-forge 安装增加镜像体积与求解时间] → 与既有 `av` 安装方式一致；ITN 依赖缺失时按透传策略降级。
- [`text` 字段由原始拼接改为规范化拼接，影响既有下游比对] → 在 README 中记录该口径变化，并保留 `segments[].segment_text` 的原始粒度对照。
- [Whisper 模型 5.8 GB 显著增大镜像] → 仅在开启法语能力时要求提供，默认构建不受影响。
- [Mac 的算子调度设施未同步] → 明确列入 Non-Goals，避免接口对齐被误读为整体运行时对齐。

## Migration Plan

1. 合并代码变更，默认配置 `open_mul_lang = false`，现网部署行为与变更前一致（中英文转写，`text` 字段口径变为规范化拼接）。
2. 需要法语能力时，将 `faster-whisper-large-v3` 放入 `model/`，并把 `config.docker.toml` 的 `open_mul_lang` 置为 `true`，重新构建镜像。
3. 回滚方式：将 `open_mul_lang` 置回 `false` 即可关闭法语链路，其余能力不受影响；如需完全回滚，按 `git revert` 撤销本次提交。

## Open Questions

- 无。若后续决定将默认 `open_mul_lang` 改为 `true`，只需调整配置与模型清单，不改变本次接口契约。
