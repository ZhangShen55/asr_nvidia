# Proposal

## Why

12 服务器上的 `/v1.1.8/seacraft_asr` 在上一轮精简中与小语种 Whisper、文本规范化（ITN）一起被移除，与 Mac 上 `asr_offline` 的同名接口出现行为分叉：语言分流、`showRoleIdentify` 默认值、`text` 字段口径都不一致，同一份音频在两侧会得到不同结果。现在需要把两侧的 `v1.1.8` 契约对齐，只保留部署环境差异（说话人与情绪的默认值）。

## What Changes

- 将 `POST /v1.1.8/seacraft_asr` 的语言分流对齐 Mac 实现：`auto`/`zh`/`en` 走 Paraformer，`fr` 走 Whisper，其他语言统一返回 `{"msg": "不支持的语言: <lang>", "code": 4009}`。
- 语言参数在分发前统一做去空格与小写归一化（`" FR "` → `fr`）；显式提交空 `language` 与未提交 `language` 语义不同，前者按不支持语言处理，后者保持默认 `auto`。
- 恢复 Whisper 小语种转写链路：`faster-whisper` 依赖、模型懒加载、`fr` 转写函数与响应结构，与 Mac 侧保持一致；未开启 `open_mul_lang` 或模型未就绪时返回 `{"msg": "未开启小语种识别或模型未就绪", "code": 4003}`。
- 恢复 ITN 文本规范化：`text` 字段改为按分段拼接并经 WeTextProcessing 规范化，中英文口径与 Mac 一致；依赖缺失时按 `failure_mode = "passthrough"` 降级为原始文本，不影响请求成功。
- 将 `showRoleIdentify` 默认值改为 `false`，与 Mac 一致；`showSpk`、`showEmotion` 在 12 服务器继续保持 `false`（Mac 为 `true`），此项为有意保留的部署差异。
- 语速统计的内容词计数改为 Unicode 感知实现，使法语等含重音字母的文本在两侧得到一致的语速结果。
- 依赖清单补回 `faster-whisper`、`ctranslate2` 与 `WeTextProcessing`；Docker 依赖层通过 conda-forge 安装 `pynini` 以支持 ITN 规则包。
- 镜像内新增可选的 `faster-whisper-large-v3` 模型目录约定：仅当配置开启 `open_mul_lang` 时构建期强制校验该目录存在，否则不要求提供。
- `model/README.md` 与 README 同步更新，明确 `model/` 需放置的模型目录以及法语能力的开启方式。
- 新增接口契约测试，覆盖参数默认值、语言分流与 Whisper 响应结构。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `offline-asr-service`: `/v1.1.8/seacraft_asr` 的参数默认值、语言分流与错误码、Whisper 小语种转写、ITN 文本规范化与响应字段口径需要对齐 Mac 实现；同时修正“移除低频能力依赖”要求中对 Whisper 的绝对禁用表述。
- `container-model-packaging`: 模型目录集合增加可选的 `faster-whisper-large-v3`，并随开启条件校验；镜像依赖增加 Whisper 与 ITN 运行时。

## Impact

- API：`POST /v1.1.8/seacraft_asr` 的 `language` 参数新增 `fr` 支持，未知语言新增 `4009`；`showRoleIdentify` 默认值由 `true` 改为 `false`；`text` 字段改为规范化后的分段拼接结果。
- 代码：`entity/data.py`、`api/routes/asr_v18.py`、`api/routes/asr_common.py`、`core/config.py`、`core/models.py`、`utils/itn.py`（新增）、`utils/feature_utils.py`。
- 模型：新增可选 `faster-whisper-large-v3`（约 5.8 GB），不提供时法语请求返回 `4003`，其余语言不受影响。
- 依赖：新增 `faster-whisper==1.1.1`、`ctranslate2==4.4.0`、`WeTextProcessing==1.2.0`，Docker 依赖层新增 conda-forge 的 `pynini`。
- 部署：`config.toml` 与 `config.docker.toml` 增加 `whisper_model_dir`、`open_mul_lang` 与 `[itn]` 段；默认仍关闭法语能力，与 Mac 的默认配置一致。
- 文档：`README.md`、`model/README.md` 同步接口契约与模型清单。
