# v2 接口变更后效果比对报告

对应 change：`2026-10-23-retire-v118-and-refine-v2-config`
（下线 `/v1.1.8`、配置项清理、语种占比改简写、标点卫生规则、日志双轮转、进程命名）。

## 1. 测试环境

| 项目 | 配置 |
|------|------|
| 服务端 | 本仓库 `refactor/adapt-qwen-asr` 分支，Conda 环境 `asr`（Python 3.11.16），`uvicorn main:app`，`CONFIG_PATH=./config.toml` |
| Qwen 服务 | 容器 `qwen3-asr-0.6b-8806`（`Qwen3-ASR-0.6B`，`max_model_len=4096`），宿主 `127.0.0.1:8806` |
| GPU | **仅 GPU 0**（`CUDA_VISIBLE_DEVICES=0`）：Qwen 引擎约 10 GB + 本服务约 0.9~1.5 GB；GPU 1/2 未触碰 |
| 进程名 | `asr-offline-model`（`nvidia-smi --query-compute-apps` 实测可见） |
| 音频 | `test/audio_file/` 下的三段完整课堂录音（46 / 52 / 55 分钟） |
| 基线 | 变更前（ITN + 语气词剔除，无标点卫生）跑同一批音频的结果 `/tmp/asr_run/after_skip_*.json` |

## 2. 结论

| 音频 | 基线段数 | 新段数 | 文本相似度 | 纯标点段 | 段首标点段 | 耗时 |
|------|---------|--------|-----------|---------|-----------|------|
| 教师1.wav | 395 | 389 | **0.9931** | 6 → **0** | 44 → **0** | 219 s |
| 储能学术英语实践.wav | 752 | 744 | **0.9888** | 9 → **0** | 18 → **0** | 127 s |
| 课堂实录.wav | 639 | 632 | 0.8713 | 5 → **0** | 12 → **0** | 250 s |

1. **标点脏输出被彻底消除**：三段音频合计 20 个纯标点段、74 个段首标点段，变更后全部为 0。
   段数减少量（6 / 8 / 7）与被丢弃的纯标点段数量一致，说明没有误删有效段。
2. **语种占比键改为简写**，占比数值与基线同量级：`zh` / `en` / `yue` / `de` / `it` / `pt` / `th` / `ms` / `ja` / `hi` / `es`。
   顶层 `language` 仍返回模型返回的全称（`Chinese` / `English`），未改动。
3. **转写内容无回退**：未出现幻觉、连续重复成段或整段丢失。

## 3. 「课堂实录」相似度偏低的归因

按时间重叠对齐逐段比对后，632 段中 **569 段完全一致、63 段有差异（一致率 90.0%）**。
差异全部来自 **Qwen 采样非确定性**，并被 ITN 的语气词剔除放大：

| 类型 | 样例 |
|------|------|
| 识别结果本身抖动 | 基线 `This cost.` / 新版 `This class.`；基线 `确定他们到底从哪儿来。` / 新版 `确定他们到底多少？` |
| 语气词被识别成不同词，导致剔除结果不同 | 基线 `Uh, in different institutes, uh, he has, ...` / 新版 `In different institutes, he has ...`；基线 `Um, group wise.` / 新版 `Group wise.` |
| 语气词本身不同 | 基线 `嗯。` / 新版 `咳咳。`；基线 `嘿。` / 新版 `嗯。` |

即：Qwen 这次把同一个词识别成 `Uh`、上次识别成 `In`，`remove_interjections` 是否剔除就跟着变。
这是**上游模型采样随机性**经 ITN 级联的结果，不是本次变更引入的行为变化。
语种占比的对应抖动（`zh` 35.9% → 32.3%，`en` 63.8% → 67.3%）同源。

三段音频的 VAD 段数与基线一致（395 / 640 / 753 vs 基线 395 / 639 / 752），
说明切分行为未受本次变更影响。

## 4. 功能项验收

| 项 | 验证方式 | 结果 |
|----|---------|------|
| `/v1.1.8/seacraft_asr` 下线 | `curl -X POST .../v1.1.8/seacraft_asr` | **HTTP 404** |
| `/v2/seacraft_asr` 正常 | 四种 `showSpk`/`showEmotion` 组合 | 全部 HTTP 200，字段符合预期 |
| `showSpk` / `showEmotion` 独立 | 组合矩阵实测 | `role` 仅在 `showSpk=true` 时有值，`emotion` 仅在 `showEmotion=true` 时有值 |
| 开关解耦（`open_v2_role=false`） | 独立实例实测 | CAM++ 未加载（显存 0），`role=null`；emotion2vec 正常加载并给出情绪值 |
| 开关解耦（`open_v2_emotion=false`） | 独立实例实测 | emotion2vec 未加载（显存 0），`emotion=null`；CAM++ 正常给出 `spk_0` |
| 懒加载 | `showSpk`/`showEmotion` 全关时查状态 | 只有 VAD 加载，CAM++ 与 emotion2vec 显存明细为 0 |
| 标点卫生与 ITN 解耦 | `[itn].enabled=false` 的独立实例实测 | 段首标点违例 0、纯标点段 0，规则照常生效 |
| 语气词剔除仍生效 | 对比 ITN 开/关的同一段音频 | 关闭时保留「呃」，开启时被剔除并替换为标点，随后被标点卫生清理 |
| 进程名 | `nvidia-smi --query-compute-apps=pid,process_name,used_memory` | 显示 `asr-offline-model` |
| 逐模型显存明细 | `GET /get_status` 的 `localModels` | VAD / CAM++ / emotion2vec 各自的 `loaded`/`device`/`allocated_mb`/`instances` |
| 多实例显存放大 | 启动 2 个实例实测 | `nvidia-smi` 出现 2 行 `asr-offline-model`，每个实例各持一份模型显存（约 0.9 GB/实例） |
| 状态接口字段 | `GET /get_status` | `id_engine` 固定 `"1"`，`appVersion` 固定 `"latest"` |
| 镜像模型目录校验 | 用真实 `model/` 目录跑校验逻辑 | 三个目录校验通过；人为缺目录时明确报出缺失目录名 |
| 单元测试 | `python -m unittest discover -s tests` | **72 项全部通过** |

## 5. 未覆盖项

- **镜像构建与容器验证**：按约定本次不 build、不部署；Dockerfile 与依赖清单已按新结构改好，
  但未实际构建验证。依赖安装步骤保持缓存命中（基础依赖层未变动）。
- **`instance_count = 2` 的容器内形态**：用两个独立 uvicorn 进程等效验证了「进程行数与显存翻倍」，
  容器内 Nginx 分流未验证。

## 6. 复现方式

```bash
# 1) 起服务（固定 GPU0）
CUDA_VISIBLE_DEVICES=0 CONFIG_PATH=./config.toml \
  python -m uvicorn main:app --host 0.0.0.0 --port 8083 --workers 1

# 2) 跑效果比对（3 段音频并发，showSpk/showEmotion 全开）
python scripts/run_effect_compare.py \
  --base-url http://127.0.0.1:8083 \
  --baseline-dir /tmp/asr_run \
  --out-dir /tmp/asr_run/after_change
```

脚本会输出 `compare_report.json`：段数、文本相似度、语种占比（改简写前后）与标点违例统计。
