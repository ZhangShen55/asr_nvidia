# Tasks

## 1. v3 接口迁移

- [x] 1.1 将 Qwen 路由注册、模块命名和内部入口从 /v2/seacraft_asr 迁移为 /v3/seacraft_asr，保留 multipart 参数、响应字段、A/B 准入和错误码语义；通过路由枚举与单请求 smoke test 验证 v3 可访问且 v2 返回 404。
- [x] 1.2 更新应用入口、脚本、示例、测试和运行文档中的活动 v2 URL 与符号引用，保留历史 OpenSpec 变更记录；通过 rg 审查活动代码并运行 Qwen 路由、能力错误和并发准入测试。
- [x] 1.3 为 v3 增加接口回归测试，覆盖默认 show 参数、v1.1.8 路由不受影响、v2 不注册和 v3 正常响应结构；通过 pytest 目标测试集验证。

## 2. CUDA profile 配置与 TensorRT 闸门

- [x] 2.1 新增或整理 Docker 专用配置模板，固定镜像内三个模型路径、open_v2_role/open_v2_emotion、两个 TensorRT 开关为 true、精度为 fp16，并保留 audio_concurrency 与 audio_spk_emo_concurrency；通过 TOML 解析和配置校验测试验证字段类型与路径。
- [x] 2.2 实现 CUDA Docker profile 的 fp16 engine/manifest、SHA256、GPU 架构、CUDA 和 TensorRT 兼容性启动闸门；通过缺失资产、校验失败、不兼容环境和完整资产四组测试验证失败原因及就绪状态。
- [x] 2.3 确保该 profile 的 TensorRT 初始化失败不回退 PyTorch，同时 CPU、NPU 和非 Docker 运行继续遵循既有后端选择；通过后端选择、部分实例失败和非 CUDA 设备测试验证。
- [x] 2.4 更新状态输出和启动日志，展示实际配置路径、fp16 后端、engine 校验摘要与失败原因；通过 /get_status 快照测试确认查询不触发模型加载。

## 3. Cython 构建与最小运行时

- [x] 3.1 调整 Cython 模块清单，覆盖 core、api、utils、entity 的业务模块并保留 __init__.py 与 main.py；通过构建脚本检查每个目标模块生成扩展且编译失败会终止构建。
- [x] 3.2 增加 builder/runtime 分离流程，runtime 不复制业务 .py、.c、build、缓存和开发工具，固定 venv 启动依赖并移除 Conda 假设；通过无源码 runtime 导入检查和入口启动 smoke test 验证。
- [x] 3.3 为 Cython 产物补充可诊断的版本/构建元数据，记录 Python、Cython、编译器和依赖版本；通过镜像元数据检查确认可追溯。

## 4. Docker 构建与 compose 部署文件

- [x] 4.1 在 docker/ 提供目标 CUDA Ubuntu Dockerfile 和白名单 staging 构建脚本，复制三个必需模型及 fp16 资产并排除无关模型、测试样本和缓存；通过干净 staging 清单检查确认运行时文件集合。
- [x] 4.2 实现镜像构建、资产校验和 asr_offline:v3_YYYYMMDDHHMMSS tag 生成，构建失败时不产生可发布镜像；通过 docker build（或等价 CI 构建）和 tag 格式断言验证。
- [x] 4.3 提供 docker/docker-compose.yaml，声明 NVIDIA GPU、端口、只读 CONFIG_PATH 挂载、健康检查和可覆盖的 ASR_IMAGE；通过 compose config 和无业务请求的启动检查验证语法及挂载展开。
- [x] 4.4 提供 docker/README.md，说明基础镜像、驱动/TensorRT 前置条件、构建、配置挂载、v3 探活、日志、停止和回滚；通过按 README 命令在目标 GPU 主机完成一次演练并核对文档路径。

## 5. 端到端验证与交付

- [x] 5.1 在目标 NVIDIA GPU 主机上启动新镜像，验证模型内置加载、/get_status、/v3/seacraft_asr、普通 A 请求和全增强 B 请求；保存镜像 tag、配置摘要、GPU/TensorRT 版本和启动日志。
- [x] 5.2 验证 audio_concurrency 与 audio_spk_emo_concurrency 的共享总槽位、B 槽位及能力关闭错误码在 v3 中保持既有行为；通过 3A+4B、5A+3B、8A 和关闭增强能力场景记录响应码与状态快照。
- [x] 5.3 执行 CAM++/emotion2vec TensorRT fp16 端到端质量与并发验收，覆盖八个完整音频、实际模型调用、P50/P95、资源占用和失败/拒绝记录；将结果写入中文测试报告并确认不把准入排队当作并发处理。
- [x] 5.4 检查运行镜像不含指定目录业务源码、测试文件和无关模型，且 v3 路由及健康检查可用；通过 docker image 文件清单、HTTP 回归和最终 pytest/静态检查完成发布门禁。

## Workflow follow-up

- 用户确认规划产物后，运行 $openspec-apply-change 开始实施。
- 实施和验收完成后，运行 $openspec-archive-change 归档本 change，并复核归档后的主规格。
