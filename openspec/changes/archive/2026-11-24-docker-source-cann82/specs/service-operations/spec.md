# Spec Delta

## MODIFIED Requirements

### Requirement: 明确实例数量的语义
`instance_count` MUST 继续表示完整 HTTP 服务进程数，缺省1，不表示 Qwen 数。本版无代理源码容器 MUST 仅支持 HTTP1，`instance_count != 1` 在创建模型/worker 前明确拒绝。单 HTTP 内 Qwen 数由 local_instance_count 配置，默认4，合法1～8；增加模型数不复制 HTTP/VAD/增强池。一般显式多 HTTP 架构仍按各服务独立资源评估，不属于本版容器支持范围。

#### Scenario: 配置多个实例
- **WHEN** 本版容器配置 instance_count 大于1
- **THEN** 启动明确说明只支持单 HTTP，并提示调整 Qwen 模型数量所用字段；不静默忽略、不启动 Nginx，也不在本版中复制多套服务

#### Scenario: 配置缺省
- **WHEN** 未声明 instance_count
- **THEN** 一个 HTTP 服务内默认运行四个 Qwen owner，不产生四套 HTTP/VAD/增强服务

### Requirement: 提供可复现的 v3 Docker 部署入口
`docker/` MUST 恰好提供 Dockerfile、docker-compose.yml、config.toml、README.md。README MUST 说明依赖准备、构建/重试、启动、健康、真实 v3 请求、停止、配置挂载、NPU映射和模型策略。入口 MUST 直接启动一个 HTTP 服务，不使用 Nginx 或旧项目启动/构建脚本；Compose MUST 声明 NPU runtime、特权、IPC、端口、必需只读挂载及有界正常关闭策略。

#### Scenario: 查看部署目录
- **WHEN** 运维仅查看 docker 目录
- **THEN** 能从四个文件获得完整可执行命令及前置条件，构建材料及报告路径在目录外明确说明

#### Scenario: compose 启动服务
- **WHEN** 使用 docker-compose.yml、必需宿主文件及昇腾容器 runtime
- **THEN** 单 HTTP 直接提供 v3 和健康查询，预加载完成后才就绪，配置挂载只读；健康等待覆盖实际预加载耗时，停止给设备清理预留充足时间，不能用端口可通代替模型健康

### Requirement: 镜像标签符合 v3 时间格式
本版部署 MUST 使用 `asr_offline_910b:v3_YYYYMMDD`，日期为实际发布日期，首版示例 `asr_offline_910b:v3_20261125`。Compose 默认及覆盖方法 MUST 与构建一致；交付 MUST 记录已验收的不可变 image ID、源码提交及配置摘要，最终日期标签指向同一已测 image ID，不在验收后重建或误覆盖已有不同版本。

#### Scenario: 构建并启动时间标签镜像
- **WHEN** 完成候选构建及 B4 验收
- **THEN** 最终日期标签引用已测 image ID，Compose 可按该标签启动，报告能核对源码、配置与镜像身份

#### Scenario: 标签格式错误
- **WHEN** 未覆盖默认值且标签不符合指定仓库名及八位日期
- **THEN** 构建或部署说明明确指出错误，不继续使用旧仓库名或时间格式作为本版默认

### Requirement: 交付文件准备与部署执行分阶段跟踪
交付 MUST 分别记录规划、文件准备、依赖/资产检查、镜像构建、真实 NPU 前向、完整音频验收、正常关闭及构建资源清理。仅规划或文件存在 MUST NOT 被称为已构建/部署。实际构建和容器验证属于本 change 后续 apply，生产常驻上线及人工质量认证另行区分，不以 B4 处理完成代替。

#### Scenario: 规划或交付文件完成
- **WHEN** proposal 或 Docker 交付文件已完成但尚未实际构建
- **THEN** 仅报告对应准备阶段完成，不勾选镜像、推理或四音频验收任务

## ADDED Requirements

### Requirement: 构建资源归属和缓存清理可核验
构建 MUST 记录本轮拥有的中间镜像、构建缓存、辅助容器/镜像及临时资源，失败期间保留缓存并定位原因重试。B4、镜像内容及正常关闭均通过后 MUST 清理本轮全部可独立移除的中间镜像和缓存；基础/最终镜像及其必需共享层、原有镜像和服务、日志报告 MUST 保留。禁止全局 prune 或依赖模糊名称归属删除。

#### Scenario: 构建失败后重试
- **WHEN** 构建步骤失败
- **THEN** 保存实际构建进程状态及完整错误，修正根因后复用同一任务缓存；不因 SSH 断开误判失败，不在重试前删除缓存，也不同时启动竞争构建

#### Scenario: 成功验收后清理
- **WHEN** 同一候选 image ID 完成全部验收且测试资源正常退出
- **THEN** 按事先登记清单及引用检查逐项清理本轮缓存和临时构建资源，记录剩余资源；最终和基础镜像身份不变，共享服务保持，用户原有镜像不被删除

#### Scenario: 验收失败或归属不明
- **WHEN** 完整音频、正常退出或内容审计失败，或待删资源归属/引用无法确认
- **THEN** 保留失败证据及可复用缓存，继续诊断；归属不明项不删除，不宣称镜像可用或清理完成

### Requirement: 本版源码镜像必须完成真实B4验收
候选镜像 MUST 经真实 HTTP 接口同时处理四条独立完整 WAV，启用 showSpk/showEmotion，全部 Qwen owner 在正式计时前完成真实前向覆盖。报告 MUST 给出逐文件状态、完成率与严格无警告率、客户端 P50/P95、墙钟、HBM、增强真实调用/skip 和段级异常；仅导入、短片、离线段测试或 HTTP200 不能替代验收。结束后 MUST 正常回收本轮服务及 worker。

#### Scenario: 四完整音频完成
- **WHEN** 四条已校验 SHA256 的 canonical a1～a4 同时从宿主向候选容器 v3 接口发送双增强请求
- **THEN** 保存四份完整 JSON、逐请求耗时及配置/镜像证据，核对四份 Qwen 和 CAM4/emo1 实际工作，分开统计警告与不可恢复段；4/4完整处理不等于4/4严格无警告或人工准确率达标

#### Scenario: 出现慢请求或真实故障
- **WHEN** 任一完整请求超过336秒，或出现真实 OOM、设备/服务故障
- **THEN** 停止该轮并有界取证及正常收尾，列出未完成/失败，不只统计成功快请求；修复原因后可重验，保留原轮证据，B4不设200秒硬门槛

#### Scenario: 验收结束释放服务
- **WHEN** 测试结束或被中止
- **THEN** 按拥有的容器及 PID/出生身份核对正常退出、无孤儿、私有音频缓存为空和测试端口连续空闲至少5秒，保留共享服务；强制关闭或非零退出不计正常收尾通过
