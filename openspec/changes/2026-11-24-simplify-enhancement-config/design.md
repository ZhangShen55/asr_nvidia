# Design

## Context

见proposal.md。当前HEAD d3f419b，已具备Qwen4/CAM4/emo1、启动预加载及异常复读折叠；根config.toml另有用户未提交vad_segment_decibel_thres=-100改动，本change不得覆盖或提交该原有改动。当前双配置源允许非空数组同时被features=false屏蔽，直接删除旧字段会改变其语义。

## Goals / Non-Goals

**Goals:** 两个设备数组唯一表达增强能力启用与实例数量；4项冗余模板配置清理；移除增强固定batch显式模板值；全链路一致的预加载、准入前拒绝、直接入口保护和旧字段迁移诊断。

**Non-Goals:** 不改prefetch、复读、retry_count、VAD阈值、N/S/B默认、权重、精度、聚类、情绪映射或错误码数值。不删除legacy执行链，不改历史归档/报告，不执行Docker构建上线，不以短clip宣称完整课程质量或吞吐提升。既有三个内置本地模型的资产预检契约保留，空数组关闭不等于镜像可省略相应资产。

## Decisions

1. **启用状态单一来源。** CAM++ enabled=bool(validated speech_campplus_devices)，emo同理；先严格校验数组类型及元素，不把字符串或非法设备误作开启。缺省缺失数组按已有缺省[]关闭。状态区分configured=0和configured>0但healthy=0，加载故障不转换成配置关闭。
2. **统一入口。** 启动预加载、路由能力校验、池enabled检查及直接speaker/emotion入口复用同一派生规则；showSpk/showEmotion/showRoleIdentify只选择执行，不开启空数组模型。两项都为空且同时请求时4018；单独请求关闭能力4015/4016；只请求另一项开启能力可正常进入B。关闭校验在整音频准入前，拒绝不占owner/活动/等待。非空数组全部加载失败仍走原4015/4016故障路径，不标成both_disabled。
3. **旧键明确拒绝。** 删除features开关的读取/Settings入口，配置加载时在VAD/Qwen/CAM/emo出生前检查残留。旧开关true→删除键、保持数组；false→删除键并清空相应数组。固定batch_size不再在正式模板展示，仍保留实际batch1与既有非1值拒绝约束。选择拒绝而不是忽略或双轨兼容，防止旧false无声变开启，并通过模板和文档去除“增强可调batch”错觉。
4. **legacy分支隔离。** 两项Qwen组批参数仅从根/Docker等当前continuous模板去掉；Settings/legacy scheduler显式参数和原范围校验保留。活跃可执行样例中的被删除增强旧键需要迁移；归档中的参数只作为历史证据，不重写。两个固定增强batch的既有1值约束保留，actual_batch_size=1状态不变；四项本轮仅清理当前模板显式值。
5. **现有改动归属。** apply前复核用户VAD差异；本change编辑同一个config.toml时保持-100原值，提交使用定向暂存只纳入本次6键清理；不得提交用户VAD差异、stash/reset覆盖它。规划阶段仅提交change目录。
6. **验证边界。** 先CPU派生/迁移/路由/直接入口/预加载和预取/复读回归，再隔离完整CPU。实机只用NPU7启动当前正式配置，首次音频前GET核CAM4/emo1已加载；短真实B请求核两增强forward与复用，再纯A核增强调用不增。默认预加载不等于forward暖身；采样、加载和请求计时分离。

## Risks / Trade-offs

- 旧false加非空数组被误开 → 旧键启动前拒绝，并给逐字段迁移说明，测试worker/model创建次数为0。
- 空数组被运行入口或请求参数绕过 → 四种数组组合、showSpk/Emotion/RoleIdentify和直接入口均覆盖；复用同一派生状态。
- 模型全部加载失败误判配置关闭 → 按配置容量与健康容量分别测试，保留4015/4016与部分实例降级。
- 旧样例残留导致测试误判 → 更新当前可执行样例和fixture，历史证据原样保留并说明需迁移。
- 参数清理被误当性能优化 → 不改执行并发/预取/生成行为，不承诺200秒或质量改善。

## Multi-agent Work Split

- Agent A：core/config.py派生/迁移及配置契约测试；只写配置层测试与实现。
- Agent B：core/models.py、相关路由/直接入口和预加载/能力测试；配置派生接口先由主agent冻结，避免同文件并改。
- Agent C：根/Docker及活跃样例、README/AGENT/部署说明和旧字段审计；不动历史归档，不提交用户VAD差异。
- 主agent：审查各产物、完整CPU、NPU7唯一占用、owned服务生命周期、提交推送与规格同步归档。测试/NPU不得与子agent争抢，所有shared服务保留。

## Migration Plan

发布前先迁移残留增强旧键；保留prefetch与复读等有效配置。源码重启生效；镜像/保护构建后续纳入。若回退实现，必须同时回退对应配置格式并保留用户VAD改动，避免旧版本在缺少功能开关时默认启用不符合预期。验收后按项目流程提交推送、同步四能力并独立归档；旧未完成change不因此完成。
