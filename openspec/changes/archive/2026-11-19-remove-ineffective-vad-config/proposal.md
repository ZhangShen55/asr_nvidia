# Proposal
## Why
当前VAD配置包含不会覆盖真实FSMN选项的nested声明，用户容易误以为调参有效，要求删除死配置和读取代码。
## What Changes
- 删除四项无效VAD键及Settings属性和默认nested转发，清理当前配置模板。
- 保留有效段级SNR/能量过滤，保留模型原始构造选项和所有资产。
- 根配置中误放在vad表的复读参数移动到qwen_asr，不改变有效值。
## Capabilities
### New Capabilities
无。
### Modified Capabilities
无；本次仅删除无效声明及冗余转发，实际推理/接口行为不变，skip_specs=true。
## Impact
core/config.py、core/vad_pool.py、当前TOML模板、相关测试与文档。需下次启动读取新配置，不重启当前服务；不改依赖或模型格式。
