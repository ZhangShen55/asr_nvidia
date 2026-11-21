# Spec Delta

## ADDED Requirements

### Requirement: 连续转写明确段级异常完成状态
continuous模式 MUST 保留现有v3字段、类型、时间边界及排序，并添加processing_status（complete或completed_with_warnings）与segment_warnings列表。每条警告 MUST 含原段segment_index、bg、ed、reason、action、recovered；警告不改变其它段归属，不包含内部原始生成轨迹。legacy响应字段保持原样。

#### Scenario: 段级复读已处理
- **WHEN** 某段发生可恢复生成异常且其它段正常完成
- **THEN** 保留可恢复文本与其它段，返回completed_with_warnings及该段时间边界/原因/动作；无法恢复的段即使不进入segments仍须出现在警告中

#### Scenario: 全部正常
- **WHEN** continuous音频所有段正常处理且无警告
- **THEN** processing_status为complete且segment_warnings为空；既有语言/语速/计时口径不变

#### Scenario: legacy路径
- **WHEN** generation_mode为legacy
- **THEN** 保持既有响应字段与失败规则，不因候选段级恢复添加新字段
