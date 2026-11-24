# N5_B4_vs_N4_B4

仅比较summary中status=completed的完整响应与声明源SHA。模型间相对一致/不一致不是人工准确率、召回率或WER/CER；censored、缺失和无法精确对齐者为unknown。性能是否通过由原summary独立判断。

原N5/CAM1/emo1补测B4对最终N4/CAM4/emo1 B4，均为同a1～a4双增强且showRoleIdentify=false。两轮均为B，Qwen与CAM实例数/运行状态不同；角色ID为各请求局部聚类标签，标签差异不能直接解释成人物识别错误或准确率变化。

匹配边界文本变化单列在JSON的 `text_change_details`；HTML勾选“仅匹配边界的文本变化”可独立查阅。

| 音频 | 状态 | 文本变化 | 差异边界 | role / emotion变化 | 左 / 右独有边界 | 说明 |
|---|---|---:|---:|---:|---:|---|
| a1 | comparable | 23 | 23 | 0 / 0 | 0 / 0 | 完整同源配对，SHA一致 |
| a2 | comparable | 61 | 61 | 0 / 0 | 0 / 0 | 完整同源配对，SHA一致 |
| a3 | comparable | 30 | 34 | 2 / 0 | 0 / 2 | 完整同源配对，SHA一致 |
| a4 | comparable | 60 | 86 | 23 / 0 | 2 / 2 | 完整同源配对，SHA一致 |

完整实际内容见 [N5_B4_vs_N4_B4.html](N5_B4_vs_N4_B4.html) 与 JSON；HTML可搜索并展开完整字段。

文本、角色/情绪字段与警告变化分别保存；同A/同B相对差异同样不等于准确率变化，无人工GT不认证准确率或召回。

警告明细见JSON的 `warning_changes`，增强字段明细见 `enhancement_field_changes`；原始警告数组始终保留。
