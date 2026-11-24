# N4_A4_vs_N4_A4_resident

仅比较summary中status=completed的完整响应与声明源SHA。模型间相对一致/不一致不是人工准确率、召回率或WER/CER；censored、缺失和无法精确对齐者为unknown。性能是否通过由原summary独立判断。

最终N4/CAM4/emo1：同a1～a4纯A4对B4完成后全部权重继续驻留的纯A4。两轮均不请求增强；对照驻留状态下的具体输出波动，不套用旧A6实验说明，也不把相对一致当质量认证。

匹配边界文本变化单列在JSON的 `text_change_details`；HTML勾选“仅匹配边界的文本变化”可独立查阅。

| 音频 | 状态 | 文本变化 | 差异边界 | role / emotion变化 | 左 / 右独有边界 | 说明 |
|---|---|---:|---:|---:|---:|---|
| a1 | comparable | 37 | 38 | 0 / 0 | 1 / 0 | 完整同源配对，SHA一致 |
| a2 | comparable | 55 | 55 | 0 / 0 | 0 / 0 | 完整同源配对，SHA一致 |
| a3 | comparable | 38 | 39 | 0 / 0 | 1 / 0 | 完整同源配对，SHA一致 |
| a4 | comparable | 61 | 66 | 0 / 0 | 2 / 3 | 完整同源配对，SHA一致 |

完整实际内容见 [N4_A4_vs_N4_A4_resident.html](N4_A4_vs_N4_A4_resident.html) 与 JSON；HTML可搜索并展开完整字段。

文本、角色/情绪字段与警告变化分别保存；同A/同B相对差异同样不等于准确率变化，无人工GT不认证准确率或召回。

警告明细见JSON的 `warning_changes`，增强字段明细见 `enhancement_field_changes`；原始警告数组始终保留。
