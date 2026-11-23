# baseline_vs_formal_A6

仅比较summary中status=completed的完整响应与声明源SHA。模型间相对一致/不一致不是人工准确率、召回率或WER/CER；censored、缺失和无法精确对齐者为unknown。性能是否通过由原summary独立判断。

旧实验A与正式A6的同源输出对照；不把隔离实验视为本次正式性能验收。

匹配边界文本变化单列在JSON的 `text_change_details`；HTML勾选“仅匹配边界的文本变化”可独立查阅。

| 音频 | 状态 | 文本变化 | 差异边界 | role / emotion变化 | 左 / 右独有边界 | 说明 |
|---|---|---:|---:|---:|---:|---|
| a1 | comparable | 30 | 31 | 0 / 0 | 0 / 1 | 完整同源配对，SHA一致 |
| a2 | comparable | 66 | 67 | 0 / 0 | 1 / 0 | 完整同源配对，SHA一致 |
| a3 | comparable | 35 | 37 | 0 / 0 | 1 / 1 | 完整同源配对，SHA一致 |
| a4 | comparable | 70 | 78 | 0 / 0 | 2 / 6 | 完整同源配对，SHA一致 |
| a5 | comparable | 34 | 34 | 0 / 0 | 0 / 0 | 完整同源配对，SHA一致 |
| a6 | comparable | 22 | 22 | 0 / 0 | 0 / 0 | 完整同源配对，SHA一致 |

完整实际内容见 [baseline_vs_formal_A6.html](baseline_vs_formal_A6.html) 与 JSON；HTML可搜索并展开完整字段。

A/B开关导致的角色、情绪字段差异与文本差异分别保留；具体质量需人工听音/真值，不能据此称准确率提高。
