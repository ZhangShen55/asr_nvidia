# 规格覆盖与后续同步顺序

当前仅新建提案，不同步主规格、不归档三个旧change，不改旧任务、报告或性能结论。新delta的MODIFIED目标只取现有main，不能假定open delta已经合并。

| 冲突来源 | 现有含义 | 新正式continuous范围 | 保留边界 |
|---|---|---|---|
| main local-qwen按需加载 | 所有转写启动不加载 | preload=true默认顺序全体加载；false仍lazy；状态不触发 | CAM/emo lazy不变 |
| main local-model-runtime主服务承载 | 主进程持有Qwen/CAM/emo | 主协调+N个owned Qwen进程，增强/VAD各自独立 | 不复制整套HTTP服务 |
| main adapter音频默认8、global独立 | 三层配置单独维护 | continuous audio=N/global=N×S，B默认4 | legacy显式路径及既有码保留 |
| throughput pending按段跨副本均衡 | 反对完整音频绑定，先测单再2/4/8 | 请求生命周期内独占owner，释放后动态复用 | 禁SHA/文件名永久绑定；旧矩阵仍未验收 |
| throughput pending首次并行加载 | 允许多副本并行初始化 | 新默认顺序owned preload避免启动峰值 | 旧实验结果不重写 |
| continuous pending单实例design | 一个模型权重/执行owner | N个owner，各自单设备执行循环 | 独立EOS/refill、bounded KV、警告恢复和不重发已接受段继续有效 |
| feasibility pending隔离实验 | 不能推广正式默认 | 新change负责正式代码及A6/B4实测 | 194.81秒仅参考，实验任务26/32仍open |

## 合并预览与审查

创建临时规划副本，以main为基底应用本change的六份delta，严格校验各能力；再按本表审查throughput、continuous未归档条款的潜在覆盖，不把未验收的旧delta整体导入。预览文件不得写回main。检查所有MODIFIED名字在main存在、完整场景未被误删、ADDED没有与main重名。

未来同步必须先确定各旧delta哪些已经验收、哪些被明确替代：形成`main → 可接受throughput → 可接受continuous → 新multi-instance`覆盖账本，新正式调度/默认必须最终生效。仅存在旧open delta的条目按届时main重新定位，不能机械套用本次MODIFIED。feasibility继续限实验。若以后归档旧change，先rebase其冲突条款/记录被替代范围，不允许恢复单实例或禁止请求租约。

旧44/60、20/25、26/32是当前检查点，不会因本次提案或A6/B4将来通过而勾选旧A8、单实例真实128、16/128、1/2/4/8及420秒失败矩阵。归档仍依其自身验收要求，用户明确变更范围时另记决策。

## 本轮规划校验

2026-11-24完成：change严格校验1/1通过；临时main+本change预览九份规格严格9/9通过。六份delta共36条MODIFIED、17条ADDED、184个场景；MODIFIED目标均存在main且保留原Scenario标题，ADDED不与main重名。预览位于`/tmp/qwen-multi-proposal-preview-20261124-kyn_u8_3`，未写回main。29项实施任务全部未开始，apply状态ready；规划阶段未运行模型、服务或性能测试。日期沿用2026-11-23开始创建的change名，跨日校验据实记录。
