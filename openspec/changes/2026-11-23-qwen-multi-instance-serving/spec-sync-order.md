# 规格覆盖与后续同步顺序

当前为2026-11-24用户采纳最终N4/CAM4/emo1后的规划修订，不同步主规格、不归档本change或三个旧change，不改旧任务、报告或性能结论。新delta的MODIFIED目标只取现有main，不能假定open delta已经合并。

| 冲突来源 | 现有含义 | 新正式continuous范围 | 保留边界 |
|---|---|---|---|
| main local-qwen按需加载 | 所有转写启动不加载 | preload=true默认顺序全体加载；false仍lazy；状态不触发 | CAM/emo lazy不变 |
| main local-model-runtime主服务承载 | 主进程持有Qwen/CAM/emo | 主协调+N个owned Qwen进程，增强/VAD各自独立 | 不复制整套HTTP服务 |
| main adapter音频默认8、global独立 | 三层配置单独维护 | continuous audio=N/global=N×S，B默认4 | legacy显式路径及既有码保留 |
| throughput pending按段跨副本均衡 | 反对完整音频绑定，先测单再2/4/8 | 请求生命周期内独占owner，释放后动态复用 | 禁SHA/文件名永久绑定；旧矩阵仍未验收 |
| throughput pending首次并行加载 | 允许多副本并行初始化 | 新默认顺序owned preload避免启动峰值 | 旧实验结果不重写 |
| continuous pending单实例design | 一个模型权重/执行owner | N个owner，各自单设备执行循环 | 独立EOS/refill、bounded KV、警告恢复和不重发已接受段继续有效 |
| feasibility pending隔离实验 | 不能推广正式默认 | 新change负责正式能力；N6/N5历史保留，最终N4/CAM4/emo1 B4及增强驻留A4验收 | 194.81秒仅参考，实验任务26/32仍open |
| 本change初始N6范围 | 默认N6/S16/global96及六Qwen共驻标定 | 用户2026-11-24明确替代为最终N4/S16/global64、CAM4/emo1，合法N1～8仍保留 | 原N6纯A6/B4成功、共驻OOM及N5补测原样保留；不得写成N6通过 |
| 原N6总6/B4探针 | 4B仍能接2A，总未满第五B4017 | 最终N4总4/B4同时满，新A/B均4013优先；该配置无法单独触发4017 | 通用4017契约与原N6实测历史保留，不伪造N4可达场景 |
| 最终增强模板 | 初始N6/CAM1/emo1待标定 | loader/根/Docker/说明最终Qwen4/CAM4/emo1并以同一配置验收冻结 | 不以原1/1或N5结果交付未验收4/1；不按N复制池 |

## 合并预览与审查

创建临时规划副本，以main为基底应用本change的六份delta，严格校验各能力；再按本表审查throughput、continuous未归档条款的潜在覆盖，不把未验收的旧delta整体导入。预览文件不得写回main。检查所有MODIFIED名字在main存在、完整场景未被误删、ADDED没有与main重名。

未来同步必须先确定各旧delta哪些已经验收、哪些被明确替代：形成`main → 可接受throughput → 可接受continuous → 新multi-instance`覆盖账本，新正式调度及用户确认的最终N4/CAM4/emo1默认必须最后生效，禁止旧N6初始文本或其它open delta覆盖。仅存在旧open delta的条目按届时main重新定位，不能机械套用本次MODIFIED。feasibility继续限实验。若以后归档旧change，先rebase其冲突条款/记录被替代范围，不允许恢复单实例或禁止请求租约。

旧44/60、20/25、26/32是当前检查点，不会因本次提案或A6/B4将来通过而勾选旧A8、单实例真实128、16/128、1/2/4/8及420秒失败矩阵。归档仍依其自身验收要求，用户明确变更范围时另记决策。

## 本轮规划校验

2026-11-24完成：change严格校验1/1通过；临时main+本change预览九份规格严格9/9通过。六份delta共36条MODIFIED、17条ADDED、184个场景；MODIFIED目标均存在main且保留原Scenario标题，ADDED不与main重名。预览位于`/tmp/qwen-multi-proposal-preview-20261124-kyn_u8_3`，未写回main。当时29项实施任务全部未开始，apply状态ready；该规划校验时未运行模型、服务或性能测试。这是历史检查点，不代表后续实施/验收尚未发生。日期沿用2026-11-23开始创建的change名，跨日校验据实记录。

## 2026-11-24最终范围覆盖账本（尚未同步）

后续root仅在最终实测通过后，重新生成临时合并预览并严格校验六能力及所有保留Scenario。本轮不写main、不归档。覆盖顺序为main → 届时可接受throughput → 届时可接受continuous → 本change已实现通用多owner能力 → 本change用户最终N4范围；最后一层覆盖初始N6默认/剩余矩阵，不覆盖N6/N5历史结果。

| 能力/产物 | 最终必须生效 | 不得被覆盖或伪造 |
|---|---|---|
| local-qwen-asr-inference | continuous/process、缺省N4、合法1～8、S16、总4/global64；每owner预算与动态请求租约 | 不恢复单实例限制、按段跨owner迁移或N6默认；历史N6/N5不是最终N4证据 |
| qwen-asr-v2-adapter | N4总4/B4满额新A/B均4013；通用总满优先与4017保留 | 原N6的4B+2A/4017仍是显式N6已测历史，不能写成本次N4可达状态 |
| local-model-runtime / service-operations | 最终Qwen4/CAM4/emo1、独立池及同配置B4/共驻A4实测冻结；根/Docker/loader/说明一致 | 不用旧N6/CAM1/emo1或N5成绩替代4/1，不恢复“六Qwen标定才能交付”的旧剩余范围 |
| offline-asr-service | 原v3字段/错误码/边界及段警告、取消和安全资源所有权 | 不因范围修订降低生成上限、删段、改阈值或改变警告质量口径 |
| resource-pressure-diagnostics / test-plan | 最终同a1～a4 B4与增强驻留A4完整/资源/关闭；纯A约200目标单列、B4无200硬门槛，336/OOM/故障停止 | 原formal_matrix_complete继续代表N6历史，不伪造N6共驻通过；新final_n4_functional_matrix_complete配合root原始背景回归审查，不修改raw unknown/delta |
| tasks / implementation-status | 5.3最终N4/CAM4/emo1标定冻结、5.6同源增强驻留A4，实测前未勾；已勾历史事实保留 | 不删除原N6 OOM/N5结果，不勾旧change或宣称旧矩阵完成 |

此前1/1 change与9/9 main/临时预览、36 MODIFIED/17 ADDED/184场景只是原N6文本校验记录，不能代替本轮N4修订后的重新严格校验或最终实测。只有root确认新范围完整、安全、正常关闭与实际配置一致后，才另行执行sync/archive；约200目标达标与否、警告和人工质量限制始终独立保留。
