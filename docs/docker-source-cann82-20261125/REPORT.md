# CANN8.2源码镜像验收（2026-11-25）

**通过镜像可用性验收。** `asr_offline_910b:v3_20261125`，arm64，17,200,608,906字节（约16.02GiB）。不可变ID `sha256:ef7b8add6a42c0297dc66eaa58718a67df8fba688d1738f1c314fe59b8565c5a`；已测候选原样加tag，验收后未重建。

真实v3双增强B4完成4/4（100%），**P50 182.44s / P95 196.25s / 墙钟198.65s**。严格无警告2/4；16个警告含9段折叠恢复、7段不可恢复并显式丢弃。不是零丢段或人工准确率认证。服务正常停止，本轮中间镜像/缓存/辅助资源全部清理。

## 输入、配置和测量

镜像源码54c3ee82fd832af6520e844afae13d33ae400d80，host验收工具a007018；后者不入镜像，推理算法不改。指定本机底座ID 634118c33bcc879ee709510a532f3e8f90cb0d6a884922b5996367f9a6a7d864，实际CANN8.2.RC1/oppkernel8.2.0.0.201，driver25.0.rc1.1、910B3、Python3.10/aarch64；传输前后config digest一致，未替换外网同名镜像。宿主CANN8.1/asr环境、驱动、权重未升级。

HTTP1/Qwen4 process/每owner16/全局64，CAM4 process/emo1，VAD8 CPU，Qwen BF16；原VAD阈值、恢复、预算及精度不变。根config SHA 28cfe37dcbf45bf19c2b4f2c00521fc955968f083252d0da9c30b92d251118bc与用户原VAD差异保持。只使用物理7，两个ASCEND可见变量均为7；配置逻辑npu:0。最终镜像实际count1、BF16矩阵、SDK导入通过，probe出生身份/NSpid与物理7表匹配。

canonical四条完整WAV重读原文件名编码、SHA和时长，双增强true/showRoleIdentify=false，同时上传真实容器POST /v3/seacraft_asr（loopback18083），非离线段/截断替代。第一次/重启HTTP就绪415.69/419.43s单列；四路8.01s语音暖身30.20s，全部4owner实际prefill/completed、CAM4/emo1均真实调用；归零后冻结计数。

| 输入 | 完整文件 | 耗时s | 输出段 | 警告/丢弃 | RTF |
| --- | --- | ---: | ---: | ---: | ---: |
| a1 | 储能学术英语-多人（中英）.wav | 158.11 | 729 | 0/0 | 0.0507 |
| a2 | 反常积分与级数（中法）.wav | 182.68 | 569 | 4/1 | 0.0556 |
| a3 | 多模态机器学习1（中英）.wav | 182.20 | 659 | 0/0 | 0.0551 |
| a4 | 多模态机器学习2(中英).wav | 198.65 | 622 | 12/6 | 0.0605 |

N=4全部请求线性插值；墙钟首个发送到最后完整响应，含上传，不剔除慢请求。原时长3117.312/3287.296/3306.048/3284.416s。336s诊断线未触发，无8GiB free停止线；小样本不能外推生产P95。CAM实际2775次（683/714/677/701），emo2226次，另549段既有时长skip；empty/failed/cancelled均0，Qwen/CAM及emo+skip逐请求守恒，所有输出role/emotion非空，无owner更换或容量越限，正式B4无OOM/设备故障。

物理7目标2s采样，实际B4 76点、最大间隔2.81s；峰used 44998原始MB，最低free 20538原始MB（约20.06GiB）。包含共享背景，是采样极值，非绝对峰值；停后used 15909原始MB。启动/暖身与B4分相。

## 结果与异常

四份输出文本指纹不同，时间边界有序、非负并在原音频内，字段/请求归属无结构错误。首尾文字及完整输出留存，无逐句人工真值。下面是全部段异常；不可恢复项不伪造其原始文本，恢复仍标completed_with_warnings。未为Docker调整复读/生成策略或隐藏警告。

| 输入 | 原VAD index | 边界s | 原因 | 处理 | 实际保留正文 |
| --- | ---: | --- | --- | --- | --- |
| a2 | 8 | 64.08–72.09 | repetition_stop | discard_unrecoverable | 无可恢复正文；响应显式警告 |
| a2 | 445 | 2364.42–2371.43 | repetition_stop | repetition_collapsed | 保留：C'est quoi la question de l'exercice? Trouvez un équivalent simple. Donc, ici |
| a2 | 544 | 2886.43–2894.44 | repetition_stop | repetition_collapsed | 保留：不，还要给我的，给我 |
| a2 | 581 | 3131.32–3139.33 | length_limit | repetition_collapsed | 保留：我这，稍微把它改一下，嗯 |
| a4 | 29 | 148.49–156.50 | repetition_stop | discard_unrecoverable | 无可恢复正文；响应显式警告 |
| a4 | 262 | 1394.97–1398.24 | repetition_stop | discard_unrecoverable | 无可恢复正文；响应显式警告 |
| a4 | 268 | 1411.07–1412.74 | repetition_stop | discard_unrecoverable | 无可恢复正文；响应显式警告 |
| a4 | 515 | 2415.56–2421.26 | repetition_stop | repetition_collapsed | 保留：十六还是多少？一二十六， |
| a4 | 545 | 2571.64–2579.65 | repetition_stop | repetition_collapsed | 保留：没事，没事，来，来，就拍吧， |
| a4 | 547 | 2582.98–2589.87 | repetition_stop | discard_unrecoverable | 无可恢复正文；响应显式警告 |
| a4 | 560 | 2644.88–2648.92 | repetition_stop | discard_unrecoverable | 无可恢复正文；响应显式警告 |
| a4 | 573 | 2706.99–2711.42 | repetition_stop | repetition_collapsed | 保留：没事 |
| a4 | 581 | 2739.08–2742.88 | repetition_stop | repetition_collapsed | 保留：我们边是送 |
| a4 | 620 | 2937.87–2941.61 | repetition_stop | discard_unrecoverable | 无可恢复正文；响应显式警告 |
| a4 | 622 | 2948.70–2954.40 | repetition_stop | repetition_collapsed | 保留：他他，不是， |
| a4 | 667 | 3136.77–3144.78 | repetition_stop | repetition_collapsed | 保留：我 |

完整明细见[warning-details.json](warning-details.json)。历史B4为12warning/11discard，但不同源码/依赖，不作质量优劣归因；本轮没有同源码宿主对照，也不凭历史200.10s推断容器加速或性能不变。验收口径是已约定的“带显式段警告的完整业务完成”，人工准确率/长期稳定另验。

## 失败修复与内容审计

- frontend外网超时：daemon取得固定官方摘要，任务私有loopback registry传输；原base保持。
- APT403定位底座01proxy遗留Squid：仅构建层使用同一官方HTTPS、显式DIRECT，签名校验保留；宿主APT不改。
- CANN metadata误列标准库/部分SDK WHEEL带cp311：ASR独立prefix严格pip check；固定补真实缺项absl-py2.3.1/ml-dtypes0.5.1/tornado6.5.1并核来源摘要，SDK实际导入/编译/前向另验，不靠忽略metadata冒充成功。
- 首个可构建候选NumPy来自旧Conda2.2缓存，CPU审计失败，未用于NPU。改实际prefix唯一源、Conda仅提供重定位类型，45,049成员逐文件SHA通过；精确crcmod SHA例外经实际readelf验证RPATH=/opt/asr/lib、CRC向量和关键原生导入通过。
- 特权模式只设ASCEND_VISIBLE7仍count8，在张量前拦截；补RT_VISIBLE7及count1/logical0启动门禁，最终镜像实测通过，宿主多卡语义保持。
- 首轮host验证器把未首次请求的空调度器当忙，未发暖身/B4；正常exit0但未创建缓存目录使工具cleanup=false。仅修host严格cold-idle与精确固定路径不存在处理，真实状态重放/反例通过；失败v1留存，v2同一镜像完成。

真实BuildKit上下文精确49允许文件；命名依赖/模型上下文另有清单。最终/app精确59文件（48必要.py+11小模型资产），无烘焙config，Qwen只读挂载；项目tests/test/docs/scripts/openspec/.git、样本、旧业务.so、Cython工具及Nginx服务未带入。底座和必要第三方资源与项目测试分开审计，不虚称整个OS无同名目录。

关键Python/torch/torch-npu/transformers/torchvision/numpy/av/pynini/soundfile来自/opt/asr；torch2.5.1/torch-npu2.5.1.post1精确门禁、中英ITN、ASR严格pip check通过。Torio有可选FFmpeg6/5探测失败、部分FFmpeg4回退DEBUG；实际av/soundfile解码及完整请求成功，不计作正式业务失败。

CPU全量1419passed/379subtests、2既有保护专属skip/1硬件deselected，242.96s；host工具修复后82passed。隔离TMPDIR、禁NPU/权重，源码无变化、owned无残留、共享缓存无写入。README漏项和CPU控制器参数错误等失败原件保留；本版未做Cython保护构建。

## 正常关闭与范围内清理

容器正常exit0，无OOMKilled；全部登记出生身份消失，端口空闲≥5s，停止容器PCM tar头为空。最终8个VAD关闭记录均exit0/forced=false/cleanup_complete=true；Qwen/CAM逐worker最终码未直接打印，由关闭契约、parent0、无异常日志及身份消失间接核对，不补写未知码。共享PID1601397/出生1791963491.5及8007监听保持。

已删除本轮11个probe/build/test容器、私有builder及state缓存volume、loopback registry及data volume、Compose网络、2旧候选镜像、3新helper镜像、传输/候选tag、10个staging目录。先保留其中来源JSON/日志/audit并核SHA；原11镜像引用、14容器及状态保持，base/final完整可inspect。未用全局prune，不删其它项目资源。测试服务已释放。

Docker严格四文件（Dockerfile/docker-compose.yml/config.toml/README.md），配置必需只读bind至/config.toml，缺失/目录失败，无fallback；修改后重建容器。运行命令见[Docker README](../../docker/README.md)。

## 证据索引

[metrics.json](metrics.json)为逐请求/RTF/模型阶段/HBM/镜像ID，[warning-details.json](warning-details.json)为全部异常。原始logs/docker-source-cann82-20261125及workspace.txt指向的W保留全部7轮构建/失败/CPU/原生/NPU、输入与资源manifest、四完整JSON、采样、关闭及cleanup-before/after/result；载荷和本轮缓存已清，日志审计不删。旧NVIDIA报告原字节迁为legacy-validation-report-20261103.md。

本轮仅短时源码容器可用性，生产上线及人工质量另评。所有任务完成后同步四delta并独立归档；旧open change及历史OOM不改。
