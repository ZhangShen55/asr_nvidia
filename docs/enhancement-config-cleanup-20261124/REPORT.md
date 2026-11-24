# 增强配置精简与数组启用验证（2026-11-24）

## 结果

已移除features.open_v2_role/open_v2_emotion读取，CAM++和emotion2vec分别由严格校验的设备数组是否非空决定启用，缺省空数组即关闭。启动预加载、路由准入及直接运行入口使用同一规则；showSpk/showEmotion/showRoleIdentify只选择执行能力。空数组配置关闭和非空数组全加载失败分开，原4015/4016/4018、能力关闭优先于容量、A/B分类、段级降级与取消收尾保持。

旧features键不论true/false均在创建任何worker/模型前拒绝并提示迁移：true删除旧键保持数组，false删除旧键并将对应数组设为空。根/Docker continuous模板省略两项legacy组批参数和两个固定batch1参数；显式legacy参数支持及固定batch1/非1拒绝约束保留。5个活跃TOML样例迁移旧开关，历史归档/报告不改。两个prefetch、所有复读参数、retry_count、队列与其它VAD配置不变。

## CPU与规格验证

- 针对子集528 passed、1个既有保护专属skip、288 subtests，pytest109.22秒；补齐采样测试依赖后独立sampler25 passed。
- 最终完整源码CPU：1378 passed, 2 skipped, 1 deselected, 2 warnings, 379 subtests passed in 253.03s (0:04:13)。外层257.45秒。两个skip为既有保护构建专属，另排除真实模型验证文件和既有直接npu-smi硬件探针。
- CPU使用私有TMPDIR/config/log/stats与既有device/cache guard，并补充禁止真实HF资产IO；Transformers可选NPU包发现仅在CPU测试进程设不可用。无源码/config漂移、无owned残留、共享cache写删尝试0。完整命令、隔离位置、源SHA及结果见verification.json。
- 首轮完整CPU为1375通过/3失败，资源采样测试只模拟预加载，未隔离随后get_qwen_client/bind_owner_pool，触及真实HF模型元数据被guard拦截。只补齐该测试依赖，不放开guard；独立25项及完整重跑通过。失败日志保留/tmp/enhancement-config-cleanup-full-20261124-v1，不算通过。
- change严格校验通过；四能力共16条修改Requirement的合并预览严格9/9通过。正常代码open_v2_*仅余迁移拒绝定义，测试旧键只用于迁移及不存在断言。

## NPU7真实服务冒烟

使用当前正式源码main.py、当前工作区config.toml、conda asr与NPU7；独立loopback18083、私有cwd/log/stats/TMPDIR，CANN路径保留，未接管原服务。启动加载耗时205.906秒，与短请求分开。首次音频前GET核四Qwen健康、CAM4/emo1全部加载成功，增强calls均0，load_attempts均1。随后两次B请求showSpk/showEmotion=true、showRoleIdentify=false，CAM/emo成功调用各从0增1再增2；模型出生身份和加载次数保持。再A请求三个增强参数false，两增强calls和outcomes不增加。

| 请求 | HTTP | 客户端耗时秒 | 返回段数 |
|---|---:|---:|---:|
| request_1 | 200 | 4.608 | 1 |
| request_2 | 200 | 4.287 | 1 |
| request_A | 200 | 2.393 | 1 |

正常TERM后exit0、无强制/孤儿收尾、owned身份退出、18083连续释放至少5秒、服务专属PCM缓存为空；既有PID1601397及8007保持。源/config前后SHA一致。原始请求/响应和服务日志在logs/enhancement-config-cleanup-20261124/npu-smoke-v1；精简状态/关闭证据同目录文档保留。

## 用户原有配置差异与交付边界

本轮实测使用当前工作区VAD阈值-100，三段级过滤开关均false；用户原有-40→-100差异原样保留且不纳入本change提交，提交配置仍保留原默认-40。该差异不作为性能结论依据。逐键模板证据表明本轮仅移除批准的六项，其余配置保持。

此验证是配置/生命周期及短真实音频功能冒烟，不代表完整B4课程吞吐、200秒目标、全shape暖身、长期稳定性或准确率/WER/召回率认证；没有人工真值。只要求增强真实forward与复用，不把READY/加载当实际推理。未改历史课程JSON、权重、精度、依赖或构建镜像/保护包。源码重启后生效，镜像/保护构建在后续重建纳入；本次服务已释放。
