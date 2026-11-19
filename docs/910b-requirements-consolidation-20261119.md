# 910B 依赖清单整理验证

日期：2026-11-19。仅整理依赖文件布局及当前文档，不修改业务源码、依赖版本、模型资产、配置默认值或现用环境。

## 结果

- 原 requirements-pip.txt、requirements-pip-runtime.txt、requirements-npu.txt、requirements-pip-extra.txt 合为 requirements.txt；与提交前 Git 内容比较，30 项依赖集合完全一致，无版本变化。
- Conda explicit 与 pip-owned 完整锁移至 environment/910b/；与 Git 原内容逐字节一致。开发及构建清单保留，模型目录不动。
- rebuild_npu_env.sh 引用新路径，保留独立前缀、Conda av/pynini、pip --no-deps、版本比较及导入校验流程。
- README、docker/README、AGENT 与环境说明同步；历史报告/归档保留原记录。

## 验证

- bash -n scripts/rebuild_npu_env.sh、--help、git diff --check 通过。
- 独立短 TMPDIR，加载 CANN set_env.sh 后运行 tests/test_docker_deployment.py、tests/test_docker_profile.py、tests/test_npu_environment.py：33 passed in 20.24s。
- 首轮未加载 CANN，31 passed / 2 failed（libhccl.so 缺失及后续 torch 重复注册）；修正测试环境后同组全部通过。
- 本次只做相关交付与 CPU 替身检查，未跑完整 pytest、未重建环境、未构建镜像或启动模型/服务，不作为新的 NPU 或吞吐验收。

预先存在的 fusion_result.json 运行产物变更保留，不纳入提交。吞吐 change 验收状态不变；本次属于交付文件整理，没有业务行为规格变更。
