# 910B 环境复现材料

目标：Linux aarch64、Python 3.10、CANN 8.1.RC1。

根目录 `requirements.txt` 是唯一运行依赖入口；本目录的 Conda 显式清单和 pip-owned 完整锁用于精确重建，包含参考环境的传递依赖及部分测试包，不应直接合并成生产依赖入口。

从项目根目录运行 `scripts/rebuild_npu_env.sh /不存在的绝对前缀`。脚本保持 Conda 提供 av/pynini、pip --no-deps 和完整版本比较流程。基线元数据仍在 `docs/npu-910b-migration-20261117/environment-reference.json`；不原地升级现用环境。
