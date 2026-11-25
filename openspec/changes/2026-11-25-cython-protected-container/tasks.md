# Tasks

- [x] 1.1 接入独立 Cython 编译阶段、版本与 ABI 校验、精确运行产物白名单。
- [x] 1.2 同步构建说明、独立标签、配置挂载与仓库契约检查。
- [x] 2.1 实际构建，保留镜像和编译日志，查验最终文件与镜像层。
- [x] 2.2 用真实扩展执行适用的 CPU 契约、入口及子进程回归。
- [x] 2.3 在有可用 NPU 时核对真实 forward、B4 和正常收尾，记录验收范围。

首次构建与 CPU 证据见 docs/docker-cython-cann82-20261125/REPORT.md。日期标签重建、NPU 真实前向、A4/B4 及测试收尾见 docs/docker-cython-cann82-20261125/A4-B4-REPORT.md。
