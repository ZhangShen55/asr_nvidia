# CANN 8.2 RC1 源码容器交付验收（2026-11-25）

状态：实施中，尚未完成真实构建或B4，不能据文件准备认定镜像可用。

## 冻结范围

- Change：2026-11-24-docker-source-cann82；起始提交b9bddb2。
- 指定本地底座ID：sha256:634118c33bcc879ee709510a532f3e8f90cb0d6a884922b5996367f9a6a7d864，arm64，实际CANN8.2.RC1/opp kernel8.2.0.0.201。
- 宿主driver25.0.rc1.1，开发asr原依赖保留；容器单独torch2.5.1/torch-npu2.5.1.post1，必要源码运行，无Cython、Nginx或项目测试/文档/脚本进入上下文。
- HTTP1/Qwen4/S16/总4与逻辑段64，CAM4 process/emo1，VAD8 CPU，物理7限定后容器逻辑0待真实验证；不改精度、阈值及恢复策略。
- 用户根config SHA28cfe37dcbf45bf19c2b4f2c00521fc955968f083252d0da9c30b92d251118bc；其未提交VAD差异保留，不纳入提交。
- 四完整输入沿canonical a1～a4清单，分别3117.312/3287.296/3306.048/3284.416秒，SHA已重读验证；正式请求双增强，clip仅暖身。

## 已知构建准备问题及处理

1. 独立conda-pack0.8.1工具依赖pkg_resources，而其setuptools84已移除此模块；仅在工具子进程注入已验证setuptools80.9.0 wheel解包副本，不安装到宿主或运行镜像。
2. 宿主Conda历史元数据五项与实际文件不符（pip/wheel/setuptools/packaging/numpy）；精确审核允许集合，保留原错误，打包实际存在文件；归档、实际distribution和来源有指纹，镜像内解包/ABI/真实前向仍须独立验证。
3. Buildx官方二进制首次传输超时；保留部分文件断点续传，最终SHA f7d867e9f1a3c00b32dd580f56594e229df05e3fb1b083b7099c91c2e7d2ce1e验证通过，无全局工具升级。
4. 首次CPU全量1387passed/379subtests、1failed、2skip、1deselected，失败仅README漏写CONFIG_PATH，已补；后续完整复验待记录。新工具测试首次缺pytest-asyncio插件，改用项目既有anyio/asyncio backend，不安装新依赖。

## 证据与后续阶段

原始证据：logs/docker-source-cann82-20261125/；任务私有workspace路径见该目录workspace.txt。资源ID、基底传输身份、工具/模型/环境清单、所有失败日志均保留。

待执行：依赖解包和镜像构建、内容及实际导入审计、物理映射/NPU矩阵与真实前向、全部owner暖身、完整B4、正常收尾、最终tag及本轮缓存清理。P50/P95、墙钟、HBM峰值和文本/警告差异尚无本轮数据。人工准确率与生产长期运行不在本轮可用性认证范围。

部署/验收工具专项45项通过。真实BuildKit上下文首次审计发现父目录例外会放行缓存及Docker配置，已删除宽泛父目录例外；v2实际导出恰好49个允许文件，精确清单匹配，无配置/README/缓存。首个audit非正式候选，正式候选尚待构建。完整CPU复验v2进行中，失败记录保留。

首个正式构建v1在外部Dockerfile frontend解析时因registry-1.docker.io网络超时失败，尚未进入业务依赖层。通过daemon获取官方固定frontend，并与base相同方式传入本机registry、核对config digest；重试通过BUILDKIT_SYNTAX固定本地frontend，保持原builder及缓存。追溯ARG/LABEL移至昂贵层之后，避免发布元数据使重试依赖缓存失效。

完整CPU复验v2：1401passed、379subtests、2个既有保护专属skip、1个硬件节点deselected，240.28秒；无源码变化、owned残留或全局缓存写入。构建v2已完成底座解包，但Ubuntu HTTP源被网络代理403拒绝，改为同一官方ports.ubuntu.com HTTPS源并保持签名验证，重试复用底座缓存；apt索引仅在本层安装完成后移除，不清理BuildKit缓存。

构建v3的HTTPS同样403。进一步审计确认底座/etc/apt/apt.conf.d/01proxy硬编码旧Squid 10.30.6.2:3128，非软件包缺失；宿主直连同一官方HTTPS源返回200。构建层移除这份遗留配置，APT显式DIRECT，仍校验HTTPS与Ubuntu签名；同时pip使用--no-compile防止post1安装生成不必要.pyc。原底座与宿主APT均未修改。

构建v4进入独立环境后，pip check因继承CANN PYTHONPATH读到SDK供应方元数据而失败：op-compile-tool将标准库getopt/inspect/multiprocessing误列pip依赖，te/op_test_frame的WHEEL带cp311标签；另确有absl-py/ml-dtypes/tornado缺项。已依据实际Requires-Dist补齐三个固定依赖（2.3.1/0.5.1/6.5.1，验证Python3.10/aarch64及NumPy1.26.4兼容），不安装虚假标准库包。ASR环境单独严格pip check，SDK真实导入/编译及模型forward须另验，不能以忽略供应方metadata代替实际运行通过。APT索引和包缓存改用专属BuildKit cache mounts，不进入最终层，保留至最终验收后统一清理。

候选v5构建成功（17,199,859,464字节）但CPU审计失败，因此未交付、未进行NPU：archive中NumPy __init__字节来自Conda旧2.2缓存，与宿主实际1.26.4不同；ignore_missing_files容错不足，前述“实际前缀快照”假设不成立。v2改为真实prefix唯一源，Conda元数据仅提供重定位类型，45,049成员逐文件SHA验证通过、无package-cache来源，NumPy init重新核对与宿主SHA完全一致。一个crcmod原生RPATH例外按真实readelf及精确SHA审核，实际容器仍须核验/导入。旧archive、错误候选和失败原件保留，runtime_env_v2另建；Python开发config脚本无需运行，解包后从交付环境排除，真实解释器保持原库处理。

最终r3候选已完成构建/精确59文件/ABI导入/pip check/ITN/NPU矩阵+SDK/物理7身份验证，ID ef7b8add6a42c0297dc66eaa58718a67df8fba688d1738f1c314fe59b8565c5a，源码54c3ee8。CPU全量1419/379subtests通过，2既有skip/1硬件deselected；target-v8为82通过。

首轮service冷启动415.69秒，HTTP及4owner健康，但验证器将冷态未创建continuous调度器误判仍忙，未发暖身或正式完整请求；服务正常exit0/全部PID消失/端口释放/shared保持，缓存目录尚不存在导致检查器cleanup字段false。已仅修host工具的严格cold-idle与精确目录不存在处理，实际cold状态重放/反例测试通过，不改镜像或业务调度；v1原始失败保留，新v2输出独立。正式B4/最终cleanup/tag尚未通过，不提前宣称镜像交付。
