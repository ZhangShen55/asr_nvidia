# CANN 8.2 源码容器环境

本文件说明容器专用环境，宿主 `requirements.txt` / asr 前缀仍使用 CANN8.1 配套，不在原环境升级。

- 平台：Linux aarch64、Python3.10，独立解释器 `/opt/asr/bin/python`；禁止使用系统 Python 加上 asr site-packages。
- 底座：本机 `cann:8.2.rc1-910b-ubuntu22.04-py3.10` 的已核验 image ID；独立builder通过任务私有registry传输后固定manifest digest，不从公网取同名底座。
- 配套：torch2.5.1 / torch-npu2.5.1.post1；其余顶层运行版本见 `requirements-container-cann82.txt`。官方依据：[Ascend PyTorch 配套表](https://raw.githubusercontent.com/Ascend/pytorch/v2.5.1-7.1.0/README.zh.md)。该表不代替本机真实forward验收。

## 准备和来源

在仓库外选择持久workspace，运行宿主侧 `scripts/prepare_container_context.py --workspace PATH --tool-python BUILD_TOOL_PYTHON`；可用 `--runtime-prefix` 指定已经核验的原始运行前缀。此脚本只读原前缀和三个模型、不调用Docker、不安装包。它产出两个命名上下文：

- `runtime_env`：conda-pack运行归档、PyPI官方torch-npu2.5.1.post1的cp310/aarch64 wheel、实际distribution精确快照和wheel来源/SHA。
- `model_assets`：VAD的model.pt/config.yaml/configuration.json/am.mvn；CAM的campplus_cn_en_common.pt/config.yaml/configuration.json；emo的model.pt/config.yaml/configuration.json/tokens.txt。其它模型资料不复制。

完整输入清单、归档摘要及来源保存在workspace根，不进入镜像。wheel通过PyPI固定版本JSON选择唯一平台文件并核对官方SHA256；已核验文件在重试时复用，不用pip隐式解析其它版本。构建层使用 `pip --no-index --no-deps` 安装该wheel，并执行依赖/版本检查。镜像内保留 `/opt/asr-metadata/` 的实际运行版本与wheel来源供核对，源码revision和构建日期为OCI labels。

当前外部构建工具conda-pack0.8.1依赖pkg_resources，而工具前缀的setuptools84已移除它。准备脚本只在workspace下载/解包setuptools80.9.0，作为conda-pack子进程临时PYTHONPATH；不安装至任何现有环境、不打进运行归档。这个工具兼容处理不能用于给业务运行时叠加系统包。

## 实际前缀字节打包与失败记录

首次严格conda-pack发现宿主历史Conda元数据不一致；曾允许五项已确认缺失管理记录。但真实候选CPU导入失败进一步证明，conda-pack默认从旧package cache读取受管理文件，造成NumPy2.x与实际1.26.4混装。版本元数据/pip check通过不能证明字节正确。该旧方法已停用，坏归档、失败镜像和日志保留，不允许将旧state当成可复用实际快照。

当前固定 `pack_method=actual-prefix-v2`：宿主准备工具调用 `scripts/pack_actual_prefix.py`，以真实前缀现存路径集合为唯一输入，每个File.source均指向该前缀，完全不从package cache读取payload。前缀自身conda-meta.paths_data只提供text/binary重定位类型，占位符改为实际旧前缀；使用原库CondaEnv/File和conda-unpack完成重定位。不修改宿主环境、权重或元数据。

未知二进制含旧前缀时默认阻止打包，必须先审计。当前唯一例外是crcmod的原生扩展，已用readelf确认真实DT_RPATH指向原环境lib，仅对精确路径和SHA登记binary类型，仍需候选容器验证新RPATH及实际导入。任何其它路径或内容变化继续失败。符号链接只允许相对且解析后位于前缀内，绝对/越界链接拒绝。

所有源文件与归档成员保存SHA比对；仅允许库自身确定性的shebang与Conda元数据重写，以及固定版本生成的激活/conda-unpack模板。原生文件在归档中保持源字节，重定位记录进入conda-unpack；解包后再按库规则及真实native/ITN导入验证。源文件、归档SHA审计全部通过方可生成已完成state，不能仅靠包版本复用。

失败旧目录保持为runtime_env；修复使用 `--runtime-context-name runtime_env_v2`，新归档及7文件上下文位于该目录。state为 `environment-pack-actual-prefix-v2-runtime_env_v2.json`，完整输入清单为 `context-preparation-runtime_env_v2.json`，逐文件审计为 `actual-prefix-v2-runtime_env_v2-audit.json`；新鲜workspace可使用默认runtime_env名称，但仍须v2方法state。不允许覆盖已有未知或旧方法归档。

## 构建和运行检查

Dockerfile不运行Cython、不复制项目脚本/测试/文档或配置；运行文件由根 `.dockerignore` 白名单控制。运行归档排除已知保护/测试工具及Python缓存；第三方运行依赖自带的必要原生扩展不属于项目旧业务.so。

容器设置 `ASR_CONTAINER_RUNTIME=cann82-source`，精确拒绝torch-npu2.5.1（未带post1）、不可读配置及HTTP实例数不为1的配置；主程序预检仍保留。只有完成NPU映射/矩阵、三个模型实际forward、四owner暖身覆盖和完整B4后才可称镜像可用。构建失败保留任务私有缓存，成功验收后按任务归属清理，具体构建/部署命令见docker/README.md。

## CANN 8.2 SDK 真实缺项补充

实际底座SDK的schedule-search0.0.1声明需要decorator、numpy、absl-py；te0.4.0基础依赖包含attrs、cloudpickle、decorator、ml-dtypes、numpy、psutil、scipy、tornado。现有asr快照缺少absl-py、ml-dtypes、tornado，因此只在容器构建层安装固定absl-py2.3.1、ml-dtypes0.5.1、tornado6.5.1，不修改宿主或原asr归档。PyPI官方JSON与wheel METADATA分别验证Requires-Python、CPython3.10/aarch64/manylinux平台及NumPy1.26.4；ml-dtypes的Python3.10运行约束是NumPy>=1.21.2，不引入其dev额外依赖。tornado使用兼容Python>=3.9的abi3 wheel，拒绝musllinux和x86产物。

runtime_env因此精确包含7文件：原asr归档、torch-npu及上述三个补充wheel、runtime-packages.json和wheel-source.json。补充包来源、摘要及依赖检查记录在wheel-source.json的supplemental列表，实际版本快照同步加入三包；context-preparation.json记录更新后的完整输入摘要。额外文件、目录或符号链接均拒绝进入构建。

底座op-compile-tool把标准库getopt/inspect/multiprocessing错误声明为pip包，te/op_test_frame元数据又带cp311 wheel标签；这些vendor元数据问题单独记录，不下载同名第三方包冒充标准库、不删除SDK文件。独立asr的pip一致性检查仅该子进程清空CANN PYTHONPATH，实际运行和算子编译保持CANN环境，仍须验证真实导入、编译和NPU前向。不能将通过独立pip check称为SDK全部兼容。

复用v2归档必须同时存在已通过的v2字节审计，并重新核对全部实际源文件SHA及相对链接目标；保持相同版本、文件大小和mtime也不能跳过字节核验。缺少审计、来源不符或任一源变化均拒绝复用。所有workspace/staging的词法路径及祖先在resolve或写入前检查符号链接，models/runtime/tools已有条目递归检查；workspace不能位于运行前缀内，避免将打包产物写回原环境。新准备步骤不会覆盖旧方法失败归档。
