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

## 已审核的宿主元数据偏差

当前前缀实际distribution为numpy1.26.4，pip check无依赖冲突，但旧conda-meta仍记录numpy2.2.6；pip26.2.1、wheel0.48.0、setuptools84.0.0、packaging26.3的旧文件映射含lib/python3.1路径。首次严格conda-pack错误必须保留。仅当错误集合精确等于这五项已审核偏差时，准备工具才允许忽略缺失的历史管理记录并打包实际存在文件；出现任何其它缺项立即失败。不得修改宿主conda-meta，不能将忽略元数据解释为忽略依赖或运行错误。

环境归档产生后记录SHA和实际包快照；相同workspace仅在指纹与归档SHA一致时复用，指纹变化则要求另选workspace，不能覆盖已有缓存。最终仍须在独立解包环境核验numpy1.26.4、torch/torchaudio/torchvision、pynini和ITN数据、实际导入路径及原生ABI，不能仅靠安装记录证明可用。

## 构建和运行检查

Dockerfile不运行Cython、不复制项目脚本/测试/文档或配置；运行文件由根 `.dockerignore` 白名单控制。运行归档排除已知保护/测试工具及Python缓存；第三方运行依赖自带的必要原生扩展不属于项目旧业务.so。

容器设置 `ASR_CONTAINER_RUNTIME=cann82-source`，精确拒绝torch-npu2.5.1（未带post1）、不可读配置及HTTP实例数不为1的配置；主程序预检仍保留。只有完成NPU映射/矩阵、三个模型实际forward、四owner暖身覆盖和完整B4后才可称镜像可用。构建失败保留任务私有缓存，成功验收后按任务归属清理，具体构建/部署命令见docker/README.md。
