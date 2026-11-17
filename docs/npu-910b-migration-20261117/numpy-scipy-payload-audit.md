# 旧build前缀NumPy/SciPy payload只读审计

结论：`/tmp/asr-rebuild-20261117-build`同时加载了NumPy1.26.4的`numpy/core`扩展和NumPy2.2.6遗留的`numpy/_core`扩展，产生两个不同的`numpy.ufunc`类型。SciPy从后者构造ufunc，却按前者的`np.ufunc`校验，直接触发MultiUFunc类型错误。正式`/home/zhangs/conda-envs/asr`没有该残留二进制，兼容路径回到同一1.26.4扩展，SciPy导入成功。

这是**已运行的独立新进程只读证据**，没有pytest、模块替身或Tensor.to修改，没有导入torch/torch_npu、NPU/服务/真实项目模型。没有修改、删除或安装任何环境文件；旧build失败原件保留。安装历史中的clone重新链接与本次残留内容相符，但本探针不重演或伪造安装过程。

| 项目 | 旧build前缀 | 正式asr |
| --- | --- | --- |
| NumPy实际`__version__`/metadata | 1.26.4 / 1.26.4 | 1.26.4 / 1.26.4 |
| SciPy metadata | 1.15.3 | 1.15.3 |
| `numpy/core/_multiarray_umath...so` | 正常1.26.4扩展，4,597,857bytes，SHA `63dc32deab0dc4e20aad0eed4e5c9963ddc9b1a9b56a8346bdca22944c73318b` | 同字节数/同SHA |
| `numpy/_core/_multiarray_umath...so` | **存在且实际载入**，4,672,864bytes，SHA `75d48ca05dee60aa68fb07e9c00fdd1492c5857ceb97efc69abbddc0c8cd22c3`，不在NumPy1.26.4 RECORD | 不存在；`numpy._core._multiarray_umath.__file__`实际指向上行1.26.4扩展 |
| `_core/_multiarray_umath.py`兼容文件 | 159bytes、在1.26.4 RECORD，但实际载入的是并存`.so` | 159bytes、同SHA，兼容路径正常 |
| `type(_core.add) is np.ufunc` | false | true |
| SciPy`sph_legendre_p`各ufunc | 类型与`_core.add`相同，与`np.ufunc`不同，`isinstance=False` | 与`np.ufunc`相同，`isinstance=True` |
| `import scipy.special` | MultiUFunc ValueError（原始traceback保留） | 成功 |
| conda-meta | 仍登记numpy2.2.6 | 同样仍登记2.2.6，说明conda metadata本身不能证明实际payload |

上述类型身份比较都在**各自单个进程内部**完成，不将不同进程的数值ID互相比对。build残留`.so`的SHA与`/data/anaconda3/pkgs/numpy-2.2.6-py310h6e5608f_0/lib/python3.10/site-packages/numpy/_core/_multiarray_umath.cpython-310-aarch64-linux-gnu.so`完全相同；conda原artifact SHA为`d7234b9c45e4863c7d4c5221c1e91d69b0e0009464bf361c3fea47e64dc4adc2`。这不是仅由报错文本猜测的NumPy版本问题。

## 原始证据与复算

- 聚合JSON：`/tmp/asr-numpy-scipy-readonly-comparison.json`，SHA256 `c87bc4dc86297fa962147c70b418aa46ee4a3c0482858d71773308becb360385`。
- 单进程记录：`/tmp/asr-numpy-scipy-build.json`、`/tmp/asr-numpy-scipy-asr.json`及对应stderr。
- 探针源码：`/tmp/asr_numpy_scipy_readonly_probe.py`，SHA `750293d979debb1fa4272069c8c6de20c5cc64a7dc8990fbc76f6d135c5a9f1b`；执行器`/tmp/asr_run_numpy_scipy_readonly.py`，SHA `abd6ffe4da302643cb2c4988fb8f4698e2ffcbb915886bcc6923bd5e9c5f4e0a`。
- 执行器使用fresh SSH `ControlPath=none`，分别source CANN、设置`TORCH_DEVICE_BACKEND_AUTOLOAD=0`、用build/asr的绝对解释器执行同一只读脚本；记录实际扩展路径、NumPy RECORD归属、conda-cache文件SHA、SciPy异常和ufunc类型关系。
- 本轮正式asr对源码/最终frontguard `.so` 的10+10 CPU回归和输入dtype输出parity另见`/tmp/asr-qwen-frontguard-final-cpu-parity.json`（SHA `6cf3c832e9ab3caf12489433ef04e41a083341b76b49148d1ce4279a93f5c0e0`）。旧build失败不归因给生产源码或`.so`，也不由此CPU通过替代NPU/API稳态。

## 给clean build的检查建议（未实施）

1. 保留旧前缀/报告供追溯；新建不存在的独立前缀，使用完整conda explicit+pip lock入口，不再以旧conda metadata的clone加force-reinstall作为干净payload证明。不改正式asr。当前入口用法为`ASR_CONDA_EXE=/data/anaconda3/bin/conda ASR_WITH_BUILD_TOOLS=1 scripts/rebuild_npu_env.sh /独立/新前缀`，由负责构建的流程执行。
2. 保留113包严格metadata对照和构建工具允许差异，同时增加实际`numpy`、`scipy.special`、`scipy.optimize`导入与来源记录。CANN支持路径保留；仅pip metadata子命令按既有方式排除外部PYTHONPATH，不能靠把主asr包叠进新前缀制造成功。
3. 针对锁定1.26.4验证`numpy._core._multiarray_umath`兼容路径与`numpy.core._multiarray_umath`使用同一ufunc类型，检查NumPy目录内未在RECORD的native `.so`残留；版本字符串、pip list和TBE导入各自不足以覆盖这个故障。
4. 做公开CPU数值冒烟，例如`scipy.special.expit(0)==0.5`、`scipy.special.eval_legendre(2,0.5)==-0.125`，再跑真实TinyHF/CPU特征前端及保护模块import。不要以跳过SciPy、隐藏其异常、替换其类型检查或清sys.modules获取通过。
5. 将新前缀实际imports/数值/模块来源及source/artifact指纹写入新证据；当前旧build报告的imports_ok仅限当时已验证范围，不能提升为完整运行时健康。是否用clean build重新构建stage另由最终构建/验收流程决定，不能未经新指纹核对继承旧stage结果。
