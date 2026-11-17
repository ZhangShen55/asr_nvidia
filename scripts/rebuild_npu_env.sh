#!/usr/bin/env bash
# 使用完整 conda explicit + pip-owned lock 重建，不修改源 asr。
set -euo pipefail
if [[ "${1:-}" == "--help" || $# -ne 1 ]]; then
  echo "用法: ASR_CONDA_EXE=/data/anaconda3/bin/conda $0 /独立/新前缀"
  echo "默认按锁定参考重建；ASR_WITH_DEV_TOOLS=1 / ASR_WITH_BUILD_TOOLS=1 可选。"
  echo "离线 wheel 源: ASR_WHEELHOUSE=/绝对目录；无此参数时使用 pip 配置的索引。"
  [[ "${1:-}" == "--help" ]] && exit 0 || exit 2
fi
ASR_TARGET_PREFIX="$1"
[[ "$ASR_TARGET_PREFIX" == /* && ! -e "$ASR_TARGET_PREFIX" && ! -L "$ASR_TARGET_PREFIX" ]] || {
  echo "[ERROR] 目标必须是不存在的绝对路径，拒绝覆盖: $ASR_TARGET_PREFIX" >&2; exit 2;
}
ASR_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASR_CONDA_EXE="${ASR_CONDA_EXE:-$(command -v conda || true)}"
[[ -x "$ASR_CONDA_EXE" ]] || { echo '[ERROR] 请设置 ASR_CONDA_EXE 为 conda 可执行文件' >&2; exit 2; }
[[ "$(uname -m)" == aarch64 ]] || { echo '[ERROR] 此锁定环境仅适用于 linux-aarch64' >&2; exit 2; }
ASR_CANN_ENV="${ASR_CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
[[ -r "$ASR_CANN_ENV" ]] || { echo "[ERROR] CANN 环境脚本不存在: $ASR_CANN_ENV" >&2; exit 2; }
set +u
source "$ASR_CANN_ENV"
set -u
"$ASR_CONDA_EXE" create --yes --prefix "$ASR_TARGET_PREFIX" --file "$ASR_REPO_ROOT/requirements-conda-linux-aarch64.txt"
ASR_PYTHON="$ASR_TARGET_PREFIX/bin/python"
ASR_PIP_SOURCE=()
if [[ -n "${ASR_WHEELHOUSE:-}" ]]; then
  [[ "$ASR_WHEELHOUSE" == /* && -d "$ASR_WHEELHOUSE" ]] || { echo '[ERROR] ASR_WHEELHOUSE 必须是已有绝对目录' >&2; exit 2; }
  ASR_PIP_SOURCE=(--no-index --find-links "$ASR_WHEELHOUSE")
fi
# pip元数据操作只看独立环境，避免CANN PYTHONPATH暴露外部te/op-compile-tool发行元数据。
# 当前shell仍完整保留CANN PYTHONPATH，后续真实import/forward不可清空。
# conda 提供 av=14.2.0/pynini=2.1.7 及二进制库；pip 不接管这些包，不解析漂移依赖。
env -u PYTHONPATH "$ASR_PYTHON" -m pip install "${ASR_PIP_SOURCE[@]}" --no-deps --report "$ASR_TARGET_PREFIX/asr-pip-install-report.json" -r "$ASR_REPO_ROOT/requirements-lock-npu.txt"
if [[ "${ASR_WITH_DEV_TOOLS:-0}" == 1 ]]; then
  env -u PYTHONPATH "$ASR_PYTHON" -m pip install "${ASR_PIP_SOURCE[@]}" --no-deps -r "$ASR_REPO_ROOT/requirements-dev.txt"
fi
if [[ "${ASR_WITH_BUILD_TOOLS:-0}" == 1 ]]; then
  env -u PYTHONPATH "$ASR_PYTHON" -m pip install "${ASR_PIP_SOURCE[@]}" --no-deps -r "$ASR_REPO_ROOT/requirements-build.txt"
fi
env -u PYTHONPATH "$ASR_PYTHON" -m pip check
ASR_MANIFEST_DIR="$ASR_TARGET_PREFIX/asr-env-manifest"
mkdir -p "$ASR_MANIFEST_DIR"
env -u PYTHONPATH "$ASR_PYTHON" -m pip list --format=json > "$ASR_MANIFEST_DIR/pip-list.json"
"$ASR_CONDA_EXE" list --explicit --prefix "$ASR_TARGET_PREFIX" > "$ASR_MANIFEST_DIR/conda-explicit.txt"
"$ASR_PYTHON" - "$ASR_REPO_ROOT" "$ASR_MANIFEST_DIR" "${ASR_WITH_BUILD_TOOLS:-0}" <<'PYCODE'
import json, sys
from pathlib import Path
root, manifest = map(Path, sys.argv[1:3])
normalize = lambda name: name.lower().replace('_', '-').replace('.', '-')
expected = {normalize(x['name']): x['version'] for x in json.loads((root/'docs/npu-910b-migration-20261117/environment-reference.json').read_text())['pip_list']}
actual = {normalize(x['name']): x['version'] for x in json.loads((manifest/'pip-list.json').read_text())}
allowed = {'cython', 'conda-pack'} if sys.argv[3] == '1' else set()
missing = {key: value for key, value in expected.items() if actual.get(key) != value}
extra = {key: value for key, value in actual.items() if key not in expected and key not in allowed}
report = {'missing_or_different': missing, 'unexpected': extra, 'build_tools_allowed': sorted(allowed), 'matches_reference': not missing and not extra}
(manifest/'package-comparison.json').write_text(json.dumps(report, indent=2)+'\n')
if not report['matches_reference']:
    raise SystemExit(f'package reference differs; inspect {manifest}/package-comparison.json')
# metadata一致不能证明二进制一致：克隆后残留NumPy 2的_core扩展曾让SciPy的ufunc类型分裂。
# 新环境必须验证真实科学计算导入、CPU数值和NumPy RECORD归属，失败时禁止用于打包。
import importlib.metadata as metadata
import numpy as np
import scipy, scipy.special, scipy.optimize
numpy_dist = metadata.distribution('numpy')
tracked = {Path(numpy_dist.locate_file(item)).resolve() for item in numpy_dist.files or ()}
numpy_root = Path(np.__file__).resolve().parent
untracked_binary = sorted(str(item) for item in numpy_root.rglob('*.so') if item.resolve() not in tracked)
if untracked_binary:
    (manifest/'numpy-untracked-binaries.json').write_text(json.dumps(untracked_binary, indent=2)+'\n')
    raise RuntimeError(f'NumPy contains binaries outside installed RECORD: {untracked_binary}')
expit = scipy.special.expit(np.array([-1., 0., 1.]))
assert np.allclose(expit, [0.2689414213699951, 0.5, 0.7310585786300049], rtol=0, atol=1e-14)
root_value = scipy.optimize.brentq(lambda value: value*value-2, 1., 2.)
assert abs(root_value-2**0.5) < 1e-12
# 仅验证导入与CPU数值，不探测可用卡/初始化设备/执行模型。
import torch, torch_npu, torchaudio, transformers, funasr, av, pynini, tbe, te, attr
assert not torch_npu.npu.is_initialized()
(manifest/'imports.json').write_text(json.dumps({
    'imports_ok': True, 'torch': torch.__version__, 'av': av.__version__,
    'numpy': np.__version__, 'numpy_file': np.__file__, 'scipy': scipy.__version__,
    'scipy_file': scipy.__file__, 'numpy_untracked_binaries': untracked_binary,
    'scipy_cpu_expit': expit.tolist(), 'scipy_cpu_brentq': root_value,
    'npu_initialized': False, 'npu_forward_checked': False}, indent=2)+'\n')
PYCODE
"$ASR_PYTHON" "$ASR_REPO_ROOT/scripts/check_npu_environment.py" --metadata-only > "$ASR_MANIFEST_DIR/preflight-metadata.json"
echo "[INFO] 已创建独立环境: $ASR_TARGET_PREFIX；metadata/import不等于真实前向通过"
