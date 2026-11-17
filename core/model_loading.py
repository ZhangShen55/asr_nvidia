"""模型构造共用进程锁，隔离 Transformers 临时修改的全局 torch 精度。"""
from __future__ import annotations

from contextlib import contextmanager
import threading


_MODEL_LOADING_LOCK = threading.RLock()


@contextmanager
def model_loading_guard(dtype=None):
    """锁内保存/恢复默认精度，避免等待者误恢复另一个加载器的临时值。

    HF 的 BF16 构造上下文修改的是进程全局状态，所有项目模型构造必须
    共用此锁。原生网络传入 torch.float32；HF 自行选择构造精度。
    本模块导入时不导入 torch，也不初始化设备。
    """
    with _MODEL_LOADING_LOCK:
        import torch

        previous = torch.get_default_dtype()
        try:
            if dtype is not None:
                torch.set_default_dtype(dtype)
            yield
        finally:
            torch.set_default_dtype(previous)


def validate_parameter_dtype(model, expected_dtype):
    """验证实际浮点权重；库有意保留的 FP32 缓冲区不能被统一降精度。"""
    floating_counts, nonfloating, mismatches = {}, 0, []
    for name, parameter in model.named_parameters():
        if parameter.is_floating_point():
            key = str(parameter.dtype)
            floating_counts[key] = floating_counts.get(key, 0) + 1
            if parameter.dtype != expected_dtype:
                mismatches.append(f"{name}={parameter.dtype}")
        else:
            nonfloating += 1
    if not floating_counts:
        raise RuntimeError("loaded model has no floating weight parameters to validate")
    if mismatches:
        details = ", ".join(mismatches[:12])
        suffix = f" (and {len(mismatches) - 12} more)" if len(mismatches) > 12 else ""
        raise RuntimeError(f"model weight dtype mismatch: expected {expected_dtype}; {details}{suffix}")
    return {"expected_dtype": str(expected_dtype), "floating_parameter_dtype_counts": floating_counts,
            "nonfloating_parameter_count": nonfloating, "buffers_cast_or_checked": False}


__all__ = ["model_loading_guard", "validate_parameter_dtype"]
