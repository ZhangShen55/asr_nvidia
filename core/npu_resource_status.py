"""Live process allocator counters without importing torch or creating an NPU context."""
from __future__ import annotations

from datetime import datetime, timezone
import os
import re
import sys


def get_live_allocator_status(device):
    """Read only the configured device after the process already initialized NPU.

    Physical/logical mapping is owned by deployment preflight. This function
    never scans devices, imports torch/torch_npu, synchronizes kernels, resets
    peaks or empties caches. Counters cover this process/device, not one model
    and not all HBM (runtime/non-PyTorch allocations are excluded).
    """
    result = {
        "available": False, "device": device, "pid": os.getpid(),
        "timestamp": datetime.now(timezone.utc).isoformat(), "initialized": None,
        "torch_default_dtype": None,
        "scope": "current_service_process_configured_device_pytorch_allocator",
        "excludes": "runtime/non-PyTorch HBM, other processes; not per-model attribution",
        "peak_scope": "process/device allocator peak since initialization or external peak reset",
        "sampling": "sequential counters; concurrent inference may change values between reads",
        "allocated_bytes": None, "reserved_bytes": None,
        "peak_allocated_bytes": None, "peak_reserved_bytes": None,
        "allocated_mb": None, "reserved_mb": None,
        "peak_allocated_mb": None, "peak_reserved_mb": None,
    }
    if not isinstance(device, str) or not re.fullmatch(r"npu:\d+", device.strip().lower()):
        return dict(result, reason="invalid_device", detail="allocator sampling requires an explicit npu:N configuration")
    device = f"npu:{int(device.strip().split(':')[1])}"
    result["device"] = device
    torch = sys.modules.get("torch")
    if torch is None:
        return dict(result, reason="torch_not_imported", detail="status query does not import torch")
    # Reading the module dictionary avoids torch.__getattr__ lazy imports.
    getter = vars(torch).get("get_default_dtype")
    if callable(getter):
        try:
            result["torch_default_dtype"] = str(getter())
        except Exception:
            pass  # metadata failure must not create a backend or mask allocator diagnostics
    api = vars(torch).get("npu")
    if api is None:
        return dict(result, reason="npu_api_not_present", detail="status query does not import/register torch_npu")
    try:
        result["initialized"] = bool(api.is_initialized())
        if not result["initialized"]:
            return dict(result, reason="npu_not_initialized", detail="status query must not create a device context")
        # Set only the caller's explicit configured target; never query a
        # default current_device() or select another card as a fallback.
        api.set_device(device)
        for field, method in (("allocated", "memory_allocated"), ("reserved", "memory_reserved"),
                              ("peak_allocated", "max_memory_allocated"), ("peak_reserved", "max_memory_reserved")):
            value = getattr(api, method)(device)
            if type(value) is not int or value < 0:
                raise ValueError(f"{method} returned invalid byte counter {value!r}")
            result[f"{field}_bytes"] = value
            result[f"{field}_mb"] = value / 2**20
    except Exception as exc:
        return dict(result, reason="allocator_sampling_failed", detail=f"{type(exc).__name__}: {exc}")
    result["available"] = True
    return result


__all__ = ["get_live_allocator_status"]
