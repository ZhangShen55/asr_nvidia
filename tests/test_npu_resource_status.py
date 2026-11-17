"""No torch/NPU import is needed to verify live allocator status boundaries."""
import builtins
import sys
from types import SimpleNamespace

import pytest

from core.npu_resource_status import get_live_allocator_status


class FakeApi:
    def __init__(self, initialized):
        self.initialized = initialized
        self.calls = []
    def is_initialized(self):
        self.calls.append(("is_initialized",))
        return self.initialized
    def set_device(self, device):
        self.calls.append(("set_device", device))
    def memory_allocated(self, device):
        self.calls.append(("allocated", device))
        return 2**20
    def memory_reserved(self, device):
        self.calls.append(("reserved", device))
        return 3 * 2**20
    def max_memory_allocated(self, device):
        self.calls.append(("peak_allocated", device))
        return 2 * 2**20
    def max_memory_reserved(self, device):
        self.calls.append(("peak_reserved", device))
        return 4 * 2**20


def test_absent_torch_does_not_import_or_create_context(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    imported = []
    original_import = builtins.__import__
    def guard(name, *args, **kwargs):
        imported.append(name)
        assert name not in {"torch", "torch_npu"}
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guard)
    result = get_live_allocator_status("npu:7")
    assert result["available"] is False and result["reason"] == "torch_not_imported"
    assert result["allocated_bytes"] is result["reserved_bytes"] is None
    assert "torch" not in sys.modules and not imported


def test_existing_torch_without_npu_does_not_trigger_lazy_attributes(monkeypatch):
    class LazyTorch:
        def __getattr__(self, name):
            raise AssertionError(f"lazy attribute must not be queried: {name}")
    monkeypatch.setitem(sys.modules, "torch", LazyTorch())
    result = get_live_allocator_status("npu:7")
    assert result["reason"] == "npu_api_not_present" and result["available"] is False


def test_uninitialized_npu_never_selects_device_or_samples(monkeypatch):
    api = FakeApi(False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=api))
    result = get_live_allocator_status("npu:7")
    assert result["reason"] == "npu_not_initialized"
    assert result["initialized"] is False
    assert api.calls == [("is_initialized",)]


@pytest.mark.parametrize("device", ["npu:7", "npu:0"])
def test_initialized_allocator_selects_only_configured_host_or_mapped_target(monkeypatch, device):
    api = FakeApi(True)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=api))
    first = get_live_allocator_status(device)
    assert first["available"] is True and first["initialized"] is True
    assert api.calls == [("is_initialized",), ("set_device", device), ("allocated", device),
                         ("reserved", device), ("peak_allocated", device), ("peak_reserved", device)]
    assert [first[name + "_mb"] for name in ("allocated", "reserved", "peak_allocated", "peak_reserved")] == [1, 3, 2, 4]
    assert first["scope"] == "current_service_process_configured_device_pytorch_allocator"
    assert "not per-model" in first["excludes"] and "non-PyTorch" in first["excludes"]
    api.memory_allocated = lambda d: 2 * 2**20
    second = get_live_allocator_status(device)
    assert second["allocated_mb"] == 2  # live, not a cached loading-window estimate


@pytest.mark.parametrize("device", ["cpu", "cuda:7", "npu", "npu:-1", "npu:7junk", None, 7])
def test_invalid_device_is_diagnostic_without_touching_torch(monkeypatch, device):
    api = FakeApi(True)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=api))
    result = get_live_allocator_status(device)
    assert result["reason"] == "invalid_device" and not result["available"]
    assert not api.calls


def test_sampler_failure_does_not_fall_back_to_another_card_or_fake_zero(monkeypatch):
    api = FakeApi(True)
    def fail(device):
        raise RuntimeError("allocator unavailable")
    api.memory_allocated = fail
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=api))
    result = get_live_allocator_status("npu:7")
    assert result["reason"] == "allocator_sampling_failed" and not result["available"]
    assert result["allocated_bytes"] is None
    assert "allocator unavailable" in result["detail"]
    assert api.calls == [("is_initialized",), ("set_device", "npu:7")]


def test_missing_peak_api_reports_incomplete_sample(monkeypatch):
    api = FakeApi(True)
    api.max_memory_reserved = None
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=api))
    result = get_live_allocator_status("npu:7")
    assert result["available"] is False
    assert result["peak_reserved_mb"] is None
    assert result["allocated_mb"] == 1


def test_default_dtype_metadata_is_read_without_importing_or_initializing_npu(monkeypatch):
    api = FakeApi(False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=api, get_default_dtype=lambda: "torch.float32"))
    result = get_live_allocator_status("npu:7")
    assert result["torch_default_dtype"] == "torch.float32"
    assert result["reason"] == "npu_not_initialized"
    assert api.calls == [("is_initialized",)]
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(get_default_dtype=lambda: "torch.bfloat16"))
    result = get_live_allocator_status("npu:7")
    assert result["torch_default_dtype"] == "torch.bfloat16"
    assert result["reason"] == "npu_api_not_present"


def test_default_dtype_metadata_missing_on_cpu_fake_is_unknown(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=FakeApi(False)))
    result = get_live_allocator_status("npu:7")
    assert result["torch_default_dtype"] is None
    assert result["available"] is False
