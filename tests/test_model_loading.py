"""CPU thread regressions for HF process-default dtype and native FP32 ownership."""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from core.model_loading import model_loading_guard, validate_parameter_dtype


@contextlib.contextmanager
def preserve_default_dtype():
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        yield
    finally:
        torch.set_default_dtype(original)


def test_waiter_captures_previous_dtype_only_after_acquiring_shared_lock(monkeypatch):
    owner_entered, release_owner = threading.Event(), threading.Event()
    waiter_started, waiter_entered = threading.Event(), threading.Event()
    reads = []
    get_dtype = torch.get_default_dtype
    def observed_get():
        value = get_dtype()
        reads.append((threading.current_thread().name, value))
        return value
    monkeypatch.setattr(torch, "get_default_dtype", observed_get)
    def owner():
        threading.current_thread().name = "guard-owner"
        with model_loading_guard():
            torch.set_default_dtype(torch.bfloat16)
            owner_entered.set()
            assert release_owner.wait(3)
    def waiter():
        threading.current_thread().name = "guard-waiter"
        waiter_started.set()
        with model_loading_guard(dtype=torch.float32):
            waiter_entered.set()
            assert get_dtype() == torch.float32
    with preserve_default_dtype(), ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(owner)
        assert owner_entered.wait(3)
        second = pool.submit(waiter)
        assert waiter_started.wait(3)
        try:
            time.sleep(.025)
            assert not waiter_entered.is_set()
            assert not [row for row in reads if row[0] == "guard-waiter"]
        finally:
            release_owner.set()
        first.result(3)
        second.result(3)
        assert [value for name, value in reads if name == "guard-waiter"] == [torch.float32]
        assert get_dtype() == torch.float32


def test_guard_restores_default_on_exception_and_reentrant_nested_context():
    with preserve_default_dtype():
        with model_loading_guard(dtype=torch.float64):
            assert torch.get_default_dtype() == torch.float64
            with model_loading_guard(dtype=torch.bfloat16):
                assert torch.get_default_dtype() == torch.bfloat16
            assert torch.get_default_dtype() == torch.float64
            with pytest.raises(RuntimeError, match="controlled failure"):
                with model_loading_guard(dtype=torch.float32):
                    torch.set_default_dtype(torch.bfloat16)  # emulate a bad third-party loader
                    raise RuntimeError("controlled failure")
            assert torch.get_default_dtype() == torch.float64
        assert torch.get_default_dtype() == torch.float32


def test_parameter_validation_checks_all_weights_and_preserves_fp32_buffers():
    model = torch.nn.Linear(2, 2).to(dtype=torch.bfloat16)
    model.register_buffer("intentional_fp32", torch.tensor([.123456], dtype=torch.float32))
    result = validate_parameter_dtype(model, torch.bfloat16)
    assert result["floating_parameter_dtype_counts"] == {"torch.bfloat16": 2}
    assert result["buffers_cast_or_checked"] is False
    assert model.intentional_fp32.dtype == torch.float32
    model.bias = torch.nn.Parameter(model.bias.float())
    with pytest.raises(RuntimeError, match="bias=torch.float32"):
        validate_parameter_dtype(model, torch.bfloat16)


def tiny_hf_model(tmp_path, monkeypatch):
    # Transformers' installed-backend discovery imports torch_npu even with
    # autoload disabled. Only disable that availability probe; the real HF
    # from_pretrained/default-dtype/checkpoint path stays untouched.
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    monkeypatch.setattr(utils, "is_torch_npu_available", lambda *args, **kwargs: False)
    monkeypatch.setattr(imports, "is_torch_npu_available", lambda *args, **kwargs: False)
    from transformers import PreTrainedModel
    from transformers.configuration_utils import PreTrainedConfig
    class TinyConfig(PreTrainedConfig):
        model_type = "project_dtype_guard_cpu_test"
    class TinyModel(PreTrainedModel):
        config_class = TinyConfig
        gate = None
        constructor_records = []
        def __init__(self, config):
            super().__init__(config)
            gate = type(self).gate
            if gate is not None:
                type(self).constructor_records.append((threading.current_thread().name, torch.get_default_dtype()))
                gate["entered"].set()
                assert gate["release"].wait(3)
                if gate.get("fail"):
                    raise RuntimeError("controlled HF constructor failure")
            self.linear = torch.nn.Linear(2, 2)
            self.register_buffer("intentional_fp32", torch.tensor([.123456], dtype=torch.float32), persistent=False)
            self.post_init()
        def forward(self, input_ids=None):
            return self.linear(input_ids)
    directory = tmp_path / "tiny-hf"
    TinyModel(TinyConfig()).save_pretrained(directory)
    return TinyModel, directory


def test_real_hf_bf16_load_serializes_native_fp32_construct_and_checkpoint_copy(tmp_path, monkeypatch):
    with preserve_default_dtype():
        Tiny, directory = tiny_hf_model(tmp_path, monkeypatch)
        gate = {"entered": threading.Event(), "release": threading.Event()}
        Tiny.gate = gate
        native_waiting, native_entered = threading.Event(), threading.Event()
        state = {"weight": torch.tensor([[.1234567, 1.00001], [-.234567, .333333]], dtype=torch.float32),
                 "bias": torch.tensor([.1234567, -.345678], dtype=torch.float32)}
        def hf():
            with model_loading_guard():
                model = Tiny.from_pretrained(directory, dtype=torch.bfloat16, local_files_only=True)
                validate_parameter_dtype(model, torch.bfloat16)
                assert model.intentional_fp32.dtype == torch.float32
                return model
        def native():
            native_waiting.set()
            with model_loading_guard(dtype=torch.float32):
                native_entered.set()
                model = torch.nn.Linear(2, 2).float()  # before load_state_dict
                model.load_state_dict(state)
                validate_parameter_dtype(model, torch.float32)
                return model
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(hf)
            assert gate["entered"].wait(3)
            assert torch.get_default_dtype() == torch.bfloat16
            second = pool.submit(native)
            assert native_waiting.wait(3)
            try:
                time.sleep(.025)
                assert not native_entered.is_set()
            finally:
                gate["release"].set()
            first.result(5)
            actual = second.result(5)
        assert torch.get_default_dtype() == torch.float32
        assert torch.equal(actual.weight, state["weight"])
        assert torch.equal(actual.bias, state["bias"])


def test_two_real_hf_bf16_loads_share_guard_and_keep_every_weight_bf16(tmp_path, monkeypatch):
    with preserve_default_dtype():
        Tiny, directory = tiny_hf_model(tmp_path, monkeypatch)
        gate = {"entered": threading.Event(), "release": threading.Event()}
        Tiny.gate = gate
        second_started = threading.Event()
        def load(name):
            threading.current_thread().name = name
            if name == "second-HF":
                second_started.set()
            with model_loading_guard():
                model = Tiny.from_pretrained(directory, dtype=torch.bfloat16, local_files_only=True)
                validate_parameter_dtype(model, torch.bfloat16)
                return model
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(load, "first-HF")
            assert gate["entered"].wait(3)
            second = pool.submit(load, "second-HF")
            assert second_started.wait(3)
            try:
                time.sleep(.025)
                assert Tiny.constructor_records == [("first-HF", torch.bfloat16)]
            finally:
                gate["release"].set()
            models = [first.result(5), second.result(5)]
        assert Tiny.constructor_records == [("first-HF", torch.bfloat16), ("second-HF", torch.bfloat16)]
        assert torch.get_default_dtype() == torch.float32
        assert all(parameter.dtype == torch.bfloat16 for model in models for parameter in model.parameters())
        assert all(model.intentional_fp32.dtype == torch.float32 for model in models)


def test_real_hf_constructor_error_restores_global_default_dtype(tmp_path, monkeypatch):
    with preserve_default_dtype():
        Tiny, directory = tiny_hf_model(tmp_path, monkeypatch)
        gate = {"entered": threading.Event(), "release": threading.Event(), "fail": True}
        gate["release"].set()
        Tiny.gate = gate
        with pytest.raises(RuntimeError, match="controlled HF constructor failure"):
            with model_loading_guard():
                Tiny.from_pretrained(directory, dtype=torch.bfloat16, local_files_only=True)
        assert gate["entered"].is_set()
        assert torch.get_default_dtype() == torch.float32


def test_qwen_lazy_transformers_resolver_waits_for_shared_import_guard(tmp_path, monkeypatch):
    """FunASR还在导入HF时，Qwen不得在锁外查询HF懒加载属性。"""
    from types import ModuleType
    from core.qwen_asr_runtime import LocalQwenRuntime
    owner_entered, release_owner = threading.Event(), threading.Event()
    load_started, resolved = threading.Event(), threading.Event()
    ready = []
    api = SimpleNamespace(set_device=lambda d: None, device=lambda d: contextlib.nullcontext(),
                          synchronize=lambda d: None, is_available=lambda: True, device_count=lambda: 8,
                          memory_allocated=lambda d: 0, memory_reserved=lambda d: 0,
                          conv=SimpleNamespace(allow_hf32=False), matmul=SimpleNamespace(allow_hf32=False))
    model = torch.nn.Linear(2, 2).to(dtype=torch.bfloat16)
    model.to = lambda *args, **kwargs: model  # 本用例不能分配NPU张量
    module = ModuleType("transformers")
    def resolve(name):
        if name in {"AutoProcessor", "Qwen3ASRForConditionalGeneration"}:
            resolved.set()
            assert ready, "HF lazy resolver accessed while the other loader still owns initialization"
            value = object() if name == "AutoProcessor" else model
            return SimpleNamespace(from_pretrained=lambda *args, **kwargs: value)
        raise AttributeError(name)
    module.__getattr__ = resolve
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(torch, "npu", api, raising=False)
    runtime = LocalQwenRuntime(tmp_path, "npu:7")
    def import_owner():
        with model_loading_guard():
            owner_entered.set()
            assert release_owner.wait(3)
            ready.append(True)
    def qwen_load():
        load_started.set()
        runtime.ensure_loaded()
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(import_owner)
        assert owner_entered.wait(3)
        waiter = pool.submit(qwen_load)
        assert load_started.wait(3)
        try:
            time.sleep(.025)
            assert not resolved.is_set()
        finally:
            release_owner.set()
        owner.result(5)
        waiter.result(5)
    assert resolved.is_set() and runtime.loaded
    assert runtime.parameter_dtype_check["floating_parameter_dtype_counts"] == {"torch.bfloat16": 2}


def test_loaded_qwen_protects_only_cpu_processor_not_model_forward(tmp_path, monkeypatch):
    """热实例的CPU Hann等factory不能与另一实例BF16初始化窗口交叠。"""
    import io
    import wave
    import numpy as np
    from core.qwen_asr_runtime import LocalQwenRuntime
    from core import qwen_asr_runtime as runtime_module
    owner_entered, release_owner = threading.Event(), threading.Event()
    started, processor_entered = threading.Event(), threading.Event()
    entered_dtypes = []
    runtime = LocalQwenRuntime(tmp_path, "npu:7")
    runtime.api = SimpleNamespace(set_device=lambda d: None, device=lambda d: contextlib.nullcontext(), synchronize=lambda d: None)
    inputs = {"input_ids": torch.tensor([[1, 2]])}
    def prepare(**kwargs):
        entered_dtypes.append(torch.get_default_dtype())
        processor_entered.set()
        return inputs
    def generate(**kwargs):
        # 独立线程此时仍能取得全局guard，证明NPU前向没有持有它。
        acquired = threading.Event()
        def acquire():
            with model_loading_guard():
                acquired.set()
        thread = threading.Thread(target=acquire)
        thread.start()
        assert acquired.wait(3), "global model loading guard incorrectly held through model forward"
        thread.join(3)
        return torch.tensor([[1, 2, 3]])
    runtime.model = SimpleNamespace(dtype=torch.bfloat16, generate=generate)
    runtime.processor = SimpleNamespace(apply_transcription_request=prepare,
                                         batch_decode=lambda *args, **kwargs: ["language English<asr_text>done"])
    monkeypatch.setattr(runtime_module, "move_inputs", lambda *args, **kwargs: inputs)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setparams((1, 2, 16000, 0, "NONE", ""))
        writer.writeframes(np.zeros(1600, dtype="<i2").tobytes())
    def owner():
        with model_loading_guard():
            torch.set_default_dtype(torch.bfloat16)
            owner_entered.set()
            assert release_owner.wait(3)
    def transcribe():
        started.set()
        return runtime.transcribe(buffer.getvalue())
    with preserve_default_dtype(), ThreadPoolExecutor(max_workers=2) as pool:
        loading = pool.submit(owner)
        assert owner_entered.wait(3)
        working = pool.submit(transcribe)
        assert started.wait(3)
        try:
            time.sleep(.025)
            assert not processor_entered.is_set()
        finally:
            release_owner.set()
        loading.result(5)
        assert working.result(5) == "language English<asr_text>done"
        assert entered_dtypes == [torch.float32]
        assert torch.get_default_dtype() == torch.float32


def test_real_whisper_hann_frontend_stays_fp32_under_guard(tmp_path, monkeypatch):
    import numpy as np
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    monkeypatch.setattr(utils, "is_torch_npu_available", lambda *args, **kwargs: False)
    monkeypatch.setattr(imports, "is_torch_npu_available", lambda *args, **kwargs: False)
    from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
    extractor = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, n_fft=400, hop_length=160, dither=0.)
    audio = (np.sin(np.arange(6560, dtype=np.float32) * .03) * .1).reshape(1, -1)
    windows = []
    create_window = torch.hann_window
    def window(*args, **kwargs):
        value = create_window(*args, **kwargs)
        windows.append(value.dtype)
        return value
    monkeypatch.setattr(torch, "hann_window", window)
    with preserve_default_dtype():
        reference = extractor._torch_extract_fbank_features(audio, device="cpu")
        torch.set_default_dtype(torch.bfloat16)
        with model_loading_guard(dtype=torch.float32):
            actual = extractor._torch_extract_fbank_features(audio, device="cpu")
        assert torch.get_default_dtype() == torch.bfloat16
    assert windows == [torch.float32, torch.float32]
    np.testing.assert_array_equal(actual, reference)


def test_real_qwen_frontend_waits_for_hf_bf16_window_and_matches_fp32_reference(tmp_path, monkeypatch):
    """实际Qwen前端必须先等HF临时BF16窗口结束，再构造相同FP32特征。"""
    import numpy as np
    from core.qwen_asr_runtime import move_inputs
    with preserve_default_dtype():
        Tiny, directory = tiny_hf_model(tmp_path, monkeypatch)
        from transformers.models.qwen3_asr.feature_extraction_qwen3_asr import Qwen3ASRFeatureExtractor
        extractor = Qwen3ASRFeatureExtractor(feature_size=128, sampling_rate=16000, n_fft=400,
                                              hop_length=160, dither=0.)
        audio = (np.sin(np.arange(65600, dtype=np.float32) * .03) * .1)
        reference = extractor(audio, sampling_rate=16000, padding="longest", return_tensors="pt", return_attention_mask=True)
        gate = {"entered": threading.Event(), "release": threading.Event()}
        Tiny.gate = gate
        waiting, prepared = threading.Event(), threading.Event()
        window_dtypes = []
        create_window = torch.hann_window
        def window(*args, **kwargs):
            value = create_window(*args, **kwargs)
            window_dtypes.append(value.dtype)
            return value
        monkeypatch.setattr(torch, "hann_window", window)
        def load():
            with model_loading_guard():
                return Tiny.from_pretrained(directory, dtype=torch.bfloat16, local_files_only=True)
        def frontend():
            waiting.set()
            with model_loading_guard(dtype=torch.float32):
                value = extractor(audio, sampling_rate=16000, padding="longest", return_tensors="pt", return_attention_mask=True)
                prepared.set()
                return value
        with ThreadPoolExecutor(max_workers=2) as pool:
            loading = pool.submit(load)
            assert gate["entered"].wait(3)
            assert torch.get_default_dtype() == torch.bfloat16
            working = pool.submit(frontend)
            assert waiting.wait(3)
            try:
                time.sleep(.025)
                assert not prepared.is_set()
                assert window_dtypes == []
            finally:
                gate["release"].set()
            loading.result(5)
            actual = working.result(5)
        assert torch.get_default_dtype() == torch.float32
        assert window_dtypes == [torch.float32]
        assert actual["input_features"].dtype == torch.float32
        assert torch.equal(actual["input_features"], reference["input_features"])
        assert torch.equal(actual["attention_mask"], reference["attention_mask"])
        moved = move_inputs(actual, device="cpu", dtype=torch.bfloat16)
        assert moved["input_features"].dtype == torch.bfloat16
        assert moved["attention_mask"].dtype == actual["attention_mask"].dtype == torch.int32
        assert torch.equal(moved["attention_mask"], reference["attention_mask"])
