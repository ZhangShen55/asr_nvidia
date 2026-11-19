"""Real CPU CAMPPlus construction and controlled HF dtype windows; no NPU/model assets."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import transformers.utils as hf_utils
import transformers.utils.import_utils as hf_imports

# The installed Transformers backend probe imports torch_npu even when autoload
# is disabled. Disable only that probe; use the real HF dtype context and real
# FunASR CAMPPlus constructor, checkpoint loader and CPU tensors below.
with patch.object(hf_utils, "is_torch_npu_available", return_value=False), \
        patch.object(hf_imports, "is_torch_npu_available", return_value=False):
    from transformers.modeling_utils import local_torch_dtype
    from funasr.models.campplus.model import CAMPPlus

from core.local_inference import load_native_network, TorchRunner, validate_fp32_network
from core.model_loading import model_loading_guard
from core.config import settings
from core.models import ModelPool


SMALL_CONFIG = {"feat_dim": 8, "embedding_size": 4, "growth_rate": 2,
                "bn_size": 1, "init_channels": 8, "memory_efficient": False}


@contextmanager
def held_hf_bf16_window():
    """The actual HF context changes process-global dtype; hold it deterministically."""
    entered, release = threading.Event(), threading.Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        def hf():
            with model_loading_guard():
                with local_torch_dtype(torch.bfloat16):
                    entered.set()
                    if not release.wait(5):
                        raise TimeoutError("controlled HF dtype window")
        future = executor.submit(hf)
        if not entered.wait(2):
            release.set()
            raise AssertionError("HF dtype window did not start")
        try:
            yield release
        finally:
            release.set()
            future.result(timeout=5)


class NativeLoadingDtypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(4)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.previous = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, self.previous)
        torch.set_default_dtype(torch.float32)
        directory = tempfile.TemporaryDirectory(prefix="native-dtype-")
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        (self.directory / "config.yaml").write_text(json.dumps({"model_conf": SMALL_CONFIG}))
        with model_loading_guard(dtype=torch.float32):
            network = CAMPPlus(**SMALL_CONFIG).float()
            self.expected = {key: value.clone() for key, value in network.state_dict().items()}
            self.key = next(name for name, parameter in network.named_parameters() if parameter.is_floating_point())
            # These are not BF16-representable: load into BF16 followed by float
            # cannot pass the exact checkpoint-value assertion below.
            self.expected[self.key].reshape(-1)[:4] = torch.tensor([.1234567, -.2345678, .3456789, -.4567891])
            torch.save({"state_dict": self.expected}, self.directory / "campplus_cn_en_common.pt")

    def tearDown(self):
        torch.set_default_dtype(self.previous)

    def assert_native_fp32_and_exact(self, network):
        self.assertIsInstance(network, CAMPPlus)
        self.assertTrue(validate_fp32_network(network))
        for name, value in network.state_dict().items():
            if value.is_floating_point():
                self.assertEqual(value.dtype, torch.float32)
            else:
                self.assertEqual(value.dtype, self.expected[name].dtype)
            self.assertTrue(torch.equal(value, self.expected[name]), name)

    def test_hf_bf16_window_cannot_pollute_actual_native_cpu_construction(self):
        attempting, constructed = threading.Event(), threading.Event()
        observed = []

        def construct(**config):
            constructed.set()
            observed.append(torch.get_default_dtype())
            return CAMPPlus(**config)

        def load():
            attempting.set()
            return load_native_network("speech_campplus", self.directory)

        with patch("funasr.models.campplus.model.CAMPPlus", side_effect=construct):
            with held_hf_bf16_window() as release, ThreadPoolExecutor(max_workers=1) as executor:
                self.assertEqual(torch.get_default_dtype(), torch.bfloat16)
                pending = executor.submit(load)
                self.assertTrue(attempting.wait(2))
                self.assertFalse(constructed.wait(.08))
                self.assertFalse(pending.done())
                release.set()
                network = pending.result(timeout=5)
        self.assertEqual(observed, [torch.float32])
        self.assertEqual(torch.get_default_dtype(), torch.float32)
        self.assert_native_fp32_and_exact(network)

    def test_float_before_state_copy_preserves_checkpoint_if_constructor_returns_bf16(self):
        def explicitly_half(**config):
            return CAMPPlus(**config).to(dtype=torch.bfloat16)
        with patch("funasr.models.campplus.model.CAMPPlus", side_effect=explicitly_half):
            network = load_native_network("speech_campplus", self.directory)
        self.assert_native_fp32_and_exact(network)

    def test_cpu_automodel_factory_uses_same_guard_and_reports_verified_storage(self):
        attempting, constructed = threading.Event(), threading.Event()
        observed = []

        def factory(**kwargs):
            observed.append(torch.get_default_dtype())
            constructed.set()
            model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
            return SimpleNamespace(model=model, generate=lambda **kw: [])

        pool = ModelPool("speech_campplus", "synthetic", devices_getter=lambda: ["cpu"])
        def load():
            attempting.set()
            return pool._build_instance("cpu")
        # 这里只测真实CPU工厂与HF dtype锁，明确选择原thread构造路径。
        thread_pool = dict(settings._cfg.get("model_device_pool", {}), speech_campplus_execution_mode="thread")
        with patch.dict(settings._cfg, {"model_device_pool": thread_pool}), \
                patch("core.models.AutoModel", side_effect=factory):
            with held_hf_bf16_window() as release, ThreadPoolExecutor(max_workers=1) as executor:
                pending = executor.submit(load)
                self.assertTrue(attempting.wait(2))
                self.assertFalse(constructed.wait(.08))
                self.assertFalse(pending.done())
                release.set()
                runner, device, allocated = pending.result(timeout=5)
        self.assertEqual(observed, [torch.float32])
        self.assertEqual(torch.get_default_dtype(), torch.float32)
        self.assertEqual((device, allocated, runner.actual_precision), ("cpu", 0.0, "fp32"))
        self.assertTrue(validate_fp32_network(runner.model.model))
        runner.close()

    def test_mismatched_floating_buffer_is_rejected_before_fp32_claim(self):
        network = torch.nn.Linear(2, 2)
        network.register_buffer("floating_cache", torch.ones(2, dtype=torch.bfloat16))
        with self.assertRaisesRegex(ValueError, "buffer:floating_cache=torch.bfloat16"):
            TorchRunner(SimpleNamespace(model=network), "cpu")
        network.float()
        network.weight = torch.nn.Parameter(network.weight.to(dtype=torch.bfloat16))
        with self.assertRaisesRegex(ValueError, "parameter:weight=torch.bfloat16"):
            validate_fp32_network(network)

    def test_opaque_compatibility_handle_does_not_claim_verified_fp32(self):
        runner = TorchRunner(SimpleNamespace(generate=lambda **kwargs: []), "cpu")
        self.assertIsNone(runner.actual_precision)


class InferenceDtypeBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(4)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        self.addCleanup(torch.set_default_dtype, self.previous)

    def during_hf_window(self, operation, entered):
        attempting = threading.Event()
        def call():
            attempting.set()
            return operation()
        with held_hf_bf16_window() as release, ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(call)
            try:
                self.assertTrue(attempting.wait(2))
                self.assertFalse(entered.wait(.08))
                self.assertFalse(pending.done())
                self.assertEqual(torch.get_default_dtype(), torch.bfloat16)
            finally:
                release.set()
            result = pending.result(timeout=5)
        self.assertEqual(torch.get_default_dtype(), torch.float32)
        return result

    def test_cam_fbank_waits_for_hf_window_and_matches_original_fp32_features(self):
        import numpy as np
        from torchaudio.compliance.kaldi import fbank
        from core.local_inference import campplus_features
        wave = torch.from_numpy(np.sin(np.arange(16000, dtype=np.float32) * np.float32(.07)))
        expected = campplus_features(wave)
        entered = threading.Event()
        def observed(*args, **kwargs):
            entered.set()
            return fbank(*args, **kwargs)
        with patch("torchaudio.compliance.kaldi.fbank", side_effect=observed):
            actual = self.during_hf_window(lambda: campplus_features(wave), entered)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.dtype, torch.float32)

    def test_emotion_bias_factory_cache_is_instance_local_and_cannot_round_in_bf16_window(self):
        from functools import partial
        from funasr.models.emotion2vec.base import get_alibi_bias
        from core.local_inference import adapt_emotion_dtype_boundaries
        expected = get_alibi_bias({}, 1, 98, 16, torch.float32, "cpu")
        entered, cache = threading.Event(), {}
        def observed(*args, **kwargs):
            entered.set()
            return get_alibi_bias(*args, **kwargs)
        encoder, other = torch.nn.Module(), torch.nn.Module()
        encoder.get_alibi_bias = partial(observed, alibi_biases=cache)
        other.get_alibi_bias = partial(get_alibi_bias, alibi_biases={})
        original_other = other.get_alibi_bias
        network = torch.nn.Sequential(encoder)
        adapt_emotion_dtype_boundaries(network)
        wrapper = encoder.get_alibi_bias
        adapt_emotion_dtype_boundaries(network)
        self.assertIs(encoder.get_alibi_bias, wrapper)
        self.assertIs(other.get_alibi_bias, original_other)
        actual = self.during_hf_window(
            lambda: encoder.get_alibi_bias(batch_size=1, time_steps=98, heads=16, dtype=torch.float32, device="cpu"), entered)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(cache)
        # The cached values remain exact after HF restores the process default.
        self.assertTrue(torch.equal(encoder.get_alibi_bias(batch_size=1, time_steps=98, heads=16,
                                                         dtype=torch.float32, device="cpu"), expected))
        # Caller-requested low precision remains explicit; the wrapper protects
        # construction constants, not a forced output/weight precision policy.
        expected_half = get_alibi_bias({}, 1, 98, 16, torch.bfloat16, "cpu")
        with model_loading_guard(dtype=torch.bfloat16):
            actual_half = encoder.get_alibi_bias(batch_size=1, time_steps=98, heads=16,
                                                dtype=torch.bfloat16, device="cpu")
            self.assertEqual(torch.get_default_dtype(), torch.bfloat16)
        self.assertEqual(actual_half.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(actual_half, expected_half))

    def test_cpu_automodel_generate_keeps_its_internal_fbank_fp32(self):
        import numpy as np
        from torchaudio.compliance.kaldi import fbank
        wave = torch.from_numpy(np.sin(np.arange(16000, dtype=np.float32) * np.float32(.07)))
        expected = fbank(wave.unsqueeze(0), num_mel_bins=80).mean(dim=0)
        entered = threading.Event()
        def generate(**kwargs):
            entered.set()
            return [{"spk_embedding": fbank(wave.unsqueeze(0), num_mel_bins=80).mean(dim=0)}]
        runner = TorchRunner(SimpleNamespace(model=torch.nn.Linear(2, 2), generate=generate), "cpu")
        try:
            actual = self.during_hf_window(lambda: runner.generate(input=None), entered)[0]["spk_embedding"]
            self.assertTrue(torch.equal(actual, expected))
        finally:
            runner.close()

    def test_npu_network_forward_does_not_hold_the_global_loading_guard(self):
        import contextlib
        import numpy as np
        from core.local_inference import TorchNpuRunner, campplus_features
        entered = threading.Event()
        futures = []
        runner = TorchNpuRunner.__new__(TorchNpuRunner)
        runner.device, runner.name = "npu:7", "speech_campplus"
        runner.api = SimpleNamespace(set_device=lambda device: None, device=lambda device: contextlib.nullcontext(),
                                     synchronize=lambda device: None, conv=SimpleNamespace(allow_hf32=False),
                                     matmul=SimpleNamespace(allow_hf32=False))
        class Feature:
            def to(self, device):
                return self
        def frontend(wave):
            campplus_features(wave)  # exercise the real protected CPU boundary
            return Feature()  # do not transfer a tensor or touch an NPU
        with ThreadPoolExecutor(max_workers=1) as executor:
            def independent_loader():
                with model_loading_guard(dtype=torch.float32):
                    entered.set()
            def forward(features):
                futures.append(executor.submit(independent_loader))
                self.assertTrue(entered.wait(1), "device forward held the global loading guard")
                return torch.ones((1, 192), dtype=torch.float32)
            runner.model = forward
            try:
                with patch("core.local_inference.campplus_features", side_effect=frontend):
                    output = runner.generate(input=np.zeros(16000, dtype=np.float32), fs=16000)
                self.assertEqual(output[0]["spk_embedding"].shape, (1, 192))
            finally:
                # If the assertion failed, generate has unwound before waiting
                # for the other thread, so no cleanup/guard deadlock is hidden.
                for future in futures:
                    future.result(timeout=5)
                runner.close()
