"""NPU 池化替代算法可在 CPU 对照原始算子，避免用维度正确冒充数值等价。"""
import unittest
from unittest.mock import Mock, patch
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from core.local_inference import segment_pooling, adapt_campplus_pooling, backend_for_device, synchronize_npu, InstanceFailure, device_scope, configure_npu_precision, adapt_emotion_gelu
from core.models import device_allocated_mb


class PoolingEquivalenceTests(unittest.TestCase):
    def test_ceil_pooling_tail_and_negative_max_match_original(self):
        generator = torch.Generator().manual_seed(42)
        for length in (1, 20, 99, 100, 101, 199, 200, 201, 1201):
            x = -torch.rand((2, 5, length), generator=generator)
            for kind, original in (("avg", F.avg_pool1d), ("max", F.max_pool1d)):
                with self.subTest(length=length, kind=kind):
                    reference = original(x, kernel_size=100, stride=100, ceil_mode=True)
                    reference = reference.unsqueeze(-1).expand(*reference.shape, 100).reshape(2, 5, -1)[..., :length]
                    actual = segment_pooling(x, 100, kind)
                    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)

    def test_patch_is_instance_local(self):
        class Layer(torch.nn.Module):
            def seg_pooling(self, x, seg_len=100, stype="avg"):
                return x + 1
        first, untouched = Layer(), Layer()
        adapt_campplus_pooling(first)
        x = torch.full((1, 1, 2), -2.0)
        self.assertTrue(torch.equal(first.seg_pooling(x), x))
        self.assertTrue(torch.equal(untouched.seg_pooling(x), x + 1))

    def test_empty_or_invalid_pooling_is_explicit(self):
        for x, size, kind in ((torch.zeros(1, 2, 0), 100, "avg"),
                              (torch.zeros(1, 2, 3), 0, "avg"),
                              (torch.zeros(1, 2, 3), 100, "median")):
            with self.assertRaises(ValueError):
                segment_pooling(x, size, kind)


class DeviceMemoryTests(unittest.TestCase):
    def test_target_is_selected_before_context_or_allocator_initialization(self):
        from contextlib import nullcontext
        events = []

        class Api:
            def __init__(self):
                from types import SimpleNamespace
                self.conv = SimpleNamespace(allow_hf32=True)
                self.matmul = SimpleNamespace(allow_hf32=False)
            def set_device(self, device):
                events.append(("set_device", device))
            def device(self, device):
                self_test.assertEqual(events[-1], ("set_device", device))
                events.append(("context", device))
                return nullcontext()
            def memory_allocated(self, device):
                self_test.assertEqual(events[-1], ("set_device", device))
                events.append(("memory", device))
                return 1048576

        self_test = self
        api = Api()
        with patch("core.local_inference.npu_api", return_value=api):
            with device_scope("npu:7"):
                pass
        with patch("core.models.npu_api", return_value=api):
            self.assertEqual(device_allocated_mb("npu:7"), 1.0)
        self.assertEqual(events, [("set_device", "npu:7"), ("context", "npu:7"),
                                  ("set_device", "npu:7"), ("memory", "npu:7")])

    def test_strict_fp32_math_mode_is_process_once_and_thread_safe(self):
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace
        from unittest.mock import Mock

        class Flag:
            def __init__(self, initial):
                self.value, self.writes = initial, 0
            @property
            def allow_hf32(self):
                return self.value
            @allow_hf32.setter
            def allow_hf32(self, value):
                self.writes += 1
                self.value = value

        api = SimpleNamespace(set_device=Mock(), conv=Flag(True), matmul=Flag(False))
        with ThreadPoolExecutor(max_workers=4) as executor:
            reports = list(executor.map(lambda _: configure_npu_precision(api, "npu:7"), range(8)))
        self.assertEqual(api.conv.writes, 1)
        self.assertEqual(api.matmul.writes, 1)
        self.assertTrue(all(row["conv_allow_hf32"] is False and row["matmul_allow_hf32"] is False for row in reports))
        self.assertTrue(all(call.args == ("npu:7",) for call in api.set_device.call_args_list))

    def test_failed_device_selection_invalidates_the_instance(self):
        from unittest.mock import Mock
        api = Mock()
        api.set_device.side_effect = RuntimeError("device unavailable")
        with patch("core.local_inference.npu_api", return_value=api), self.assertRaises(InstanceFailure):
            device_scope("npu:7")
        api.device.assert_not_called()

    def test_cpu_does_not_load_npu_and_sampling_failure_is_unknown(self):
        with patch("core.models.npu_api", side_effect=AssertionError("must not load")):
            self.assertEqual(device_allocated_mb("cpu"), 0.0)
            self.assertIsNone(device_allocated_mb("npu:7"))

    def test_invalid_and_legacy_device_names_fail(self):
        for name in ("cuda:0", "npu:-1", "npu:x", "npu", "gpu:0"):
            with self.assertRaises(ValueError):
                backend_for_device(name)


class DeviceCompletionTests(unittest.TestCase):
    def test_async_device_error_is_instance_failure(self):
        api = Mock()
        api.synchronize.side_effect = RuntimeError("asynchronous kernel failed")
        with self.assertRaisesRegex(InstanceFailure, "NPU 同步失败"):
            synchronize_npu(api, "npu:7")


class EmotionExactGeluTests(unittest.TestCase):
    def test_patch_is_instance_local_preserves_weights_and_tanh_mode(self):
        import torch
        import torch.nn.functional as F
        original_global = F.gelu
        original_class = torch.nn.GELU.forward
        first = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.GELU(), torch.nn.GELU(approximate="tanh"))
        other = torch.nn.GELU()
        state = {key: value.clone() for key, value in first.state_dict().items()}
        adapt_emotion_gelu(first)
        self.assertEqual(first._npu_exact_gelu_count, 1)
        self.assertIn("forward", first[1].__dict__)
        self.assertNotIn("forward", first[2].__dict__)
        self.assertNotIn("forward", other.__dict__)
        self.assertIs(F.gelu, original_global)
        self.assertIs(torch.nn.GELU.forward, original_class)
        self.assertEqual(set(state), set(first.state_dict()))
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, state[key]))

    def test_cpu_exact_math_unchanged_for_negative_and_extreme_values(self):
        import math
        import torch
        import torch.nn.functional as F
        x = torch.cat((torch.linspace(-8, 8, 1001), torch.tensor([-100.0, 0.0, 100.0])))
        reference = F.gelu(x, approximate="none")
        module = adapt_emotion_gelu(torch.nn.Sequential(torch.nn.GELU()))
        with patch("core.local_inference.npu_api", side_effect=AssertionError("CPU imported NPU")):
            actual = module(x)
        self.assertTrue(torch.equal(actual, reference))
        mathematical = 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
        torch.testing.assert_close(actual, mathematical, rtol=1e-5, atol=1e-6)

    def test_npu_custom_schema_explicitly_requests_none_without_global_patch(self):
        import torch
        from types import SimpleNamespace
        module = adapt_emotion_gelu(torch.nn.Sequential(torch.nn.GELU()))[0]
        value = SimpleNamespace(device=SimpleNamespace(type="npu"))
        sentinel = object()
        with patch.object(torch.ops.npu, "npu_gelu", return_value=sentinel, create=True) as op:
            self.assertIs(module(value), sentinel)
        op.assert_called_once_with(value, approximate="none")
