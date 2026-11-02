"""每实例独立 context/stream/buffer；返回前等待设备与 CPU 结果读取完成。"""
import importlib

from core.local_inference import (
    InferenceRunner, InstanceFailure, campplus_features, emotion_labels, waveform_tensor,
)
from core.model_assets import CONTRACTS, validate_manifest


class TensorRTRunner(InferenceRunner):
    effective_backend = "tensorrt"

    def __init__(self, name, model_dir, device, precision):
        self.device = device
        self.context = self.engine = self.runtime = self.stream = None
        self.buffers = {}
        try:
            self._initialize(name, model_dir, device, precision)
        except BaseException:
            self.close()
            raise

    def _initialize(self, name, model_dir, device, precision):
        import torch

        self.device = device
        self.name = name
        self.actual_precision = precision
        self.healthy = True
        self.failure = None
        self.context = self.engine = self.runtime = self.stream = None
        self.buffers = {}
        trt = importlib.import_module("tensorrt")
        with torch.cuda.device(device):
            environment = {"tensorrt": trt.__version__, "cuda": torch.version.cuda,
                           "compute_capability": list(torch.cuda.get_device_capability(device))}
            engine_path, meta = validate_manifest(model_dir, name, precision, environment=environment)
            self.labels = emotion_labels(model_dir) if name == "emotion2vec" else None
            self.logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
            if self.engine is None:
                raise RuntimeError(f"无法反序列化 TensorRT engine: {engine_path}")
            self.contract = CONTRACTS[name]
            tensors = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
            if tensors != {self.contract["input"], *self.contract["outputs"]}:
                raise ValueError(f"engine I/O 名称不匹配: {tensors}")
            for tensor in tensors:
                if self.engine.get_tensor_dtype(tensor) != trt.float32:
                    raise ValueError(f"engine {tensor} I/O 必须为 FP32")
                expected_mode = (trt.TensorIOMode.INPUT if tensor == self.contract["input"]
                                 else trt.TensorIOMode.OUTPUT)
                if self.engine.get_tensor_mode(tensor) != expected_mode:
                    raise ValueError(f"engine {tensor} I/O 方向不匹配")
            shapes = [list(x) for x in self.engine.get_tensor_profile_shape(self.contract["input"], 0)]
            if shapes != self.contract["profile"] or self.engine.num_optimization_profiles != 1:
                raise ValueError(f"engine profile 不匹配: {shapes}")
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("无法创建 TensorRT execution context")
            self.stream = torch.cuda.Stream(device=device)
            self.engine_info = {
                "path": str(engine_path), "sha256": meta["engine_sha256"],
                "compatible": True, "environment": environment,
                "built_at": meta["built_at"], "profile": shapes,
            }
            free, total = torch.cuda.mem_get_info(device)
            self.memory = {
                "scope": "TensorRT context bytes; device usage includes other processes/allocators",
                "context_mb": self.engine.device_memory_size / (1024 ** 2),
                "engine_file_mb": engine_path.stat().st_size / (1024 ** 2),
                "device_used_mb_at_load": (total - free) / (1024 ** 2),
                "allocator_total_mb": None,
                "allocator_total_reason": "TensorRT does not expose all native allocations",
            }

    def _execute(self, cpu):
        import torch

        shape = list(cpu.shape)
        minimum, _, maximum = self.contract["profile"]
        if cpu.dtype != torch.float32 or len(shape) != len(minimum) or any(
            size < lo or size > hi for size, lo, hi in zip(shape, minimum, maximum)
        ):
            raise ValueError(f"TensorRT 输入 dtype/shape 超出 profile: {cpu.dtype} {shape}")
        if not self.healthy:
            raise InstanceFailure(self.failure or "TensorRT 实例不可用")
        try:
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                if not self.context.set_input_shape(self.contract["input"], tuple(shape)):
                    raise ValueError(f"TensorRT 拒绝输入 shape: {shape}")
                # 租约保证本实例互斥；同 shape 缓冲可复用，不与其他实例共享。
                previous = self.buffers.get(self.contract["input"])
                if previous is None or tuple(previous.shape) != tuple(shape):
                    self.buffers[self.contract["input"]] = torch.empty(shape, device=self.device, dtype=torch.float32)
                self.buffers[self.contract["input"]].copy_(cpu)
                for output, expected_shape in self.contract["outputs"].items():
                    if list(self.context.get_tensor_shape(output)) != expected_shape:
                        raise ValueError(f"TensorRT 输出 {output} shape 不匹配")
                    if output not in self.buffers:
                        self.buffers[output] = torch.empty(expected_shape, device=self.device, dtype=torch.float32)
                for tensor, buffer in self.buffers.items():
                    if not self.context.set_tensor_address(tensor, buffer.data_ptr()):
                        raise InstanceFailure(f"TensorRT 绑定 {tensor} buffer 失败")
                if not self.context.execute_async_v3(self.stream.cuda_stream):
                    raise InstanceFailure("TensorRT execute_async_v3 失败")
                # CPU 结果必须与本实例 stream 上的计算保持顺序。
                outputs = {key: value.cpu() for key, value in self.buffers.items()
                           if key in self.contract["outputs"]}
                self.stream.synchronize()
                if not all(bool(torch.isfinite(value).all()) for value in outputs.values()):
                    raise ValueError("TensorRT 输出包含非有限值")
                return outputs
        except ValueError:
            raise
        except Exception as exc:
            self.healthy = False
            self.failure = str(exc)
            raise InstanceFailure(f"TensorRT 实例执行失败: {exc}") from exc
        finally:
            # 异常也必须排空该 stream，不能让待执行设备工作越过租约边界。
            try:
                with torch.cuda.device(self.device):
                    self.stream.synchronize()
            except Exception as exc:
                self.healthy = False
                self.failure = str(exc)
                raise InstanceFailure(f"TensorRT stream 同步失败: {exc}") from exc

    def generate(self, *, input, fs=16000, **kwargs):
        wave = waveform_tensor(input, fs)
        cpu = campplus_features(wave) if self.name == "speech_campplus" else wave.unsqueeze(0)
        output = self._execute(cpu)
        if self.name == "speech_campplus":
            return [{"spk_embedding": output["embedding"]}]
        return [{"labels": self.labels, "scores": output["scores"][0].tolist()}]

    def close(self):
        import torch

        try:
            if self.stream is not None:
                with torch.cuda.device(self.device):
                    self.stream.synchronize()
        finally:
            self.buffers.clear()
            self.context = self.engine = self.runtime = self.stream = None
