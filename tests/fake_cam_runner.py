"""Explicit CPU-only CAM spawn fixture; never imports torch_npu or loads model weights."""
import os
from pathlib import Path
import threading
import time


class FakeAPI:
    def __init__(self,device,fail,post_failure=False):self.index=int(device.split(':')[1]);self.fail=fail;self.post_failure=post_failure;self.stream_calls=0
    def set_device(self,device):assert int(device.split(':')[1])==self.index
    def current_stream(self,device):
        self.stream_calls+=1
        if self.post_failure and self.stream_calls==2:raise RuntimeError('controlled post-forward metadata failure')
        from types import SimpleNamespace
        return SimpleNamespace(device_index=self.index,stream_id=0,npu_stream=123)
    def synchronize(self,device):
        self.set_device(device)
        if self.fail:raise RuntimeError('controlled NPU synchronize failure')


class Runner:
    _cam_cpu_test_double = True
    actual_precision = 'fp32'
    math_mode = 'fp32_ops_hf32_disabled'
    operator_adaptations = {'campplus_segment_pooling': False, 'explicit_CPU_test_double': True}
    healthy = True
    failure = None

    def __init__(self, name, model_dir, device):
        assert name == 'speech_campplus' and device.startswith('npu:')
        self.device = device; self.mode = Path(model_dir).name; self.calls = 0
        if self.mode in ('model_error_fenced','sync_failure','post_metadata_error','input_shape_error'):self.api=FakeAPI(device,self.mode=='sync_failure',self.mode=='post_metadata_error')
        if self.mode == 'startup_crash': os._exit(19)
        if self.mode == 'startup_error': raise RuntimeError('controlled startup failure')
        if self.mode == 'startup_slow': time.sleep(.3)
        if self.mode == 'startup_bad_device': self.device = 'npu:99'
        if self.mode == 'startup_bad_precision': self.actual_precision = 'float16'
        if self.mode == 'startup_unmarked': self._cam_cpu_test_double = False
        if self.mode == 'idle_exit': threading.Timer(.2, lambda: os._exit(23)).start()

    def generate(self, *, input, fs):
        import numpy as np
        import torch
        from core.local_inference import InstanceFailure
        assert input.dtype == np.float32 and input.ndim == 1 and fs == 16000
        self.calls += 1
        if self.mode == 'crash': os._exit(23)
        if self.mode in ('slow','slow_error'): time.sleep(.3)
        if self.mode == 'input_shape_error' and self.calls == 1:raise RuntimeError('invalid input shape')
        if self.mode in ('model_error','slow_error','model_error_fenced','sync_failure') and self.calls == 1:
            raise RuntimeError('controlled ordinary model error')
        if self.mode == 'instance_error':
            self.healthy = False; self.failure = 'controlled NPU device resource failure'
            raise InstanceFailure(self.failure)
        result = (torch.arange(192, dtype=torch.float32) + float(input[0])).reshape(1,192)
        if self.mode == 'wrong_shape': return [{'spk_embedding':result.reshape(-1)}]
        if self.mode == 'nonfinite': result[0,1] = float('nan')
        return [{'spk_embedding':result}]

    def close(self):
        if self.mode == 'close_slow': time.sleep(.3)
        if self.mode in ('close_error','close_error_slow'):
            if self.mode == 'close_error_slow':time.sleep(.1)
            raise RuntimeError('controlled close error')
        if self.mode == 'ack_nonzero':
            def exit_after_ack():
                time.sleep(.05); os._exit(7)
            threading.Thread(target=exit_after_ack,daemon=False).start()


def make(name, model_dir, device):
    return Runner(name,model_dir,device)
