"""Replay two real previously abnormal segments; root alone executes NPU:7."""
import argparse
import io
import json
from pathlib import Path
import time
import wave
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--samples-dir',default='/tmp/qc-failed-segments')
    parser.add_argument('--output',required=True)
    parser.add_argument('--modes',default='static,continuous-off,continuous-on')
    args=parser.parse_args()
    from core.qwen_asr_process import LocalQwenProcessRuntime
    from core.qwen_continuous_process import LocalQwenContinuousProcessRuntime
    source=Path(args.samples_dir)
    samples=np.load(source/'samples.npz')
    audios=[]
    for key in ['audio_a2','audio_a4']:
        stream=io.BytesIO()
        with wave.open(stream,'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
            w.writeframes((samples[key]*32768).clip(-32768,32767).astype('<i2').tobytes())
        audios.append(stream.getvalue())
    output=dict(manifest=json.loads((source/'manifest.json').read_text()),runs={},owners=[])
    runtime=None
    try:
        for mode in args.modes.split(','):
            if mode=='static':
                runtime=LocalQwenProcessRuntime('/home/model/Qwen3-ASR-1.7B-hf','npu:7',_startup_timeout_s=120,_close_timeout_s=120)
            else:
                runtime=LocalQwenContinuousProcessRuntime('/home/model/Qwen3-ASR-1.7B-hf','npu:7',
                    max_active=16,max_batch_size=16,online_repetition_enabled=mode=='continuous-on',
                    _startup_timeout_s=120,_close_timeout_s=120)
            runtime.ensure_loaded()
            started=time.monotonic()
            values=[]
            if mode=='static':
                for value in runtime.transcribe_batch(audios,[None]*2):
                    values.append(dict(text=value,warning=None) if isinstance(value,str) else dict(error=str(value),error_type=type(value).__name__))
            else:
                futures=[runtime.submit(audio,request_id=mode,segment_id=str(i)) for i,audio in enumerate(audios)]
                for i,future in enumerate(futures):
                    try:
                        value=future.result(timeout=120)
                        row=dict(value) if isinstance(value,dict) else dict(text=value,warning=None)
                        row.update(index=i,tokens=getattr(future,'generated_tokens',None))
                    except Exception as exc:
                        row=dict(index=i,error_type=type(exc).__name__,reason=getattr(exc,'reason',None))
                    values.append(row)
            output['runs'][mode]=dict(wall_s=time.monotonic()-started,outputs=values,state=runtime.snapshot())
            runtime.close();output['owners'].append(runtime.snapshot());runtime=None
            Path(args.output).parent.mkdir(parents=True,exist_ok=True)
            Path(args.output).write_text(json.dumps(output,ensure_ascii=False,indent=2))
    finally:
        if runtime is not None:
            try:runtime.close()
            except Exception as exc:output['cleanup_error']=type(exc).__name__+': '+str(exc)
            output['owners'].append(runtime.snapshot())
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
