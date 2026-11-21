"""Owned NPU:7 small real-segment correctness/admission check, not a throughput claim."""
import argparse
import concurrent.futures
import io
import json
from pathlib import Path
import time
import wave
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--samples', default='/home/zhangs/qwen-om-smoke-20261120/bench128-20261120/samples.npz')
    p.add_argument('--manifest', default='/home/zhangs/qwen-om-smoke-20261120/bench128-20261120/manifest.json')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    from core.qwen_asr_process import LocalQwenProcessRuntime
    from core.qwen_continuous_process import LocalQwenContinuousProcessRuntime
    rows = json.loads(Path(args.manifest).read_text())['samples']
    order = sorted(rows, key=lambda x:x['duration_s'])
    chosen = [order[-1],order[2],order[len(order)//2]]
    samples = np.load(args.samples)
    audios = []
    for row in chosen:
        stream = io.BytesIO()
        with wave.open(stream, 'wb') as writer:
            writer.setnchannels(1); writer.setsampwidth(2); writer.setframerate(16000)
            writer.writeframes((np.clip(samples[row['audio_key']], -1, 32767/32768)*32768).astype('<i2').tobytes())
        audios.append(stream.getvalue())
    result = dict(samples=chosen, static=None, continuous={}, owners=[], error=None)
    runtime = None
    try:
        runtime = LocalQwenProcessRuntime('/home/model/Qwen3-ASR-1.7B-hf','npu:7',
            _startup_timeout_s=120,_close_timeout_s=120)
        runtime.ensure_loaded()
        result['static'] = [str(x) for x in runtime.transcribe_batch(audios, [None]*3)]
        runtime.close()
        result['owners'].append(runtime.snapshot())
        runtime = LocalQwenContinuousProcessRuntime('/home/model/Qwen3-ASR-1.7B-hf','npu:7',
            max_active=16,max_batch_size=16,online_repetition_enabled=False,_close_timeout_s=120)
        runtime.ensure_loaded()
        started=time.monotonic()
        futures = {str(i):runtime.submit(audios[i],request_id='smoke',segment_id=str(i)) for i in (0,1)}
        done, _ = concurrent.futures.wait(futures.values(),return_when=concurrent.futures.FIRST_COMPLETED)
        first_key = next(k for k,v in futures.items() if v in done)
        result['first_done'] = first_key
        result['long_done_when_replacing'] = futures['0'].done()
        futures['2'] = runtime.submit(audios[2],request_id='smoke',segment_id='2')
        for key,future in futures.items():
            try:
                value=future.result(timeout=120)
                value=value if isinstance(value,dict) else dict(text=value,warning=None)
                result['continuous'][key] = dict(value, tokens=getattr(future,'generated_tokens',None), wall_s=time.monotonic()-started)
            except Exception as exc:
                result['continuous'][key] = dict(error=str(exc))
        result['state_before_close'] = runtime.snapshot()
        result['text_equal'] = [result['continuous'][str(i)].get('text')==result['static'][i] for i in range(3)]
    except BaseException as exc:
        result['error'] = type(exc).__name__+': '+str(exc)
        raise
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except Exception as exc:
                result['cleanup_error'] = str(exc)
            result['owners'].append(runtime.snapshot())
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
