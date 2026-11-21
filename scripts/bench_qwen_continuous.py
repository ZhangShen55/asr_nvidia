"""Owned static/continuous same-PCM capacity pair, 420s total gate."""
import argparse
import concurrent.futures
import io
import json
from pathlib import Path
import time
import wave
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--counts',default='16,128')
    parser.add_argument('--gate-s',type=float,default=420)
    args=parser.parse_args()
    from core.qwen_asr_process import LocalQwenProcessRuntime
    from core.qwen_continuous_process import LocalQwenContinuousProcessRuntime
    data=Path('/home/zhangs/qwen-om-smoke-20261120/bench128-20261120')
    rows=json.loads((data/'manifest.json').read_text())['samples']
    samples=np.load(data/'samples.npz')
    audios=[]
    for row in rows:
        stream=io.BytesIO()
        with wave.open(stream,'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
            w.writeframes((np.clip(samples[row['audio_key']],-1,32767/32768)*32768).astype('<i2').tobytes())
        audios.append(stream.getvalue())
    out=dict(counts={},owners=[],gate_s=args.gate_s,error=None)
    deadline=time.monotonic()+args.gate_s
    runtime=None
    def remaining():
        value=deadline-time.monotonic()
        if value<=0:raise TimeoutError('420s total gate reached')
        return value
    def save():
        target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(out,ensure_ascii=False,indent=2))
    try:
        for count in map(int,args.counts.split(',')):
            out['counts'][str(count)]={}
            for mode in ('static','continuous'):
                remaining()
                cls=LocalQwenProcessRuntime if mode=='static' else LocalQwenContinuousProcessRuntime
                kw=dict(_startup_timeout_s=min(120,remaining()),_close_timeout_s=120)
                if mode=='continuous':kw.update(max_active=count,max_batch_size=count,
                    online_repetition_enabled=False,max_batch_tokens=80000,max_batch_audio_frames=115200)
                runtime=cls('/home/model/Qwen3-ASR-1.7B-hf','npu:7',**kw)
                runtime.ensure_loaded()
                rounds=[];out['counts'][str(count)][mode]=rounds
                for iteration in range(2):
                    remaining();started=time.monotonic()
                    if mode=='static':
                        # Background control checks the total gate while preserving ownership.
                        pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
                        try:
                            values=pool.submit(runtime.transcribe_batch,audios[:count],[None]*count).result(timeout=remaining())
                        except concurrent.futures.TimeoutError:
                            runtime.begin_closing();runtime.abort_owned('capacity benchmark420s gate');raise
                        finally:pool.shutdown(wait=True)
                        outputs=[dict(index=i,text=v) if isinstance(v,str) else dict(index=i,error=str(v))
                                 for i,v in enumerate(values)]
                    else:
                        futures=[runtime.submit(audios[i],request_id=str(iteration),segment_id=str(i)) for i in range(count)]
                        outputs=[]
                        for i,future in enumerate(futures):
                            try:
                                value=future.result(timeout=remaining())
                                value=value if isinstance(value,dict) else dict(text=value,warning=None)
                                outputs.append(dict(value,index=i,tokens=getattr(future,'generated_tokens',None)))
                            except concurrent.futures.TimeoutError:
                                raise TimeoutError('continuous benchmark total gate')
                            except Exception as exc:outputs.append(dict(index=i,error=str(exc)))
                        runtime.release_request(str(iteration))
                    rounds.append(dict(phase='warm' if iteration==0 else 'steady',wall_s=time.monotonic()-started,
                        outputs=outputs,state=runtime.snapshot()))
                    save()
                    if any('error' in v for v in outputs):raise RuntimeError('member failure; no further scaling')
                runtime.close();out['owners'].append(runtime.snapshot());runtime=None;save()
            a=out['counts'][str(count)]['static'][-1]['outputs']
            b=out['counts'][str(count)]['continuous'][-1]['outputs']
            out['counts'][str(count)]['text_differences']=[dict(index=i,static=x.get('text'),continuous=y.get('text'))
                for i,(x,y) in enumerate(zip(a,b)) if x.get('text')!=y.get('text')]
            save()
    except BaseException as exc:
        out['error']=type(exc).__name__+': '+str(exc)
        raise
    finally:
        if runtime is not None:
            try:runtime.close()
            except Exception as exc:out['cleanup_error']=str(exc)
            out['owners'].append(runtime.snapshot())
        save()


if __name__=='__main__':main()
