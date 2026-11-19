"""Actual CPU Uvicorn processes; no model weights, hardware preflight or VAD workers."""
from pathlib import Path
import json
import os
import signal
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=r'''
import json,sys
from pathlib import Path
from types import SimpleNamespace
root,output,port,mode=sys.argv[1:]
output=Path(output);sys.path.insert(0,root)
import main

def event(name):
    with (output/'events.jsonl').open('a') as stream:stream.write(json.dumps({'event':name})+'\n')
async def start():event('start')
async def stop_vad():
    event('stop_vad')
    if mode=='failed':raise RuntimeError('synthetic CPU shutdown failure')
async def close_models():event('close_models')
async def close_qwen():event('close_qwen')
main.settings=SimpleNamespace(resource_sample_interval_s=1.0,qwen_execution_mode="thread",qwen_audio_spk_emo_concurrency=1,validate_model_backends=lambda:None,log_path=str(output/'unused.log'))
main.validate_npu_runtime=lambda *_: {'checked':False,'cpu_fixture':True}
main._set_process_name=lambda:None
main.setup_logging=lambda *_:None
main.cleanup_stale_service_cache=lambda:None
main.reset_stats=lambda:None
main.start_vad_pool=start;main.stop_vad_pool=stop_vad
main.close_v3_model_pools=close_models;main.close_qwen_client=close_qwen
try:main.run_server(['--host','127.0.0.1','--port',port,'--workers','1','--timeout-graceful-shutdown','2'])
finally:
    backend=sys.modules.get('torch_npu')
    initialized=backend.npu.is_initialized() if backend is not None else False
    (output/'cpu-runtime.json').write_text(json.dumps({'npu_initialized':initialized,'project_models_loaded':False})+'\n')
'''


def listening(port):
    with socket.socket() as client:return client.connect_ex(('127.0.0.1',port))==0


def exercise_cpu_main(directory,mode):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with socket.socket() as candidate:
        candidate.bind(('127.0.0.1',0));port=candidate.getsockname()[1]
    fixture=directory/'cpu_main.py';fixture.write_text(FIXTURE)
    temp=directory/'tmp';temp.mkdir()
    env=dict(os.environ,TORCH_DEVICE_BACKEND_AUTOLOAD='0',TMPDIR=str(temp),
             ASR_RUNTIME_DIR=str(temp),ASR_STATS_PATH=str(directory/'stats.json'))
    with (directory/'service.log').open('w') as log:
        process=subprocess.Popen([sys.executable,str(fixture),str(ROOT),str(directory),str(port),mode],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline=time.monotonic()+30
            while not listening(port):
                if process.poll() is not None or time.monotonic()>deadline:
                    raise AssertionError((directory/'service.log').read_text(errors='replace'))
                time.sleep(.02)
            process.send_signal(signal.SIGTERM);code=process.wait(timeout=15)
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=3)
    text=(directory/'service.log').read_text(errors='replace')
    assert not listening(port)
    runtime=json.loads((directory/'cpu-runtime.json').read_text());assert runtime['npu_initialized'] is False
    events=[json.loads(line)['event'] for line in (directory/'events.jsonl').read_text().splitlines()]
    assert events==['start','stop_vad','close_models','close_qwen']
    return {'exit_code':code,'application_failed':'Application shutdown failed' in text,
            'port_listening':False,'events':events,'npu_initialized':False,'log':str(directory/'service.log')}


def test_real_cpu_main_normal_term_exits_zero(tmp_path):
    result=exercise_cpu_main(tmp_path/'normal','normal')
    assert result['exit_code']==0 and not result['application_failed']


def test_real_cpu_main_shutdown_failure_is_nonzero(tmp_path):
    result=exercise_cpu_main(tmp_path/'failed','failed')
    assert result['application_failed'] and result['exit_code']!=0



def test_run_server_uses_the_served_app_and_restores_term_handler(monkeypatch):
    import main
    import uvicorn
    original=signal.getsignal(signal.SIGTERM)
    seen=[]
    def fake_run(target, **kwargs):
        seen.append(target)
        assert kwargs['workers']==1 and kwargs['reload'] is False
        assert signal.getsignal(signal.SIGTERM) is not original
    monkeypatch.setattr(uvicorn,'run',fake_run)
    monkeypatch.setattr(main,'_set_process_name',lambda:None)
    assert main.run_server(['--port','12345']) is None
    assert seen==[main.app]
    assert signal.getsignal(signal.SIGTERM)==original
