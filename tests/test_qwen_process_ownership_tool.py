"""CPU stubs only; subprocesses read stdin and exit0, never load models/NPU."""
import asyncio
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import wave

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/validate_qwen_process_ownership.py'
spec = importlib.util.spec_from_file_location('process_ownership_tool', SCRIPT)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def wav_bytes():
    stream = io.BytesIO()
    with wave.open(stream, 'wb') as audio:
        audio.setparams((1, 2, 16000, 0, 'NONE', ''))
        audio.writeframes(bytes(65600*2))
    return stream.getvalue()


class CpuProxy:
    """Proxy interface double; child is an owned CPU-only exit0 lifecycle witness."""
    instances = []

    def __init__(self, *args):
        import psutil
        self.loaded = self.loading = False
        self.failure = None
        self.last_batch_timings = None
        self._rpc_id = 0
        self.process = subprocess.Popen([sys.executable, '-u', '-c',
            'import sys\nfor line in sys.stdin:\n if line.strip()=="close":break\n'], stdin=subprocess.PIPE, text=True)
        origin = str(Path(__file__).resolve())
        self.meta = {'worker_pid':self.process.pid, 'worker_create_time':psutil.Process(self.process.pid).create_time(),
            'worker_start_ticks':'cpu-fixture', 'launch_id':'cpu-launch', 'ready_verified':True,
            'worker_alive':True, 'start_method':'spawn', 'worker_entry_module_file':origin,
            'launch_config':{'worker_entry_identity':{'module_origin':origin, 'module_sha256':hashlib.sha256(Path(origin).read_bytes()).hexdigest()}}}
        self.instances.append(self)

    def _rpc(self, operation, **kwargs):
        from core.qwen_asr_runtime import LocalUnsupportedLanguage
        self._rpc_id += 1
        if operation == 'ensure_loaded':
            self.loaded = True
            return None
        languages = kwargs.get('languages') or [None]*len(kwargs['audios'])
        good = [index for index, language in enumerate(languages) if language != '__not_a_supported_language__']
        self.last_batch_timings = {'batch_size':len(good), 'valid_member_indices':good,
            'generate_started':True, 'generate_completed':True, 'generate_wall_ms':20,
            'output_error_indices':[], 'output_error_count':0}
        time.sleep(.02)
        return ['language Chinese<asr_text>CPU替身' if index in good else LocalUnsupportedLanguage('unsupported')
                for index in range(len(languages))]

    def ensure_loaded(self):
        self._rpc('ensure_loaded')

    def transcribe_batch(self, audios, languages=None):
        return self._rpc('transcribe_batch', audios=audios, languages=languages)

    def snapshot(self):
        return {'loaded':self.loaded, 'loading':False, 'failure':None, 'device':'npu:7', 'dtype':'bfloat16',
            'memory_allocated_mb':0, 'memory_reserved_mb':0, 'model_worker':copy.deepcopy(self.meta)}

    def close(self):
        if self.process.poll() is None:
            self.process.communicate('close\n', timeout=3)
        self.loaded = False
        self.meta.update(worker_alive=False, resource_reaped=True, normal_exit_ok=True,
            worker_exitcode=self.process.returncode, cleanup_status='passed', forced_cleanup=False)


def test_three_phases_retain_owned_parent_gate_and_exit0_cpu_child():
    async def check():
        try:
            report = await tool.run_validation(wav_bytes(), model_dir='cpu-double', runtime_base=CpuProxy,
                observation_timeout_s=5, gate_timeout_s=5)
            assert report['status'] == report['cleanup']['status'] == 'passed'
            assert len(report['phases']) == 3
            assert report['phases'][0]['codes'] == [4009]+[None]*7
            assert report['phases'][1]['codes'] == [4012]*8
            assert report['cleanup']['model_worker']['worker_exitcode'] == 0
            assert report['runtime_kind'] == 'cpu_test_double'
            events = report['observer']['events']
            for phase in ('partial_cancel_invalid', 'timeout_waits_done', 'cancel_aclose'):
                rows = [event for event in events if event.get('phase') == phase]
                done = next(event for event in rows if event['event'] == 'model_done_rpc_read_complete')
                returned = next(event for event in rows if event['event'] == 'parent_owned_thread_returned')
                released = next(event for event in rows if event['event'] == 'client_instance_returned')
                assert done['monotonic_s'] <= returned['monotonic_s'] <= released['monotonic_s']
                rpc = next(event for event in rows if event['event'] == 'rpc_response_fully_read_and_verified')
                assert rpc['member_ids'] == list(range(8))
                assert done['runtime_timings']['batch_size'] == (7 if phase == 'partial_cancel_invalid' else 8)
        finally:
            for runtime in CpuProxy.instances:
                runtime.close()
    asyncio.run(check())


def test_cleanup_refuses_forced_or_wrong_entry_and_never_washes_failure():
    runtime = CpuProxy()
    try:
        ready = runtime.snapshot()
        assert tool.verify_worker(ready, real=False)['worker_alive']
        with pytest.raises(AssertionError, match='protected .so'):
            tool.verify_worker(ready, real=False, require_so=True)
        ready['model_worker']['launch_config']['worker_entry_identity']['module_sha256'] = 'f'*64
        with pytest.raises(AssertionError, match='entry hash'):
            tool.verify_worker(ready, real=False)
        runtime.close()
        closed = runtime.snapshot()
        closed['model_worker']['forced_cleanup'] = True
        with pytest.raises(AssertionError, match='normally'):
            tool.verify_worker(closed, real=False, closed=True)
    finally:
        runtime.close()


def test_plan_cannot_start_runtime_and_preserves_existing_evidence(tmp_path):
    config = tmp_path / 'config.toml'
    source = tmp_path / 'real.wav'
    output = tmp_path / 'plan.json'
    config.write_text('[qwen_asr]\nlocal_device="npu:7"\nlocal_dtype="bfloat16"\n')
    source.write_bytes(wav_bytes())
    with patch.object(tool, 'run_validation', side_effect=AssertionError('plan cannot load runtime')):
        assert tool.main(['--config',str(config),'--wav',str(source),'--output',str(output),'--require-worker-so']) == 0
    result = json.loads(output.read_text())
    assert result['configuration']['execution_mode'] == 'process'
    assert result['configuration']['require_worker_so'] is True
    with pytest.raises(SystemExit):
        tool.main(['--config',str(config),'--wav',str(source),'--output',str(output)])


def test_supervisor_reaps_own_orphan_but_preserves_other_session(tmp_path):
    import psutil
    output = tmp_path / 'summary.json'
    trace = tmp_path / 'owned-pid.txt'
    tool.checkpoint(output, {'status':'passed'})
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'], start_new_session=True)
    launcher = ('import subprocess,sys;from pathlib import Path;'
        'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]);'
        'Path(sys.argv[1]).write_text(str(p.pid))')
    try:
        assert tool.supervise([sys.executable, '-c', launcher, str(trace)], output, 5) == 1
        report = json.loads(output.read_text())
        assert report['status'] == 'failed' and report['supervisor']['forced_cleanup']
        assert not report['supervisor']['remaining_after_force']
        own_pid = int(trace.read_text())
        assert any(row['pid'] == own_pid for row in report['supervisor']['owned_descendants'])
        assert unrelated.poll() is None
        own = next(row for row in report['supervisor']['owned_descendants'] if row['pid'] == own_pid)
        assert tool.same_process(own) is None
        wrong_birth = {'pid':unrelated.pid, 'create_time':psutil.Process(unrelated.pid).create_time()+1}
        assert tool.same_process(wrong_birth) is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=3)
