"""所有权工具CPU替身；不创建NPU、权重或监听服务。"""
import asyncio
import io
import json
import tempfile
import subprocess
import time
import unittest
import wave
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from scripts.validate_qwen_batch_ownership import main, run_validation


def wav_data():
    stream=io.BytesIO()
    with wave.open(stream,'wb') as handle:
        handle.setparams((1,2,16000,0,'NONE','not compressed'))
        handle.writeframes(bytes(65600*2))
    return stream.getvalue()


class CpuRuntime:
    instances=[]
    def __init__(self,*args):
        self.loaded=False;self.loading=False;self.failure=None;self.last_batch_timings=None
        self.processor=SimpleNamespace(batch_decode=lambda ids,**kwargs:['language Chinese<asr_text>真实替身测试']*len(ids))
        self.original_decode=self.processor.batch_decode
        self.instances.append(self)
    def ensure_loaded(self):self.loaded=True
    def transcribe_batch(self,audios,languages=None):
        from core.qwen_asr_runtime import LocalUnsupportedLanguage
        self.ensure_loaded();time.sleep(.002)
        good=[i for i,lang in enumerate(languages or [None]*len(audios)) if lang!='__not_a_supported_language__']
        texts=self.processor.batch_decode([1]*len(good))
        output_errors=[i for i,text in zip(good,texts) if '<asr_text>' not in text]
        self.last_batch_timings={'batch_size':len(good),'valid_member_indices':good,
            'generate_started':True,'generate_completed':True,'generate_wall_ms':2,
            'output_error_indices':output_errors,'output_error_count':len(output_errors)}
        values=[LocalUnsupportedLanguage('unsupported language') for _ in audios]
        for i,text in zip(good,texts):
            values[i]=RuntimeError('malformed output') if i in output_errors else text
        return values
    def snapshot(self):return {'loaded':self.loaded,'loading':self.loading,'failure':self.failure,'memory_allocated_mb':0,'memory_reserved_mb':0}
    def close(self):self.loaded=False


class BatchOwnershipToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_controlled_phases_retain_single_runtime_until_completion(self):
        report=await run_validation(wav_data(),model_dir='cpu-fake',runtime_base=CpuRuntime,
                                    observation_timeout_s=3,gate_timeout_s=3)
        self.assertEqual(report['status'],'passed')
        self.assertEqual(report['cleanup']['status'],'passed')
        self.assertEqual(len(report['phases']),8)
        self.assertEqual(report['observer']['calls'],9)
        self.assertEqual(report['observer']['peak'],1)
        self.assertEqual(report['observer']['active'],0)
        self.assertTrue(all(phase['status']=='passed' for phase in report['phases']))
        self.assertEqual(report['phases'][1]['codes'],[4009]+[None]*7)
        self.assertFalse(report['cleanup']['client']['loaded'])
        self.assertEqual(report['cleanup']['client']['batch_scheduler']['inflight'],0)
        events=report['observer']['events']
        for call in range(1,10):
            rows=[row for row in events if row.get('call')==call]
            kinds={row['event']:row['monotonic_s'] for row in rows}
            batch_id=next(row['batch_id'] for row in rows if row['event']=='real_runtime_started')
            self.assertLessEqual(kinds['real_runtime_started'],kinds['real_runtime_read_complete_after_sync'])
            self.assertLessEqual(kinds['real_runtime_read_complete_after_sync'],kinds['owned_thread_finished'])
            released=next(row for row in events if row['event']=='instance_release_observed' and row['batch_id']==batch_id)
            self.assertLessEqual(kinds['owned_thread_finished'],released['monotonic_s'])
            self.assertEqual(released['busy'],0)
            self.assertEqual(released['available'],1)
        self.assertEqual(report['runtime_kind'],'cpu_test_double')
        self.assertIs(CpuRuntime.instances[-1].processor.batch_decode,CpuRuntime.instances[-1].original_decode)
        phases={row['phase']:row for row in report['phases']}
        self.assertEqual(phases['retry_partial_cancel']['actual_generate_members'],[8,7])
        self.assertEqual(phases['retry_all_cancel']['actual_generate_members'],[8])
        self.assertEqual(phases['malformed_output_member_isolation']['codes'],[4012]+[None]*7)
        for name in ('retry_partial_cancel','retry_all_cancel'):
            phase=phases[name]
            self.assertTrue(all(not row['hardware_execution_confirmed'] for row in phase['actual_read_events']))
            self.assertLess(phase['actual_read_events'][0]['monotonic_s'],phase['cancel_requested_monotonic_s'])
            if name=='retry_partial_cancel':
                second_start=next(row for row in events if row['event']=='real_runtime_started' and row.get('phase')==name and row['phase_call']==2)
                self.assertLess(phase['cancel_requested_monotonic_s'],second_start['monotonic_s'])
                self.assertGreaterEqual(second_start['monotonic_s']-phase['actual_read_events'][0]['monotonic_s'],.99)
            for key in ('busy','waiting'):
                self.assertEqual(phase['after'][key],0)
            for key in ('waiting','running_members','inflight'):
                self.assertEqual(phase['after']['batch_scheduler'][key],0)

    async def test_stopped_plan_drains_and_closes_without_running_batch(self):
        report={}
        with self.assertRaises(InterruptedError):
            await run_validation(wav_data(),model_dir='cpu-fake',runtime_base=CpuRuntime,
                                 report=report,should_stop=lambda:True,observation_timeout_s=3,gate_timeout_s=3)
        self.assertEqual(report['cleanup']['status'],'passed')
        self.assertFalse(report['cleanup']['client']['loaded'])


class PlanOnlyTests(unittest.TestCase):
    def test_default_mode_never_constructs_inference_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=root/'config.toml';wav=root/'sample.wav';out=root/'plan.json'
            config.write_text('[qwen_asr]\nlocal_device="npu:7"\n');wav.write_bytes(wav_data())
            with patch('scripts.validate_qwen_batch_ownership.run_validation',side_effect=AssertionError('must not execute')):
                self.assertEqual(main(['--config',str(config),'--wav',str(wav),'--output',str(out)]),0)
            report=json.loads(out.read_text())
            self.assertEqual(report['status'],'planned')
            self.assertEqual(report['configuration']['physical_batch'],8)
            with self.assertRaises(SystemExit):
                main(['--config',str(config),'--wav',str(wav),'--output',str(out)])

    def test_non7_card_and_watchdog_out_of_range_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=root/'config.toml';wav=root/'sample.wav';out=root/'plan.json'
            config.write_text('[qwen_asr]\nlocal_device="npu:0"\n');wav.write_bytes(wav_data())
            with self.assertRaises(ValueError):main(['--config',str(config),'--wav',str(wav),'--output',str(out)])
            with self.assertRaises(SystemExit):main(['--config',str(config),'--wav',str(wav),'--output',str(out),'--watchdog-s','421'])


class SupervisorTests(unittest.TestCase):
    def test_total_watchdog_only_terminates_owned_child_and_preserves_failure(self):
        class Child:
            pid=987654321
            returncode=None
            waits=0
            def poll(self):return self.returncode
            def wait(self,timeout):
                self.waits+=1
                if self.waits==1:raise subprocess.TimeoutExpired('owned-fake',timeout)
                self.returncode=-9
                return self.returncode
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=root/'config.toml';wav=root/'sample.wav';out=root/'report.json'
            config.write_text('[qwen_asr]\nlocal_device="npu:7"\n');wav.write_bytes(wav_data())
            child=Child()
            with patch('scripts.validate_qwen_batch_ownership.subprocess.Popen',return_value=child), \
                 patch('scripts.validate_qwen_batch_ownership.os.killpg') as kill, \
                 patch('scripts.validate_qwen_batch_ownership.time.monotonic',side_effect=[0,3,3,3.2,3.3]), \
                 patch('scripts.validate_qwen_batch_ownership.run_validation',side_effect=AssertionError('CPU supervisor must not execute runtime')):
                code=main(['--config',str(config),'--wav',str(wav),'--output',str(out),'--execute','--watchdog-s','5'])
            self.assertEqual(code,1)
            self.assertEqual([call.args[0] for call in kill.call_args_list],[child.pid,child.pid])
            result=json.loads(out.read_text())
            self.assertEqual(result['status'],'failed')
            self.assertTrue(result['supervisor']['term_sent'])
            self.assertTrue(result['supervisor']['forced_kill'])
            self.assertTrue(result['supervisor']['child_exited'])
            self.assertLessEqual(result['supervisor']['elapsed_s'],5)
