#!/usr/bin/env python3
"""单Qwen spawn worker的8成员所有权验证；默认计划，显式execute才运行。"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_qwen_batch_ownership import Probe, checkpoint, error_details, read_plan


def same_process(identity):
    import psutil
    try:
        process = psutil.Process(identity['pid'])
        return process if process.create_time() == identity['create_time'] and process.status() != psutil.STATUS_ZOMBIE else None
    except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
        return None


def require(value, reason):
    if not value:
        raise AssertionError(reason)


def verify_worker(state, *, require_so=False, closed=False, real=True):
    meta = state.get('model_worker', {})
    required = ('worker_pid', 'worker_create_time', 'worker_start_ticks', 'launch_id')
    require(all(meta.get(key) is not None for key in required), 'owned worker birth identity missing')
    require(meta.get('ready_verified') is True and meta.get('start_method') == 'spawn', 'worker READY was not verified')
    entry = meta.get('worker_entry_module_file')
    config = meta.get('launch_config', {})
    entry_identity = config.get('worker_entry_identity', {})
    require(entry and entry_identity.get('module_origin') == entry, 'READY actual entry differs from launch source')
    require(isinstance(entry_identity.get('module_sha256'), str) and len(entry_identity['module_sha256']) == 64,
            'worker entry source hash missing')
    require(Path(entry).is_file() and hashlib.sha256(Path(entry).read_bytes()).hexdigest() == entry_identity['module_sha256'],
            'READY entry hash does not match actual source artifact')
    if require_so:
        require(Path(entry).suffix == '.so', 'actual worker entry did not come from protected .so')
    if real:
        require(meta.get('worker_ready_identity', {}).get('pid') == meta['worker_pid']
                and meta.get('worker_ready_identity', {}).get('start_ticks') == meta['worker_start_ticks'],
                'READY worker birth identity differs from retained owner')
        require(config.get('device') == 'npu:7' and config.get('dtype') == 'bfloat16', 'worker device or precision changed')
        require(config.get('factory_identity', {}).get('factory') == 'core.qwen_asr_runtime:LocalQwenRuntime', 'worker is not original HF runtime')
    identity = {'pid': meta['worker_pid'], 'create_time': meta['worker_create_time']}
    if closed:
        require(meta.get('resource_reaped') is True and meta.get('normal_exit_ok') is True
                and meta.get('worker_exitcode') == 0 and meta.get('cleanup_status') == 'passed'
                and meta.get('worker_alive') is False and not meta.get('forced_cleanup') and not meta.get('close_error'),
                'worker did not close normally with exit0')
        require(same_process(identity) is None, 'owned PID/creationtime remains alive after close')
    else:
        require(meta.get('worker_alive') is True and same_process(identity) is not None, 'READY owned worker is not alive')
    return copy.deepcopy(meta)


def observed_runtime(base, probe):
    class Observed(base):
        def _rpc(self, operation, **kwargs):
            phase = probe.phase
            before = self._rpc_id
            if operation == 'transcribe_batch':
                probe.rpc_started.set()
            probe.event('rpc_entered', phase=phase, operation=operation, previous_rpc_id=before)
            value = super()._rpc(operation, **kwargs)
            meta = self.snapshot().get('model_worker', {})
            probe.event('rpc_response_fully_read_and_verified', phase=phase, operation=operation,
                        rpc_id=self._rpc_id, launch_id=meta.get('launch_id'),
                        member_ids=list(range(len(kwargs['audios']))) if operation == 'transcribe_batch' else None,
                        member_id_scope='original private proxy verifies response RPC/launch/member IDs; observer does not modify or read the pipe')
            return value

        def transcribe_batch(self, audios, languages=None):
            with probe.lock:
                probe.calls += 1
                probe.active += 1
                probe.peak = max(probe.peak, probe.active)
            phase = probe.phase
            probe.event('parent_owned_call_entered', phase=phase, input_members=len(audios),
                        wav_sha256=[hashlib.sha256(audio).hexdigest() for audio in audios])
            probe.entered.set()
            try:
                values = super().transcribe_batch(audios, languages)
                timings = copy.deepcopy(self.last_batch_timings)
                expected = [index for index, language in enumerate(languages or [None]*len(audios))
                            if language != '__not_a_supported_language__']
                require(timings and timings.get('generate_started') and timings.get('generate_completed')
                        and timings.get('batch_size') == len(expected)
                        and timings.get('valid_member_indices') == expected,
                        'original physical generate/member timing evidence incomplete')
                probe.event('model_done_rpc_read_complete', phase=phase, rpc_id=self._rpc_id,
                            runtime_timings=timings, model_worker=self.snapshot().get('model_worker'),
                            outputs=[error_details(value) if isinstance(value, BaseException) else value for value in values])
                probe.read_done.set()
                # 仅父观察wrapper的测试gate；原HF generate/read/sync及Pipe已真实完成。
                if not probe.released.wait(probe.gate_timeout_s):
                    raise TimeoutError('parent post-DONE observation gate timed out')
                return values
            finally:
                probe.event('parent_owned_thread_returned', phase=phase)
                with probe.lock:
                    probe.active -= 1
    return Observed


async def run_validation(wav_bytes, *, model_dir, dtype='bfloat16', max_new_tokens=256,
                         runtime_base=None, client_base=None, report=None, checkpoint_fn=None,
                         should_stop=None, require_worker_so=False, observation_timeout_s=90, gate_timeout_s=60):
    from core.async_utils import await_completion
    from core.qwen_asr_client import QwenAsrClient
    from scripts.validate_qwen_cancellation import wait_until
    real = runtime_base is None
    if runtime_base is None:
        from core.qwen_asr_process import LocalQwenProcessRuntime
        runtime_base = LocalQwenProcessRuntime
    client_base = client_base or QwenAsrClient
    probe = Probe(gate_timeout_s)
    probe.rpc_started = threading.Event()
    report = {} if report is None else report
    checkpoint_fn = checkpoint_fn or (lambda: None)
    should_stop = should_stop or (lambda: False)

    class Client(client_base):
        async def _execute_batch(self, members, batch_id):
            probe.event('logical_batch_dispatched', phase=probe.phase, batch_id=batch_id,
                        members=[{'request_id':member.request_id, 'segment_index':member.segment_index} for member in members])
            try:
                return await super()._execute_batch(members, batch_id)
            finally:
                probe.event('client_instance_returned', phase=probe.phase, batch_id=batch_id,
                            busy=self.snapshot()['busy'], available=self._available.qsize())

    # 此处验证legacy静态批RPC读完后的进程所有权，不发送continuous submit。
    client = Client(generation_mode='legacy', local_model_dir=model_dir, local_device='npu:7', local_dtype=dtype,
                    local_instance_count=1, max_new_tokens=max_new_tokens, retry_count=0,
                    request_timeout_s=120, max_inference_batch_size=8, batch_wait_timeout_ms=30,
                    max_batch_audio_frames=102400, max_batch_tokens=78336, batch_max_duration_ratio=4,
                    execution_mode='process', runtime_factory=observed_runtime(runtime_base, probe))
    tasks = []
    report.update(status='running', phases=[], cleanup={'status':'pending'},
                  runtime_kind='actual_HF_spawn_model_worker' if real else 'cpu_test_double',
                  observation_scope='parent gate only after original model DONE, sync and complete verified RPC read; gate is not kernel delay',
                  ports_started=False)

    def save():
        report['observer'] = probe.snapshot()
        report['client'] = client.snapshot()
        checkpoint_fn()

    def idle():
        state = client.snapshot()
        scheduler = state['batch_scheduler']
        require(state['busy'] == state['waiting'] == scheduler['waiting'] == scheduler['running_members'] == scheduler['inflight'] == 0,
                'client/scheduler ownership not drained')
        require(client._available.qsize() == 1 and probe.active == 0, 'instance returned early or more than once')

    def launch(name, invalid=False):
        require(not should_stop(), 'stop requested')
        idle()
        probe.prepare('hold', name)
        probe.rpc_started = threading.Event()
        group = [asyncio.create_task(client.transcribe_bytes(wav_bytes,
                    language='__not_a_supported_language__' if invalid and i == 0 else None,
                    request_id=f'{name}-{i}', segment_index=i)) for i in range(8)]
        tasks.extend(group)
        return group

    async def wait_owned(name):
        await wait_until(probe.rpc_started.is_set, observation_timeout_s, name+' RPC started')

    def held(group):
        require(client.snapshot()['busy'] == 1 and client._available.qsize() == 0 and not any(task.done() for task in group),
                'active model worker ownership released before parent owned return')

    try:
        await client.discover_model()
        report['ready_worker'] = verify_worker(client._runtimes[0].snapshot(), require_so=require_worker_so, real=real)
        save()
        group = launch('partial_cancel_invalid', invalid=True)
        await wait_owned('partial_cancel_invalid')
        cancelled_at = time.monotonic()
        group[1].cancel()
        await asyncio.sleep(0)
        group[1].cancel()
        await asyncio.sleep(0)
        held(group)
        await wait_until(probe.read_done.is_set, observation_timeout_s, 'original model DONE read')
        held(group)
        phase = {'phase':probe.phase, 'cancel_requested_monotonic_s':cancelled_at, 'before_release':client.snapshot()}
        report['phases'].append(phase)
        save()
        probe.released.set()
        values = await asyncio.gather(*group, return_exceptions=True)
        require(getattr(values[0], 'code', None) == 4009 and isinstance(values[1], asyncio.CancelledError)
                and all(isinstance(value, str) and '<asr_text>' in value for value in values[2:]), 'partial cancel/4009 leaked to other members')
        idle()
        phase.update(status='passed', codes=[getattr(value, 'code', None) for value in values], cancelled_indices=[1])

        client.request_timeout_s = .01
        group = launch('timeout_waits_done')
        await wait_owned('timeout_waits_done')
        await asyncio.sleep(.03)
        held(group)
        await wait_until(probe.read_done.is_set, observation_timeout_s, 'timeout original model DONE read')
        held(group)
        phase = {'phase':probe.phase, 'request_timeout_s':.01, 'before_release':client.snapshot()}
        report['phases'].append(phase)
        save()
        probe.released.set()
        values = await asyncio.gather(*group, return_exceptions=True)
        require(all(getattr(value, 'code', None) == 4012 for value in values), 'timeout must map to4012 only after DONE/drain')
        idle()
        phase.update(status='passed', codes=[getattr(value, 'code', None) for value in values])
        client.request_timeout_s = 120

        group = launch('cancel_aclose')
        await wait_owned('cancel_aclose')
        closer = asyncio.create_task(client.aclose())
        tasks.append(closer)
        await wait_until(lambda:client._close_task is not None, observation_timeout_s, 'owned client close started')
        closer.cancel()
        await asyncio.sleep(0)
        closer.cancel()
        await asyncio.sleep(0)
        held(group)
        await wait_until(probe.read_done.is_set, observation_timeout_s, 'close original DONE read')
        held(group)
        probe.released.set()
        values = await asyncio.gather(*group, return_exceptions=True)
        result = await asyncio.gather(closer, return_exceptions=True)
        require(all(isinstance(value, str) and '<asr_text>' in value for value in values)
                and isinstance(result[0], asyncio.CancelledError), 'cancelled aclose abandoned original members or swallowed cancellation')
        idle()
        report['closed_worker'] = verify_worker(client._runtimes[0].snapshot(), require_so=require_worker_so, closed=True, real=real)
        report['phases'].append({'phase':probe.phase, 'status':'passed', 'normal_worker_exitcode':0})
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', error=error_details(exc))
        raise
    finally:
        probe.released.set()
        try:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await await_completion(asyncio.gather(*tasks, return_exceptions=True))
            await await_completion(client.aclose())
            idle()
            closed = verify_worker(client._runtimes[0].snapshot(), require_so=require_worker_so, closed=True, real=real)
            report['cleanup'] = {'status':'passed', 'model_worker':closed, 'ports_started':False}
        except BaseException as exc:
            report.update(status='failed', cleanup={'status':'failed', 'error':error_details(exc)})
            raise
        finally:
            save()
    return report


def supervise(command, output, watchdog_s):
    """Only this launch's exact PID/ctime descendants; outer deadline includes cleanup."""
    import psutil
    registry = {}
    stops = []
    handlers = {}
    child = None
    forced = terminated = False
    started = time.monotonic()
    for number in (signal.SIGINT, signal.SIGTERM):
        handlers[number] = signal.signal(number, lambda n, f:stops.append(n))
    try:
        with output.with_suffix(output.suffix+'.worker.log').open('w') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            parent = psutil.Process(child.pid)
            identity = {'pid':child.pid, 'create_time':parent.create_time()}
            checkpoint(output.with_suffix(output.suffix+'.ownership.json'), identity)
            deadline = started + watchdog_s
            reserve = min(30, watchdog_s/2)
            while child.poll() is None and not stops and time.monotonic() < deadline-reserve:
                process = same_process(identity)
                if process:
                    for descendant in process.children(recursive=True):
                        try:
                            registry[descendant.pid] = {'pid':descendant.pid, 'create_time':descendant.create_time()}
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                time.sleep(.05)
            if child.poll() is None:
                process = same_process(identity)
                if process:
                    terminated = True
                    process.send_signal(signal.SIGTERM)
                try:
                    child.wait(timeout=max(.01, deadline-time.monotonic()-2))
                except subprocess.TimeoutExpired:
                    forced = True
                    for item in [*registry.values(), identity]:
                        process = same_process(item)
                        if process:
                            process.kill()
                    child.wait(timeout=max(.01, deadline-time.monotonic()))
    finally:
        if child is not None and child.poll() is None:
            process = same_process(identity)
            if process:
                forced = True
                process.kill()
            child.wait(timeout=max(.01, min(2, started+watchdog_s-time.monotonic())))
        # Session belongs exclusively to this start_new_session launch, including
        # a spawn child whose parent exited before the next ancestry sample.
        if child is not None:
            for process in psutil.process_iter(['pid', 'create_time']):
                try:
                    if (process.info['create_time'] >= identity['create_time']
                            and os.getsid(process.pid) == identity['pid']
                            and os.getpgid(process.pid) == identity['pid']):
                        registry[process.pid] = {'pid':process.pid, 'create_time':process.info['create_time']}
                except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
                    pass
        # Give the multiprocessing resource-tracker EOF a bounded natural exit.
        psutil.wait_procs([process for item in registry.values() if (process := same_process(item)) is not None],
                          timeout=max(0, min(1, started+watchdog_s-time.monotonic())))
        remaining = []
        for item in registry.values():
            process = same_process(item)
            if process:
                remaining.append(item)
                forced = True
                process.kill()
        psutil.wait_procs([process for item in registry.values() if (process := same_process(item)) is not None],
                          timeout=max(0, min(1, started+watchdog_s-time.monotonic())))
        remaining_after_force = [item for item in registry.values() if same_process(item) is not None]
        for number, handler in handlers.items():
            signal.signal(number, handler)
    report = json.loads(output.read_text())
    report['supervisor'] = {'ownership':identity, 'returncode':child.returncode, 'owned_descendants':list(registry.values()),
        'remaining_after_child_exit':remaining, 'remaining_after_force':remaining_after_force, 'forced_cleanup':forced, 'term_sent':terminated,
        'stop_signals':stops, 'elapsed_s':time.monotonic()-started, 'child_exited':child.poll() is not None}
    if forced or terminated or child.returncode or report.get('status') != 'passed':
        report['status'] = 'failed'
    checkpoint(output, report)
    return int(report['status'] != 'passed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--wav', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--require-worker-so', action='store_true')
    parser.add_argument('--watchdog-s', type=float, default=420)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 5 <= args.watchdog_s <= 420:
        parser.error('watchdog includes cleanup and must be5..420')
    if not args.worker and args.output.exists():
        parser.error('use a new output path; preserve previous evidence')
    report = read_plan(args)
    require(report['configuration']['dtype'] == 'bfloat16', 'ownership smoke keeps original BF16')
    for key in ('retry_phase_retry_count', 'retry_phase_count', 'production_retry_backoff_s'):
        report['configuration'].pop(key, None)
    report['configuration'].update(require_worker_so=args.require_worker_so, execution_mode='process', phase_count=3)
    report['experiment_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['scope'] = 'owned core process worker; no ports; parent post-DONE test gate, no kernel-delay or protocol injection'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.execute:
        checkpoint(args.output, report)
        return 0
    if args.worker:
        os.environ['CONFIG_PATH'] = str(args.config.resolve())
        stops = []
        async def work():
            task = asyncio.current_task()
            loop = asyncio.get_running_loop()
            def stop(number):
                stops.append(number)
                task.cancel()
            for number in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(number, stop, number)
            qwen = report['configuration']
            await run_validation(args.wav.read_bytes(), model_dir=qwen['model_dir'], dtype=qwen['dtype'],
                max_new_tokens=qwen['max_new_tokens'], report=report, checkpoint_fn=lambda:checkpoint(args.output, report),
                should_stop=lambda:bool(stops), require_worker_so=args.require_worker_so)
        try:
            asyncio.run(work())
            return int(report['status'] != 'passed')
        except BaseException as exc:
            report.update(status='failed', fatal_error=error_details(exc))
            return 1
        finally:
            report['stop_signals'] = stops
            checkpoint(args.output, report)
    checkpoint(args.output, report)
    command = [sys.executable, str(Path(__file__).resolve()), '--config', str(args.config.resolve()),
               '--wav', str(args.wav.resolve()), '--output', str(args.output.resolve()), '--execute', '--worker',
               '--watchdog-s', str(args.watchdog_s)]
    if args.require_worker_so:
        command.append('--require-worker-so')
    return supervise(command, args.output, args.watchdog_s)


if __name__ == '__main__':
    raise SystemExit(main())
