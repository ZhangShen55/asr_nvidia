"""宿主控制器：只验收指定的owned源码容器，真实暖身/B4及正常停止。"""
from __future__ import annotations
import argparse
import asyncio
import base64
import hashlib
import ipaddress
import re
import json
import os
from pathlib import Path, PurePosixPath
import socket
import signal
import subprocess
import sys
import tarfile
import threading
import time
from urllib.parse import urlparse
import wave

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import httpx
import psutil
import tomli
from scripts.benchmark_qwen_multi_instance import (
    batch, covered_owners, enhancement_validation, stage_evidence, shared_identity,
    get_status, background_snapshot, device7_hbm,
)
from scripts.benchmark_asr_throughput import write_json


def emit(phase, **value):
    print(json.dumps(dict(phase=phase, **value), ensure_ascii=True), flush=True)


def command(args, *, timeout=30):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{args!r}: exit={result.returncode}: {result.stderr[-3000:]}")
    return result.stdout


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(path):
    source = json.loads(Path(path).read_text())
    if [row.get('sample_id') for row in source[:4]] != ['a1', 'a2', 'a3', 'a4']:
        raise ValueError('必须使用canonical a1～a4完整清单')
    result, seen = [], set()
    for original in source[:4]:
        row = dict(original)
        row['path'] = os.fsdecode(base64.b64decode(row['path_bytes_b64'], validate=True))
        if sha(row['path']) != row['sha256'] or row['sha256'] in seen:
            raise ValueError('完整文件SHA不匹配或重复')
        with wave.open(row['path'], 'rb') as wav:
            duration = wav.getnframes()/wav.getframerate()
        if abs(duration-row['audio_seconds']) > .001 or duration < 3000:
            raise ValueError('不是约定的完整音频或时长不匹配')
        row.update(verified_duration_s=duration, size_bytes=Path(row['path']).stat().st_size)
        seen.add(row['sha256']); result.append(row)
    return result


def config_contract(path):
    cfg = tomli.loads(Path(path).read_text())
    q, pools = cfg.get('qwen_asr', {}), cfg.get('model_device_pool', {})
    expected = ('continuous', 'process', 4, 16, 'npu:0', 4)
    actual = tuple(q.get(key, 4 if key == 'audio_spk_emo_concurrency' else None) for key in
                   ('generation_mode', 'execution_mode', 'local_instance_count', 'segment_concurrency', 'local_device', 'audio_spk_emo_concurrency'))
    if actual != expected or type(cfg.get('instance_count', 1)) is not int or cfg.get('instance_count', 1) != 1:
        raise ValueError('只接受HTTP1/Qwen4/S16/逻辑npu0/B4配置')
    if pools.get('speech_campplus_devices') != ['npu:0']*4 or pools.get('emotion2vec_devices') != ['npu:0']:
        raise ValueError('只接受CAM4/emo1共驻配置')
    return cfg


class DockerOwner:
    def __init__(self, name, output, *, expected_image_id, task_id, base_url):
        item = json.loads(command(['docker', 'inspect', name]))[0]
        # 在取得stop资格前只读inspect，不能因误传共享容器名而执行任何生命周期操作。
        if (not re.fullmatch(r'sha256:[0-9a-f]{64}', expected_image_id or '') or item.get('Image') != expected_image_id
                or not task_id or item.get('Config', {}).get('Labels', {}).get('asr.build.task') != task_id):
            raise ValueError('容器image ID或asr.build.task归属不匹配；禁止停止该容器')
        if not item.get('State', {}).get('Running'):
            raise ValueError('指定owned容器未运行')
        if not re.fullmatch(r'[0-9a-f]{64}', item.get('Id', '')):
            raise ValueError('无法取得immutable container ID；禁止停止')
        parsed = urlparse(base_url)
        try:
            address = ipaddress.ip_address(parsed.hostname or '')
        except ValueError as exc:
            raise ValueError('base-url必须为明确的loopback IP，避免localhost多地址误连共享服务') from exc
        if (parsed.scheme != 'http' or not address.is_loopback or parsed.username or parsed.password
                or parsed.path not in ('', '/') or parsed.query or parsed.fragment):
            raise ValueError('base-url必须为无认证/路径的loopback HTTP地址')
        published = item.get('NetworkSettings', {}).get('Ports', {}).get('9000/tcp') or []
        host_ips = {str(address), '0.0.0.0' if address.version == 4 else '::'}
        if not any(row.get('HostIp') in host_ips and row.get('HostPort') == str(parsed.port or 80) for row in published):
            raise ValueError('base-url与目标容器9000/tcp的实际发布端口不匹配；禁止请求或停止')
        self.id, self.image, self.output = item['Id'], item['Image'], output
        self.initial = item
        self.pids, self.last, self.state = {}, 0., item['State']
        self.process = self
        self.verified_owned = True
        # 登记和写证据必须由execute在try内执行；构造后不再进行可失败I/O。


    def register(self):
        if time.monotonic()-self.last < 2:
            return
        item = json.loads(command(['docker', 'inspect', self.id], timeout=10))[0]
        if item['Image'] != self.image:
            raise RuntimeError('候选镜像身份改变')
        self.state, self.last = item['State'], time.monotonic()
        if not self.state['Running']:
            return
        parent = psutil.Process(self.state['Pid'])
        for proc in [parent, *parent.children(recursive=True)]:
            try:
                text = Path(f'/proc/{proc.pid}/status').read_text()
                nspid = next((list(map(int, line.split()[1:])) for line in text.splitlines() if line.startswith('NSpid:')), [])
                record = {'host_pid': proc.pid, 'birth': proc.create_time(), 'nspid': nspid,
                          'pid_namespace': os.readlink(f'/proc/{proc.pid}/ns/pid'), 'name': proc.name()}
                self.pids[f'{proc.pid}:{record["birth"]}'] = record
            except (psutil.Error, OSError):
                continue
        write_json(self.output/'owned-identities.json', list(self.pids.values()))

    def poll(self):
        return None if self.state.get('Running') else self.state.get('ExitCode', -1)

    def map_owner(self, pid, birth):
        matches = [p for p in self.pids.values() if p['nspid'] and p['nspid'][-1] == pid
                   and abs(p['birth']-birth) < .1]
        return matches[0] if len(matches) == 1 else None


def validate_status(status, owner, *, idle=False, resident=False, baseline=None):
    errors = []
    q = status.get('qwenModel', {}); pool = q.get('owner_pool', {})
    checks = [(q.get('generation_mode'), 'continuous'), (q.get('execution_mode'), 'process'),
              (q.get('device'), 'npu:0'), (pool.get('configured'), 4), (pool.get('healthy'), 4),
              (pool.get('segment_capacity_per_owner'), 16), (pool.get('configured_global_segments'), 64),
              (status.get('audioConcurrency', {}).get('configured_total'), 4),
              (status.get('spkEmoConcurrency', {}).get('effective'), 4)]
    if any(observed != expected for observed, expected in checks):
        errors.append({'configuration_mismatch': checks})
    instances = q.get('instances', [])
    if len(instances) != 4:
        errors.append('four owner records required')
    mappings, cold_uninitialized = {}, []
    pool_owners = {row.get('index'): row for row in pool.get('owners', [])}
    pcm = status.get('resourceDiagnostics', {}).get('segmentPipeline', {})
    cold_global_idle = (baseline is None and pool.get('leased') == 0 and pool.get('available') == 4
                        and len(pool_owners) == 4
                        and status.get('audioConcurrency', {}).get('active_total') == 0
                        and status.get('spkEmoConcurrency', {}).get('active') == 0
                        and q.get('batch_scheduler', {}).get('running_members') == 0
                        and pcm.get('active') == 0 and pcm.get('audio_bytes') == 0)
    for item in instances:
        worker = item.get('model_worker', {}); entry = worker.get('worker_entry_module_file')
        launch = worker.get('launch_config', {})
        pid, birth = worker.get('worker_pid'), worker.get('worker_create_time')
        mapped = owner.map_owner(pid, birth) if type(pid) is int and type(birth) in (int, float) else None
        mappings[str(item.get('index'))] = mapped
        if not mapped or not isinstance(entry, str) or not PurePosixPath(entry).is_relative_to('/app/core') or not entry.endswith('.py'):
            errors.append(f'owner {item.get("index")}: source/host PID mapping missing')
        if launch.get('device') != 'npu:0' or launch.get('continuous_options', {}).get('max_active') != 16:
            errors.append('owner launch device/window mismatch')
        peak = (item.get('continuous') or {}).get('peak_active')
        if baseline is not None and (type(peak) is not int or not 0 <= peak <= 16):
            errors.append('owner peak window unknown or exceeds16')
        continuous = item.get('continuous')
        capacity_owner = pool_owners.get(item.get('index'), {})
        # 预加载完成不代表调度器已创建。仅在冷态、身份健康且整个服务确实无租约/
        # 在途/PCM时，将明确的未初始化状态记录为冷空闲，绝不将unknown普遍当成0。
        cold_idle = (cold_global_idle and continuous is None
                     and item.get('loaded') is True and item.get('leased') is False
                     and item.get('busy') is False and item.get('request_id') is None and not item.get('failure')
                     and item.get('active_sequences') in (None, 0)
                     and worker.get('ready_verified') is True and worker.get('worker_alive') is True
                     and worker.get('closing') is False and worker.get('closed') is False
                     and capacity_owner.get('loaded') is True and capacity_owner.get('healthy') is True
                     and capacity_owner.get('available') is True and capacity_owner.get('leased') is False
                     and capacity_owner.get('releasing') is False and capacity_owner.get('closing') is False
                     and 'active_sequences' in capacity_owner and capacity_owner['active_sequences'] is None
                     and not capacity_owner.get('failure'))
        if idle and (not isinstance(continuous, dict) or continuous.get('active_sequences') != 0):
            if cold_idle:
                cold_uninitialized.append(item.get('index'))
            else:
                errors.append('owner not idle')
    if baseline is not None:
        current = covered_owners(status)
        if set(current) != set(baseline) or any((current[k]['pid'], current[k]['birth']) !=
                (baseline[k]['pid'], baseline[k]['birth']) for k in current.keys() & baseline.keys()):
            errors.append('owner identity/coverage changed')
    if baseline is not None:
        peak = q.get('batch_scheduler', {}).get('peak_inflight')
        if type(peak) is not int or not 0 <= peak <= 64:
            errors.append('global peak window unknown or exceeds64')
    if idle:
        pcm = status.get('resourceDiagnostics', {}).get('segmentPipeline', {})
        values = [status.get('audioConcurrency', {}).get('active_total'),
                  status.get('spkEmoConcurrency', {}).get('active'), pool.get('leased'),
                  q.get('batch_scheduler', {}).get('running_members'), pcm.get('active'), pcm.get('audio_bytes')]
        if any(value != 0 for value in values):
            errors.append({'not_idle_or_unknown': values})
    for name, count in [('speech_campplus', 4), ('emotion2vec', 1)]:
        model = status.get('localModels', {}).get(name, {})
        if model.get('configured_instances') != count or model.get('instances') != count or model.get('failed_instances', 0):
            errors.append(f'{name}: health mismatch')
        if resident and model.get('called_instances') != count:
            errors.append(f'{name}: real instance coverage missing')
    return {'passed': not errors, 'errors': errors, 'host_namespace_mappings': mappings,
            'cold_uninitialized_idle_owners': cold_uninitialized}


async def idle_snapshot(client, owner, *, baseline=None, resident=False):
    last = {}
    for _ in range(30):
        owner.register(); status = await get_status(client)
        last = validate_status(status, owner, idle=True, resident=resident, baseline=baseline)
        if last['passed']:
            return status, last
        await asyncio.sleep(1)
    raise RuntimeError(f'idle/health evidence unavailable: {last}')


def stopped_cache(container):
    """只消费tar目录头；遇首个文件即停，绝不将音频复制到宿主。"""
    process = subprocess.Popen(['docker', 'cp', f'{container}:/tmp/asr-offline/seacraft_asr', '-'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timer = threading.Timer(15, process.kill); timer.start()
    records = []
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            for member in archive:
                if not member.isdir():
                    records.append({'name': member.name, 'size': member.size, 'type': str(member.type)})
                    break
        if records:
            process.terminate()
        _, stderr = process.communicate(timeout=3)
        return {'passed': not records and process.returncode == 0, 'non_directory_entries': records,
                'copy_exit': process.returncode, 'stderr': stderr.decode(errors='replace'), 'scope': 'stopped container cache tar headers; no audio saved'}
    except tarfile.ReadError as exc:
        stdout, stderr = process.communicate(timeout=3)
        message = stderr.decode(errors='replace').strip()
        missing = f'Error response from daemon: Could not find the file /tmp/asr-offline/seacraft_asr in container {container}'
        # 完全无请求时目录尚未创建；只接受固定路径、当前immutable容器和精确daemon错误。
        absent = process.returncode == 1 and str(exc) == 'empty file' and not stdout and message == missing
        return {'passed': absent, 'directory_absent': absent, 'copy_exit': process.returncode,
                'stderr': message, 'error': None if absent else repr(exc),
                'scope': 'fixed upload PCM cache directory absent' if absent else 'unknown cache read failure'}
    except Exception as exc:
        return {'passed': False, 'status': 'unknown', 'error': repr(exc)}
    finally:
        timer.cancel()
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=3)


def cleanup(owner, url, shared_before):
    if getattr(owner, 'verified_owned', False) is not True:
        return {'passed': False, 'not_stopped': True, 'reason': 'ownership not verified'}
    errors = []
    try:
        owner.last = 0; owner.register()
    except Exception as exc:
        errors.append({'registration_error': repr(exc)})
        emit('cleanup_registration_failed', error=repr(exc))
    # 即使/proc登记或证据写入失败，也必须按已验证immutable ID正常停止。
    try:
        stop = command(['docker', 'stop', '--time', '240', owner.id], timeout=255)
    except Exception as exc:
        stop = None
        errors.append({'stop_error': repr(exc)})
        emit('normal_stop_failed', error=repr(exc))
    item = json.loads(command(['docker', 'inspect', owner.id]))[0]
    state = item['State']
    alive = []
    for _ in range(30):
        alive = []
        for identity in owner.pids.values():
            try:
                proc = psutil.Process(identity['host_pid'])
                if abs(proc.create_time()-identity['birth']) < .1:
                    alive.append(dict(identity, status=proc.status()))
            except psutil.Error:
                pass
        if not alive:
            break
        time.sleep(.5)
    parsed = urlparse(url); host, port = parsed.hostname, parsed.port or 80
    end, since = time.monotonic()+20, None
    while time.monotonic() < end:
        try:
            with socket.create_connection((host, port), timeout=.3):
                since = None
        except OSError:
            since = time.monotonic() if since is None else since
        if since is not None and time.monotonic()-since >= 5:
            break
        time.sleep(.25)
    port_free = since is not None and time.monotonic()-since >= 5
    after = shared_identity()
    shared_ok = (shared_before.get('available') and after.get('available') and after.get('alive')
                 and shared_before.get('birth') == after.get('birth')
                 and bool(shared_before.get('listening_ports'))
                 and shared_before.get('listening_ports') == after.get('listening_ports'))
    cache = stopped_cache(owner.id)
    passed = (not errors and not state['Running'] and state.get('ExitCode') == 0 and not state.get('OOMKilled')
              and not state.get('Error') and not alive and port_free and shared_ok and cache['passed'])
    return {'passed': bool(passed), 'state': state, 'errors': errors, 'stop_stdout': stop, 'remaining_owned': alive,
            'port_free_5s': port_free, 'cache': cache, 'shared_before': shared_before,
            'shared_after': after, 'shared_preserved': bool(shared_ok)}


async def execute(args):
    output = Path(args.output)
    owner, shared_before = None, {}
    result = {'passed': False, 'scope': 'B4 source container, not human accuracy certification'}
    config_digest, output_claimed = None, False
    try:
        owner = DockerOwner(args.container, output, expected_image_id=args.expected_image_id, task_id=args.task_id, base_url=args.base_url)
        output.mkdir(parents=True, exist_ok=False)
        output_claimed = True
        owner.register()
        write_json(output/'container-before.json', owner.initial)
        shared_before = shared_identity()
        config_contract(args.config)
        inputs = load_inputs(args.manifest)
        write_json(output/'inputs.json', inputs)
        config_digest = sha(args.config)
        write_json(output/'provenance.json', {'config_sha256': config_digest, 'tool_sha256': sha(__file__),
                   'clip_sha256': sha(args.clip), 'helper_sha256': sha(ROOT/'scripts/benchmark_qwen_multi_instance.py'),
                   'source_revision': command(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']).strip(),
                   'physical_npu': 7, 'logical_npu': 0, 'hard_s': 336, 'no_progress_s': 120})
        mounts = {row['Destination']: row for row in owner.initial.get('Mounts', [])}
        mounted = mounts.get('/config.toml', {})
        if mounted.get('RW') is not False or Path(mounted.get('Source', '')).resolve() != Path(args.config).resolve():
            raise ValueError('actual readonly config mount differs from validated config')
        if mounts.get('/app/qwen-model', {}).get('RW') is not False:
            raise ValueError('Qwen mount missing or writable')
        environment = dict(item.split('=', 1) for item in owner.initial.get('Config', {}).get('Env', []) if '=' in item)
        if (environment.get('ASCEND_VISIBLE_DEVICES') != '7' or environment.get('ASCEND_RT_VISIBLE_DEVICES') != '7'
                or environment.get('ASR_CONTAINER_RUNTIME') != 'cann82-source'):
            raise ValueError('physical device/container runtime marker mismatch')
        if not shared_before.get('available'):
            raise RuntimeError('shared service identity unavailable')
        with wave.open(args.clip, 'rb') as clip:
            if not 1 <= clip.getnframes()/clip.getframerate() <= 30:
                raise ValueError('暖身需要1～30秒真实语音clip')
        clip_samples = [{'sample_id': f'w{i}', 'path': args.clip, 'sha256': sha(args.clip)} for i in range(4)]
        async with httpx.AsyncClient(base_url=args.base_url, trust_env=False, timeout=httpx.Timeout(370, connect=10), limits=httpx.Limits(max_connections=10)) as client:
            await get_status(client)
            # 给异步资源采样一次刷新机会，不能将首个unknown当作零。
            await asyncio.sleep(2)
            initial, _ = await idle_snapshot(client, owner)
            write_json(output/'initial-status.json', initial)
            for index in range(2):
                emit('warmup_start', round=index)
                warm = await batch(client, clip_samples, 'B', output/f'warmup-{index}', owner, owner_count=4)
                if not warm['passed']:
                    raise RuntimeError(f'warmup failed: {warm.get("trigger")}')
                current, _ = await idle_snapshot(client, owner)
                coverage = covered_owners(current)
                write_json(output/f'warmup-{index}-coverage.json', coverage)
                if len(coverage) == 4 and all(current.get('localModels', {}).get(name, {}).get('called_instances') == count
                        for name, count in [('speech_campplus', 4), ('emotion2vec', 1)]):
                    break
            else:
                raise RuntimeError('bounded warmup did not cover every owner/CAM/emo instance')
            before, readiness = await idle_snapshot(client, owner, baseline=coverage, resident=True)
            write_json(output/'warm-baseline-status.json', before)
            write_json(output/'readiness.json', readiness)
            host_before = await background_snapshot()
            write_json(output/'host-before-B4.json', host_before)
            result['physical7_before_B4'] = device7_hbm(host_before)
            if not result['physical7_before_B4']['available']:
                raise RuntimeError('physical7 HBM observation unavailable')
            emit('B4_start', container_id=owner.id, image_id=owner.image)
            complete = await batch(client, inputs, 'B', output/'B4', owner, owner_count=4)
            result['B4'] = complete
            if not complete['passed']:
                raise RuntimeError(f'B4 failed: {complete.get("trigger")}')
            after, validation = await idle_snapshot(client, owner, baseline=coverage, resident=True)
            enhancement = enhancement_validation(before, after, complete['records'])
            stages = stage_evidence(before, after, complete['records'], 'B')
            write_json(output/'final-status.json', after)
            write_json(output/'B4-enrichment.json', {'enhancement': enhancement, 'stages': stages, 'service': validation})
            result.update(B4=complete, enhancement=enhancement, stages=stages, service=validation,
                          passed=bool(complete['passed'] and enhancement['passed'] and stages['passed'] and validation['passed']))
            emit('B4_finished', complete=complete['completed'], wall_s=complete['wall_s'], passed=result['passed'])
    except BaseException as exc:
        result.update(passed=False, error=repr(exc))
        if isinstance(exc, asyncio.CancelledError):
            result['cancel_reason'] = repr(exc)
        emit('failed', error=repr(exc))
    finally:
        async def finalize():
            try:
                if owner is not None and owner.verified_owned:
                    result['cleanup'] = await asyncio.to_thread(cleanup, owner, args.base_url, shared_before)
                else:
                    result['cleanup'] = {'passed': False, 'not_stopped': True, 'reason': 'ownership not verified'}
            except BaseException as exc:
                result['cleanup'] = {'passed': False, 'error': repr(exc)}
            try:
                result['config_unchanged'] = config_digest is not None and sha(args.config) == config_digest
            except OSError as exc:
                result['config_unchanged'] = False
                result['config_error'] = repr(exc)
            result['passed'] = bool(result['passed'] and result['cleanup']['passed'] and result['config_unchanged'])
            try:
                host_after = await background_snapshot()
                result['physical7_after_stop'] = device7_hbm(host_after)
                result['passed'] = bool(result['passed'] and result['physical7_after_stop']['available'])
                if output_claimed:
                    write_json(output/'host-after-stop.json', host_after)
            except BaseException as exc:
                result.update(passed=False, final_snapshot_error=repr(exc))
                emit('final_snapshot_failed', error=repr(exc))
            try:
                if output_claimed:
                    write_json(output/'result.json', result)
                else:
                    emit('result_without_output_directory', result=result)
            except BaseException as exc:
                result.update(passed=False, final_evidence_error=repr(exc))
                emit('evidence_write_failed', error=repr(exc), result=result)
            emit('finished', passed=result['passed'], cleanup=result['cleanup']['passed'])
        # finalizer拥有独立Task；二次TERM只取消外层观察，不能丢掉stop结果或最后的证据。
        finalizer = asyncio.create_task(finalize())
        while not finalizer.done():
            try:
                await asyncio.shield(finalizer)
            except asyncio.CancelledError as exc:
                result['passed'] = False
                result.setdefault('finalizer_cancellations', []).append(repr(exc))
        finalizer.result()
    return 0 if result['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', required=True)
    parser.add_argument('--expected-image-id', required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:18083')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--clip', required=True)
    args = parser.parse_args()
    async def supervised():
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        # TERM/INT只取消控制器的HTTP观察，finally仍须等待正常容器关闭。
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(signum, task.cancel)
        return await execute(args)
    raise SystemExit(asyncio.run(supervised()))


if __name__ == '__main__':
    main()
