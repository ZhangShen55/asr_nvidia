#!/usr/bin/env python3
"""在仓库外准备源码镜像命名上下文；不安装包、不改原环境/模型、不调用Docker。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

MODELS = {
    'speech_fsmn_vad_zh-cn-16k-common-pytorch': ('model.pt', 'config.yaml', 'configuration.json', 'am.mvn'),
    'speech_campplus_sv_zh_en_16k-common_advanced': ('campplus_cn_en_common.pt', 'config.yaml', 'configuration.json'),
    'emotion2vec_plus_large': ('model.pt', 'config.yaml', 'configuration.json', 'tokens.txt'),
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def wheel(project, version, target, predicate):
    api = f'https://pypi.org/pypi/{project}/{version}/json'
    with urllib.request.urlopen(api, timeout=60) as response: info = json.load(response)
    choices = [entry for entry in info['urls'] if predicate(entry['filename'])]
    if len(choices) != 1: raise RuntimeError(f'wheel选择不唯一: {project} {choices}')
    entry = choices[0]; dest = target / entry['filename']; expected = entry['digests']['sha256']
    if not dest.is_file() or sha(dest) != expected:
        part = dest.with_suffix(dest.suffix + '.part')
        with urllib.request.urlopen(entry['url'], timeout=120) as response, part.open('wb') as out:
            shutil.copyfileobj(response, out)
        if sha(part) != expected: raise RuntimeError(f'wheel摘要不匹配: {dest.name}')
        part.rename(dest)
    return {'project': project, 'version': version, 'filename': dest.name, 'url': entry['url'], 'metadata_url': api, 'sha256': expected}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--runtime-prefix', default='/home/zhangs/conda-envs/asr')
    parser.add_argument('--tool-python', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    work = Path(args.workspace).resolve()
    if work == repo or repo in work.parents: raise SystemExit('workspace必须在仓库外')
    work.mkdir(parents=True, exist_ok=True)
    runtime = work / 'runtime_env'; models = work / 'model_assets'; tools = work / 'tools'
    for directory in (runtime, models, tools): directory.mkdir(exist_ok=True)
    # 显式白名单先筛选，再将目录交给构建器，避免无关模型资料进入上下文。
    model_records = []
    allowed = set()
    for name, files in MODELS.items():
        (models / name).mkdir(exist_ok=True)
        for filename in files:
            source = repo / 'model' / name / filename; dest = models / name / filename
            if not source.is_file() or not source.stat().st_size: raise RuntimeError(f'缺少资产: {source}')
            digest = sha(source)
            if not dest.is_file() or sha(dest) != digest: shutil.copyfile(source, dest)
            allowed.add(dest.relative_to(models).as_posix())
            model_records.append({'path': dest.relative_to(models).as_posix(), 'sha256': digest, 'bytes': dest.stat().st_size})
    actual = {p.relative_to(models).as_posix() for p in models.rglob('*') if p.is_file()}
    if actual != allowed: raise RuntimeError(f'model_assets含非白名单文件: {actual - allowed}')
    save(work / 'model-assets-manifest.json', model_records)
    npu = wheel('torch-npu', '2.5.1.post1', runtime, lambda n: 'cp310-cp310' in n and 'aarch64' in n and n.endswith('.whl'))
    # conda-pack0.8.1需要pkg_resources；只为构建子进程解包兼容工具，不安装到任何现有环境。
    setuptools = wheel('setuptools', '80.9.0', tools, lambda n: n.endswith('py3-none-any.whl'))
    compat = tools / 'setuptools-80.9.0'
    if not (compat / 'pkg_resources' / '__init__.py').is_file():
        compat.mkdir(exist_ok=True)
        with zipfile.ZipFile(tools / setuptools['filename']) as archive:
            for name in archive.namelist():
                if name.startswith('/') or '..' in Path(name).parts: raise RuntimeError('wheel非法路径')
            archive.extractall(compat)
    tool_code = "import importlib.metadata as m,importlib.util,hashlib,json,pathlib; p=pathlib.Path(importlib.util.find_spec('conda_pack').origin); print(json.dumps({'version':m.version('conda-pack'),'module':str(p),'module_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'core_sha256':hashlib.sha256(p.with_name('core.py').read_bytes()).hexdigest()}))"
    tool_env = os.environ.copy(); tool_env['PYTHONDONTWRITEBYTECODE'] = '1'
    tool_record = json.loads(subprocess.check_output([args.tool_python, '-c', tool_code], env=tool_env, text=True))
    if tool_record['version'] != '0.8.1': raise RuntimeError('本交付固定conda-pack0.8.1，请使用已核验工具前缀')
    save(work / 'pack-tool-provenance.json', {'conda_pack':tool_record,'compatibility_wheel':setuptools})
    prefix = Path(args.runtime_prefix).resolve()
    package_code = 'import importlib.metadata as m,json;print(json.dumps(dict(sorted((d.metadata["Name"],d.version) for d in m.distributions()))))'
    env = os.environ.copy(); env['PYTHONDONTWRITEBYTECODE'] = '1'; env['PYTHONNOUSERSITE'] = '1'
    packages = json.loads(subprocess.check_output([str(prefix/'bin/python'), '-c', package_code], env=env, text=True))
    # 文件列表及元数据也进入指纹，不能只凭版本号复用发生过文件变更的环境。
    source_files = []
    for item in sorted(prefix.rglob('*')):
        if item.is_file():
            st = item.stat()
            source_files.append({'path': item.relative_to(prefix).as_posix(), 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns, 'symlink': os.readlink(item) if item.is_symlink() else None})
    save(work / 'source-prefix-files.json', source_files)
    file_fingerprint = hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps(packages, sort_keys=True).encode()).hexdigest()
    archive = runtime / 'asr-env.tar.gz'; state = work / 'environment-pack.json'
    reusable = False
    if archive.is_file() and state.is_file():
        old = json.loads(state.read_text())
        reusable = old.get('package_fingerprint') == fingerprint and old.get('source_file_fingerprint') == file_fingerprint and old.get('sha256') == sha(archive)
    if not reusable:
        if archive.exists(): raise RuntimeError('现有归档不可复用，请选择新workspace，不能覆盖未核验缓存')
        pack_env = env.copy(); pack_env['PYTHONPATH'] = str(compat)
        excludes = ['**/__pycache__', '**/__pycache__/*', '**/*.pyc', '**/*.pyo', 'bin/pytest', 'bin/py.test', 'bin/cython*', 'bin/conda-pack', 'lib/python3.10/site-packages/pytest*', 'lib/python3.10/site-packages/_pytest*', 'lib/python3.10/site-packages/Cython*', 'lib/python3.10/site-packages/cython*', 'lib/python3.10/site-packages/conda_pack*']
        code = """import conda_pack,sys,json,re,pathlib
filters=[('exclude',p) for p in json.loads(sys.argv[3])]
try:
    conda_pack.CondaEnv.from_prefix(sys.argv[1])
    audit={'missing_conda_metadata':[],'ignore_missing_files':False}
except conda_pack.CondaPackException as exc:
    message=str(exc)
    observed=set(re.findall(r'^- ([^ ]+) ([^:]+):$',message,re.M))
    allowed={('pip','26.2.1'),('wheel','0.48.0'),('setuptools','84.0.0'),('packaging','26.3'),('numpy','2.2.6')}
    if observed != allowed or 'Files managed by conda' not in message:
        raise
    audit={'missing_conda_metadata':sorted(observed),'ignore_missing_files':True,'original_error':message,'reason':'已审核宿主历史Conda元数据偏差；按实际存在文件打包，实际distribution锁及解包验证为准'}
pathlib.Path(sys.argv[4]).write_text(json.dumps(audit,ensure_ascii=False,indent=2))
conda_pack.pack(prefix=sys.argv[1],output=sys.argv[2],filters=filters,ignore_missing_files=audit['ignore_missing_files'])
"""
        subprocess.run([args.tool_python, '-c', code, str(prefix), str(archive), json.dumps(excludes), str(work / "conda-metadata-audit.json")], env=pack_env, check=True)
        with tarfile.open(archive) as packed:
            names = {n.name.lstrip('./') for n in packed}
            if not {'bin/python', 'bin/conda-unpack'} <= names: raise RuntimeError('环境缺少重定位入口')
        save(state, {'source_prefix': str(prefix), 'tool_python': args.tool_python, 'tool_compatibility_wheel': setuptools, 'package_fingerprint': fingerprint, 'source_file_fingerprint': file_fingerprint, 'packages': packages, 'sha256': sha(archive), 'bytes': archive.stat().st_size, 'excludes': excludes})
    expected = dict(packages); expected['torch-npu'] = '2.5.1.post1'
    expected.pop('torch_npu', None)
    for key in list(expected):
        if key.lower().replace('_','-') in ('cython','conda-pack','pytest'): expected.pop(key)
    save(runtime / 'runtime-packages.json', expected)
    save(runtime / 'wheel-source.json', npu)
    expected_names = {'asr-env.tar.gz', npu['filename'], 'runtime-packages.json', 'wheel-source.json'}
    actual_names = {p.name for p in runtime.iterdir()}
    if actual_names != expected_names or any(not p.is_file() for p in runtime.iterdir()):
        raise RuntimeError(f'runtime_env含非白名单输入: {actual_names - expected_names}')
    with tarfile.open(archive) as packed:
        members = [{'path':m.name,'bytes':m.size,'type':m.type.decode('ascii'),'link':m.linkname} for m in packed]
    save(work / 'packed-files.json', members)
    save(work / 'context-preparation.json', {'runtime_env': str(runtime), 'model_assets': str(models), 'runtime_env_files': [{'path': p.name, 'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(runtime.iterdir())], 'models': model_records, 'host_environment_modified': False})
    print(json.dumps({'runtime_env': str(runtime), 'model_assets': str(models), 'environment_reused': reusable}, ensure_ascii=False))

if __name__ == '__main__': main()
