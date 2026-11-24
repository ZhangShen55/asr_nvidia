#!/usr/bin/env python3
"""只从真实前缀取字节，Conda元数据仅提供重定位类型；不读package cache内容。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile

METHOD = 'actual-prefix-v2'
# 只对已用readelf审计过的原生RPATH登记精确内容；其它未知二进制仍失败。
AUDITED_BINARY = {
    'lib/python3.10/site-packages/crcmod/_crcfunext.cpython-310-aarch64-linux-gnu.so': {
        'sha256':'ebb7d8414aa14a151be30efd6c35bae8482ebad84e111808a02086ec5f58d12c',
        'reason':'ELF DT_RPATH=/home/zhangs/conda-envs/asr/lib，NEEDED仅libc.so.6；按已知binary重定位，候选容器须核验实际RPATH和crcmod导入',
    },
}

def digest(data): return hashlib.sha256(data).hexdigest()

def relocation_modes(prefix):
    modes = {}
    for path in sorted((prefix/'conda-meta').glob('*.json')):
        info = json.loads(path.read_text())
        for item in info.get('paths_data', {}).get('paths', []):
            name, mode = item.get('_path'), item.get('file_mode')
            if not isinstance(name, str) or name.startswith('/') or '..' in Path(name).parts:
                raise RuntimeError(f'非法Conda管理路径: {name}')
            if mode in ('text', 'binary'):
                if name in modes and modes[name] != mode: raise RuntimeError(f'重定位类型冲突: {name}')
                modes[name] = mode
    return modes


def source_record(prefix, target, modes, core):
    source = prefix/target
    if Path(target).is_absolute() or '..' in Path(target).parts: raise RuntimeError('源路径越界')
    if source.is_symlink():
        link = os.readlink(source)
        if os.path.isabs(link) or not source.resolve(strict=True).is_relative_to(prefix):
            raise RuntimeError(f'禁止绝对或越界symlink: {target} -> {link}')
        return {'path':target,'type':'symlink','link':link,'source_sha256':digest(link.encode()),'archive_sha256':None,'transformation':'relative internal symlink'}, None
    if not source.resolve(strict=True).is_relative_to(prefix): raise RuntimeError(f'源不在实际prefix: {target}')
    if source.is_dir(): return {'path':target,'type':'directory'}, None
    data = source.read_bytes(); mode = modes.get(target, 'unknown')
    original = digest(data); expected = data; transformation = 'identity'; record = None
    if target.startswith('conda-meta/') and target.endswith('.json'):
        mode = None
        expected, _ = core.rewrite_conda_meta(str(source)); transformation = 'rewrite_conda_meta'
    elif mode == 'unknown':
        if core.is_binary_file(data):
            # 库对Unix未知二进制不重定位；不能将含宿主路径的文件悄悄打入镜像。
            if str(prefix).encode() in data:
                reviewed = AUDITED_BINARY.get(target)
                if not reviewed or reviewed['sha256'] != original:
                    raise RuntimeError(f'未知二进制含原前缀，须审计其重定位类型: {target}')
                mode = 'binary'
        else:
            mode = 'text'
    if mode in ('text','binary'):
        fixed = False
        if mode == 'text' and target.startswith('bin'):
            expected, fixed = core.rewrite_shebang(data, target, str(prefix))
            if fixed: transformation = 'rewrite_shebang'
        if not fixed and not re.match('bin/python', target): record = (target,str(prefix),mode)
    return {'path':target,'type':'file','bytes':len(data),'source_sha256':original,'archive_sha256':digest(expected),'mode':mode,'transformation':transformation,'audited_binary_override':AUDITED_BINARY.get(target) if mode=='binary' and target not in modes else None}, record


def pack(prefix, output, audit_path, excludes):
    import conda_pack
    from conda_pack import core
    from fnmatch import fnmatch
    prefix = prefix.resolve(strict=True)
    if output.exists(): raise RuntimeError('v2不覆盖已有归档')
    modes = relocation_modes(prefix)
    generated = {target for _,target in core._scripts} | {'bin/conda-unpack'}
    inputs = sorted(core.load_files(str(prefix)))
    selected = [p for p in inputs if p not in generated and not any(fnmatch(p,rule) for rule in excludes)]
    manifest, files, records, errors = {}, [], [], []
    for target in selected:
        try:
            entry, relocation = source_record(prefix,target,modes,core)
            manifest[target] = entry
            if relocation: records.append(relocation)
            mode = entry.get('mode', 'unknown')
            files.append(conda_pack.File(str(prefix/target),target,is_conda=target.startswith('conda-meta/'),file_mode=mode,prefix_placeholder=str(prefix) if mode in ('text','binary') else None))
        except Exception as exc: errors.append({'path':target,'error':repr(exc)})
    audit = {'pack_method':METHOD,'source_prefix':str(prefix),'all_sources_from_actual_prefix':True,'package_cache_used':False,'source_files':list(manifest.values()),'errors':errors,'passed':False}
    audit_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    if errors: raise RuntimeError(f'源审计失败，见 {audit_path}: {errors[:8]}')
    for source,target in core._scripts:
        manifest[target] = {'path':target,'type':'file','archive_sha256':digest(Path(source).read_bytes()),'transformation':'conda-pack generated activation template'}
    prefixes_py = Path(core.__file__).with_name('prefixes.py').read_text()
    unpack = core._conda_unpack_template.format(shebang='#!/usr/bin/env python',prefixes_py=prefixes_py,prefix_records=',\n'.join(repr(r) for r in records),version=conda_pack.__version__)
    manifest['bin/conda-unpack'] = {'path':'bin/conda-unpack','type':'file','archive_sha256':digest(unpack.encode()),'transformation':'conda-pack generated relocation template'}
    conda_pack.CondaEnv(str(prefix),files).pack(output=str(output),verbose=True)
    observed, archive_rows = {}, []
    with tarfile.open(output,mode='r|gz') as archive:
        for member in archive:
            target = member.name.rstrip('/')
            if target in observed: raise RuntimeError(f'归档重复路径: {target}')
            expected = manifest.get(target)
            if expected is None: raise RuntimeError(f'归档出现未批准文件: {target}')
            if member.isfile():
                h=hashlib.sha256()
                stream=archive.extractfile(member)
                for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
                value=h.hexdigest()
            elif member.islnk():
                value=observed[member.linkname]
            elif member.issym():
                if expected.get('type')!='symlink' or member.linkname!=expected.get('link'):raise RuntimeError(f'symlink改变: {target}')
                value=None
            elif member.isdir(): value=None
            else: raise RuntimeError(f'归档类型不支持: {target}')
            if value != expected.get('archive_sha256'):raise RuntimeError(f'归档字节不匹配: {target}: {value} != {expected.get("archive_sha256")}')
            observed[target]=value;archive_rows.append({'path':target,'sha256':value,'transformation':expected.get('transformation')})
    if set(observed)!=set(manifest):raise RuntimeError(f'归档遗漏: {set(manifest)-set(observed)}')
    audit.update(source_files=list(manifest.values()),archive_files=archive_rows,passed=True,verified_count=len(observed))
    audit_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'pack_method':METHOD,'source_archive_sha_passed':True,'files':len(observed)}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',required=True);p.add_argument('--output',required=True);p.add_argument('--audit',required=True);p.add_argument('--excludes-json',required=True)
    a=p.parse_args();pack(Path(a.prefix),Path(a.output),Path(a.audit),json.loads(a.excludes_json))

if __name__=='__main__':main()
