"""真实前缀来源及重定位回归，防止Conda缓存旧包覆盖现用文件。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import pytest
from scripts import pack_actual_prefix as packer


def binary(data):
    try: data.decode('utf-8'); return False
    except UnicodeDecodeError: return True

CORE = SimpleNamespace(is_binary_file=binary)


def test_metadata_only_selects_modes_never_cached_payload(tmp_path):
    prefix=tmp_path/'prefix'; cache=tmp_path/'cache'; prefix.mkdir();cache.mkdir();(prefix/'conda-meta').mkdir()
    target='lib/numpy/__init__.py';(prefix/target).parent.mkdir(parents=True);(cache/target).parent.mkdir(parents=True)
    actual=b'__version__ = "1.26.4"\n';(prefix/target).write_bytes(actual);(cache/target).write_bytes(b'from ._expired_attrs_2_0 import x\n')
    (prefix/'conda-meta/numpy.json').write_text(json.dumps({'link':{'source':str(cache)},'paths_data':{'paths':[{'_path':target}]}}))
    entry, relocation=packer.source_record(prefix,target,packer.relocation_modes(prefix),CORE)
    assert entry['source_sha256']==hashlib.sha256(actual).hexdigest()
    assert entry['archive_sha256']==entry['source_sha256']
    assert relocation==(target,str(prefix),'text')


def test_binary_with_unknown_prefix_fails_but_declared_binary_relocates(tmp_path):
    prefix=tmp_path/'prefix';prefix.mkdir();data=b'\xff\x00'+str(prefix).encode()+b'/lib/native.so\x00';(prefix/'native.so').write_bytes(data)
    with pytest.raises(RuntimeError,match='未知二进制含原前缀'):
        packer.source_record(prefix,'native.so',{},CORE)
    entry, record=packer.source_record(prefix,'native.so',{'native.so':'binary'},CORE)
    assert entry['archive_sha256']==hashlib.sha256(data).hexdigest()
    assert record==('native.so',str(prefix),'binary')


def test_symlink_must_be_relative_and_within_prefix(tmp_path):
    prefix=tmp_path/'prefix';prefix.mkdir();(prefix/'real').write_bytes(b'real');(prefix/'safe').symlink_to('real')
    entry, _=packer.source_record(prefix,'safe',{},CORE)
    assert entry['link']=='real'
    (prefix/'absolute').symlink_to(prefix/'real')
    with pytest.raises(RuntimeError,match='禁止绝对'):
        packer.source_record(prefix,'absolute',{},CORE)
    (tmp_path/'outside').write_bytes(b'outside');(prefix/'escaping').symlink_to('../outside')
    with pytest.raises(RuntimeError,match='越界'):
        packer.source_record(prefix,'escaping',{},CORE)


def test_real_pack_keeps_actual_numpy_payload_and_full_byte_audit(tmp_path):
    tool=os.getenv('ASR_CONDA_PACK_TOOL_PYTHON')
    if not tool: pytest.skip('真实打包集成需显式独立conda-pack工具解释器；普通业务环境不安装构建工具')
    prefix=tmp_path/'prefix';cache=tmp_path/'cache';prefix.mkdir();cache.mkdir();(prefix/'conda-meta').mkdir()
    # conda-pack0.8.1用Conda history的mtime生成conda-unpack头；真实前缀存在此文件。
    (prefix/'conda-meta/history').write_text('fixture environment creation\n')
    target='lib/python3.10/site-packages/numpy/__init__.py'
    actual=b'__version__="1.26.4"\n'
    for base,payload in [(prefix,actual),(cache,b'from ._expired_attrs_2_0 import x\n')]:
        path=base/target;path.parent.mkdir(parents=True);path.write_bytes(payload)
    (prefix/'conda-meta/numpy.json').write_text(json.dumps({'name':'numpy','version':'2.2.6','link':{'source':str(cache)},'paths_data':{'paths':[{'_path':target}]}}))
    output=tmp_path/'env.tar.gz';audit=tmp_path/'audit.json';env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    if os.getenv('ASR_CONDA_PACK_TOOL_PYTHONPATH'):env['PYTHONPATH']=os.environ['ASR_CONDA_PACK_TOOL_PYTHONPATH']
    completed=subprocess.run([tool,str(Path(packer.__file__)),'--prefix',str(prefix),'--output',str(output),'--audit',str(audit),'--excludes-json','[]'],env=env,check=False,capture_output=True,text=True)
    assert completed.returncode==0, completed.stdout+'\n'+completed.stderr
    with tarfile.open(output) as archive:
        assert archive.extractfile(target).read()==actual
    report=json.loads(audit.read_text())
    assert report['passed'] and report['package_cache_used'] is False and report['pack_method']=='actual-prefix-v2'
    assert not report['errors']


def test_audited_binary_override_is_bound_to_exact_bytes(tmp_path, monkeypatch):
    prefix=tmp_path/'prefix';prefix.mkdir();target='native.so'
    payload=b'\xff'+str(prefix).encode()+b'/lib\x00';(prefix/target).write_bytes(payload)
    monkeypatch.setattr(packer,'AUDITED_BINARY',{target:{'sha256':hashlib.sha256(payload).hexdigest(),'reason':'fixture audited RPATH'}})
    entry, record=packer.source_record(prefix,target,{},CORE)
    assert entry['audited_binary_override']['reason']=='fixture audited RPATH' and record[2]=='binary'
    (prefix/target).write_bytes(payload+b'changed')
    with pytest.raises(RuntimeError,match='未知二进制'):
        packer.source_record(prefix,target,{},CORE)
