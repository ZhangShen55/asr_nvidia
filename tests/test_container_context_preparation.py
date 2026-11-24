"""上下文供应链边界：平台/依赖/摘要和额外文件均须失败关闭。"""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import pytest
from scripts import prepare_container_context as preparation


def entry(name):
    return {'filename': name, 'yanked': False}


def test_target_wheel_accepts_abi3_but_rejects_musl_other_python_and_ambiguity():
    valid = 'tornado-6.5.1-cp39-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl'
    incompatible = ['tornado-6.5.1-cp39-abi3-musllinux_1_2_aarch64.whl',
                    'tornado-6.5.1-cp311-cp311-manylinux_2_17_aarch64.whl',
                    'tornado-6.5.1-cp39-abi3-manylinux_2_17_x86_64.whl']
    info = {'info': {'requires_python': '>=3.9'}, 'urls': [entry(n) for n in [*incompatible, valid]]}
    assert preparation.select_wheel(info, 'tornado', '6.5.1')['filename'] == valid
    info['urls'].append(entry('tornado-6.5.1-py3-none-any.whl'))
    with pytest.raises(RuntimeError, match='不唯一'):
        preparation.select_wheel(info, 'tornado', '6.5.1')
    info['info']['requires_python'] = '>=3.11'
    with pytest.raises(RuntimeError, match='Requires-Python'):
        preparation.select_wheel(info, 'tornado', '6.5.1')


def test_metadata_preserves_numpy_pin_and_ignores_unrequested_dev_extra(tmp_path):
    wheel = tmp_path/'fixture.whl'
    def write(requirement):
        with zipfile.ZipFile(wheel, 'w') as out:
            out.writestr('ml_dtypes-0.5.1.dist-info/METADATA', 'Name: ml-dtypes\nVersion: 0.5.1\nRequires-Python: >=3.9\nRequires-Dist: '+requirement+'\nRequires-Dist: pytest; extra == "dev"\n')
    write('numpy>=1.21.2; python_version >= "3.10"')
    result = preparation.validate_wheel_metadata(wheel, 'ml-dtypes', '0.5.1', {'numpy':'1.26.4'})
    assert result['target_python'] == '3.10.22'
    write('numpy>=2')
    with pytest.raises(RuntimeError, match='容器锁不兼容'):
        preparation.validate_wheel_metadata(wheel, 'ml-dtypes', '0.5.1', {'numpy':'1.26.4'})


def test_bad_wheel_download_cannot_replace_verified_destination(tmp_path, monkeypatch):
    name = 'absl_py-2.3.1-py3-none-any.whl'
    expected = hashlib.sha256(b'expected').hexdigest()
    info = {'info':{}, 'urls':[dict(entry(name), digests={'sha256':expected}, url='https://files.pythonhosted.org/fake.whl')]}
    def retrieve(url, **kwargs):
        return io.BytesIO(json.dumps(info).encode() if url.endswith('/json') else b'corrupted')
    monkeypatch.setattr(preparation.urllib.request, 'urlopen', retrieve)
    with pytest.raises(RuntimeError, match='摘要不匹配'):
        preparation.wheel('absl-py', '2.3.1', tmp_path)
    assert not (tmp_path/name).exists()


def test_context_refuses_hidden_extra_and_symlink(tmp_path):
    (tmp_path/'archive').write_bytes(b'payload')
    preparation.validate_context_files(tmp_path, {'archive'})
    (tmp_path/'.secret').write_bytes(b'never build')
    with pytest.raises(RuntimeError, match='文件集合'):
        preparation.validate_context_files(tmp_path, {'archive'})
    (tmp_path/'.secret').unlink()
    (tmp_path/'archive').unlink()
    target = tmp_path.parent/'outside-context-fixture'
    target.write_bytes(b'external')
    (tmp_path/'archive').symlink_to(target)
    with pytest.raises(RuntimeError, match='符号链接'):
        preparation.validate_context_files(tmp_path, {'archive'})


def valid_byte_audit(prefix, source):
    return {'pack_method':'actual-prefix-v2','passed':True,'source_prefix':str(prefix),
            'all_sources_from_actual_prefix':True,'package_cache_used':False,'errors':[],
            'source_files':[{'path':source.name,'type':'file','source_sha256':preparation.sha(source)}]}


def test_reuse_rehashes_source_even_if_size_and_mtime_are_preserved(tmp_path):
    prefix=tmp_path/'prefix';prefix.mkdir();source=prefix/'native.so';source.write_bytes(b'first')
    audit=tmp_path/'audit.json';audit.write_text(json.dumps(valid_byte_audit(prefix,source)))
    preparation.validate_reuse_audit(prefix,audit)
    before=source.stat();source.write_bytes(b'other')
    import os
    os.utime(source,ns=(before.st_atime_ns,before.st_mtime_ns))
    with pytest.raises(RuntimeError,match='源SHA已变化'):
        preparation.validate_reuse_audit(prefix,audit)


@pytest.mark.parametrize('present',[False,True])
def test_reuse_requires_present_successful_v2_audit(tmp_path,present):
    prefix=tmp_path/'prefix';prefix.mkdir();source=prefix/'file';source.write_bytes(b'bytes');audit=tmp_path/'audit.json'
    if present:
        value=valid_byte_audit(prefix,source);value['passed']=False;audit.write_text(json.dumps(value))
    with pytest.raises(RuntimeError,match='审计'):
        preparation.validate_reuse_audit(prefix,audit)


def test_staging_rejects_ancestor_and_existing_directory_symlinks(tmp_path):
    outside=tmp_path/'outside';outside.mkdir();(tmp_path/'redirect').symlink_to(outside,target_is_directory=True)
    with pytest.raises(RuntimeError,match='符号链接'):
        preparation.reject_staging_symlinks(tmp_path/'redirect'/'new'/'file')
    stage=tmp_path/'stage';stage.mkdir();(stage/'extra').symlink_to(outside,target_is_directory=True)
    with pytest.raises(RuntimeError,match='已有符号链接'):
        preparation.reject_staging_symlinks(stage,recursive=True)
    assert not (outside/'new').exists()


def test_workspace_inside_runtime_prefix_fails_before_any_write(tmp_path,monkeypatch):
    prefix=tmp_path/'prefix';prefix.mkdir();workspace=prefix/'new-staging'
    monkeypatch.setattr(preparation.sys,'argv',['prepare','--workspace',str(workspace),'--runtime-prefix',str(prefix),'--tool-python',preparation.sys.executable])
    with pytest.raises(SystemExit,match='不得位于运行前缀内'):
        preparation.main()
    assert not workspace.exists()
