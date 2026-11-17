"""NPU 预检的文件/配置/设备边界，用替身避免初始化真实设备。"""
from pathlib import Path
from types import SimpleNamespace
import pytest
from core.config import Settings
from core import deployment


def config(tmp_path, device='cpu', qwen=False):
    dirs = {}
    for name, files in [('vad',['model.pt','config.yaml','am.mvn']),
                        ('spk',['campplus_cn_en_common.pt','config.yaml']),
                        ('emo',['model.pt','config.yaml','tokens.txt'])]:
        d=tmp_path/name;d.mkdir()
        for file in files:(d/file).write_bytes(b'asset')
        dirs[name]=d
    p=tmp_path/'config.toml'
    p.write_text(f'''[model_paths]
vad_model_dir="{dirs['vad']}"
spk_model_dir="{dirs['spk']}"
emotion_model_dir="{dirs['emo']}"
[vad]
vad_device=["{device}"]
[qwen_asr]
enabled={'true' if qwen else 'false'}
local_device="npu:7"
local_model_dir="{tmp_path}/qwen-missing"
[model_device_pool]
speech_campplus_devices=["{device}"]
emotion2vec_devices=[]
''')
    return Settings(config_path=str(p))


def test_cpu_preflight_checks_files_without_importing_npu(tmp_path, monkeypatch):
    cfg=config(tmp_path)
    monkeypatch.setattr(deployment.importlib, 'import_module', lambda name: pytest.fail(name))
    result=deployment.validate_npu_runtime(cfg)
    assert result['checked'] and result['devices']==[]
    assert result['models_loaded_by_preflight'] is False


def test_asset_failure_precedes_readiness(tmp_path):
    cfg=config(tmp_path);(Path(cfg.vad_model_dir)/'am.mvn').unlink()
    with pytest.raises(deployment.DeploymentPreflightError,match='am.mvn'):
        deployment.validate_npu_runtime(cfg)


def test_npu_preflight_checks_visible_index_without_loading_qwen(tmp_path, monkeypatch):
    cfg=config(tmp_path,'npu:7',True)
    monkeypatch.setattr(deployment,'_runtime_versions',lambda:{'torch':'2.5.1'})
    api=SimpleNamespace(is_available=lambda:True, device_count=lambda:8)
    monkeypatch.setattr(deployment.importlib,'import_module',lambda name:SimpleNamespace(npu=api))
    assert deployment.validate_npu_runtime(cfg)['devices']==['npu:7']
    api.device_count=lambda:1
    with pytest.raises(deployment.DeploymentPreflightError,match='npu:7.*不可见'):
        deployment.validate_npu_runtime(cfg)


def test_old_cuda_profile_fails_before_hardware(tmp_path,monkeypatch):
    cfg=config(tmp_path);monkeypatch.setenv('ASR_DEPLOYMENT_PROFILE','cuda-tensorrt-fp16')
    with pytest.raises(ValueError,match='迁移'):
        cfg.validate_model_backends()
