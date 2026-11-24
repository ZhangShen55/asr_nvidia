"""源码容器入口门禁：用元数据及设备替身，不加载权重或初始化NPU。"""
from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from core import deployment


@pytest.fixture(autouse=True)
def isolated_runtime_marker(monkeypatch):
    monkeypatch.delenv("ASR_CONTAINER_RUNTIME", raising=False)


def _enable(tmp_path, monkeypatch, text="instance_count = 1\n"):
    config = tmp_path / "mounted.toml"
    config.write_text(text)
    monkeypatch.setenv("ASR_CONTAINER_RUNTIME", "cann82-source")
    monkeypatch.setenv("CONFIG_PATH", str(config))
    return config


def _versions(monkeypatch, npu):
    packages = {"torch": "2.5.1", "torch-npu": npu, "torchaudio": "2.5.1",
                "transformers": "5.13.0", "numpy": "1.26.4", "funasr": "1.2.6"}
    monkeypatch.setattr(deployment.metadata, "version", lambda name: packages[name])
    monkeypatch.setattr(deployment.importlib.util, "find_spec", lambda name: object())


@pytest.mark.parametrize("npu", ["2.5.1.post1", "2.5.1.post1+vendor"])
def test_container_exact_post1_and_full_version_evidence(tmp_path, monkeypatch, npu):
    _enable(tmp_path, monkeypatch)
    _versions(monkeypatch, npu)
    assert deployment._runtime_versions()["torch-npu"] == npu


@pytest.mark.parametrize("npu", ["2.5.1", "2.5.1.post2", "2.6.0", "2.5.1.post10"])
def test_container_rejects_nonmatching_distribution_before_device_import(tmp_path, monkeypatch, npu):
    _enable(tmp_path, monkeypatch)
    _versions(monkeypatch, npu)
    monkeypatch.setattr(deployment.importlib, "import_module", lambda name: pytest.fail(name))
    cfg = SimpleNamespace(validate_model_backends=lambda: None, vad_device=[],
                          model_device_speech_campplus_devices=[],
                          model_device_emotion2vec_devices=[], qwen_asr_enabled=False)
    # 依赖门禁应早于模型目录检查，防止错误版本伪装成资产错误。
    cfg.vad_model_dir = cfg.spk_model_dir = cfg.emotion_model_dir = ""
    with pytest.raises(deployment.DeploymentPreflightError, match="torch-npu.*2.5.1.post1"):
        deployment.validate_npu_runtime(cfg)


def test_host_original_distribution_stays_valid_without_mounted_config(monkeypatch):
    monkeypatch.delenv("CONFIG_PATH", raising=False)
    _versions(monkeypatch, "2.5.1")
    assert deployment.validate_source_container_config() is None
    assert deployment._runtime_versions()["torch-npu"] == "2.5.1"


@pytest.mark.parametrize("value", ["2", "0", "true", '"1"', "1.0", "-1"])
def test_container_rejects_coercible_or_multiple_http_values(tmp_path, monkeypatch, value):
    _enable(tmp_path, monkeypatch, f"instance_count={value}\n")
    with pytest.raises(deployment.DeploymentPreflightError, match="instance_count=1.*真实整数"):
        deployment.validate_source_container_config()


def test_container_default_http_one_and_actual_config_source(tmp_path, monkeypatch):
    path = _enable(tmp_path, monkeypatch, "[qwen_asr]\nlocal_instance_count=4\n")
    result = deployment.validate_source_container_config()
    assert result["instance_count"] == 1
    assert result["config_path"] == str(path.resolve())


@pytest.mark.parametrize("kind", ["unset", "empty", "missing", "directory", "unreadable"])
def test_container_config_requires_readable_regular_file(tmp_path, monkeypatch, kind):
    path = _enable(tmp_path, monkeypatch)
    if kind == "unset":
        monkeypatch.delenv("CONFIG_PATH")
    elif kind == "empty":
        monkeypatch.setenv("CONFIG_PATH", "")
    elif kind == "missing":
        path.unlink()
    elif kind == "directory":
        path.unlink(); path.mkdir()
    else:
        monkeypatch.setattr(deployment.os, "access", lambda *args: False)
    with pytest.raises(deployment.DeploymentPreflightError, match="CONFIG_PATH.*普通"):
        deployment.validate_source_container_config()


def test_container_invalid_toml_reports_configuration(tmp_path, monkeypatch):
    _enable(tmp_path, monkeypatch, "instance_count = [broken")
    with pytest.raises(deployment.DeploymentPreflightError, match="无法读取容器配置"):
        deployment.validate_source_container_config()


def test_unknown_container_marker_does_not_disable_gate(monkeypatch):
    monkeypatch.setenv("ASR_CONTAINER_RUNTIME", "cann82-souce")
    with pytest.raises(deployment.DeploymentPreflightError, match="ASR_CONTAINER_RUNTIME"):
        deployment.validate_source_container_config()


def test_main_rejects_missing_mount_before_settings_or_model_import(tmp_path):
    root = Path(__file__).resolve().parents[1]
    program = """import builtins, runpy, sys
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'core.config' or name.startswith(('core.models', 'core.vad_pool', 'core.qwen_asr_client')):
        raise AssertionError('configuration or models imported before mount gate')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
runpy.run_path(sys.argv[1], run_name='container_import_probe')
"""
    env = dict(os.environ, ASR_CONTAINER_RUNTIME="cann82-source", CONFIG_PATH=str(tmp_path / "absent.toml"),
               PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(root), TMPDIR=str(tmp_path))
    result = subprocess.run([sys.executable, "-c", program, str(root / "main.py")],
                            env=env, cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert "CONFIG_PATH" in result.stderr and "普通挂载配置文件" in result.stderr
    assert "before mount gate" not in result.stderr


def _fake_npu_settings(tmp_path, device):
    directories = {}
    for name, files in [('vad', ['model.pt', 'config.yaml', 'am.mvn']),
                        ('spk', ['campplus_cn_en_common.pt', 'config.yaml']),
                        ('emo', ['model.pt', 'config.yaml', 'tokens.txt'])]:
        directory = tmp_path / name
        directory.mkdir()
        for filename in files:
            (directory / filename).write_bytes(b'asset')
        directories[name] = str(directory)
    return SimpleNamespace(validate_model_backends=lambda: None, vad_device=['cpu'],
                           model_device_speech_campplus_devices=[device],
                           model_device_emotion2vec_devices=[device], qwen_asr_enabled=True,
                           qwen_local_device=device, vad_model_dir=directories['vad'],
                           spk_model_dir=directories['spk'], emotion_model_dir=directories['emo'],
                           deployment_profile='npu-910b')


def _fake_device_count(monkeypatch, count):
    observations = []
    def available():
        observations.append('is_available')
        return True
    def device_count():
        observations.append('device_count')
        return count
    def forbidden(*_a, **_k):
        pytest.fail('preflight must not set_device or allocate tensor')
    api = SimpleNamespace(is_available=available, device_count=device_count, set_device=forbidden)
    torch = SimpleNamespace(npu=api, tensor=forbidden)
    extension = SimpleNamespace(npu=api)
    def module(name):
        observations.append(name)
        return {'torch': torch, 'torch_npu': extension}[name]
    monkeypatch.setattr(deployment.importlib, 'import_module', module)
    return observations


def test_container_rejects_all_visible_cards_before_device_selection(tmp_path, monkeypatch):
    _enable(tmp_path, monkeypatch)
    _versions(monkeypatch, '2.5.1.post1')
    observations = _fake_device_count(monkeypatch, 8)
    cfg = _fake_npu_settings(tmp_path, 'npu:0')
    with pytest.raises(deployment.DeploymentPreflightError, match='只能有1.*实际 8.*ASCEND_RT_VISIBLE_DEVICES'):
        deployment.validate_npu_runtime(cfg)
    assert observations == ['torch', 'torch_npu', 'is_available', 'device_count']


def test_container_accepts_single_visible_card_and_logical_zero(tmp_path, monkeypatch):
    _enable(tmp_path, monkeypatch)
    _versions(monkeypatch, '2.5.1.post1')
    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES', '7')
    observations = _fake_device_count(monkeypatch, 1)
    result = deployment.validate_npu_runtime(_fake_npu_settings(tmp_path, 'npu:0'))
    assert result['devices'] == ['npu:0']
    assert result['container_runtime']['visible_npu_count'] == 1
    assert result['container_runtime']['ascend_rt_visible_devices'] == '7'
    assert observations == ['torch', 'torch_npu', 'is_available', 'device_count']


def test_single_card_container_rejects_host_physical_index(tmp_path, monkeypatch):
    _enable(tmp_path, monkeypatch)
    _versions(monkeypatch, '2.5.1.post1')
    _fake_device_count(monkeypatch, 1)
    with pytest.raises(deployment.DeploymentPreflightError, match='仅允许配置逻辑 npu:0'):
        deployment.validate_npu_runtime(_fake_npu_settings(tmp_path, 'npu:7'))


def test_host_retains_eight_visible_cards_and_physical_seven(tmp_path, monkeypatch):
    _versions(monkeypatch, '2.5.1')
    observations = _fake_device_count(monkeypatch, 8)
    result = deployment.validate_npu_runtime(_fake_npu_settings(tmp_path, 'npu:7'))
    assert result['devices'] == ['npu:7'] and result['container_runtime'] is None
    assert observations == ['torch', 'torch_npu', 'is_available', 'device_count']
