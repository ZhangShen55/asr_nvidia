from pathlib import Path

import pytest

from core.config import Settings
from core.deployment import DeploymentPreflightError, validate_cuda_tensorrt_runtime


ROOT = Path(__file__).resolve().parents[1]


def test_docker_profile_is_strict_fp16_and_uses_image_models(monkeypatch):
    monkeypatch.setenv("ASR_DEPLOYMENT_PROFILE", "cuda-tensorrt-fp16")
    settings = Settings(config_path=str(ROOT / "docker" / "config.toml"))
    settings.validate_model_backends()
    assert settings.is_cuda_tensorrt_profile
    assert settings.deployment_profile == "cuda-tensorrt-fp16"
    assert settings.speech_campplus_use_tensorrt is True
    assert settings.emotion2vec_use_tensorrt is True
    assert settings.speech_campplus_tensorrt_precision == "fp16"
    assert settings.emotion2vec_tensorrt_precision == "fp16"
    assert settings.spk_model_dir == "/app/model/speech_campplus_sv_zh_en_16k-common_advanced"
    assert settings.emotion_model_dir == "/app/model/emotion2vec_plus_large"
    assert settings.qwen_audio_concurrency == 8
    assert settings.qwen_audio_spk_emo_concurrency == 4


def test_cuda_profile_rejects_non_fp16_or_non_cuda_pool(monkeypatch):
    monkeypatch.setenv("ASR_DEPLOYMENT_PROFILE", "cuda-tensorrt-fp16")
    settings = Settings(config_path=str(ROOT / "docker" / "config.toml"))
    settings._cfg["model_device_pool"]["emotion2vec_tensorrt_precision"] = "fp32"
    with pytest.raises(ValueError, match="emotion2vec.*fp16"):
        settings.validate_model_backends()

    settings._cfg["model_device_pool"]["emotion2vec_tensorrt_precision"] = "fp16"
    settings._cfg["model_device_pool"]["emotion2vec_devices"] = ["cpu"]
    with pytest.raises(ValueError, match="emotion2vec_devices"):
        settings.validate_model_backends()


def test_cuda_profile_preflight_fails_without_visible_gpu(monkeypatch):
    monkeypatch.setenv("ASR_DEPLOYMENT_PROFILE", "cuda-tensorrt-fp16")
    settings = Settings(config_path=str(ROOT / "docker" / "config.toml"))
    try:
        validate_cuda_tensorrt_runtime(settings)
    except DeploymentPreflightError as exc:
        assert "GPU" in str(exc) or "TensorRT" in str(exc)
    else:
        # This assertion is only reached on a real CUDA host; the preflight
        # still proves that it completed without silently selecting PyTorch.
        assert settings.is_cuda_tensorrt_profile


def test_cuda_profile_preflight_validates_existing_fp16_assets(monkeypatch):
    monkeypatch.setenv("ASR_DEPLOYMENT_PROFILE", "cuda-tensorrt-fp16")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA host")
    source = (ROOT / "docker" / "config.toml").read_text()
    config_path = ROOT / ".pytest_cuda_profile.toml"
    config_path.write_text(source.replace("/app/model", str(ROOT / "model")))
    try:
        result = validate_cuda_tensorrt_runtime(Settings(config_path=str(config_path)))
    finally:
        config_path.unlink(missing_ok=True)
    assert result["checked"] is True
    assert {item["model"] for item in result["assets"]} == {"speech_campplus", "emotion2vec"}
    assert all(item["compatible"] for item in result["assets"])
