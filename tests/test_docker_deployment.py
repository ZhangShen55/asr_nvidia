from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DOCKER = ROOT / "docker"


def test_docker_delivery_files_and_base_image_are_present():
    assert (DOCKER / "Dockerfile").is_file()
    assert (DOCKER / "docker-compose.yaml").is_file()
    assert (DOCKER / "README.md").is_file()
    dockerfile = (DOCKER / "Dockerfile").read_text()
    assert "docker.1ms.run/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04" in dockerfile
    assert "COPY --from=builder /src/model /app/model" in dockerfile
    assert "COPY --from=builder /src/cython-build.json" in dockerfile


def test_compose_declares_gpu_and_read_only_config_mount():
    compose = (DOCKER / "docker-compose.yaml").read_text()
    assert "driver: nvidia" in compose
    assert "capabilities: [gpu]" in compose
    assert "${ASR_CONFIG:-./config.toml}:/config.toml:ro" in compose
    assert "ASR_IMAGE" in compose
    assert "get_status" in compose


def test_build_script_and_docs_use_v3_time_tag_and_v3_route():
    script = (DOCKER / "build.sh").read_text()
    readme = (DOCKER / "README.md").read_text()
    assert re.search(r"asr_offline:v3_", script)
    assert "asr_offline:v3_<YYYYMMDDHHMMSS>" in readme
    assert "/v3/seacraft_asr" in readme
    assert "/v2/seacraft_asr" in readme
    assert "CONFIG_PATH" in readme
