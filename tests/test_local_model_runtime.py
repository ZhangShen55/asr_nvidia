"""本地模型运行时约定：进程命名、状态接口的模型显存明细与配置默认值。"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from core.config import Settings, settings
from core.status import APP_VERSION, ID_ENGINE_PLACEHOLDER

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class StatusContractTests(unittest.TestCase):
    def _status(self):
        from main import create_app

        # 用上下文管理器进入，确保 lifespan（记录启动时刻）被执行；
        # 单测不真的拉起 VAD worker 进程（真实模型加载要数秒）
        with patch("main.start_vad_pool", new=AsyncMock()), \
                patch("main.stop_vad_pool", new=AsyncMock()):
            with TestClient(create_app()) as client:
                return client.get("/get_status").json()

    def test_status_contains_fixed_engine_id_and_version(self):
        payload = self._status()
        self.assertEqual(payload["id_engine"], ID_ENGINE_PLACEHOLDER)
        self.assertEqual(payload["appVersion"], APP_VERSION)
        self.assertEqual(payload["appVersion"], "latest")
        for key in ("onlineDone", "queuedTaskList", "processingTaskList"):
            self.assertNotIn(key, payload)

    def test_status_contains_local_model_details(self):
        payload = self._status()
        models = payload["localModels"]
        self.assertEqual(set(models.keys()), {"speech_campplus", "emotion2vec"})
        for entry in models.values():
            self.assertIn("loaded", entry)
            self.assertIn("allocated_mb", entry)
            self.assertIn("instances", entry)
            self.assertIn("configured_instances", entry)
            self.assertIn("available_instances", entry)
            self.assertIn("failed_instances", entry)
            self.assertIn("busy", entry)
            self.assertIn("waiting", entry)
        self.assertEqual(models["speech_campplus"]["configured_instances"], 3)
        self.assertEqual(models["emotion2vec"]["configured_instances"], 1)
        self.assertEqual(models["speech_campplus"]["instances"], 0)
        self.assertEqual(models["emotion2vec"]["instances"], 0)

    def test_status_exposes_vad_pool_runtime(self):
        payload = self._status()
        pool = payload["vadPool"]
        self.assertEqual(pool["devices"], settings.vad_device)
        self.assertEqual(pool["configured_instances"], len(settings.vad_device))
        self.assertIn("loaded", pool)
        self.assertIn("loaded_instances", pool)
        self.assertIn("allocated_mb", pool)
        self.assertIn("busy", pool)
        self.assertIn("waiting", pool)
        self.assertNotIn("pool_max", pool)

    def test_status_exposes_resource_diagnostics_without_removing_legacy_fields(self):
        payload = self._status()
        diagnostics = payload["resourceDiagnostics"]
        self.assertIn("host", diagnostics)
        self.assertIn("vmstat", diagnostics)
        self.assertIn("gpu", diagnostics)
        self.assertIn("cache", diagnostics)
        self.assertIn("requestTelemetry", diagnostics)
        self.assertIn("processes", diagnostics)
        # 采集失败必须有明确标志，不能用 0 伪造成功采样。
        self.assertIn("available", diagnostics["host"])
        self.assertIn("available", diagnostics["gpu"])

    def test_v118_route_is_removed(self):
        from main import create_app

        paths = TestClient(create_app()).get("/openapi.json").json()["paths"]
        self.assertNotIn("/v1.1.8/seacraft_asr", paths)
        self.assertIn("/v3/seacraft_asr", paths)


class ConfigDefaultsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        config = Path(directory.name) / "config.toml"
        config.write_text("", encoding="utf-8")
        self.defaults = Settings(config_path=str(config))

    def test_v2_switches_default_true(self):
        self.assertTrue(self.defaults.open_v2_role)
        self.assertTrue(self.defaults.open_v2_emotion)

    def test_instance_count_default_is_one(self):
        self.assertEqual(self.defaults.instance_count, 1)

    def test_log_limits_present(self):
        self.assertEqual(self.defaults.log_max_days, 7)
        self.assertEqual(self.defaults.log_max_size_mb, 100)


class ConfigFileStructureTests(unittest.TestCase):
    def _load(self, name):
        import tomli

        with open(os.path.join(REPO_ROOT, name), "rb") as handle:
            return tomli.load(handle)

    def test_host_and_docker_configs_share_structure(self):
        host = self._load("config.toml")
        docker = self._load("docker/config.toml")
        self.assertEqual(set(host.keys()), set(docker.keys()))
        sections = ("log", "model_paths", "speech_rate", "features", "itn", "qwen_asr", "model_device_pool")
        for section in sections:
            self.assertEqual(set(host[section].keys()), set(docker[section].keys()))

    def test_removed_config_keys_are_absent(self):
        for name in ("config.toml", "docker/config.toml"):
            cfg = self._load(name)
            removed = ("id_engine", "version", "device", "ngpu", "ncpu",
                       "concurrency", "hotword_path", "audio_chunk")
            for key in removed:
                self.assertNotIn(key, cfg, f"{name} 不应再包含 {key}")
            features = cfg["features"]
            for key in ("open_spk", "open_emotion", "ban_hotword", "open_mul_lang"):
                self.assertNotIn(key, features, f"{name} 不应再包含 features.{key}")
            paths = cfg["model_paths"]
            for key in ("asr_model_dir", "punc_model_dir", "whisper_model_dir"):
                self.assertNotIn(key, paths, f"{name} 不应再包含 model_paths.{key}")

    def test_model_paths_point_to_three_models(self):
        pairs = (("config.toml", "./model"), ("docker/config.toml", "/app/model"))
        for name, prefix in pairs:
            paths = self._load(name)["model_paths"]
            expected = {"vad_model_dir", "spk_model_dir", "emotion_model_dir"}
            self.assertEqual(set(paths.keys()), expected)
            for value in paths.values():
                self.assertTrue(value.startswith(prefix))

    def test_itn_remove_interjections_is_explicitly_true(self):
        for name in ("config.toml", "docker/config.toml"):
            self.assertIs(self._load(name)["itn"]["remove_interjections"], True)

    def test_model_device_arrays_pin_gpu0(self):
        for name in ("config.toml", "docker/config.toml"):
            devices = self._load(name)["model_device_pool"]
            self.assertEqual(devices["emotion2vec_devices"], ["cuda:0"])
            self.assertEqual(devices["speech_campplus_devices"], ["cuda:0", "cuda:0", "cuda:0"])

    def test_log_path_and_vad_threshold_are_nested(self):
        expected_log_paths = {
            "config.toml": "./asr_service.log",
            "docker/config.toml": "/tmp/asr-offline/asr_service.log",
        }
        for name, expected_log_path in expected_log_paths.items():
            cfg = self._load(name)
            self.assertEqual(cfg["log"]["log_path"], expected_log_path)
            self.assertNotIn("log_path", {key for key in cfg if key != "log"})
            self.assertEqual(cfg["qwen_asr"]["vad_speech_noise_thres"], 0.6)


if __name__ == "__main__":
    unittest.main()
