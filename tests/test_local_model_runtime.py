"""本地模型运行时约定：进程命名、状态接口的模型显存明细与配置默认值。"""

import os
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from core.config import settings
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

    def test_v118_route_is_removed(self):
        from main import create_app

        paths = TestClient(create_app()).get("/openapi.json").json()["paths"]
        self.assertNotIn("/v1.1.8/seacraft_asr", paths)
        self.assertIn("/v2/seacraft_asr", paths)


class ConfigDefaultsTests(unittest.TestCase):
    def test_v2_switches_default_true(self):
        self.assertFalse(settings.open_v2_role)
        self.assertFalse(settings.open_v2_emotion)

    def test_instance_count_default_is_one(self):
        self.assertEqual(settings.instance_count, 1)

    def test_log_limits_present(self):
        self.assertEqual(settings.log_max_days, 7)
        self.assertEqual(settings.log_max_size_mb, 100)


class ConfigFileStructureTests(unittest.TestCase):
    def _load(self, name):
        import tomli

        with open(os.path.join(REPO_ROOT, name), "rb") as handle:
            return tomli.load(handle)

    def test_host_and_docker_configs_share_structure(self):
        host = self._load("config.toml")
        docker = self._load("config.docker.toml")
        self.assertEqual(set(host.keys()), set(docker.keys()))
        sections = ("log", "model_paths", "speech_rate", "features", "itn", "qwen_asr", "model_device_pool")
        for section in sections:
            self.assertEqual(set(host[section].keys()), set(docker[section].keys()))

    def test_removed_config_keys_are_absent(self):
        for name in ("config.toml", "config.docker.toml"):
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
        pairs = (("config.toml", "./model"), ("config.docker.toml", "/app/model"))
        for name, prefix in pairs:
            paths = self._load(name)["model_paths"]
            expected = {"vad_model_dir", "spk_model_dir", "emotion_model_dir"}
            self.assertEqual(set(paths.keys()), expected)
            for value in paths.values():
                self.assertTrue(value.startswith(prefix))

    def test_itn_remove_interjections_is_explicitly_true(self):
        for name in ("config.toml", "config.docker.toml"):
            self.assertIs(self._load(name)["itn"]["remove_interjections"], True)

    def test_model_device_arrays_pin_gpu0(self):
        for name in ("config.toml", "config.docker.toml"):
            devices = self._load(name)["model_device_pool"]
            self.assertEqual(devices["emotion2vec_devices"], ["cuda:0"])
            self.assertEqual(devices["speech_campplus_devices"], ["cuda:0", "cuda:0", "cuda:0"])

    def test_log_path_and_vad_threshold_are_nested(self):
        for name in ("config.toml", "config.docker.toml"):
            cfg = self._load(name)
            self.assertEqual(cfg["log"]["log_path"], "./asr_service.log")
            self.assertNotIn("log_path", {key for key in cfg if key != "log"})
            self.assertEqual(cfg["qwen_asr"]["vad_speech_noise_thres"], 0.6)


if __name__ == "__main__":
    unittest.main()
