import json
from pathlib import Path
import tempfile
import unittest

from core.model_assets import CONTRACTS, sha256_file, validate_manifest


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.env = {"tensorrt": "10.7.0.post1", "cuda": "12.4", "compute_capability": [8, 9]}
        self.engine = self.root / "tensorrt/fp16/engine.plan"
        self.engine.parent.mkdir(parents=True)
        self.engine.write_bytes(b"engine fixture")
        self.meta = {**self.env, "schema_version": 1, "model_name": "speech_campplus",
                     "precision_policy": "fp16", "profile": CONTRACTS["speech_campplus"]["profile"],
                     "input_name": "features", "outputs": {"embedding": [1, 192]},
                     "io_dtype": "float32", "tf32": False, "built_at": "2026-11-02T00:00:00Z",
                     "model_sources": [], "export_sources": [], "engine_sha256": sha256_file(self.engine)}
        for filename, key in [("config.yaml", "model_sources"), ("campplus_cn_en_common.pt", "model_sources"),
                              ("export.py", "export_sources")]:
            path = self.root / filename
            path.write_bytes(filename.encode())
            self.meta[key].append({"path": filename, "sha256": sha256_file(path)})

    def validate(self):
        self.engine.with_name("manifest.json").write_text(json.dumps(self.meta))
        return validate_manifest(self.root, "speech_campplus", "fp16", environment=self.env)

    def test_valid_asset(self):
        path, meta = self.validate()
        self.assertEqual(path, self.engine)
        self.assertEqual(meta["model_name"], "speech_campplus")

    def test_compatibility_and_io_rejected_before_runtime(self):
        for key, wrong in [("tensorrt", "10.8.0"), ("cuda", "12.1"),
                           ("compute_capability", [8, 6]), ("io_dtype", "float16"),
                           ("profile", [[1, 1, 80]]), ("precision_policy", "fp32")]:
            with self.subTest(key=key):
                original = self.meta[key]
                self.meta[key] = wrong
                with self.assertRaises(ValueError):
                    self.validate()
                self.meta[key] = original

    def test_modified_weight_and_corrupt_engine_rejected(self):
        for path in [self.root / "campplus_cn_en_common.pt", self.engine]:
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(b"corrupt")
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    self.validate()
                path.write_bytes(original)

    def test_missing_asset_and_missing_provenance_rejected(self):
        self.engine.unlink()
        with self.assertRaises(FileNotFoundError):
            self.validate()
        self.meta["export_sources"] = []
        with self.assertRaisesRegex(ValueError, "来源"):
            self.validate()

    def test_source_path_cannot_escape_model_directory(self):
        self.meta["export_sources"] = [{"path": "../export.py", "sha256": "x"}]
        with self.assertRaisesRegex(ValueError, "越界"):
            self.validate()
