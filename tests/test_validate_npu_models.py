"""Numerical/report contracts only; no model loading and no NPU initialization."""
import argparse
import hashlib
import json
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.validate_npu_models import (
    DEVICE, TOLERANCES, compare_output, compare_vad, create_torchair_backend,
    manifest_samples, read_clip, run, vector_metrics, write_record,
)


class NpuModelValidationTests(unittest.TestCase):
    def test_read_only_gbk_path_hex_and_independent_clip_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.fsencode(directory) + b"/" + "样本.wav".encode("gb18030")
            pcm = np.arange(32000, dtype="<i2").tobytes()
            with wave.open(os.fsdecode(path), "wb") as source:
                source.setnchannels(1)
                source.setsampwidth(2)
                source.setframerate(16000)
                source.writeframes(pcm)
            before = Path(os.fsdecode(path)).read_bytes()
            sample = {"sample_id": "synthetic", "path_hex": path.hex(), "sha256": hashlib.sha256(before).hexdigest()}
            audio, metadata = read_clip(sample, 0.5, 0.25, verify_source_hash=True)
            self.assertEqual(audio.dtype, np.float32)
            self.assertEqual(len(audio), 4000)
            self.assertEqual(audio[0], 8000 / 32768)
            self.assertEqual(metadata["pcm_sha256"], hashlib.sha256(pcm[16000:24000]).hexdigest())
            self.assertEqual(metadata["source_sha256_verified"], sample["sha256"])
            self.assertEqual(Path(os.fsdecode(path)).read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "超出"):
                read_clip(sample, 1.9, 0.2)
            sample["sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "SHA256"):
                read_clip(sample, 0.0, 0.2, verify_source_hash=True)

    def test_nonfinite_candidate_evidence_is_explicit_and_json_replayable(self):
        import io
        handle = io.StringIO()
        write_record(handle, {"output": np.asarray([np.nan, np.inf, 0.1], dtype=np.float32), "passed": False})
        record = json.loads(handle.getvalue())
        self.assertEqual(record["output"][:2], ["nan", "inf"])
        self.assertFalse(record["passed"])

    def test_manifest_selection_preserves_explicit_order_and_missing_id_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"samples": [{"sample_id": "a", "path_hex": "2f61"}, {"sample_id": "b", "path_hex": "2f62"}]}))
            self.assertEqual([row["sample_id"] for row in manifest_samples(path, ["b", "a"])], ["b", "a"])
            with self.assertRaisesRegex(ValueError, "不存在"):
                manifest_samples(path, ["missing"])

    def test_fixed_cam_gate_rejects_scale_error_despite_identical_cosine(self):
        self.assertTrue(compare_output("speech_campplus", {"values": [1, 2]}, {"values": [1, 2]}, "fp32")["passed"])
        result = compare_output("speech_campplus", {"values": [1, 2]}, {"values": [2, 4]}, "fp32")
        self.assertAlmostEqual(result["cosine"], 1)
        self.assertFalse(result["passed"])
        self.assertFalse(vector_metrics([1], [float("nan")])["valid"])
        self.assertFalse(vector_metrics([1], [1, 2])["valid"])

    def test_emotion_gate_requires_seven_labels_and_top1(self):
        reference = {"labels": list("abcdefg"), "values": [0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05]}
        self.assertTrue(compare_output("emotion2vec", reference, reference, "fp32")["passed"])
        wrong_labels = dict(reference, labels=list("bacdefg"))
        self.assertFalse(compare_output("emotion2vec", reference, wrong_labels, "fp32")["passed"])
        wrong_top = dict(reference, values=[0.2, 0.4, 0.1, 0.1, 0.1, 0.05, 0.05])
        self.assertFalse(compare_output("emotion2vec", reference, wrong_top, "bf16")["passed"])

    def test_vad_count_boundary_and_silence_evidence_are_distinct(self):
        self.assertTrue(compare_vad([[10, 100]], [[20, 110]])["passed"])
        self.assertFalse(compare_vad([[10, 100]], [[31, 100]])["passed"])
        self.assertFalse(compare_vad([[10, 100]], [])["passed"])
        self.assertFalse(compare_vad([], [])["speech_evidence"])
        self.assertFalse(compare_vad([[10, 100]], [[11, 100]], 0)["passed"])

    def test_torchair_unavailable_reports_exact_reason_without_installing(self):
        with patch("scripts.validate_npu_models.importlib.import_module", side_effect=ModuleNotFoundError("No module named torchair")):
            with self.assertRaisesRegex(RuntimeError, "TorchAir unavailable: ModuleNotFoundError: No module named torchair"):
                create_torchair_backend()

    def test_plan_writes_tolerances_before_any_manifest_or_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.jsonl"
            args = argparse.Namespace(device=DEVICE, output=path, precisions=["fp32", "bf16"], compilers=["eager", "torchair"], plan_only=True)
            with patch("scripts.validate_npu_models.manifest_samples", side_effect=AssertionError("plan must not read audio")):
                self.assertEqual(run(args), 0)
            record = json.loads(path.read_text())
            self.assertEqual(record["tolerances"], TOLERANCES)
            self.assertIn("CUDA golden not provided", record["reference"])
            self.assertEqual(record["device"], "npu:7")
            args.device = "npu:0"
            with self.assertRaises(SystemExit):
                run(args)


if __name__ == "__main__":
    unittest.main()
