import importlib.util
import io
import os
from pathlib import Path
import sys
import wave

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_audio_manifest", ROOT / "scripts/build_audio_manifest.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def wav_bytes(rate=16000, channels=1, frames=1600):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as audio:
        audio.setnchannels(channels);audio.setsampwidth(2);audio.setframerate(rate)
        audio.writeframes(b"\x00\x00" * channels * frames)
    return stream.getvalue()


def test_recursive_manifest_hashes_duplicates_and_keeps_sources(tmp_path):
    root = tmp_path / "inputs";nested = root / "nested space";nested.mkdir(parents=True)
    data = wav_bytes();(root / "one.wav").write_bytes(data);(nested / "copy.WAV").write_bytes(data)
    manifest = module.build_manifest(root)
    assert manifest["summary"]["sample_count"] == 2
    assert len(manifest["duplicates"]) == 1
    assert len({x["sample_id"] for x in manifest["samples"]}) == 2
    assert all(x["media"]["evidence"] == "wav_header" for x in manifest["samples"])
    assert not manifest["full_decode_checked"] and not manifest["labels_inferred_from_filename"]
    assert (root / "one.wav").read_bytes() == data
    module.write_manifest(manifest, tmp_path / "report" / "manifest.json")
    assert (nested / "copy.WAV").read_bytes() == data


def test_sample_id_stable_after_root_relocation(tmp_path):
    ids = []
    for name in ["tree-a", "tree-b"]:
        root = tmp_path / name;root.mkdir();(root / "same.wav").write_bytes(wav_bytes())
        ids.append(module.build_manifest(root)["samples"][0]["sample_id"])
    assert ids[0] == ids[1]


@pytest.mark.skipif(sys.platform == "darwin" or os.name != "posix", reason="raw GBK filesystem byte path is verified on target Linux")
def test_gbk_path_round_trips_without_renaming(tmp_path):
    root = tmp_path / "inputs";root.mkdir()
    directory = os.fsencode(root) + b"/" + "音频文件".encode("gb18030")
    os.mkdir(directory)
    path = directory + b"/" + "课堂 1.wav".encode("gb18030")
    data = wav_bytes()
    with open(path, "wb") as stream:stream.write(data)
    item = module.build_manifest(root)["samples"][0]
    assert item["display_path"] == "音频文件/课堂 1.wav"
    assert "gb18030" in item["display_encodings"]
    assert bytes.fromhex(item["path_hex"]) == path
    assert item["sample_id"].isascii()
    with open(bytes.fromhex(item["path_hex"]), "rb") as stream:assert stream.read() == data


def test_invalid_header_is_reported_without_claiming_decode(tmp_path):
    root = tmp_path / "inputs";root.mkdir();(root / "bad.wav").write_bytes(b"bad header")
    manifest = module.build_manifest(root)
    assert manifest["summary"]["error_count"] == 1
    assert manifest["samples"][0]["sha256"]
    assert manifest["errors"][0]["stage"] == "media_metadata"
    assert not manifest["full_decode_checked"]


def test_mp3_probe_unavailable_records_reason_and_hash(tmp_path):
    root = tmp_path / "inputs";root.mkdir();(root / "clip.mp3").write_bytes(b"not-a-real-mp3")
    manifest = module.build_manifest(root, ffprobe="")
    assert manifest["summary"]["extension_counts"] == {".mp3": 1}
    assert "ffprobe" in manifest["errors"][0]["message"]
    assert manifest["samples"][0]["sha256"]


def test_output_inside_inputs_is_rejected(tmp_path):
    root = tmp_path / "inputs";root.mkdir();source = root / "clip.wav";source.write_bytes(wav_bytes())
    before = source.read_bytes()
    with pytest.raises(ValueError, match="拒绝覆盖"):
        module.write_manifest(module.build_manifest(root), source)
    assert source.read_bytes() == before


def test_no_symlink_traversal(tmp_path):
    root = tmp_path / "inputs";root.mkdir();outside = tmp_path / "outside.wav";outside.write_bytes(wav_bytes())
    (root / "linked.wav").symlink_to(outside)
    manifest = module.build_manifest(root)
    assert manifest["summary"]["sample_count"] == 0
    assert manifest["summary"]["skipped_count"] == 1
