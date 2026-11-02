"""TensorRT manifest 校验，无加速器或可选包导入。"""
import hashlib
import json
from pathlib import Path


CONTRACTS = {
    "speech_campplus": {
        "input": "features", "profile": [[1, 50, 80], [1, 200, 80], [1, 3000, 80]],
        "outputs": {"embedding": [1, 192]}, "weight": "campplus_cn_en_common.pt",
    },
    "emotion2vec": {
        "input": "waveform", "profile": [[1, 16000], [1, 48000], [1, 160000]],
        "outputs": {"logits": [1, 9], "scores": [1, 7]}, "weight": "model.pt",
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_source(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"manifest 来源路径越界: {relative}")
    return path


def validate_manifest(model_dir, name, precision, *, environment):
    root = Path(model_dir)
    directory = root / "tensorrt" / precision
    engine_path = directory / "engine.plan"
    manifest = json.loads((directory / "manifest.json").read_text())
    contract = CONTRACTS[name]
    expected = {"schema_version": 1, "model_name": name, "precision_policy": precision,
                "profile": contract["profile"], "input_name": contract["input"],
                "outputs": contract["outputs"], "io_dtype": "float32", "tf32": False}
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"{directory}/manifest.json: {field} 不匹配，期望 {value!r}")
    for key in ("tensorrt", "cuda", "compute_capability"):
        if manifest.get(key) != environment[key]:
            raise ValueError(f"TensorRT engine 环境不兼容: {key}: "
                             f"构建={manifest.get(key)!r}, 运行={environment[key]!r}")
    if not manifest.get("built_at") or not manifest.get("export_sources"):
        raise ValueError("manifest 缺少构建时间或导出来源")
    sources = manifest.get("model_sources", [])
    required = {contract["weight"], "config.yaml"}
    if name == "emotion2vec":
        required.add("tokens.txt")
    if not required.issubset({item["path"] for item in sources}):
        raise ValueError("manifest 缺少模型权重、配置或标签来源")
    for item in sources + manifest["export_sources"]:
        path = checked_source(root, item["path"])
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"TensorRT 来源 SHA256 不匹配: {path}")
    if sha256_file(engine_path) != manifest.get("engine_sha256"):
        raise ValueError(f"TensorRT engine SHA256 不匹配: {engine_path}")
    return engine_path, manifest
