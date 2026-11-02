"""无 GPU 的交付检查；只检查实际启用能力的 CUDA TensorRT 资产。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import Settings
from core.model_assets import validate_manifest


def check(config_path):
    cfg = Settings(config_path=str(config_path))
    cfg.validate_model_backends()
    results = []
    for name, enabled, directory in [
        ("speech_campplus", cfg.open_v2_role, cfg.spk_model_dir),
        ("emotion2vec", cfg.open_v2_emotion, cfg.emotion_model_dir),
    ]:
        devices = getattr(cfg, f"model_device_{name}_devices")
        use_trt = getattr(cfg, f"{name}_use_tensorrt")
        precision = getattr(cfg, f"{name}_tensorrt_precision")
        needed = enabled and use_trt and any(d.startswith("cuda:") for d in devices)
        if not needed:
            results.append({"model": name, "requires_engine": False})
            continue
        root = Path(directory)
        meta = json.loads((root / "tensorrt" / precision / "manifest.json").read_text())
        env = {key: meta[key] for key in ("tensorrt", "cuda", "compute_capability")}
        path, _ = validate_manifest(root, name, precision, environment=env)
        results.append({"model": name, "requires_engine": True, "path": str(path),
                        "asset_integrity": "verified", "target_environment": env,
                        "runtime_compatibility": "not checked; requires actual target CUDA device"})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    args = parser.parse_args()
    print(json.dumps(check(args.config), ensure_ascii=False, indent=2))
