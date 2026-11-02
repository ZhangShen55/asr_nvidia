"""导出/构建或补齐已有候选的 manifest；导出依赖不进入服务运行依赖。"""
import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.model_assets import CONTRACTS, sha256_file


def source_record(root, path):
    return {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}


def write_manifest(root, name, precision, metadata, source_files, built_at, time_source):
    output = root / "tensorrt" / precision
    contract = CONTRACTS[name]
    archive = root / "tensorrt" / "sources"
    archive.mkdir(parents=True, exist_ok=True)
    exports = []
    for path in source_files:
        archived = archive / (sha256_file(path)[:12] + "-" + path.name)
        if not archived.exists():
            shutil.copy2(path, archived)
        exports.append(source_record(root, archived))
    files = [contract["weight"], "config.yaml"]
    if name == "emotion2vec":
        files.append("tokens.txt")
    metadata.update({
        "schema_version": 1, "model_name": name, "precision_policy": precision,
        "input_name": contract["input"], "profile": contract["profile"],
        "outputs": contract["outputs"], "io_dtype": "float32", "tf32": False,
        "built_at": built_at, "build_timestamp_source": time_source,
        "manifest_created_at": datetime.now(timezone.utc).isoformat(),
        "model_sources": [source_record(root, root / p) for p in files],
        "export_sources": exports,
        "engine_sha256": sha256_file(output / "engine.plan"),
    })
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output / "manifest.json")


def adopt(args):
    """已验证 engine 保持原字节；旧来源与哈希必须与独立实验证据一致。"""
    root = args.model_dir.resolve()
    candidate = args.adopt_from.resolve()
    kind = "cam" if args.model == "speech_campplus" else "emo"
    meta = json.loads((candidate / f"{kind}_{args.precision}_manifest.json").read_text())
    original = candidate / f"{kind}_{args.precision}.plan"
    target = root / "tensorrt" / args.precision / "engine.plan"
    if sha256_file(original) != meta["engine_sha256"] or sha256_file(target) != meta["engine_sha256"]:
        raise ValueError("候选 engine 与原始独立实验证据哈希不匹配")
    environment = json.loads((candidate / "environment.json").read_text())
    weight = root / CONTRACTS[args.model]["weight"]
    if sha256_file(weight) != environment["hashes"][str(weight)]:
        raise ValueError("当前权重与原始独立实验权重不匹配")
    onnx = candidate / ("campplus_pool_fixed.onnx" if kind == "cam" else "emotion_dynamic_logits.onnx")
    if sha256_file(onnx) != meta["onnx_sha256"]:
        raise ValueError("候选 ONNX 与原始独立实验证据不匹配")
    legacy = target.with_name("legacy-manifest.json")
    if not legacy.exists():
        shutil.copy2(target.with_name("manifest.json"), legacy)
    sources = [candidate / "asr_reference.py", candidate / "build_engine.py"]
    if kind == "cam":
        sources.append(candidate / "export_cam.py")
    package = "/campplus/" if kind == "cam" else "/emotion2vec/"
    for filename, digest in environment["hashes"].items():
        if filename.endswith(".py") and package in filename:
            path = Path(filename)
            if sha256_file(path) != digest:
                raise ValueError(f"原始模型实现版本不匹配: {path}")
            sources.append(path)
    reference = json.loads((candidate / f"{kind}_reference.json").read_text())
    meta["funasr"] = reference["funasr"]
    # 旧实验没有记录 wall-clock 构建起点，保留该限制，使用原始文件 mtime 标注。
    built_at = datetime.fromtimestamp(original.stat().st_mtime, timezone.utc).isoformat()
    write_manifest(root, args.model, args.precision, meta, sources, built_at,
                   "legacy candidate engine mtime; exact original build start not recorded")
    print("Adopted unchanged engine:", target, flush=True)


def build(args):
    import torch
    import onnx
    import tensorrt as trt
    from core.local_inference import load_native_network

    root = args.model_dir.resolve()
    output = root / "tensorrt" / args.precision
    output.mkdir(parents=True, exist_ok=True)
    target = output / "engine.plan"
    if target.exists() and not args.rebuild:
        raise FileExistsError(f"engine 已存在；复用候选或显式 --rebuild: {target}")
    contract = CONTRACTS[args.model]
    torch.cuda.set_device(args.device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    net = load_native_network(args.model, root).to(args.device)

    class EmotionExport(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, source):
            if self.inner.cfg.normalize:
                mean = source.mean(dim=1, keepdim=True)
                variance = source.var(dim=1, keepdim=True, unbiased=False)
                source = (source - mean) / torch.sqrt(variance + 1e-5)
            features = self.inner.extract_features(source, padding_mask=None)["x"]
            logits = self.inner.proj(features.mean(dim=1))
            return logits, torch.softmax(logits[:, [0, 2, 3, 4, 5, 7, 8]], dim=-1)

    wrapped = net if args.model == "speech_campplus" else EmotionExport(net).eval()
    if args.model == "emotion2vec":
        with torch.no_grad():
            wrapped(torch.zeros(1, 160000, device=args.device))
    onnx_path = output / "model.onnx"
    dummy = torch.zeros(contract["profile"][1], device=args.device)
    with torch.no_grad():
        torch.onnx.export(wrapped, dummy, str(onnx_path), opset_version=17,
                          input_names=[contract["input"]], output_names=list(contract["outputs"]),
                          dynamic_axes={contract["input"]: {1: "frames_or_samples"}},
                          do_constant_folding=True)
    graph = onnx.load(str(onnx_path))
    if args.model == "speech_campplus":
        pools = [n for n in graph.graph.node if n.op_type == "AveragePool"]
        if len(pools) != 52:
            raise ValueError(f"CAM++ AveragePool 数量变化: {len(pools)}")
        for node in pools:
            attrs = {attr.name: attr for attr in node.attribute}
            if attrs["ceil_mode"].i != 1:
                raise ValueError("CAM++ ceil_mode 必须为 1")
            attrs["count_include_pad"].i = 0
    onnx.checker.check_model(graph)
    onnx.save(graph, str(onnx_path))
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.clear_flag(trt.BuilderFlag.TF32)
    if args.precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    profile.set_shape(contract["input"], *[tuple(x) for x in contract["profile"]])
    config.add_optimization_profile(profile)
    built_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT 构建失败")
    temporary = target.with_suffix(".plan.tmp")
    temporary.write_bytes(bytes(serialized))
    temporary.replace(target)
    meta = {"tensorrt": trt.__version__, "cuda": torch.version.cuda,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(args.device),
            "compute_capability": list(torch.cuda.get_device_capability(args.device)),
            "onnx_sha256": sha256_file(onnx_path), "build_s": time.perf_counter() - started}
    import core.local_inference as local

    sources = [Path(__file__).resolve(), Path(local.__file__), Path(inspect.getfile(type(net)))]
    write_manifest(root, args.model, args.precision, meta, sources, built_at, "UTC build start")
    print("Built:", target, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(CONTRACTS))
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--adopt-from", type=Path)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    adopt(args) if args.adopt_from else build(args)
