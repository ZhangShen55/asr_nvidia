"""剖析 v2 链路本地 role / emotion 阶段的耗时构成与并发扩展性。

用法示例：

    /root/anaconda3/envs/asr/bin/python scripts/profile_role_emotion.py \
        --segments-dir /tmp/asr_run/segs --limit 40 --device cuda:2

输出内容：

1. 各变体（说话人 / 情绪 × 文件路径 / numpy 输入）的单次调用耗时；
2. 情绪阶段内部「音频加载」与「模型前向」的拆分；
3. 线程并发下的墙钟时间与加速比，用于判断是否被 GIL 串行化。
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from funasr import AutoModel


def load_segments(segments_dir: str, limit: int) -> list:
    paths = sorted(Path(segments_dir).glob("*.wav"))
    if limit:
        paths = paths[:limit]
    return paths


def read_wav(path: Path) -> tuple:
    import torchaudio

    wav, sr = torchaudio.load(str(path))
    return wav.mean(0).numpy(), sr


def time_calls(fn, items) -> list:
    latencies = []
    for item in items:
        start = time.perf_counter()
        fn(item)
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def summarize(name: str, latencies: list) -> dict:
    return {
        "name": name,
        "calls": len(latencies),
        "mean_ms": round(statistics.mean(latencies), 1),
        "p50_ms": round(statistics.median(latencies), 1),
        "p90_ms": round(sorted(latencies)[int(len(latencies) * 0.9)], 1),
        "total_s": round(sum(latencies) / 1000, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments-dir", default="/tmp/asr_run/segs")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--spk-model", default="model/speech_campplus_sv_zh_en_16k-common_advanced")
    parser.add_argument("--emotion-model", default="model/emotion2vec_plus_large")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    paths = load_segments(args.segments_dir, args.limit)
    if not paths:
        raise SystemExit(f"{args.segments_dir} 下没有 wav 片段")
    print(f"[片段] {len(paths)} 个，示例：{paths[0].name}")

    report = {"segments": len(paths), "device": args.device, "variants": {}, "scaling": {}}

    # ---------------- CAM++（说话人嵌入）----------------
    print("[加载] CAM++ ...")
    spk_model = AutoModel(
        model=args.spk_model,
        device=args.device,
        disable_update=True,
        disable_pbar=True,
    )

    arrays = [read_wav(p) for p in paths]

    def spk_numpy(item):
        audio, sr = item
        return spk_model.generate(input=audio, fs=sr)

    lat = time_calls(spk_numpy, arrays)
    report["variants"]["spk_numpy"] = summarize("CAM++ / numpy 输入", lat)
    print(f"  CAM++ numpy：{report['variants']['spk_numpy']}")

    # ---------------- emotion2vec（情绪）----------------
    print("[加载] emotion2vec (fp16) ...")
    emo_model = AutoModel(
        model=args.emotion_model,
        device=args.device,
        fp16=True,
        disable_update=True,
        disable_pbar=True,
    )

    def emo_file(item):
        path, _ = item
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            return emo_model.generate(
                input=str(path), granularity="utterance", extract_embedding=False
            )

    lat = time_calls(emo_file, [(p, None) for p in paths])
    report["variants"]["emotion_file"] = summarize("emotion2vec / 文件路径", lat)
    print(f"  emotion2vec 文件路径：{report['variants']['emotion_file']}")

    # 与生产实现一致：autocast 负责 fp16 前向，输入仍用 float32 numpy
    def emo_numpy_fp16(item):
        audio, _ = item
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            return emo_model.generate(
                input=audio,
                fs=16000,
                granularity="utterance",
                extract_embedding=False,
            )

    try:
        lat = time_calls(emo_numpy_fp16, arrays)
        report["variants"]["emotion_numpy_fp16"] = summarize("emotion2vec / float16 numpy", lat)
        print(f"  emotion2vec float16 numpy：{report['variants']['emotion_numpy_fp16']}")
    except Exception as exc:  # noqa: BLE001
        report["variants"]["emotion_numpy_fp16"] = {"error": repr(exc)}
        print(f"  emotion2vec float16 numpy 失败：{exc!r}")

    # ---------------- 情绪阶段内部拆分 ----------------
    from funasr.utils.load_utils import load_audio_text_image_video

    load_lat = []
    for path in paths:
        start = time.perf_counter()
        load_audio_text_image_video(str(path), fs=16000, audio_fs=16000, data_type="sound")
        load_lat.append((time.perf_counter() - start) * 1000)
    report["variants"]["emotion_audio_load"] = summarize("情绪阶段：仅音频加载", load_lat)
    print(f"  仅音频加载：{report['variants']['emotion_audio_load']}")

    # ---------------- 固定开销 vs 音频时长 ----------------
    # 用同一条音频的裁剪版本测不同长度，判断单次耗时是固定开销还是随音频变长
    base_audio, base_sr = arrays[0]
    fixed = []
    for seconds in (0.2, 1.0, 3.0, 10.0):
        piece = base_audio[: int(seconds * base_sr)]
        if piece.size == 0:
            continue
        start = time.perf_counter()
        spk_model.generate(input=piece, fs=base_sr)
        fixed.append((seconds, round((time.perf_counter() - start) * 1000, 1)))
    report["variants"]["spk_fixed_overhead"] = fixed
    print(f"  CAM++ 固定开销（按时长）：{fixed}")

    # ---------------- 批量推理 ----------------
    # 注意：CAM++ 的批量推理只是性能参考，不能直接用于生产。
    # batch 内长度不一的音频会被零填充，污染 StatsPool 池化，
    # 实测 16 段批量与单段的嵌入余弦相似度中位仅 0.857、最低 0.143。
    def spk_batch(item):
        batch, sr = item
        return spk_model.generate(input=list(batch), fs=sr, batch_size=len(batch))

    def _batch_embedding(result) -> np.ndarray:
        payload = result[0] if isinstance(result, list) else result
        embedding = payload["spk_embedding"]
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().float().cpu().numpy()
        return np.asarray(embedding).reshape(len(embedding), -1)

    for batch_size in (8, 16):
        sample_rate = arrays[0][1]
        batches = [
            ([arr for arr, _ in arrays[i : i + batch_size]], sample_rate)
            for i in range(0, len(arrays), batch_size)
        ]
        cosines = []
        start = time.perf_counter()
        for batch, sr in batches:
            result = spk_batch((batch, sr))
            embedded = _batch_embedding(result)
            for offset, audio in enumerate(batch):
                single = spk_model.generate(input=audio, fs=sr)
                single_vec = _batch_embedding(single)[0]
                batch_vec = embedded[offset]
                cosines.append(
                    float(
                        np.dot(
                            single_vec / (np.linalg.norm(single_vec) + 1e-12),
                            batch_vec / (np.linalg.norm(batch_vec) + 1e-12),
                        )
                    )
                )
        wall = time.perf_counter() - start
        report["variants"][f"spk_batch{batch_size}"] = {
            "total_s": round(wall, 2),
            "per_call_effective_ms": round(wall * 1000 / len(arrays), 1),
            "batch_vs_single_cosine": {
                "min": round(min(cosines), 4),
                "median": round(statistics.median(cosines), 4),
            },
        }
        print(
            f"  CAM++ 批量 {batch_size}：墙钟 {wall:.2f}s，单段等效 {wall * 1000 / len(arrays):.1f}ms，"
            f"批量 vs 单段余弦中位 {statistics.median(cosines):.4f}（<1 说明结果不等价，不可用于生产）"
        )

    def emo_batch(item):
        batch = item
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            return emo_model.generate(
                input=list(batch),
                fs=16000,
                granularity="utterance",
                extract_embedding=False,
                batch_size=len(batch),
            )

    for batch_size in (8, 16):
        batches = [
            [arr for arr, _ in arrays[i : i + batch_size]] for i in range(0, len(arrays), batch_size)
        ]
        start = time.perf_counter()
        for batch in batches:
            emo_batch(batch)
        wall = time.perf_counter() - start
        report["variants"][f"emo_batch{batch_size}"] = {
            "total_s": round(wall, 2),
            "per_call_effective_ms": round(wall * 1000 / len(arrays), 1),
        }
        print(f"  情绪批量 {batch_size}：墙钟 {wall:.2f}s，单段等效 {wall * 1000 / len(arrays):.1f}ms")

    # ---------------- 线程扩展性 ----------------
    for threads in args.threads:
        items = [(p, None) for p in paths]
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=threads) as pool:
            list(pool.map(emo_file, items))
        wall = time.perf_counter() - start
        sequential = report["variants"]["emotion_file"]["total_s"]
        report["scaling"][f"emotion_file_{threads}t"] = {
            "wall_s": round(wall, 2),
            "speedup": round(sequential / wall, 2),
            "per_call_effective_ms": round(wall * 1000 / len(items), 1),
        }
        print(
            f"  情绪阶段 {threads} 线程：墙钟 {wall:.2f}s，"
            f"加速比 {sequential / wall:.2f}x，单段等效 {wall * 1000 / len(items):.1f}ms"
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
