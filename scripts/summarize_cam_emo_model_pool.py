"""把模型池 benchmark 的 JSON 汇总成中文 Markdown 报告。"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_dir(path: Path) -> dict[int, dict]:
    result = {}
    for item in path.glob("concurrency_*.json"):
        payload = json.loads(item.read_text(encoding="utf-8"))
        result[int(payload["concurrency"])] = payload
    return result


def fmt(value):
    return "-" if value is None else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pool-size", default="unknown")
    args = parser.parse_args()
    baseline = load_dir(args.baseline)
    pool = load_dir(args.pool)
    levels = sorted(set(baseline) | set(pool))
    lines = [
        "# CAM++/emotion2vec 同卡模型池完整音频并发报告",
        "",
        f"- 测试音频：完整长音频；`showSpk=true`、`showEmotion=true`。",
        f"- 基线：CAM++/emotion2vec 各 1 副本；模型池：各 {args.pool_size} 副本，全部 `cuda:0`。",
        "- 指标来自 benchmark 原始 JSON；Qwen-asr 服务保持运行，压测只访问 8084。",
        "",
        "## 并发结果",
        "",
        "| 并发 | baseline 成功/失败 | pool 成功/失败 | baseline wall(s) | pool wall(s) | wall 变化 | baseline 音频/分钟 | pool 音频/分钟 | 吞吐变化 | 是否提升 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    improvements = []
    for level in levels:
        left, right = baseline.get(level), pool.get(level)
        if not left or not right:
            left_result = "-" if not left else f"{left.get('success', 0)}/{left.get('failed', 0)}"
            right_result = "-" if not right else f"{right.get('success', 0)}/{right.get('failed', 0)}"
            lines.append(
                f"| {level} | {left_result} | {right_result} | - | - | - | - | - | - | 待测 |"
            )
            continue
        base_t, pool_t = left.get("audio_per_min", 0), right.get("audio_per_min", 0)
        change = (pool_t / base_t - 1) * 100 if base_t else None
        improved = bool(change is not None and change > 0 and right.get("failed", 0) <= left.get("failed", 0))
        improvements.append(improved)
        lines.append(
            f"| {level} | {left.get('success', 0)}/{left.get('failed', 0)} | "
            f"{right.get('success', 0)}/{right.get('failed', 0)} | {fmt(left.get('wall_s'))} | "
            f"{fmt(right.get('wall_s'))} | {fmt((right.get('wall_s', 0) / left.get('wall_s', 1) - 1) * 100)}% | "
            f"{fmt(base_t)} | {fmt(pool_t)} | {fmt(change)}% | {'是' if improved else '否'} |"
        )
    lines.extend([
        "",
        "## 结论",
        "",
        f"- 至少一个完成档位有吞吐提升：{'是' if any(improvements) else '尚未证明'}。",
        "- 需要结合 `before/after` 中的 GPU 0 利用率、显存、模型池 busy/waiting、Qwen waiting/KV 和错误率判断同卡抢占代价。",
        "- 角色/情绪一致性与文本、分段、时间轴对比需从各档原始响应补充；模型池不能改变音频级 CAM++ 聚类边界。",
        "",
        "## 原始数据",
        "",
        f"- baseline：`{args.baseline}`",
        f"- pool：`{args.pool}`",
    ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
