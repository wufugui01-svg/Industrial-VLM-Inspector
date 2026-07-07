"""Command-line entry point for building an MMAD JSONL index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.mmad_indexer import build_mmad_index  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a flat JSONL index from local MMAD annotations."
    )
    parser.add_argument("--mmad-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of question samples to write.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = build_mmad_index(
            mmad_root=args.mmad_root,
            output_path=args.output,
            limit=args.limit,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(f"样本总数: {summary.total_samples}")
    print("前 3 条样本:")
    for sample in summary.first_samples:
        print(json.dumps(sample, ensure_ascii=False, indent=2))
    print(f"缺失图像数量（唯一路径）: {summary.missing_image_count}")
    print(f"缺失图像涉及样本数: {summary.missing_image_sample_count}")
    print("不同 task_type 的统计:")
    for task_type, count in sorted(summary.task_type_counts.items()):
        print(f"  {task_type}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
