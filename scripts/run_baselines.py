"""Run simple non-VLM baselines over a JSONL index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.majority_baseline import run_majority_baseline  # noqa: E402
from src.baselines.random_baseline import run_random_baseline  # noqa: E402


def read_index(index_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Read JSONL index rows."""

    index_path = index_path.expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"Index file does not exist: {index_path}")
    if limit is not None and limit < 0:
        raise ValueError("--limit must be zero or greater")

    samples: list[dict[str, Any]] = []
    with index_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if limit is not None and len(samples) >= limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {index_path}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} of {index_path}"
                )
            samples.append(row)
    return samples


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_baselines(
    *,
    index_path: Path,
    output_dir: Path,
    limit: int | None = None,
    seed: int = 42,
) -> dict[str, Path]:
    """Run random and majority baselines and return output paths."""

    samples = read_index(index_path, limit)
    output_dir = output_dir.expanduser().resolve()
    random_path = output_dir / "random_predictions.jsonl"
    majority_path = output_dir / "majority_predictions.jsonl"

    write_jsonl(random_path, run_random_baseline(samples, seed=seed))
    write_jsonl(majority_path, run_majority_baseline(samples))
    return {
        "random": random_path,
        "majority": majority_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple baseline predictors.")
    parser.add_argument("--index", type=Path, required=True, help="Input JSONL index.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for baseline prediction JSONL files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of samples to process.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = run_baselines(
            index_path=args.index,
            output_dir=args.output_dir,
            limit=args.limit,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(f"Random baseline: {outputs['random']}")
    print(f"Majority baseline: {outputs['majority']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
