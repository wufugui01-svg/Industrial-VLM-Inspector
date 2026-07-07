"""Run basic, industrial, and strict-JSON prompt ablation experiments."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_predictions import evaluate_predictions_file  # noqa: E402
from scripts.run_batch_infer import run_batch_inference  # noqa: E402
from src.models.base_vlm import BaseVLM  # noqa: E402
from src.models.mock_vlm import MockVLM  # noqa: E402
from src.models.qwen3vl_transformers import (  # noqa: E402
    Qwen3VLTransformers,
    QwenDependencyError,
)
from src.utils.config import configured_random_seed  # noqa: E402

PROMPT_TYPES = ("basic", "industrial", "strict_json")
SUMMARY_FIELDS = (
    "prompt_type",
    "total_samples",
    "json_valid_rate",
    "parse_success_rate",
    "repair_rate",
    "avg_latency_sec",
    "p95_latency_sec",
    "binary_accuracy",
)


def _create_backend(
    backend: str,
    model_path: str | None,
    max_new_tokens: int,
    device_map: str,
    torch_dtype: str,
    min_pixels: int | None,
    max_pixels: int | None,
) -> BaseVLM:
    if backend == "mock":
        return MockVLM()
    if backend == "qwen":
        if not model_path:
            raise ValueError("--model-path is required for --backend qwen")
        return Qwen3VLTransformers(
            model_name_or_path=model_path,
            max_new_tokens=max_new_tokens,
            device_map=device_map,
            torch_dtype=torch_dtype,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def _write_summary(rows: list[dict[str, Any]], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_stratified_selection(
    index_path: Path,
    selected_path: Path,
    limit: int | None,
    seed: int,
) -> list[str]:
    """Write one fixed, shuffled round-robin sample set for every prompt."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with index_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Index row {line_number} is not a JSON object")
            key = (
                str(row.get("task_type") or "unknown"),
                str(row.get("object_category") or "unknown"),
            )
            groups[key].append(row)

    rng = random.Random(seed)
    queues: list[deque[dict[str, Any]]] = []
    for key in sorted(groups):
        rng.shuffle(groups[key])
        queues.append(deque(groups[key]))
    rng.shuffle(queues)

    selected: list[dict[str, Any]] = []
    target = sum(len(queue) for queue in queues) if limit is None else limit
    while queues and len(selected) < target:
        next_round: list[deque[dict[str, Any]]] = []
        for queue in queues:
            if queue and len(selected) < target:
                selected.append(queue.popleft())
            if queue:
                next_round.append(queue)
        queues = next_round

    selected_path.parent.mkdir(parents=True, exist_ok=True)
    with selected_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return [str(row.get("sample_id") or "") for row in selected]


def run_prompt_ablation(
    *,
    index_path: Path,
    output_dir: Path,
    backend: str,
    model_path: str | None = None,
    limit: int | None = None,
    summary_path: Path | None = None,
    show_progress: bool = True,
    max_new_tokens: int = 128,
    dataset_root: Path | None = None,
    seed: int = 42,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> Path:
    """Run all prompt variants with one shared backend instance."""

    index_path = index_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if limit is not None and limit < 0:
        raise ValueError("--limit must be zero or greater")
    output_dir.mkdir(parents=True, exist_ok=True)
    if summary_path is None:
        summary_path = output_dir / "prompt_ablation_summary.csv"
    else:
        summary_path = summary_path.expanduser().resolve()

    selected_index = output_dir / "selected_samples.jsonl"
    selected_ids = _write_stratified_selection(
        index_path, selected_index, limit, seed
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "model_path": model_path,
        "source_index": str(index_path),
        "selected_index": str(selected_index),
        "sample_count": len(selected_ids),
        "sample_ids": selected_ids,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "prompt_types": list(PROMPT_TYPES),
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    vlm = _create_backend(
        backend,
        model_path,
        max_new_tokens,
        device_map,
        torch_dtype,
        min_pixels,
        max_pixels,
    )
    summary_rows: list[dict[str, Any]] = []

    for prompt_type in PROMPT_TYPES:
        predictions_path = output_dir / f"{prompt_type}_predictions.jsonl"
        metrics_path = output_dir / f"{prompt_type}_metrics.json"
        print(f"Running prompt_type={prompt_type}")

        run_batch_inference(
            index_path=selected_index,
            output_path=predictions_path,
            backend=backend,
            limit=None,
            show_progress=show_progress,
            prompt_type=prompt_type,
            vlm=vlm,
            dataset_root=dataset_root,
        )
        metrics = evaluate_predictions_file(predictions_path, metrics_path)
        summary_rows.append(
            {
                "prompt_type": prompt_type,
                "total_samples": metrics["total_samples"],
                "json_valid_rate": metrics["json_valid_rate"],
                "parse_success_rate": metrics["parse_success_rate"],
                "repair_rate": metrics["repair_rate"],
                "avg_latency_sec": metrics["avg_latency_sec"],
                "p95_latency_sec": metrics["p95_latency_sec"],
                "binary_accuracy": metrics["binary_accuracy"],
            }
        )

    _write_summary(summary_rows, summary_path)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run prompt ablation over three inspection prompt strategies."
    )
    parser.add_argument("--index", type=Path, required=True, help="Input JSONL index.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for per-prompt predictions and metrics.",
    )
    parser.add_argument(
        "--backend",
        choices=["mock", "qwen"],
        default="mock",
        help="Inference backend.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Required local model directory for the Qwen backend.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit applied to every prompt variant.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--seed", type=int, default=configured_random_seed())
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary_path = run_prompt_ablation(
            index_path=args.index,
            output_dir=args.output_dir,
            backend=args.backend,
            model_path=args.model_path,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
            dataset_root=args.dataset_root,
            seed=args.seed,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
    except (FileNotFoundError, ValueError, QwenDependencyError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(f"Prompt ablation summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
