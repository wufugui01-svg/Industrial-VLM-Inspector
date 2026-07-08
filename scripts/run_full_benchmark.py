"""Run baselines and VLM inspection variants in one benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_predictions import evaluate_predictions_file  # noqa: E402
from scripts.run_baselines import read_index, write_jsonl  # noqa: E402
from scripts.run_batch_infer import run_batch_inference  # noqa: E402
from scripts.run_global_local_infer import run_global_local_inference  # noqa: E402
from scripts.run_reference_infer import run_reference_inference  # noqa: E402
from src.baselines.majority_baseline import run_majority_baseline  # noqa: E402
from src.baselines.random_baseline import run_random_baseline  # noqa: E402
from src.datasets.path_utils import with_resolved_image_path  # noqa: E402
from src.models.base_vlm import BaseVLM  # noqa: E402
from src.models.mock_vlm import MockVLM  # noqa: E402
from src.models.qwen3vl_transformers import (  # noqa: E402
    Qwen3VLTransformers,
    QwenDependencyError,
)

METHODS = (
    "random_baseline",
    "majority_baseline",
    "single_vlm",
    "reference_vlm",
    "global_local_vlm",
)

SUMMARY_FIELDS = (
    "method",
    "total_samples",
    "json_valid_rate",
    "binary_accuracy",
    "precision",
    "recall",
    "f1",
    "avg_latency_sec",
    "p95_latency_sec",
    "avg_confidence",
    "error_count",
)


def _create_backend(
    backend: str,
    *,
    model_path: str | None,
    max_new_tokens: int,
) -> BaseVLM:
    if backend == "mock":
        return MockVLM()
    if backend == "qwen":
        if not model_path:
            raise ValueError("--model-path is required for --backend qwen")
        return Qwen3VLTransformers(
            model_name_or_path=model_path,
            max_new_tokens=max_new_tokens,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def _csv_value(value: Any) -> Any:
    return "" if value is None else value


def _summary_row(method: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": method,
        "total_samples": _csv_value(metrics.get("total_samples")),
        "json_valid_rate": _csv_value(metrics.get("json_valid_rate")),
        "binary_accuracy": _csv_value(metrics.get("binary_accuracy")),
        "precision": _csv_value(metrics.get("precision")),
        "recall": _csv_value(metrics.get("recall")),
        "f1": _csv_value(metrics.get("f1")),
        "avg_latency_sec": _csv_value(metrics.get("avg_latency_sec")),
        "p95_latency_sec": _csv_value(metrics.get("p95_latency_sec")),
        "avg_confidence": _csv_value(metrics.get("avg_confidence")),
        "error_count": _csv_value(metrics.get("error_count")),
    }


def _write_summary(rows: list[dict[str, Any]], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_selected_index(
    *,
    index_path: Path,
    selected_index_path: Path,
    limit: int | None,
    dataset_root: Path | None,
) -> list[dict[str, Any]]:
    samples = read_index(index_path, limit)
    if dataset_root is not None:
        samples = [with_resolved_image_path(sample, dataset_root) for sample in samples]
    write_jsonl(selected_index_path, samples)
    return samples


def run_full_benchmark(
    *,
    index_path: Path,
    output_dir: Path,
    backend: str = "mock",
    model_path: str | None = None,
    limit: int = 50,
    max_new_tokens: int = 128,
    grid: str = "2x2",
    dataset_root: Path | None = None,
    show_progress: bool = True,
) -> Path:
    """Run all benchmark methods and return the summary CSV path."""

    index_path = index_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if limit < 0:
        raise ValueError("--limit must be zero or greater")
    if backend not in {"mock", "qwen"}:
        raise ValueError(f"Unsupported backend: {backend}")

    predictions_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    crop_dir = output_dir / "crops"
    selected_index = output_dir / "selected_index.jsonl"
    summary_path = metrics_dir / "benchmark_summary.csv"

    samples = _write_selected_index(
        index_path=index_path,
        selected_index_path=selected_index,
        limit=limit,
        dataset_root=dataset_root,
    )
    vlm = _create_backend(
        backend,
        model_path=model_path,
        max_new_tokens=max_new_tokens,
    )

    summary_rows: list[dict[str, Any]] = []

    print("Running method=random_baseline")
    random_predictions = predictions_dir / "random_baseline_predictions.jsonl"
    random_metrics = metrics_dir / "random_baseline_metrics.json"
    write_jsonl(random_predictions, run_random_baseline(samples))
    metrics = evaluate_predictions_file(random_predictions, random_metrics)
    summary_rows.append(_summary_row("random_baseline", metrics))

    print("Running method=majority_baseline")
    majority_predictions = predictions_dir / "majority_baseline_predictions.jsonl"
    majority_metrics = metrics_dir / "majority_baseline_metrics.json"
    write_jsonl(majority_predictions, run_majority_baseline(samples))
    metrics = evaluate_predictions_file(majority_predictions, majority_metrics)
    summary_rows.append(_summary_row("majority_baseline", metrics))

    print("Running method=single_vlm")
    single_predictions = predictions_dir / "single_vlm_predictions.jsonl"
    single_metrics = metrics_dir / "single_vlm_metrics.json"
    run_batch_inference(
        index_path=selected_index,
        output_path=single_predictions,
        backend=backend,
        limit=None,
        show_progress=show_progress,
        prompt_type="strict_json",
        vlm=vlm,
    )
    metrics = evaluate_predictions_file(single_predictions, single_metrics)
    summary_rows.append(_summary_row("single_vlm", metrics))

    print("Running method=reference_vlm")
    reference_predictions = predictions_dir / "reference_vlm_predictions.jsonl"
    reference_metrics = metrics_dir / "reference_vlm_metrics.json"
    run_reference_inference(
        index_path=selected_index,
        output_path=reference_predictions,
        backend=backend,
        prompt_type="reference_strict",
        reference_strategy="first",
        limit=None,
        max_new_tokens=max_new_tokens,
        show_progress=show_progress,
        vlm=vlm,
    )
    metrics = evaluate_predictions_file(reference_predictions, reference_metrics)
    summary_rows.append(_summary_row("reference_vlm", metrics))

    print("Running method=global_local_vlm")
    global_local_predictions = predictions_dir / "global_local_vlm_predictions.jsonl"
    global_local_metrics = metrics_dir / "global_local_vlm_metrics.json"
    run_global_local_inference(
        index_path=selected_index,
        output_path=global_local_predictions,
        backend=backend,
        prompt_type="strict_json",
        grid=grid,
        crop_dir=crop_dir,
        limit=None,
        max_new_tokens=max_new_tokens,
        show_progress=show_progress,
        vlm=vlm,
    )
    metrics = evaluate_predictions_file(
        global_local_predictions,
        global_local_metrics,
    )
    summary_rows.append(_summary_row("global_local_vlm", metrics))

    _write_summary(summary_rows, summary_path)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full benchmark across baselines and VLM methods."
    )
    parser.add_argument("--index", type=Path, required=True, help="Input JSONL index.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory for benchmark outputs.",
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
        help="Required local model directory when --backend qwen is selected.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of samples to evaluate.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum generated tokens for the Qwen backend.",
    )
    parser.add_argument(
        "--grid",
        choices=["2x2", "3x3"],
        default="2x2",
        help="Grid crop layout for global-local VLM.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional dataset root used with image_relative_path for portability.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary_path = run_full_benchmark(
            index_path=args.index,
            output_dir=args.output_dir,
            backend=args.backend,
            model_path=args.model_path,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
            grid=args.grid,
            dataset_root=args.dataset_root,
        )
    except (FileNotFoundError, ValueError, QwenDependencyError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(f"Benchmark summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
