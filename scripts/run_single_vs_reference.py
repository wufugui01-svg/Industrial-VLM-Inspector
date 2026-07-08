"""Compare single-image and reference-based inspection pipelines."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_predictions import evaluate_predictions_file  # noqa: E402
from scripts.run_batch_infer import run_batch_inference  # noqa: E402
from scripts.run_reference_infer import run_reference_inference  # noqa: E402
from src.models.base_vlm import BaseVLM  # noqa: E402
from src.models.mock_vlm import MockVLM  # noqa: E402
from src.models.qwen3vl_transformers import (  # noqa: E402
    Qwen3VLTransformers,
    QwenDependencyError,
)

SUMMARY_FIELDS = (
    "method",
    "total_samples",
    "json_valid_rate",
    "error_count",
    "avg_latency_sec",
    "binary_accuracy",
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
    return "null" if value is None else value


def _summary_row(method: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": method,
        "total_samples": metrics.get("total_samples", 0),
        "json_valid_rate": metrics.get("json_valid_rate", 0.0),
        "error_count": metrics.get("error_count", 0),
        "avg_latency_sec": _csv_value(metrics.get("avg_latency_sec")),
        "binary_accuracy": _csv_value(metrics.get("binary_accuracy")),
    }


def _write_summary(rows: list[dict[str, Any]], summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_single_vs_reference(
    *,
    index_path: Path,
    output_dir: Path,
    backend: str = "mock",
    model_path: str | None = None,
    limit: int | None = None,
    max_new_tokens: int = 512,
    show_progress: bool = True,
) -> Path:
    """Run single-image and reference-based pipelines and summarize metrics."""

    index_path = index_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"Index file does not exist: {index_path}")
    if limit is not None and limit < 0:
        raise ValueError("--limit must be zero or greater")

    predictions_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    single_predictions = predictions_dir / "single_predictions.jsonl"
    reference_predictions = predictions_dir / "reference_predictions.jsonl"
    single_metrics_path = metrics_dir / "single_metrics.json"
    reference_metrics_path = metrics_dir / "reference_metrics.json"
    summary_path = metrics_dir / "single_vs_reference_summary.csv"

    vlm = _create_backend(
        backend,
        model_path=model_path,
        max_new_tokens=max_new_tokens,
    )

    print("Running method=single-image")
    run_batch_inference(
        index_path=index_path,
        output_path=single_predictions,
        backend=backend,
        limit=limit,
        show_progress=show_progress,
        prompt_type="strict_json",
        vlm=vlm,
    )
    single_metrics = evaluate_predictions_file(
        single_predictions,
        single_metrics_path,
    )

    print("Running method=reference-based")
    run_reference_inference(
        index_path=index_path,
        output_path=reference_predictions,
        backend=backend,
        model_path=model_path,
        prompt_type="reference_strict",
        reference_strategy="first",
        limit=limit,
        max_new_tokens=max_new_tokens,
        show_progress=show_progress,
        vlm=vlm,
    )
    reference_metrics = evaluate_predictions_file(
        reference_predictions,
        reference_metrics_path,
    )

    _write_summary(
        [
            _summary_row("single-image", single_metrics),
            _summary_row("reference-based", reference_metrics),
        ],
        summary_path,
    )
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare single-image and reference-based inspection."
    )
    parser.add_argument("--index", type=Path, required=True, help="Input JSONL index.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory for predictions/ and metrics/ outputs.",
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
        default=None,
        help="Optional maximum number of samples per method.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens for the Qwen backend.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary_path = run_single_vs_reference(
            index_path=args.index,
            output_dir=args.output_dir,
            backend=args.backend,
            model_path=args.model_path,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
        )
    except (FileNotFoundError, ValueError, QwenDependencyError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(f"Single-vs-reference summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
