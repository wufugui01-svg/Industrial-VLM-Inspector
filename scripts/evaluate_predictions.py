"""Evaluate JSONL predictions and save basic metrics as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import calculate_metrics  # noqa: E402


def read_prediction_rows(predictions_path: Path) -> list[dict[str, Any]]:
    """Read prediction rows while representing malformed lines as errors."""

    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"Predictions file does not exist: {predictions_path}"
        )

    rows: list[dict[str, Any]] = []
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError("Prediction row must be a JSON object")
            except (json.JSONDecodeError, TypeError) as exc:
                row = {
                    "sample_id": f"line_{line_number:08d}",
                    "prediction": None,
                    "final_prediction": None,
                    "global_prediction": None,
                    "ground_truth_answer": "",
                    "task_type": "unknown",
                    "object_category": "unknown",
                    "image_path": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
    return rows


def evaluate_predictions_file(
    predictions_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Evaluate a prediction JSONL file and write its metrics."""

    predictions_path = predictions_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    metrics = calculate_metrics(read_prediction_rows(predictions_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate basic metrics from batch prediction JSONL."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Input prediction JSONL path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output metrics JSON path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metrics = evaluate_predictions_file(args.predictions, args.output)
    except FileNotFoundError as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Metrics written to: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
