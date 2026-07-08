"""Tests for single-image versus reference-based comparison runner."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from scripts.run_single_vs_reference import run_single_vs_reference


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_single_vs_reference_outputs_predictions_metrics_and_summary() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        reference_image = root / "bottle-good.png"
        test_image = root / "bottle-bad.png"
        reference_image.write_bytes(b"reference")
        test_image.write_bytes(b"test")

        samples = [
            {
                "sample_id": "normal-reference",
                "image_path": str(reference_image),
                "label": "good",
                "answer": "normal",
                "task_type": "Anomaly Detection",
                "object_category": "bottle",
            },
            {
                "sample_id": "test-sample",
                "image_path": str(test_image),
                "label": "abnormal",
                "answer": "yes",
                "task_type": "Anomaly Detection",
                "object_category": "bottle",
            },
        ]
        index_path = root / "index.jsonl"
        output_dir = root / "outputs"
        index_path.write_text(
            "".join(json.dumps(sample) + "\n" for sample in samples),
            encoding="utf-8",
        )

        summary_path = run_single_vs_reference(
            index_path=index_path,
            output_dir=output_dir,
            backend="mock",
            limit=2,
            show_progress=False,
        )

        single_predictions = output_dir / "predictions" / "single_predictions.jsonl"
        reference_predictions = (
            output_dir / "predictions" / "reference_predictions.jsonl"
        )
        single_metrics = output_dir / "metrics" / "single_metrics.json"
        reference_metrics = output_dir / "metrics" / "reference_metrics.json"

        assert single_predictions.is_file()
        assert reference_predictions.is_file()
        assert single_metrics.is_file()
        assert reference_metrics.is_file()
        expected_summary = output_dir / "metrics" / "single_vs_reference_summary.csv"
        assert summary_path.resolve() == expected_summary.resolve()
        assert summary_path.is_file()

        single_rows = _read_jsonl(single_predictions)
        reference_rows = _read_jsonl(reference_predictions)
        assert len(single_rows) == 2
        assert len(reference_rows) == 2
        assert single_rows[1]["reference_image_path"] is None
        assert reference_rows[1]["reference_image_path"] == str(reference_image)

        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            summary_rows = list(csv.DictReader(handle))

    assert [row["method"] for row in summary_rows] == [
        "single-image",
        "reference-based",
    ]
    assert all(row["total_samples"] == "2" for row in summary_rows)
    assert all(row["json_valid_rate"] == "1.0" for row in summary_rows)
    assert all(row["error_count"] == "0" for row in summary_rows)
    assert all(row["binary_accuracy"] == "0.5" for row in summary_rows)
