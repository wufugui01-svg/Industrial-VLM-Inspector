"""Tests for the full benchmark runner."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from PIL import Image

from scripts.run_full_benchmark import METHODS, run_full_benchmark


def test_full_benchmark_mock_generates_all_outputs() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        image_a = root / "normal.png"
        image_b = root / "abnormal.png"
        Image.new("RGB", (48, 48), color=(100, 120, 140)).save(image_a)
        Image.new("RGB", (48, 48), color=(140, 120, 100)).save(image_b)
        samples = [
            {
                "sample_id": "normal-sample",
                "image_path": str(image_a),
                "answer": "A",
                "label": "good",
                "task_type": "Anomaly Detection",
                "object_category": "bottle",
            },
            {
                "sample_id": "abnormal-sample",
                "image_path": str(image_b),
                "answer": "B",
                "label": "abnormal",
                "task_type": "Anomaly Detection",
                "object_category": "bottle",
            },
        ]
        index_path = root / "index.jsonl"
        output_dir = root / "benchmark"
        index_path.write_text(
            "".join(json.dumps(sample) + "\n" for sample in samples),
            encoding="utf-8",
        )

        summary_path = run_full_benchmark(
            index_path=index_path,
            output_dir=output_dir,
            backend="mock",
            limit=2,
            show_progress=False,
        )

        assert summary_path == output_dir.resolve() / "metrics" / "benchmark_summary.csv"
        assert summary_path.is_file()
        assert (output_dir / "selected_index.jsonl").is_file()
        for method in METHODS:
            predictions_path = output_dir / "predictions" / f"{method}_predictions.jsonl"
            metrics_path = output_dir / "metrics" / f"{method}_metrics.json"
            assert predictions_path.is_file()
            assert metrics_path.is_file()
            assert len(predictions_path.read_text(encoding="utf-8").splitlines()) == 2
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            assert metrics["total_samples"] == 2

        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    assert [row["method"] for row in rows] == list(METHODS)
    assert all(row["total_samples"] == "2" for row in rows)
    assert all("json_valid_rate" in row for row in rows)
    assert all("avg_confidence" in row for row in rows)
    assert all("error_count" in row for row in rows)
