"""Tests for the independent MMAD multiple-choice benchmark."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts.run_mmad_benchmark import run_mmad_benchmark
from src.eval.mmad_benchmark import calculate_mmad_metrics, parse_mmad_answer


def test_parse_mmad_answer() -> None:
    options = ["A: Yes", "B: No"]
    assert parse_mmad_answer('{"answer": "B"}', options) == ("B", "success")
    assert parse_mmad_answer("The answer is {\"answer\": \"A\"}.", options) == (
        "A",
        "repaired",
    )
    assert parse_mmad_answer("C", options) == (None, "failed")


def test_mmad_metrics_use_option_exact_match() -> None:
    metrics = calculate_mmad_metrics(
        [
            {
                "prediction_answer": "A",
                "ground_truth_answer": "A",
                "parse_status": "success",
                "task_type": "Anomaly Detection",
                "object_category": "bottle",
                "error": None,
            },
            {
                "prediction_answer": "A",
                "ground_truth_answer": "B",
                "parse_status": "success",
                "task_type": "Anomaly Detection",
                "object_category": "bottle",
                "error": None,
            },
        ]
    )
    assert metrics["accuracy"] == 0.5
    assert metrics["valid_prediction_accuracy"] == 0.5
    assert metrics["coverage"] == 1.0


def test_mock_mmad_benchmark_writes_outputs() -> None:
    sample = {
        "sample_id": "sample-1",
        "image_path": "/tmp/not-used-by-mock.png",
        "question": "Is there a defect?",
        "options": ["A: Yes", "B: No"],
        "answer": "A",
        "task_type": "Anomaly Detection",
        "object_category": "bottle",
    }
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        index_path = root / "index.jsonl"
        output_path = root / "predictions.jsonl"
        metrics_path = root / "metrics.json"
        index_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

        metrics = run_mmad_benchmark(
            index_path=index_path,
            output_path=output_path,
            metrics_path=metrics_path,
            backend="mock",
            show_progress=False,
        )

        row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["prediction_answer"] == "A"
    assert metrics["accuracy"] == 1.0
