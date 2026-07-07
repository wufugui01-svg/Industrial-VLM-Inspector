"""Tests for the prompt ablation experiment runner."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from scripts.run_prompt_ablation import PROMPT_TYPES, run_prompt_ablation


def test_mock_prompt_ablation_writes_all_artifacts() -> None:
    samples = [
        {
            "sample_id": "sample-1",
            "image_path": "/tmp/one.png",
            "answer": "A",
            "task_type": "Anomaly Detection",
            "object_category": "bottle",
        },
        {
            "sample_id": "sample-2",
            "image_path": "/tmp/two.png",
            "answer": "B",
            "task_type": "Defect Classification",
            "object_category": "cable",
        },
        {
            "sample_id": "sample-3",
            "image_path": "/tmp/three.png",
            "answer": "C",
            "task_type": "Defect Localization",
            "object_category": "capsule",
        },
    ]

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        index_path = root / "index.jsonl"
        output_dir = root / "ablation"
        summary_path = root / "metrics" / "summary.csv"
        index_path.write_text(
            "".join(json.dumps(sample) + "\n" for sample in samples),
            encoding="utf-8",
        )

        returned_path = run_prompt_ablation(
            index_path=index_path,
            output_dir=output_dir,
            backend="mock",
            limit=2,
            summary_path=summary_path,
            show_progress=False,
        )

        assert returned_path == summary_path.resolve()
        manifest = json.loads(
            (output_dir / "experiment_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["sample_count"] == 2
        assert manifest["seed"] == 42
        for prompt_type in PROMPT_TYPES:
            predictions = output_dir / f"{prompt_type}_predictions.jsonl"
            metrics = output_dir / f"{prompt_type}_metrics.json"
            assert predictions.is_file()
            assert metrics.is_file()
            assert len(predictions.read_text(encoding="utf-8").splitlines()) == 2
            assert json.loads(metrics.read_text(encoding="utf-8"))[
                "total_samples"
            ] == 2

        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            summary_rows = list(csv.DictReader(handle))

    assert [row["prompt_type"] for row in summary_rows] == list(PROMPT_TYPES)
    assert all(row["total_samples"] == "2" for row in summary_rows)
    assert all(row["json_valid_rate"] == "1.0" for row in summary_rows)
    assert all(row["parse_success_rate"] == "1.0" for row in summary_rows)
    assert all(row["avg_latency_sec"] for row in summary_rows)
    assert all(row["p95_latency_sec"] for row in summary_rows)
    assert all(row["binary_accuracy"] == "" for row in summary_rows)
