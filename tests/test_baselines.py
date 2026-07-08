"""Tests for simple non-VLM baselines."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts.run_baselines import run_baselines
from src.baselines.majority_baseline import infer_majority_class, run_majority_baseline
from src.baselines.random_baseline import run_random_baseline


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_random_baseline_outputs_prediction_records() -> None:
    samples = [
        {
            "sample_id": "sample-1",
            "image_path": "/tmp/one.png",
            "answer": "yes",
            "task_type": "Anomaly Detection",
            "object_category": "bottle",
        }
    ]

    rows = run_random_baseline(samples, seed=1)

    assert len(rows) == 1
    row = rows[0]
    assert row["sample_id"] == "sample-1"
    assert row["image_path"] == "/tmp/one.png"
    assert row["ground_truth_answer"] == "yes"
    assert row["task_type"] == "Anomaly Detection"
    assert row["object_category"] == "bottle"
    assert row["method"] == "random"
    assert row["prediction"]["confidence"] == 0.5
    assert row["prediction"]["parse_status"] == "success"
    assert isinstance(row["prediction"]["is_anomaly"], bool)


def test_majority_baseline_infers_majority_anomaly_class() -> None:
    samples = [
        {"sample_id": "a", "answer": "yes"},
        {"sample_id": "b", "answer": "defective"},
        {"sample_id": "c", "answer": "normal"},
    ]

    assert infer_majority_class(samples) is True
    rows = run_majority_baseline(samples)

    assert all(row["method"] == "majority" for row in rows)
    assert all(row["prediction"]["is_anomaly"] is True for row in rows)
    assert all(row["prediction"]["confidence"] == 0.5 for row in rows)


def test_majority_baseline_defaults_to_normal_when_labels_are_ambiguous() -> None:
    samples = [
        {"sample_id": "a", "answer": "A"},
        {"sample_id": "b", "answer": "B"},
    ]

    assert infer_majority_class(samples) is False
    rows = run_majority_baseline(samples)

    assert all(row["prediction"]["is_anomaly"] is False for row in rows)
    assert all(row["prediction"]["defect_type"] == "none" for row in rows)


def test_run_baselines_writes_random_and_majority_jsonl() -> None:
    samples = [
        {
            "sample_id": "one",
            "image_path": "/tmp/one.png",
            "answer": "yes",
            "task_type": "Anomaly Detection",
            "object_category": "bottle",
        },
        {
            "sample_id": "two",
            "image_path": "/tmp/two.png",
            "answer": "normal",
            "task_type": "Anomaly Detection",
            "object_category": "bottle",
        },
        {
            "sample_id": "three",
            "image_path": "/tmp/three.png",
            "answer": "yes",
            "task_type": "Anomaly Detection",
            "object_category": "bottle",
        },
    ]

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        index_path = root / "index.jsonl"
        output_dir = root / "baselines"
        index_path.write_text(
            "".join(json.dumps(sample) + "\n" for sample in samples),
            encoding="utf-8",
        )

        outputs = run_baselines(
            index_path=index_path,
            output_dir=output_dir,
            limit=2,
        )
        random_rows = _read_jsonl(outputs["random"])
        majority_rows = _read_jsonl(outputs["majority"])

    assert outputs["random"].name == "random_predictions.jsonl"
    assert outputs["majority"].name == "majority_predictions.jsonl"
    assert len(random_rows) == 2
    assert len(majority_rows) == 2
    assert {row["method"] for row in random_rows} == {"random"}
    assert {row["method"] for row in majority_rows} == {"majority"}
