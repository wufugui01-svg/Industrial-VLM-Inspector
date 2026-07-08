"""Tests for the reference-based batch inference script."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts.run_reference_infer import run_reference_inference


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_reference_inference_writes_selected_reference_output() -> None:
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
                "answer": "A",
                "task_type": "Anomaly Detection",
                "object_category": "bottle",
            },
        ]
        index_path = root / "index.jsonl"
        output_path = root / "reference_predictions.jsonl"
        index_path.write_text(
            "".join(json.dumps(sample) + "\n" for sample in samples),
            encoding="utf-8",
        )

        summary = run_reference_inference(
            index_path=index_path,
            output_path=output_path,
            backend="mock",
            reference_strategy="first",
            limit=2,
            show_progress=False,
        )
        rows = _read_jsonl(output_path)

    assert summary.total == 2
    assert summary.succeeded == 2
    assert summary.failed == 0
    assert len(rows) == 2
    assert rows[0]["sample_id"] == "normal-reference"
    assert rows[0]["reference_image_path"] is None
    assert rows[0]["prediction"]["parse_status"] == "success"
    assert rows[0]["error"] is None
    assert rows[1]["sample_id"] == "test-sample"
    assert rows[1]["image_path"] == str(test_image)
    assert rows[1]["reference_image_path"] == str(reference_image)
    assert rows[1]["ground_truth_answer"] == "A"
    assert rows[1]["task_type"] == "Anomaly Detection"
    assert rows[1]["object_category"] == "bottle"
    assert rows[1]["prediction"]["confidence"] == 0.5
    assert rows[1]["error"] is None
    assert isinstance(rows[1]["latency_sec"], float)


def test_reference_inference_without_candidate_keeps_reference_none() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        test_image = root / "widget-bad.png"
        test_image.write_bytes(b"test")

        sample = {
            "sample_id": "only-sample",
            "image_path": str(test_image),
            "label": "abnormal",
            "answer": "A",
            "task_type": "Anomaly Detection",
            "object_category": "widget",
        }
        index_path = root / "index.jsonl"
        output_path = root / "reference_predictions.jsonl"
        index_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

        summary = run_reference_inference(
            index_path=index_path,
            output_path=output_path,
            backend="mock",
            show_progress=False,
        )
        rows = _read_jsonl(output_path)

    assert summary.total == 1
    assert summary.succeeded == 1
    assert summary.failed == 0
    assert rows[0]["sample_id"] == "only-sample"
    assert rows[0]["reference_image_path"] is None
    assert rows[0]["prediction"]["parse_status"] == "success"
    assert rows[0]["error"] is None
