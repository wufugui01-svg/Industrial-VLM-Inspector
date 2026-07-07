"""Tests for mock batch inference."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts.run_batch_infer import run_batch_inference


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_batch_inference_writes_complete_output() -> None:
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
            "task_type": "Defect Localization",
            "object_category": "cable",
        },
        {
            "sample_id": "sample-3",
            "image_path": "/tmp/three.png",
            "answer": "C",
            "task_type": "Defect Classification",
            "object_category": "capsule",
        },
    ]

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        index_path = root / "index.jsonl"
        output_path = root / "nested" / "predictions.jsonl"
        index_path.write_text(
            "".join(json.dumps(sample) + "\n" for sample in samples),
            encoding="utf-8",
        )

        summary = run_batch_inference(
            index_path=index_path,
            output_path=output_path,
            limit=2,
            show_progress=False,
        )
        rows = _read_jsonl(output_path)

    assert summary.total == 2
    assert summary.succeeded == 2
    assert summary.failed == 0
    assert len(rows) == 2
    assert rows[0]["sample_id"] == "sample-1"
    assert rows[0]["ground_truth_answer"] == "A"
    assert rows[0]["task_type"] == "Anomaly Detection"
    assert rows[0]["object_category"] == "bottle"
    assert rows[0]["image_path"] == "/tmp/one.png"
    assert rows[0]["prediction"]["confidence"] == 0.5
    assert rows[0]["error"] is None
    assert isinstance(rows[0]["latency_sec"], float)
    assert "gpu_memory_allocated_mb" in rows[0]
    assert "gpu_memory_reserved_mb" in rows[0]


def test_bad_row_is_recorded_and_batch_continues() -> None:
    valid_sample = {
        "sample_id": "valid-after-error",
        "image_path": "/tmp/valid.png",
        "answer": "A",
        "task_type": "Anomaly Detection",
        "object_category": "widget",
    }

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        index_path = root / "index.jsonl"
        output_path = root / "predictions.jsonl"
        index_path.write_text(
            "not valid json\n" + json.dumps(valid_sample) + "\n",
            encoding="utf-8",
        )

        summary = run_batch_inference(
            index_path=index_path,
            output_path=output_path,
            show_progress=False,
        )
        rows = _read_jsonl(output_path)

    assert summary.total == 2
    assert summary.succeeded == 1
    assert summary.failed == 1
    assert rows[0]["sample_id"] == "line_00000001"
    assert rows[0]["prediction"] is None
    assert rows[0]["error"].startswith("JSONDecodeError:")
    assert isinstance(rows[0]["latency_sec"], float)
    assert rows[1]["sample_id"] == "valid-after-error"
    assert rows[1]["prediction"] is not None
    assert rows[1]["error"] is None
