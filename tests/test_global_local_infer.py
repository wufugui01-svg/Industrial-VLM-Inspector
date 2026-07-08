"""Tests for global-to-local batch inference."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from PIL import Image

from scripts.run_global_local_infer import run_global_local_inference


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_global_local_inference_with_mock_backend() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        image_path = root / "sample.png"
        crop_dir = root / "crops"
        output_path = root / "global_local_predictions.jsonl"
        index_path = root / "index.jsonl"
        Image.new("RGB", (80, 60), color=(120, 140, 160)).save(image_path)
        sample = {
            "sample_id": "sample-1",
            "image_path": str(image_path),
            "answer": "yes",
            "task_type": "Anomaly Detection",
            "object_category": "bottle",
        }
        index_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

        summary = run_global_local_inference(
            index_path=index_path,
            output_path=output_path,
            backend="mock",
            grid="2x2",
            crop_dir=crop_dir,
            show_progress=False,
        )
        rows = _read_jsonl(output_path)
        crop_files_exist = [
            Path(crop["crop_path"]).is_file()
            for crop in rows[0]["crop_predictions"]
        ]

    assert summary.total == 1
    assert summary.succeeded == 1
    assert summary.failed == 0
    assert len(rows) == 1
    row = rows[0]
    assert row["sample_id"] == "sample-1"
    assert row["image_path"] == str(image_path)
    assert row["global_prediction"]["parse_status"] == "success"
    assert len(row["crop_predictions"]) == 4
    assert row["final_prediction"]["is_anomaly"] is True
    assert row["final_prediction"]["confidence"] == 0.5
    assert row["final_prediction"]["defect_location"] == (
        "top_left, top_right, bottom_left, bottom_right"
    )
    assert row["final_prediction"]["reason"].startswith("global-local aggregation")
    assert row["ground_truth_answer"] == "yes"
    assert row["task_type"] == "Anomaly Detection"
    assert row["object_category"] == "bottle"
    assert isinstance(row["latency_sec"], float)
    assert row["error"] is None
    assert all(crop_files_exist)
    for crop in row["crop_predictions"]:
        assert crop["region_name"]
        assert crop["box"]
        assert crop["prediction"]["parse_status"] == "success"


def test_global_local_single_sample_failure_does_not_abort_batch() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        valid_image = root / "valid.png"
        Image.new("RGB", (40, 40), color=(90, 100, 110)).save(valid_image)
        samples = [
            {
                "sample_id": "missing-image",
                "image_path": str(root / "missing.png"),
                "answer": "yes",
                "task_type": "Anomaly Detection",
                "object_category": "bottle",
            },
            {
                "sample_id": "valid-image",
                "image_path": str(valid_image),
                "answer": "yes",
                "task_type": "Anomaly Detection",
                "object_category": "bottle",
            },
        ]
        index_path = root / "index.jsonl"
        output_path = root / "predictions.jsonl"
        index_path.write_text(
            "".join(json.dumps(sample) + "\n" for sample in samples),
            encoding="utf-8",
        )

        summary = run_global_local_inference(
            index_path=index_path,
            output_path=output_path,
            backend="mock",
            crop_dir=root / "crops",
            show_progress=False,
        )
        rows = _read_jsonl(output_path)

    assert summary.total == 2
    assert summary.succeeded == 1
    assert summary.failed == 1
    assert rows[0]["sample_id"] == "missing-image"
    assert rows[0]["global_prediction"] is None
    assert rows[0]["final_prediction"] is None
    assert rows[0]["error"].startswith("FileNotFoundError:")
    assert rows[1]["sample_id"] == "valid-image"
    assert rows[1]["final_prediction"]["is_anomaly"] is True
    assert rows[1]["error"] is None


def test_global_local_resolves_image_relative_path_with_dataset_root() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        dataset_root = root / "dataset"
        image_path = dataset_root / "class-a" / "good" / "001.png"
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (40, 40), color=(100, 120, 140)).save(image_path)
        index_path = root / "index.jsonl"
        output_path = root / "predictions.jsonl"
        sample = {
            "sample_id": "relative-path-sample",
            "image_path": "D:\\stale\\windows\\path\\001.png",
            "image_relative_path": "class-a/good/001.png",
            "answer": "yes",
            "task_type": "Anomaly Detection",
            "object_category": "class-a",
        }
        index_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

        summary = run_global_local_inference(
            index_path=index_path,
            output_path=output_path,
            backend="mock",
            crop_dir=root / "crops",
            dataset_root=dataset_root,
            show_progress=False,
        )
        rows = _read_jsonl(output_path)

    assert summary.total == 1
    assert summary.succeeded == 1
    assert rows[0]["image_path"] == str(image_path.resolve())
    assert rows[0]["error"] is None
