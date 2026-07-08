"""Tests for infrastructure benchmark runner."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from scripts.run_infra_benchmark import (
    CSV_FIELDS,
    _parse_max_new_tokens_list,
    run_infra_benchmark,
)


def test_parse_max_new_tokens_list() -> None:
    assert _parse_max_new_tokens_list("64, 128,256") == [64, 128, 256]


def test_mock_infra_benchmark_writes_csv() -> None:
    samples = [
        {
            "sample_id": "sample-1",
            "image_path": "/tmp/missing-one.png",
            "answer": "normal",
            "task_type": "Anomaly Detection",
            "object_category": "bottle",
        },
        {
            "sample_id": "sample-2",
            "image_path": "/tmp/missing-two.png",
            "answer": "anomaly",
            "task_type": "Anomaly Detection",
            "object_category": "bottle",
        },
    ]

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        index_path = root / "index.jsonl"
        output_path = root / "infra.csv"
        index_path.write_text(
            "".join(json.dumps(sample) + "\n" for sample in samples),
            encoding="utf-8",
        )

        rows = run_infra_benchmark(
            index_path=index_path,
            output_path=output_path,
            backend="mock",
            limit=2,
            max_new_tokens_list=[64, 128],
            prompt_type="strict_json",
        )

        assert len(rows) == 2
        assert output_path.is_file()
        with output_path.open("r", encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))

    assert list(csv_rows[0].keys()) == list(CSV_FIELDS)
    assert [row["max_new_tokens"] for row in csv_rows] == ["64", "128"]
    assert all(row["backend"] == "mock" for row in csv_rows)
    assert all(row["total_samples"] == "2" for row in csv_rows)
    assert all(float(row["avg_latency_sec"]) >= 0.0 for row in csv_rows)
    assert all(float(row["throughput_samples_per_sec"]) >= 0.0 for row in csv_rows)
