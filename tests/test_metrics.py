"""Tests for basic prediction metrics."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts.evaluate_predictions import evaluate_predictions_file
from src.eval.metrics import calculate_metrics, map_binary_ground_truth


def _prediction(is_anomaly: bool) -> dict:
    return {
        "is_anomaly": is_anomaly,
        "defect_type": "unknown" if is_anomaly else "none",
        "defect_location": "unknown" if is_anomaly else "none",
        "severity": "medium" if is_anomaly else "none",
        "reason": "test prediction",
        "confidence": 0.5,
        "raw_model_answer": "{}",
        "sample_id": "sample",
        "parse_status": "success",
    }


def test_calculate_metrics_and_conservative_binary_mapping() -> None:
    rows = [
        {
            "prediction": _prediction(False),
            "ground_truth_answer": "normal",
            "task_type": "Anomaly Detection",
            "object_category": "bottle",
            "error": None,
            "latency_sec": 1.0,
            "gpu_memory_allocated_mb": 100.0,
        },
        {
            "prediction": _prediction(True),
            "ground_truth_answer": "abnormal",
            "task_type": "Anomaly Detection",
            "object_category": "bottle",
            "error": None,
            "latency_sec": 3.0,
            "gpu_memory_allocated_mb": 200.0,
        },
        {
            "prediction": _prediction(True),
            "ground_truth_answer": "A",
            "task_type": "Defect Classification",
            "object_category": "cable",
            "error": None,
            "latency_sec": 2.0,
            "gpu_memory_allocated_mb": 150.0,
        },
        {
            "prediction": None,
            "ground_truth_answer": "normal",
            "task_type": "unknown",
            "object_category": "unknown",
            "error": "model failed",
            "latency_sec": None,
            "gpu_memory_allocated_mb": None,
        },
    ]

    metrics = calculate_metrics(rows)

    assert metrics["total_samples"] == 4
    assert metrics["success_count"] == 3
    assert metrics["error_count"] == 1
    assert metrics["pipeline_error_count"] == 1
    assert metrics["json_valid_count"] == 3
    assert metrics["json_valid_rate"] == 0.75
    assert metrics["parse_success_count"] == 3
    assert metrics["parse_success_rate"] == 0.75
    assert metrics["task_type_distribution"]["Anomaly Detection"] == 2
    assert metrics["object_category_distribution"]["bottle"] == 2
    assert metrics["binary_accuracy"] == 1.0
    assert metrics["binary_evaluated_count"] == 2
    assert metrics["skipped_binary_eval_count"] == 2
    assert map_binary_ground_truth("A") is None
    assert metrics["avg_latency_sec"] == 2.0
    assert metrics["p50_latency_sec"] == 2.0
    assert metrics["p95_latency_sec"] == 2.9
    assert metrics["max_gpu_memory_allocated_mb"] == 200.0


def test_failed_parser_fallback_is_not_successful_or_scored() -> None:
    failed_prediction = _prediction(False)
    failed_prediction["parse_status"] = "failed"
    rows = [
        {
            "prediction": failed_prediction,
            "ground_truth_answer": "normal",
            "task_type": "Anomaly Detection",
            "object_category": "bottle",
            "error": None,
        }
    ]

    metrics = calculate_metrics(rows)

    assert metrics["json_valid_count"] == 1
    assert metrics["parse_failed_count"] == 1
    assert metrics["parse_success_rate"] == 0.0
    assert metrics["success_count"] == 0
    assert metrics["error_count"] == 1
    assert metrics["pipeline_error_count"] == 0
    assert metrics["binary_accuracy"] is None
    assert metrics["skipped_binary_eval_count"] == 1


def test_evaluate_predictions_file_writes_metrics_json() -> None:
    row = {
        "sample_id": "sample-1",
        "prediction": _prediction(True),
        "ground_truth_answer": "yes",
        "task_type": "Anomaly Detection",
        "object_category": "widget",
        "image_path": "/tmp/example.png",
        "error": None,
        "latency_sec": 0.25,
        "gpu_memory_allocated_mb": 64.0,
    }

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        predictions_path = root / "predictions.jsonl"
        output_path = root / "nested" / "metrics.json"
        predictions_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

        returned = evaluate_predictions_file(predictions_path, output_path)
        saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert returned == saved
    assert saved["total_samples"] == 1
    assert saved["json_valid_rate"] == 1.0
    assert saved["binary_accuracy"] == 1.0
    assert saved["skipped_binary_eval_count"] == 0
    assert saved["avg_latency_sec"] == 0.25
    assert saved["p50_latency_sec"] == 0.25
    assert saved["p95_latency_sec"] == 0.25
    assert saved["max_gpu_memory_allocated_mb"] == 64.0


def test_malformed_prediction_line_is_counted_as_error() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        predictions_path = root / "predictions.jsonl"
        output_path = root / "metrics.json"
        predictions_path.write_text("not-json\n", encoding="utf-8")

        metrics = evaluate_predictions_file(predictions_path, output_path)

    assert metrics["total_samples"] == 1
    assert metrics["success_count"] == 0
    assert metrics["error_count"] == 1
    assert metrics["json_valid_count"] == 0
    assert metrics["skipped_binary_eval_count"] == 1
    assert metrics["avg_latency_sec"] is None
    assert metrics["max_gpu_memory_allocated_mb"] is None
