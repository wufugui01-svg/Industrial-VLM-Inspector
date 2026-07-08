"""Metrics tests for multiple prediction output formats."""

from __future__ import annotations

from src.eval.metrics import calculate_metrics, map_binary_ground_truth


def _prediction(is_anomaly: bool, confidence: float = 0.8) -> dict:
    return {
        "is_anomaly": is_anomaly,
        "defect_type": "unknown" if is_anomaly else "none",
        "defect_location": "unknown" if is_anomaly else "none",
        "severity": "medium" if is_anomaly else "none",
        "reason": "test prediction",
        "confidence": confidence,
        "raw_model_answer": "{}",
        "sample_id": "sample",
        "parse_status": "success",
    }


def test_metrics_selects_final_prediction_first_then_prediction_then_global() -> None:
    rows = [
        {
            "final_prediction": _prediction(True, 0.9),
            "prediction": _prediction(False, 0.1),
            "global_prediction": _prediction(False, 0.2),
            "ground_truth_answer": "B",
            "task_type": "Anomaly Detection",
            "object_category": "bottle",
            "latency_sec": 1.0,
            "error": None,
        },
        {
            "prediction": _prediction(False, 0.7),
            "global_prediction": _prediction(True, 0.3),
            "ground_truth_answer": "A",
            "task_type": "Anomaly Detection",
            "object_category": "bottle",
            "latency_sec": 3.0,
            "error": None,
        },
        {
            "global_prediction": _prediction(True, 0.6),
            "ground_truth_answer": "正常",
            "task_type": "Defect Classification",
            "object_category": "cable",
            "latency_sec": 2.0,
            "error": None,
        },
        {
            "final_prediction": _prediction(False, 0.4),
            "ground_truth_answer": "异常",
            "task_type": "Anomaly Detection",
            "object_category": "capsule",
            "latency_sec": 4.0,
            "error": None,
        },
    ]

    metrics = calculate_metrics(rows)

    assert metrics["total_samples"] == 4
    assert metrics["success_count"] == 4
    assert metrics["error_count"] == 0
    assert metrics["json_valid_count"] == 4
    assert metrics["json_valid_rate"] == 1.0
    assert metrics["parse_success_count"] == 4
    assert metrics["parse_failed_count"] == 0
    assert metrics["avg_confidence"] == 0.65
    assert metrics["avg_latency_sec"] == 2.5
    assert metrics["p50_latency_sec"] == 2.5
    assert metrics["p95_latency_sec"] == 3.8499999999999996
    assert metrics["task_type_distribution"]["Anomaly Detection"] == 3
    assert metrics["object_category_distribution"]["bottle"] == 2
    assert metrics["binary_evaluated_count"] == 4
    assert metrics["skipped_binary_eval_count"] == 0
    assert metrics["binary_accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["false_positive_count"] == 1
    assert metrics["false_negative_count"] == 1


def test_metrics_writes_null_binary_metrics_when_no_labels_are_mappable() -> None:
    metrics = calculate_metrics(
        [
            {
                "prediction": _prediction(True, 0.5),
                "ground_truth_answer": "C",
                "task_type": "Defect Classification",
                "object_category": "bottle",
                "error": None,
            }
        ]
    )

    assert metrics["binary_accuracy"] is None
    assert metrics["precision"] is None
    assert metrics["recall"] is None
    assert metrics["f1"] is None
    assert metrics["false_positive_count"] is None
    assert metrics["false_negative_count"] is None
    assert metrics["skipped_binary_eval_count"] == 1


def test_binary_ground_truth_mapping_accepts_required_labels() -> None:
    assert map_binary_ground_truth("normal") is False
    assert map_binary_ground_truth("good") is False
    assert map_binary_ground_truth("0") is False
    assert map_binary_ground_truth("A") is False
    assert map_binary_ground_truth("正常") is False
    assert map_binary_ground_truth("abnormal") is True
    assert map_binary_ground_truth("defect") is True
    assert map_binary_ground_truth("anomaly") is True
    assert map_binary_ground_truth("1") is True
    assert map_binary_ground_truth("B") is True
    assert map_binary_ground_truth("异常") is True
