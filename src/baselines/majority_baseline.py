"""Majority-class normal/abnormal baseline."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.baselines.random_baseline import (
    build_baseline_prediction,
    build_prediction_record,
)
from src.eval.metrics import map_binary_ground_truth


def infer_majority_class(samples: list[dict[str, Any]]) -> bool:
    """Infer majority anomaly class from labels, defaulting to normal."""

    counts: Counter[bool] = Counter()
    for sample in samples:
        label = map_binary_ground_truth(
            sample.get("ground_truth_answer", sample.get("answer"))
        )
        if label is not None:
            counts[label] += 1

    if not counts:
        return False
    if counts[True] > counts[False]:
        return True
    return False


def predict_majority(sample: dict[str, Any], majority_is_anomaly: bool) -> dict[str, Any]:
    """Predict the inferred majority class."""

    sample_id = sample.get("sample_id")
    return build_baseline_prediction(
        is_anomaly=majority_is_anomaly,
        confidence=0.5,
        sample_id=str(sample_id) if sample_id is not None else None,
        reason="majority baseline prediction",
    )


def run_majority_baseline(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return prediction rows for the majority baseline."""

    majority_is_anomaly = infer_majority_class(samples)
    return [
        build_prediction_record(
            sample,
            method="majority",
            prediction=predict_majority(sample, majority_is_anomaly),
        )
        for sample in samples
    ]
