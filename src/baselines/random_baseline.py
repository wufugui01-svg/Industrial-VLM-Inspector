"""Random normal/abnormal baseline."""

from __future__ import annotations

import random
from typing import Any


def build_baseline_prediction(
    *,
    is_anomaly: bool,
    confidence: float,
    sample_id: str | None,
    reason: str,
) -> dict[str, Any]:
    """Return an InspectionResult-compatible prediction dictionary."""

    return {
        "is_anomaly": is_anomaly,
        "defect_type": "unknown" if is_anomaly else "none",
        "defect_location": "unknown" if is_anomaly else "none",
        "severity": "unknown" if is_anomaly else "none",
        "reason": reason,
        "confidence": confidence,
        "raw_model_answer": None,
        "sample_id": sample_id,
        "parse_status": "success",
    }


def build_prediction_record(
    sample: dict[str, Any],
    *,
    method: str,
    prediction: dict[str, Any],
) -> dict[str, Any]:
    """Build a prediction JSONL row compatible with existing evaluators."""

    return {
        "sample_id": sample.get("sample_id"),
        "image_path": sample.get("image_path", ""),
        "prediction": prediction,
        "ground_truth_answer": sample.get("ground_truth_answer", sample.get("answer", "")),
        "task_type": sample.get("task_type", "unknown"),
        "object_category": sample.get("object_category", "unknown"),
        "method": method,
    }


def predict_random(sample: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Predict normal/abnormal uniformly at random with confidence 0.5."""

    sample_id = sample.get("sample_id")
    return build_baseline_prediction(
        is_anomaly=bool(rng.getrandbits(1)),
        confidence=0.5,
        sample_id=str(sample_id) if sample_id is not None else None,
        reason="random baseline prediction",
    )


def run_random_baseline(
    samples: list[dict[str, Any]],
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Return prediction rows for the random baseline."""

    rng = random.Random(seed)
    return [
        build_prediction_record(
            sample,
            method="random",
            prediction=predict_random(sample, rng),
        )
        for sample in samples
    ]
