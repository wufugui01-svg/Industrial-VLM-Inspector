"""Basic metrics for batch industrial inspection predictions."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from pydantic import ValidationError

from src.agent.schema import InspectionResult

NORMAL_LABELS = {
    "0",
    "false",
    "good",
    "no",
    "no defect",
    "non-defective",
    "normal",
}
ANOMALY_LABELS = {
    "1",
    "abnormal",
    "anomalous",
    "anomaly",
    "defect",
    "defective",
    "true",
    "yes",
}
BINARY_TASK_TYPES = {
    "anomaly classification",
    "anomaly detection",
    "binary classification",
}


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _percentile(values: list[float], quantile: float) -> float | None:
    """Calculate a linearly interpolated percentile without NumPy."""

    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + (
        ordered[upper_index] - ordered[lower_index]
    ) * fraction


def map_binary_ground_truth(value: Any) -> bool | None:
    """Map only explicit normal/anomaly labels; ambiguous choices are skipped."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if not isinstance(value, str):
        return None

    normalized = value.strip().casefold()
    if normalized in NORMAL_LABELS:
        return False
    if normalized in ANOMALY_LABELS:
        return True
    return None


def _validated_prediction(value: Any) -> InspectionResult | None:
    if not isinstance(value, dict):
        return None
    try:
        return InspectionResult(**value)
    except (ValidationError, TypeError):
        return None


def calculate_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Calculate basic integrity, distribution, and optional binary metrics."""

    total_samples = 0
    success_count = 0
    error_count = 0
    pipeline_error_count = 0
    json_valid_count = 0
    parse_success_count = 0
    parse_repaired_count = 0
    parse_failed_count = 0
    binary_evaluated_count = 0
    binary_correct_count = 0
    skipped_binary_eval_count = 0
    task_types: Counter[str] = Counter()
    object_categories: Counter[str] = Counter()
    latencies: list[float] = []
    gpu_memory_allocated: list[float] = []

    for row in rows:
        total_samples += 1
        has_pipeline_error = bool(row.get("error"))
        if has_pipeline_error:
            pipeline_error_count += 1

        task_type = str(row.get("task_type") or "unknown")
        object_category = str(row.get("object_category") or "unknown")
        task_types[task_type] += 1
        object_categories[object_category] += 1

        latency = _numeric(row.get("latency_sec"))
        if latency is not None:
            latencies.append(latency)
        allocated_memory = _numeric(
            row.get("gpu_peak_memory_allocated_mb")
        )
        if allocated_memory is None:
            allocated_memory = _numeric(row.get("gpu_memory_allocated_mb"))
        if allocated_memory is not None:
            gpu_memory_allocated.append(allocated_memory)

        prediction = _validated_prediction(row.get("prediction"))
        if prediction is not None:
            json_valid_count += 1
            if prediction.parse_status == "success":
                parse_success_count += 1
            elif prediction.parse_status == "repaired":
                parse_repaired_count += 1
            else:
                parse_failed_count += 1

        inference_succeeded = (
            not has_pipeline_error
            and prediction is not None
            and prediction.parse_status != "failed"
        )
        if inference_succeeded:
            success_count += 1
        else:
            error_count += 1

        ground_truth = map_binary_ground_truth(row.get("ground_truth_answer"))
        is_binary_task = task_type.strip().casefold() in BINARY_TASK_TYPES
        if (
            not inference_succeeded
            or ground_truth is None
            or not is_binary_task
        ):
            skipped_binary_eval_count += 1
            continue

        binary_evaluated_count += 1
        if prediction.is_anomaly == ground_truth:
            binary_correct_count += 1

    json_valid_rate = (
        json_valid_count / total_samples if total_samples else 0.0
    )
    parse_valid_count = parse_success_count + parse_repaired_count
    parse_success_rate = (
        parse_valid_count / total_samples if total_samples else 0.0
    )
    repair_rate = (
        parse_repaired_count / total_samples if total_samples else 0.0
    )
    binary_accuracy = (
        binary_correct_count / binary_evaluated_count
        if binary_evaluated_count
        else None
    )
    avg_latency_sec = (
        sum(latencies) / len(latencies) if latencies else None
    )

    return {
        "total_samples": total_samples,
        "success_count": success_count,
        "error_count": error_count,
        "pipeline_error_count": pipeline_error_count,
        "json_valid_count": json_valid_count,
        "json_valid_rate": json_valid_rate,
        "parse_success_count": parse_success_count,
        "parse_repaired_count": parse_repaired_count,
        "parse_failed_count": parse_failed_count,
        "parse_success_rate": parse_success_rate,
        "repair_rate": repair_rate,
        "task_type_distribution": dict(sorted(task_types.items())),
        "object_category_distribution": dict(sorted(object_categories.items())),
        "binary_accuracy": binary_accuracy,
        "binary_evaluated_count": binary_evaluated_count,
        "binary_correct_count": binary_correct_count,
        "skipped_binary_eval_count": skipped_binary_eval_count,
        "avg_latency_sec": avg_latency_sec,
        "p50_latency_sec": _percentile(latencies, 0.50),
        "p95_latency_sec": _percentile(latencies, 0.95),
        "max_gpu_memory_allocated_mb": (
            max(gpu_memory_allocated) if gpu_memory_allocated else None
        ),
    }
