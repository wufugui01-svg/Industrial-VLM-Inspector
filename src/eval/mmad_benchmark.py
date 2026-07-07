"""Parsing and metrics for MMAD multiple-choice predictions."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Iterable

ANSWER_OBJECT = re.compile(
    r'"answer"\s*:\s*"([A-Za-z0-9]+)"',
    flags=re.IGNORECASE,
)
STANDALONE_LABEL = re.compile(r"^\s*([A-Za-z0-9]+)[\s.)]*$", re.IGNORECASE)


def option_labels(options: Any) -> set[str]:
    """Extract labels such as A/B/C/D from normalized MMAD option strings."""

    if not isinstance(options, list):
        return set()
    labels: set[str] = set()
    for option in options:
        match = re.match(r"^\s*([A-Za-z0-9]+)\s*[:.)-]", str(option))
        if match:
            labels.add(match.group(1).upper())
    return labels


def parse_mmad_answer(text: str, options: Any) -> tuple[str | None, str]:
    """Parse a model answer and return ``(label, parse_status)``."""

    allowed = option_labels(options)
    raw = str(text).strip()
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            answer = str(payload.get("answer") or "").strip().upper()
            if answer and (not allowed or answer in allowed):
                return answer, "success"
    except (json.JSONDecodeError, TypeError):
        pass

    match = ANSWER_OBJECT.search(raw)
    if match:
        answer = match.group(1).upper()
        if not allowed or answer in allowed:
            return answer, "repaired"

    match = STANDALONE_LABEL.match(raw)
    if match:
        answer = match.group(1).upper()
        if not allowed or answer in allowed:
            return answer, "repaired"
    return None, "failed"


def calculate_mmad_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Calculate overall and grouped exact-match MMAD option accuracy."""

    total = 0
    labeled = 0
    valid = 0
    correct = 0
    errors = 0
    task_total: Counter[str] = Counter()
    task_correct: Counter[str] = Counter()
    category_total: Counter[str] = Counter()
    category_correct: Counter[str] = Counter()

    for row in rows:
        total += 1
        task = str(row.get("task_type") or "unknown")
        category = str(row.get("object_category") or "unknown")
        task_total[task] += 1
        category_total[category] += 1
        ground_truth = str(row.get("ground_truth_answer") or "").strip().upper()
        if ground_truth:
            labeled += 1
        if row.get("error"):
            errors += 1
            continue
        prediction = str(row.get("prediction_answer") or "").strip().upper()
        if not prediction or not ground_truth or row.get("parse_status") == "failed":
            continue
        valid += 1
        if prediction == ground_truth:
            correct += 1
            task_correct[task] += 1
            category_correct[category] += 1

    def grouped_accuracy(
        totals: Counter[str], correct_counts: Counter[str]
    ) -> dict[str, float]:
        return {
            key: correct_counts[key] / count
            for key, count in sorted(totals.items())
            if count
        }

    return {
        "total_samples": total,
        "valid_prediction_count": valid,
        "invalid_prediction_count": total - valid,
        "error_count": errors,
        "accuracy": correct / labeled if labeled else None,
        "valid_prediction_accuracy": correct / valid if valid else None,
        "coverage": valid / total if total else 0.0,
        "labeled_sample_count": labeled,
        "correct_count": correct,
        "task_type_accuracy": grouped_accuracy(task_total, task_correct),
        "object_category_accuracy": grouped_accuracy(
            category_total, category_correct
        ),
    }
