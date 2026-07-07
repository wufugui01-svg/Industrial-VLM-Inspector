"""Parse and conservatively repair model output into an inspection result."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

from pydantic import ValidationError

from src.agent.schema import InspectionResult, default_failure_result

JSON_CODE_BLOCK = re.compile(
    r"```json\s*(.*?)\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)
TRAILING_COMMA = re.compile(r",\s*}")

FIELD_DEFAULTS: dict[str, Any] = {
    "is_anomaly": False,
    "defect_type": "unknown",
    "defect_location": "unknown",
    "severity": "unknown",
    "reason": "parse fallback",
    "confidence": 0.0,
}
ALLOWED_DEFECT_TYPES = {
    "scratch",
    "crack",
    "contamination",
    "missing_part",
    "deformation",
    "color_abnormality",
    "texture_abnormality",
    "unknown",
    "none",
}
ALLOWED_SEVERITIES = {"none", "low", "medium", "high", "unknown"}


def _repair_truncated_object(text: str) -> str | None:
    """Repair only a missing final brace or a trailing comma."""

    object_start = text.find("{")
    if object_start < 0:
        return None

    fragment = text[object_start:].strip()
    if "```" in fragment:
        fragment = fragment.split("```", maxsplit=1)[0].rstrip()
    if not fragment.endswith("}"):
        fragment += "}"
    return TRAILING_COMMA.sub("}", fragment)


def _candidate_json(text: str) -> Iterator[tuple[str, bool]]:
    """Yield direct and increasingly repaired JSON candidates."""

    direct = text.strip()
    seen: set[str] = set()

    def emit(candidate: str | None, repaired: bool) -> Iterator[tuple[str, bool]]:
        if candidate is None:
            return
        normalized = candidate.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            yield normalized, repaired

    yield from emit(direct, False)

    code_block = JSON_CODE_BLOCK.search(text)
    if code_block:
        yield from emit(code_block.group(1), True)

    decoder = json.JSONDecoder()
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            _, consumed = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        yield from emit(text[position : position + consumed], True)

    yield from emit(_repair_truncated_object(text), True)


def parse_inspection_result(
    text: str,
    sample_id: str | None = None,
) -> InspectionResult:
    """Parse direct JSON, extracted JSON, or a conservatively repaired object."""

    for candidate, candidate_was_repaired in _candidate_json(text):
        try:
            payload = json.loads(candidate)
            if not isinstance(payload, dict):
                continue

            missing_fields = [
                field for field in FIELD_DEFAULTS if field not in payload
            ]
            normalized = {**FIELD_DEFAULTS, **payload}
            normalized_fields = False
            if normalized.get("defect_type") not in ALLOWED_DEFECT_TYPES:
                normalized["defect_type"] = "unknown"
                normalized_fields = True
            if normalized.get("severity") not in ALLOWED_SEVERITIES:
                normalized["severity"] = "unknown"
                normalized_fields = True
            normalized["raw_model_answer"] = text
            normalized["sample_id"] = sample_id
            normalized["parse_status"] = (
                "repaired"
                if candidate_was_repaired or missing_fields or normalized_fields
                else "success"
            )
            return InspectionResult(**normalized)
        except (json.JSONDecodeError, TypeError, ValidationError):
            continue

    return default_failure_result(text, sample_id)
