"""Structured result schema for industrial visual inspection."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["none", "low", "medium", "high", "unknown"]
ParseStatus = Literal["success", "repaired", "failed"]
DefectType = Literal[
    "scratch",
    "crack",
    "contamination",
    "missing_part",
    "deformation",
    "color_abnormality",
    "texture_abnormality",
    "unknown",
    "none",
]


class InspectionResult(BaseModel):
    """Normalized output returned by an industrial inspection pipeline."""

    model_config = ConfigDict(extra="forbid")

    is_anomaly: bool
    defect_type: DefectType
    defect_location: str
    severity: Severity
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    raw_model_answer: str | None = None
    sample_id: str | None = None
    parse_status: ParseStatus = "success"



def default_failure_result(
    raw_answer: str,
    sample_id: str | None = None,
) -> InspectionResult:
    """Return a safe result when a model answer cannot be parsed."""

    return InspectionResult(
        is_anomaly=False,
        defect_type="unknown",
        defect_location="unknown",
        severity="unknown",
        reason="parse fallback",
        confidence=0.0,
        raw_model_answer=raw_answer,
        sample_id=sample_id,
        parse_status="failed",
    )
