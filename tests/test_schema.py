"""Tests for the industrial inspection result schema."""

import pytest
from pydantic import ValidationError

from src.agent.schema import InspectionResult, default_failure_result


def test_valid_inspection_result() -> None:
    result = InspectionResult(
        is_anomaly=True,
        defect_type="scratch",
        defect_location="upper-left surface",
        severity="medium",
        reason="A thin discontinuity is visible on the surface.",
        confidence=0.82,
        raw_model_answer='{"is_anomaly": true}',
        sample_id="mmad_00000001",
    )

    assert result.is_anomaly is True
    assert result.severity == "medium"
    assert result.confidence == 0.82
    assert result.sample_id == "mmad_00000001"
    assert result.parse_status == "success"


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_confidence_out_of_range_is_rejected(confidence: float) -> None:
    with pytest.raises(ValidationError):
        InspectionResult(
            is_anomaly=False,
            defect_type="none",
            defect_location="none",
            severity="none",
            reason="No defect detected.",
            confidence=confidence,
        )


def test_invalid_severity_is_rejected() -> None:
    with pytest.raises(ValidationError):
        InspectionResult(
            is_anomaly=True,
            defect_type="crack",
            defect_location="center",
            severity="critical",
            reason="A crack is visible.",
            confidence=0.9,
        )


def test_invalid_defect_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        InspectionResult(
            is_anomaly=True,
            defect_type="dent",
            defect_location="center",
            severity="medium",
            reason="A dent is visible.",
            confidence=0.9,
        )


def test_default_failure_result() -> None:
    result = default_failure_result(
        raw_answer="not valid JSON",
        sample_id="sample-42",
    )

    assert result.is_anomaly is False
    assert result.defect_type == "unknown"
    assert result.defect_location == "unknown"
    assert result.severity == "unknown"
    assert result.reason == "parse fallback"
    assert result.confidence == 0.0
    assert result.raw_model_answer == "not valid JSON"
    assert result.sample_id == "sample-42"
    assert result.parse_status == "failed"
