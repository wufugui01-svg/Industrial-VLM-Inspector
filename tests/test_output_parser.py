"""Tests for model-output JSON parsing and repair."""

from src.agent.output_parser import parse_inspection_result

COMPLETE_JSON = (
    '{"is_anomaly": true, "defect_type": "scratch", '
    '"defect_location": "upper-left", "severity": "low", '
    '"reason": "A thin surface line is visible.", "confidence": 0.8}'
)


def test_plain_json_is_successful() -> None:
    result = parse_inspection_result(COMPLETE_JSON, "plain")

    assert result.is_anomaly is True
    assert result.defect_type == "scratch"
    assert result.parse_status == "success"
    assert result.raw_model_answer == COMPLETE_JSON


def test_markdown_json_block_is_repaired() -> None:
    raw = f"```json\n{COMPLETE_JSON}\n```"
    result = parse_inspection_result(raw, "markdown")

    assert result.is_anomaly is True
    assert result.parse_status == "repaired"
    assert result.raw_model_answer == raw


def test_json_surrounded_by_explanation_is_repaired() -> None:
    raw = f"Inspection result follows:\n{COMPLETE_JSON}\nEnd of result."
    result = parse_inspection_result(raw, "prose")

    assert result.defect_type == "scratch"
    assert result.parse_status == "repaired"
    assert result.raw_model_answer == raw


def test_braces_inside_reason_do_not_truncate_json() -> None:
    raw = (
        'Result: {"is_anomaly": true, "defect_type": "scratch", '
        '"defect_location": "center", "severity": "low", '
        '"reason": "Mark resembles {thin line}.", "confidence": 0.6} done'
    )
    result = parse_inspection_result(raw, "braces")

    assert result.reason == "Mark resembles {thin line}."
    assert result.parse_status == "repaired"


def test_missing_fields_receive_defaults() -> None:
    raw = '{"is_anomaly": true, "defect_type": "crack"}'
    result = parse_inspection_result(raw, "missing-fields")

    assert result.is_anomaly is True
    assert result.defect_type == "crack"
    assert result.defect_location == "unknown"
    assert result.severity == "unknown"
    assert result.reason == "parse fallback"
    assert result.confidence == 0.0
    assert result.parse_status == "repaired"


def test_out_of_contract_enum_is_normalized() -> None:
    raw = (
        '{"is_anomaly": true, "defect_type": "scratch/crack", '
        '"defect_location": "rim", "severity": "critical", '
        '"reason": "A discontinuity is visible.", "confidence": 0.8}'
    )
    result = parse_inspection_result(raw, "enum-repair")

    assert result.defect_type == "unknown"
    assert result.severity == "unknown"
    assert result.parse_status == "repaired"


def test_truncated_json_with_missing_final_brace_is_repaired() -> None:
    raw = (
        '{"is_anomaly": true, "defect_type": "deformation", '
        '"defect_location": "center", "severity": "medium", '
        '"reason": "The object is visibly bent.", "confidence": 0.7'
    )
    result = parse_inspection_result(raw, "truncated")

    assert result.defect_type == "deformation"
    assert result.confidence == 0.7
    assert result.parse_status == "repaired"


def test_invalid_text_returns_failed_fallback() -> None:
    raw = "The model did not return JSON."
    result = parse_inspection_result(raw, "invalid")

    assert result.is_anomaly is False
    assert result.defect_type == "unknown"
    assert result.reason == "parse fallback"
    assert result.confidence == 0.0
    assert result.raw_model_answer == raw
    assert result.sample_id == "invalid"
    assert result.parse_status == "failed"
