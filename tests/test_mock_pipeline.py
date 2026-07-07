"""Tests for the model-free inspection pipeline."""

from src.agent.inspector_agent import InspectorAgent
from src.agent.output_parser import parse_inspection_result
from src.agent.schema import InspectionResult
from src.models.base_vlm import BaseVLM
from src.models.mock_vlm import MockVLM

SAMPLE = {
    "sample_id": "mmad_00000001",
    "image_path": "/tmp/example.png",
    "question": "Is there any defect?",
    "options": ["A: Yes", "B: No"],
    "task_type": "Anomaly Detection",
    "object_category": "bottle",
}


class InvalidJSONVLM(BaseVLM):
    def generate(self, images: list[str], prompt: str) -> str:
        del images, prompt
        return "this is not valid JSON"


def test_mock_pipeline_returns_inspection_result() -> None:
    result = InspectorAgent(MockVLM()).inspect(SAMPLE)

    assert isinstance(result, InspectionResult)
    assert result.is_anomaly is True
    assert result.defect_type == "unknown"
    assert result.severity == "medium"
    assert result.confidence == 0.5
    assert result.sample_id == "mmad_00000001"
    assert result.raw_model_answer is not None
    assert result.parse_status == "success"


def test_invalid_json_enters_fallback() -> None:
    result = InspectorAgent(InvalidJSONVLM()).inspect(SAMPLE)

    assert isinstance(result, InspectionResult)
    assert result.is_anomaly is False
    assert result.severity == "unknown"
    assert result.confidence == 0.0
    assert result.reason == "parse fallback"
    assert result.raw_model_answer == "this is not valid JSON"
    assert result.sample_id == "mmad_00000001"
    assert result.parse_status == "failed"


def test_json_code_block_is_parsed() -> None:
    raw_answer = """```json
{
  "is_anomaly": false,
  "defect_type": "none",
  "defect_location": "none",
  "severity": "none",
  "reason": "No visible defect.",
  "confidence": 0.75
}
```"""

    result = parse_inspection_result(raw_answer, "sample-code-block")

    assert result.is_anomaly is False
    assert result.severity == "none"
    assert result.confidence == 0.75
    assert result.raw_model_answer == raw_answer
    assert result.sample_id == "sample-code-block"
    assert result.parse_status == "repaired"
