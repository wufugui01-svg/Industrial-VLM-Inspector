"""Tests for industrial inspection prompt selection."""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.agent.inspector_agent import InspectorAgent
from src.agent.prompt_builder import (
    basic_anomaly_prompt,
    build_basic_prompt,
    build_prompt,
    industrial_inspection_prompt,
    reference_strict_prompt,
    strict_json_prompt,
)
from src.models.base_vlm import BaseVLM

SAMPLE = {
    "sample_id": "sample-1",
    "question": "Is there any defect?",
    "options": ["A: Yes", "B: No"],
    "task_type": "Anomaly Detection",
    "object_category": "bottle",
}


class PromptCapturingVLM(BaseVLM):
    def __init__(self) -> None:
        self.prompt = ""

    def generate(self, images: list[str], prompt: str) -> str:
        del images
        self.prompt = prompt
        return (
            '{"is_anomaly": false, "defect_type": "none", '
            '"defect_location": "none", "severity": "none", '
            '"reason": "No visible defect.", "confidence": 0.8}'
        )


def test_all_prompts_include_schema_and_output_rules() -> None:
    builders = [
        basic_anomaly_prompt,
        industrial_inspection_prompt,
        strict_json_prompt,
        reference_strict_prompt,
    ]

    for builder in builders:
        prompt = builder(SAMPLE)
        for field in (
            "is_anomaly",
            "defect_type",
            "defect_location",
            "severity",
            "reason",
            "confidence",
        ):
            assert f'"{field}"' in prompt
        assert "scratch, crack, contamination, missing_part, deformation" in prompt
        assert "none, low, medium, high, unknown" in prompt
        assert "Output only one valid JSON object" in prompt
        assert "Do not output Markdown" in prompt
        assert "Do not output an explanatory paragraph" in prompt
        assert 'defect_type "unknown"' in prompt


def test_prompt_variants_are_distinct_and_context_is_preserved() -> None:
    prompts = {
        build_prompt(SAMPLE, "basic"),
        build_prompt(SAMPLE, "industrial"),
        build_prompt(SAMPLE, "strict_json"),
        build_prompt(
            {**SAMPLE, "reference_image_path": "/tmp/ref.png"},
            "reference_strict",
        ),
    }

    assert len(prompts) == 4
    assert all("bottle" in prompt for prompt in prompts)
    assert build_basic_prompt(SAMPLE) == basic_anomaly_prompt(SAMPLE)


def test_invalid_prompt_type_is_rejected() -> None:
    try:
        build_prompt(SAMPLE, "unsupported")
    except ValueError as exc:
        assert "Unsupported prompt_type" in str(exc)
    else:
        raise AssertionError("Unsupported prompt types must be rejected")


def test_inspector_agent_uses_selected_prompt() -> None:
    vlm = PromptCapturingVLM()
    result = InspectorAgent(vlm, prompt_type="industrial").inspect(SAMPLE)

    assert "industrial visual quality inspector" in vlm.prompt
    assert result.is_anomaly is False
    assert result.defect_type == "none"


def test_reference_strict_prompt_contains_reference_and_test_image_semantics() -> None:
    sample = {
        **SAMPLE,
        "reference_image_path": "/tmp/normal-reference.png",
        "image_path": "/tmp/test-image.png",
    }

    prompt = build_prompt(sample, "reference_strict")

    assert "Image 1 is the normal reference image" in prompt
    assert "Image 2 is the test image" in prompt
    assert "Compare the two images" in prompt
    assert "local visual differences" in prompt
    assert "based on the comparison" in prompt
    assert "scratch" in prompt
    assert "crack" in prompt
    assert "contamination" in prompt
    assert "missing_part" in prompt
    assert "deformation" in prompt
    assert "color_abnormality" in prompt
    assert "texture_abnormality" in prompt
    assert "Output only one valid JSON object" in prompt
    assert "Do not output Markdown" in prompt
    assert "Do not output an explanatory paragraph" in prompt


def test_reference_strict_without_reference_falls_back_to_strict_json() -> None:
    assert build_prompt(SAMPLE, "reference_strict") == strict_json_prompt(SAMPLE)


def test_inspector_agent_accepts_reference_strict_prompt_type() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        reference_path = Path(temporary_directory) / "normal-reference.png"
        reference_path.write_bytes(b"existing reference image placeholder")
        vlm = PromptCapturingVLM()
        sample = {**SAMPLE, "reference_image_path": str(reference_path)}

        result = InspectorAgent(vlm, prompt_type="reference_strict").inspect(sample)

    assert "normal reference image" in vlm.prompt
    assert "test image" in vlm.prompt
    assert result.is_anomaly is False
