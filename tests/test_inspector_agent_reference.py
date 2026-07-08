"""Reference-image behavior for InspectorAgent."""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.agent.inspector_agent import InspectorAgent
from src.models.base_vlm import BaseVLM


class RecordingVLM(BaseVLM):
    def __init__(self) -> None:
        self.images: list[str] | None = None
        self.prompt = ""

    def generate(self, images: list[str], prompt: str) -> str:
        self.images = images
        self.prompt = prompt
        return (
            '{"is_anomaly": false, "defect_type": "none", '
            '"defect_location": "none", "severity": "none", '
            '"reason": "No visible defect.", "confidence": 0.9}'
        )


def test_agent_runs_single_image_without_reference() -> None:
    vlm = RecordingVLM()
    sample = {
        "sample_id": "single",
        "image_path": "/tmp/test-image.png",
        "object_category": "bottle",
    }

    result = InspectorAgent(vlm, prompt_type="reference_strict").inspect(sample)

    assert vlm.images == ["/tmp/test-image.png"]
    assert "normal reference image" not in vlm.prompt
    assert result.sample_id == "single"
    assert result.raw_model_answer is not None


def test_agent_runs_two_images_when_reference_file_exists() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        reference_path = root / "reference.png"
        test_path = root / "test.png"
        reference_path.write_bytes(b"not-a-real-image-but-existing")
        test_path.write_bytes(b"not-a-real-image-but-existing")

        vlm = RecordingVLM()
        sample = {
            "sample_id": "with-reference",
            "image_path": str(test_path),
            "reference_image_path": str(reference_path),
            "object_category": "bottle",
        }

        result = InspectorAgent(vlm, prompt_type="reference_strict").inspect(sample)

    assert vlm.images == [str(reference_path), str(test_path)]
    assert None not in vlm.images
    assert "normal reference image" in vlm.prompt
    assert "test image" in vlm.prompt
    assert result.sample_id == "with-reference"
    assert result.raw_model_answer is not None


def test_missing_reference_path_does_not_crash_or_enter_images() -> None:
    vlm = RecordingVLM()
    sample = {
        "sample_id": "missing-reference",
        "image_path": "/tmp/test-image.png",
        "reference_image_path": "/tmp/does-not-exist-reference.png",
        "object_category": "bottle",
    }

    result = InspectorAgent(vlm, prompt_type="reference_strict").inspect(sample)

    assert vlm.images == ["/tmp/test-image.png"]
    assert None not in vlm.images
    assert "normal reference image" not in vlm.prompt
    assert result.sample_id == "missing-reference"
    assert result.raw_model_answer is not None
