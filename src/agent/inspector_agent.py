"""Deterministic orchestration for a single industrial inspection sample."""

from __future__ import annotations

from typing import Any

from src.agent.output_parser import parse_inspection_result
from src.agent.prompt_builder import PROMPT_TYPES, build_prompt
from src.agent.schema import InspectionResult
from src.models.base_vlm import BaseVLM


class InspectorAgent:
    """Connect prompt construction, a VLM backend, and output parsing."""

    def __init__(self, vlm: BaseVLM, prompt_type: str = "basic") -> None:
        if prompt_type not in PROMPT_TYPES:
            choices = ", ".join(PROMPT_TYPES)
            raise ValueError(
                f"Unsupported prompt_type '{prompt_type}'. Choose from: {choices}"
            )
        self.vlm = vlm
        self.prompt_type = prompt_type

    def inspect(self, sample: dict[str, Any]) -> InspectionResult:
        """Inspect one indexed sample and return a normalized result."""

        prompt = build_prompt(sample, self.prompt_type)
        image_path = sample.get("image_path")
        images = [str(image_path)] if image_path else []
        raw_answer = self.vlm.generate(images=images, prompt=prompt)
        sample_id = sample.get("sample_id")
        normalized_sample_id = str(sample_id) if sample_id is not None else None
        return parse_inspection_result(raw_answer, normalized_sample_id)
