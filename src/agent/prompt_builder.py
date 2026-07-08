"""Prompt construction for industrial anomaly inspection."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

DEFECT_TYPES = (
    "scratch",
    "crack",
    "contamination",
    "missing_part",
    "deformation",
    "color_abnormality",
    "texture_abnormality",
    "unknown",
    "none",
)
SEVERITY_LEVELS = ("none", "low", "medium", "high", "unknown")
PROMPT_TYPES = ("basic", "industrial", "strict_json", "reference_strict")


def _sample_context(sample: dict[str, Any]) -> str:
    context = {
        "question": sample.get("question", ""),
        "options": sample.get("options", []),
        "task_type": sample.get("task_type", "unknown"),
        "object_category": sample.get("object_category", "unknown"),
    }
    return json.dumps(context, ensure_ascii=False)


def _json_contract() -> str:
    return (
        "{\n"
        '  "is_anomaly": true or false,\n'
        '  "defect_type": "unknown",\n'
        '  "defect_location": "brief location description",\n'
        '  "severity": "unknown",\n'
        '  "reason": "brief visual evidence",\n'
        '  "confidence": 0.0\n'
        "}"
    )


def _reference_json_contract() -> str:
    return (
        "{\n"
        '  "is_anomaly": true or false,\n'
        '  "defect_type": "scratch/crack/contamination/missing_part/'
        'deformation/color_abnormality/texture_abnormality/unknown/none",\n'
        '  "defect_location": "brief location description",\n'
        '  "severity": "none/low/medium/high/unknown",\n'
        '  "reason": "brief visual evidence based on comparison",\n'
        '  "confidence": 0.0\n'
        "}"
    )


def _output_rules() -> str:
    return (
        "Output rules:\n"
        "- Output only one valid JSON object.\n"
        "- Do not output Markdown or code fences.\n"
        "- Do not output an explanatory paragraph before or after the JSON.\n"
        f"- defect_type must be exactly one of: {', '.join(DEFECT_TYPES)}.\n"
        f"- severity must be exactly one of: {', '.join(SEVERITY_LEVELS)}.\n"
        "- confidence must be a number from 0.0 to 1.0.\n"
        "- If the image is ambiguous, lower confidence and use defect_type "
        "\"unknown\". Do not invent visual evidence."
    )


def basic_anomaly_prompt(sample: dict[str, Any]) -> str:
    """Ask for a concise anomaly decision with the shared output contract."""

    return (
        "You are an industrial anomaly detection assistant.\n"
        "Inspect the supplied image and decide whether it contains a visible "
        "defect or abnormality.\n"
        f"Task context: {_sample_context(sample)}\n\n"
        "Return this exact JSON structure:\n"
        f"{_json_contract()}\n\n"
        f"{_output_rules()}"
    )


def industrial_inspection_prompt(sample: dict[str, Any]) -> str:
    """Ask for evidence-focused inspection across common industrial defects."""

    return (
        "You are an industrial visual quality inspector.\n"
        "Examine the complete object and its local surfaces. Check for scratches, "
        "cracks, contamination, missing parts, deformation, abnormal color, and "
        "abnormal texture. Distinguish real defects from normal structure, "
        "lighting, reflections, shadows, and background artifacts.\n"
        "Base the decision only on visible evidence. Describe the defect location "
        "briefly and assign severity according to likely visual extent. If no "
        "defect is visible, use is_anomaly=false, defect_type=\"none\", "
        "severity=\"none\", and a conservative confidence.\n"
        f"Task context: {_sample_context(sample)}\n\n"
        "Required JSON structure:\n"
        f"{_json_contract()}\n\n"
        f"{_output_rules()}"
    )


def strict_json_prompt(sample: dict[str, Any]) -> str:
    """Prioritize machine-parseable output while retaining inspection context."""

    return (
        "Perform industrial defect inspection on the supplied image.\n"
        f"Task context: {_sample_context(sample)}\n\n"
        "Your entire response must be a single JSON object matching this schema:\n"
        f"{_reference_json_contract()}\n\n"
        "Every key is required. Use JSON booleans without quotes and a numeric "
        "confidence without quotes. Do not add extra keys.\n"
        f"{_output_rules()}"
    )


def reference_strict_prompt(sample: dict[str, Any]) -> str:
    """Ask for strict JSON inspection by comparing reference and test images."""

    if not sample.get("reference_image_path"):
        return strict_json_prompt(sample)

    return (
        "Perform reference-based industrial defect inspection.\n"
        "Image 1 is the normal reference image. Image 2 is the test image to "
        "inspect.\n"
        "Compare the two images carefully and decide whether the test image has "
        "local visual differences from the normal reference image.\n"
        "Focus on scratches, cracks, contamination, missing parts, deformation, "
        "color abnormality, and texture abnormality. Distinguish true defects "
        "from lighting, reflections, shadows, viewpoint changes, and background "
        "artifacts.\n"
        "The reason field must briefly describe visual evidence based on the "
        "comparison between the normal reference image and the test image.\n"
        f"Task context: {_sample_context(sample)}\n\n"
        "Your entire response must be a single JSON object matching this schema:\n"
        f"{_json_contract()}\n\n"
        "Every key is required. Use JSON booleans without quotes and a numeric "
        "confidence without quotes. Do not add extra keys.\n"
        f"{_output_rules()}"
    )


PROMPT_BUILDERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "basic": basic_anomaly_prompt,
    "industrial": industrial_inspection_prompt,
    "strict_json": strict_json_prompt,
    "reference_strict": reference_strict_prompt,
}


def build_prompt(sample: dict[str, Any], prompt_type: str = "basic") -> str:
    """Build one of the supported inspection prompts."""

    try:
        builder = PROMPT_BUILDERS[prompt_type]
    except KeyError as exc:
        choices = ", ".join(PROMPT_TYPES)
        raise ValueError(
            f"Unsupported prompt_type '{prompt_type}'. Choose from: {choices}"
        ) from exc
    return builder(sample)


def build_basic_prompt(sample: dict[str, Any]) -> str:
    """Backward-compatible alias for the original basic prompt function."""

    return basic_anomaly_prompt(sample)
