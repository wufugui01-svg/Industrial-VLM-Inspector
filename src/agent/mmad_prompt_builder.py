"""Prompt construction for the MMAD multiple-choice benchmark."""

from __future__ import annotations

import json
from typing import Any


def build_mmad_multiple_choice_prompt(sample: dict[str, Any]) -> str:
    """Build a benchmark prompt that returns one machine-readable option label."""

    question = str(sample.get("question") or "")
    options = sample.get("options")
    if not isinstance(options, list):
        options = []
    context = {
        "question": question,
        "options": options,
        "task_type": sample.get("task_type", "unknown"),
        "object_category": sample.get("object_category", "unknown"),
    }
    return (
        "You are answering one MMAD industrial visual-inspection "
        "multiple-choice question. Inspect the supplied image and use only the "
        "question and options below.\n"
        f"{json.dumps(context, ensure_ascii=False)}\n"
        'Return exactly one JSON object in this form: {"answer": "A"}\n'
        "Replace A with the single best option label. Do not output Markdown, "
        "reasoning, or any text outside the JSON object."
    )
