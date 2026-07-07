"""Deterministic VLM backend for testing without a model or GPU."""

from __future__ import annotations

import json

from src.models.base_vlm import BaseVLM


class MockVLM(BaseVLM):
    """Return a fixed, schema-compatible inspection response."""

    def generate(self, images: list[str], prompt: str) -> str:
        del images, prompt
        return json.dumps(
            {
                "is_anomaly": True,
                "defect_type": "unknown",
                "defect_location": "unknown",
                "severity": "medium",
                "reason": "mock result for pipeline testing",
                "confidence": 0.5,
            }
        )

