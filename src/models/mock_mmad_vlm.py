"""Deterministic mock backend for the MMAD benchmark runner."""

from __future__ import annotations

from src.models.base_vlm import BaseVLM


class MockMMADVLM(BaseVLM):
    """Always choose option A so benchmark plumbing can be tested without a GPU."""

    def generate(self, images: list[str], prompt: str) -> str:
        del images, prompt
        return '{"answer": "A"}'
