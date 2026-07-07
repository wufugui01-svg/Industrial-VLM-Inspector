"""Abstract interface shared by vision-language model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseVLM(ABC):
    """Minimal interface required by the inspection agent."""

    @abstractmethod
    def generate(self, images: list[str], prompt: str) -> str:
        """Generate a text response for the supplied images and prompt."""

        raise NotImplementedError

