"""Reference image selection utilities for industrial inspection samples."""

from __future__ import annotations

import random
from typing import Any, Literal

ReferenceStrategy = Literal["first", "random", "similarity"]

_CATEGORY_KEYS = ("object_category", "category")
_SELF_ID_KEYS = ("sample_id", "image_path", "image_relative_path")
_IMAGE_KEYS = ("image_path",)

_NORMAL_VALUES = {
    "0",
    "false",
    "good",
    "healthy",
    "negative",
    "no",
    "none",
    "normal",
    "ok",
    "pass",
    "passed",
}

_NORMAL_PHRASES = (
    "defect free",
    "defect-free",
    "no anomaly",
    "no defect",
    "non defective",
    "non-defective",
    "not anomalous",
    "without defect",
)


def _normalized_text(value: Any) -> str:
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


def _first_non_empty(sample: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _reference_image_path(sample: dict[str, Any]) -> str | None:
    return _first_non_empty(sample, _IMAGE_KEYS)


def _same_sample(a: dict[str, Any], b: dict[str, Any]) -> bool:
    for key in _SELF_ID_KEYS:
        a_value = a.get(key)
        b_value = b.get(key)
        if a_value is not None and b_value is not None and str(a_value) == str(b_value):
            return True
    return a is b


def is_normal_sample(sample: dict[str, Any]) -> bool:
    """Return whether a sample label indicates a normal/good item.

    The MMAD-derived indexes and external anomaly datasets use different field
    names. This helper intentionally checks only label-like fields so arbitrary
    question text does not accidentally mark a defective sample as normal.
    """

    for key in ("answer", "label", "defect_type"):
        value = sample.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            return value is False
        text = _normalized_text(value)
        if text in _NORMAL_VALUES:
            return True
        if any(phrase in text for phrase in _NORMAL_PHRASES):
            return True
    return False


def same_category(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Return whether two samples share object/category metadata."""

    a_category = _first_non_empty(a, _CATEGORY_KEYS)
    b_category = _first_non_empty(b, _CATEGORY_KEYS)
    if not a_category or not b_category:
        return False
    return _normalized_text(a_category) == _normalized_text(b_category)


class ReferenceSelector:
    """Select a normal reference image for a current inspection sample."""

    def __init__(
        self,
        strategy: ReferenceStrategy = "first",
        *,
        seed: int | None = None,
    ) -> None:
        if strategy not in {"first", "random", "similarity"}:
            raise ValueError(
                "Unsupported reference strategy "
                f"'{strategy}'. Choose from: first, random, similarity"
            )
        self.strategy = strategy
        self._random = random.Random(seed)

    def select(
        self,
        sample: dict[str, Any],
        samples: list[dict[str, Any]],
    ) -> str | None:
        """Return a reference image path, or ``None`` when no candidate exists."""

        candidates = [
            candidate
            for candidate in samples
            if not _same_sample(sample, candidate)
            and is_normal_sample(candidate)
            and _reference_image_path(candidate)
        ]
        if not candidates:
            return None

        same_category_candidates = [
            candidate for candidate in candidates if same_category(sample, candidate)
        ]
        pool = same_category_candidates or candidates

        if self.strategy == "random":
            selected = self._random.choice(pool)
        else:
            # TODO: Replace this fallback with CLIP/image-feature nearest-neighbor
            # matching when the similarity strategy is implemented.
            selected = pool[0]

        return _reference_image_path(selected)
