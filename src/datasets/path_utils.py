"""Portable image-path resolution for generated dataset indexes."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_sample_image_path(
    sample: dict[str, Any],
    dataset_root: Path | None = None,
) -> str:
    """Resolve an indexed image, preferring a supplied root and relative path."""

    relative = str(sample.get("image_relative_path") or "").strip()
    if dataset_root is not None and relative:
        candidate = dataset_root.expanduser().resolve() / Path(relative)
        return str(candidate.resolve(strict=False))
    return str(sample.get("image_path") or "")


def with_resolved_image_path(
    sample: dict[str, Any],
    dataset_root: Path | None = None,
) -> dict[str, Any]:
    """Return a shallow sample copy with its runtime image path resolved."""

    resolved = dict(sample)
    resolved["image_path"] = resolve_sample_image_path(sample, dataset_root)
    return resolved
