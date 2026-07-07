"""Dependency-free checks for Qwen backend path validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.models.qwen3vl_transformers import Qwen3VLTransformers


def test_image_validation_filters_none_and_supports_multiple_images() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        first = root / "first.png"
        second = root / "second.jpg"
        first.write_bytes(b"first")
        second.write_bytes(b"second")

        validated = Qwen3VLTransformers._validated_images(
            [None, "", str(first), str(second)]
        )

    assert validated == [str(first.resolve()), str(second.resolve())]


def test_missing_image_is_rejected() -> None:
    missing = Path("definitely-missing-image.png").resolve()
    try:
        Qwen3VLTransformers._validated_images([str(missing)])
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("A missing image path must be rejected")


def test_missing_local_model_directory_is_rejected_before_imports() -> None:
    missing = Path("definitely-missing-qwen-model").resolve()
    try:
        Qwen3VLTransformers(str(missing))
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("A missing local model directory must be rejected")

