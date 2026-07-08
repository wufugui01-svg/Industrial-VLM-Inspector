"""Tests for global-to-local crop utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from src.agent.crop_agent import make_grid_crops


def _assert_valid_crop_metadata(crops: list[dict], width: int, height: int) -> None:
    assert crops
    for crop in crops:
        crop_path = Path(crop["crop_path"])
        assert crop_path.is_file()
        assert crop["region_name"]
        x1, y1, x2, y2 = crop["box"]
        assert 0 <= x1 < x2 <= width
        assert 0 <= y1 < y2 <= height
        with Image.open(crop_path) as opened:
            assert opened.width == x2 - x1
            assert opened.height == y2 - y1


def test_make_grid_crops_2x2_outputs_four_crops() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        image_path = root / "source.png"
        output_dir = root / "crops"
        Image.new("RGB", (100, 80), color=(120, 140, 160)).save(image_path)
        original_bytes = image_path.read_bytes()

        crops = make_grid_crops(str(image_path), "2x2", str(output_dir))

        assert len(crops) == 4
        assert [crop["region_name"] for crop in crops] == [
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right",
        ]
        _assert_valid_crop_metadata(crops, width=100, height=80)
        assert image_path.read_bytes() == original_bytes


def test_make_grid_crops_3x3_outputs_nine_crops() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        image_path = root / "source.jpg"
        output_dir = root / "crops"
        Image.new("RGB", (99, 90), color=(80, 100, 120)).save(image_path)

        crops = make_grid_crops(str(image_path), "3x3", str(output_dir))

        assert len(crops) == 9
        assert [crop["region_name"] for crop in crops] == [
            "top_left",
            "top_center",
            "top_right",
            "center_left",
            "center",
            "center_right",
            "bottom_left",
            "bottom_center",
            "bottom_right",
        ]
        _assert_valid_crop_metadata(crops, width=99, height=90)


def test_make_grid_crops_rejects_unsupported_grid() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        image_path = root / "source.png"
        Image.new("RGB", (40, 40)).save(image_path)

        with pytest.raises(ValueError, match="grid must be one of"):
            make_grid_crops(str(image_path), "4x4", str(root / "crops"))
