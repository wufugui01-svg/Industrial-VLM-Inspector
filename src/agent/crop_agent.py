"""Image grid-cropping utilities for global-to-local inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

_SUPPORTED_GRIDS = {
    "2x2": 2,
    "3x3": 3,
}

_ROW_NAMES = {
    2: ("top", "bottom"),
    3: ("top", "center", "bottom"),
}

_COL_NAMES = {
    2: ("left", "right"),
    3: ("left", "center", "right"),
}


def _region_name(row: int, col: int, size: int) -> str:
    row_name = _ROW_NAMES[size][row]
    col_name = _COL_NAMES[size][col]
    if row_name == "center" and col_name == "center":
        return "center"
    return f"{row_name}_{col_name}"


def make_grid_crops(
    image_path: str,
    grid: str,
    output_dir: str,
) -> list[dict[str, Any]]:
    """Crop an image into a 2x2 or 3x3 grid and save crop files.

    Returns one metadata dictionary per crop:
    ``crop_path``, ``region_name``, and ``box`` as ``[x1, y1, x2, y2]``.
    """

    if grid not in _SUPPORTED_GRIDS:
        raise ValueError("grid must be one of: 2x2, 3x3")

    source_path = Path(image_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {source_path}")

    destination_dir = Path(output_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    grid_size = _SUPPORTED_GRIDS[grid]
    crops: list[dict[str, Any]] = []

    with Image.open(source_path) as opened:
        image = opened.convert("RGB")
        width, height = image.size

        for row in range(grid_size):
            y1 = height * row // grid_size
            y2 = height * (row + 1) // grid_size
            for col in range(grid_size):
                x1 = width * col // grid_size
                x2 = width * (col + 1) // grid_size
                region_name = _region_name(row, col, grid_size)
                crop_filename = f"{source_path.stem}_{grid}_{region_name}.png"
                crop_path = destination_dir / crop_filename
                image.crop((x1, y1, x2, y2)).save(crop_path)
                crops.append(
                    {
                        "crop_path": str(crop_path),
                        "region_name": region_name,
                        "box": [x1, y1, x2, y2],
                    }
                )

    return crops
