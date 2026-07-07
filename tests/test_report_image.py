"""Tests for visual inspection report generation."""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from src.agent.schema import InspectionResult
from src.visualization.report_image import create_report_image


def test_create_report_image_includes_original_and_summary_panel() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        source = root / "source.png"
        output = root / "nested" / "report.png"
        Image.new("RGB", (160, 100), color=(90, 120, 150)).save(source)
        result = InspectionResult(
            is_anomaly=True,
            defect_type="scratch",
            defect_location="upper-left",
            severity="medium",
            reason=(
                "A long thin surface discontinuity is visible across the object "
                "and this deliberately long explanation should wrap."
            ),
            confidence=0.87,
        )

        returned = create_report_image(str(source), result, str(output))

        assert returned == str(output.resolve())
        assert output.is_file()
        with Image.open(output) as report:
            assert report.format == "PNG"
            assert report.width == 160 + 480
            assert report.height >= 100


def test_report_does_not_draw_or_require_bbox() -> None:
    fields = (
        InspectionResult.model_fields
        if hasattr(InspectionResult, "model_fields")
        else InspectionResult.__fields__
    )
    assert "bbox" not in fields
