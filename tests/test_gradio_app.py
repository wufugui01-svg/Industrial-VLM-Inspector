"""Tests for the Gradio demo's model-independent inspection handler."""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from app.gradio_app import inspect_uploaded_image


def test_mock_uploaded_image_returns_all_outputs_without_gpu() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        image_path = Path(temporary_directory) / "sample.png"
        Image.new("RGB", (80, 60), color=(120, 140, 160)).save(image_path)
        report_path = Path(temporary_directory) / "report.png"

        outputs = inspect_uploaded_image(
            str(image_path),
            model_path="",
            backend="mock",
            prompt_type="strict_json",
            report_output_path=str(report_path),
        )

        assert report_path.is_file()
        with Image.open(report_path) as report:
            assert report.width > 80
            assert report.height >= 60

    assert len(outputs) == 9
    (
        anomaly,
        defect_type,
        location,
        severity,
        confidence,
        reason,
        raw_json,
        latency,
        returned_report_path,
    ) = outputs
    assert anomaly == "Yes"
    assert defect_type == "unknown"
    assert location == "unknown"
    assert severity == "medium"
    assert confidence == 0.5
    assert reason == "mock result for pipeline testing"
    assert '"is_anomaly": true' in raw_json
    assert latency >= 0.0
    assert returned_report_path == str(report_path.resolve())


def test_missing_upload_is_rejected() -> None:
    try:
        inspect_uploaded_image(
            None,
            model_path="",
            backend="mock",
            prompt_type="basic",
        )
    except ValueError as exc:
        assert "upload an image" in str(exc)
    else:
        raise AssertionError("A missing upload must be rejected")


def test_reference_mode_without_reference_warns_but_runs() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        image_path = Path(temporary_directory) / "sample.png"
        Image.new("RGB", (80, 60), color=(120, 140, 160)).save(image_path)

        outputs = inspect_uploaded_image(
            str(image_path),
            model_path="",
            backend="mock",
            prompt_type="reference_strict",
            detection_mode="reference",
        )

    assert outputs[0] == "Yes"
    assert "Reference image is missing" in outputs[6]


def test_reference_mode_with_reference_runs() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        test_image = Path(temporary_directory) / "test.png"
        reference_image = Path(temporary_directory) / "reference.png"
        Image.new("RGB", (80, 60), color=(120, 140, 160)).save(test_image)
        Image.new("RGB", (80, 60), color=(120, 140, 160)).save(reference_image)

        outputs = inspect_uploaded_image(
            str(test_image),
            model_path="",
            backend="mock",
            prompt_type="reference_strict",
            reference_image_path=str(reference_image),
            detection_mode="reference",
        )

    assert outputs[0] == "Yes"
    assert '"is_anomaly": true' in outputs[6]


def test_global_local_mode_returns_composite_json() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        image_path = Path(temporary_directory) / "sample.png"
        Image.new("RGB", (90, 90), color=(120, 140, 160)).save(image_path)

        outputs = inspect_uploaded_image(
            str(image_path),
            model_path="",
            backend="mock",
            prompt_type="strict_json",
            detection_mode="global_local",
            grid="2x2",
        )

    assert outputs[0] == "Yes"
    assert "global-local aggregation" in outputs[5]
    assert '"final_prediction"' in outputs[6]
    assert '"crop_predictions"' in outputs[6]
