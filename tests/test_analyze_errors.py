"""Tests for prediction error analysis."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts.analyze_errors import analyze_errors_file


def _prediction(
    *,
    is_anomaly: bool,
    confidence: float,
    parse_status: str = "success",
) -> dict[str, object]:
    return {
        "is_anomaly": is_anomaly,
        "defect_type": "unknown",
        "defect_location": "unknown",
        "severity": "unknown",
        "reason": "test prediction",
        "confidence": confidence,
        "parse_status": parse_status,
    }


def test_analyze_errors_writes_reports_and_copies_images() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        image_path = root / "sample.png"
        image_path.write_bytes(b"fake image bytes")
        predictions_path = root / "predictions.jsonl"
        output_dir = root / "analysis"
        rows = [
            {
                "sample_id": "false-positive",
                "image_path": str(image_path),
                "ground_truth_answer": "normal",
                "prediction": _prediction(is_anomaly=True, confidence=0.9),
                "error": None,
            },
            {
                "sample_id": "false-negative-low-confidence",
                "image_path": str(image_path),
                "ground_truth_answer": "anomaly",
                "prediction": _prediction(is_anomaly=False, confidence=0.4),
                "error": None,
            },
            {
                "sample_id": "parse-failed",
                "image_path": str(image_path),
                "ground_truth_answer": "anomaly",
                "prediction": _prediction(
                    is_anomaly=False,
                    confidence=0.0,
                    parse_status="failed",
                ),
                "error": None,
            },
            {
                "sample_id": "pipeline-error",
                "image_path": str(image_path),
                "ground_truth_answer": "normal",
                "prediction": None,
                "error": "RuntimeError: boom",
            },
            {
                "sample_id": "correct",
                "image_path": str(image_path),
                "ground_truth_answer": "normal",
                "prediction": _prediction(is_anomaly=False, confidence=0.8),
                "error": None,
            },
        ]
        original_text = "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        )
        predictions_path.write_text(original_text, encoding="utf-8")

        summary = analyze_errors_file(
            predictions_path=predictions_path,
            output_dir=output_dir,
            max_cases=10,
        )

        assert predictions_path.read_text(encoding="utf-8") == original_text
        assert (output_dir / "error_cases.jsonl").is_file()
        assert (output_dir / "error_summary.json").is_file()
        assert (output_dir / "error_analysis.md").is_file()
        assert summary["total_samples"] == 5
        assert summary["matched_case_count"] == 4
        assert summary["exported_case_count"] == 4
        assert summary["case_type_counts"]["false_positive"] == 1
        assert summary["case_type_counts"]["false_negative"] == 1
        assert summary["case_type_counts"]["parse_failed"] == 1
        assert summary["case_type_counts"]["low_confidence"] == 2
        assert summary["case_type_counts"]["error_record"] == 1
        assert summary["copied_image_count"] == 4
        assert len(list((output_dir / "images").iterdir())) == 4

        exported_cases = [
            json.loads(line)
            for line in (output_dir / "error_cases.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert exported_cases[0]["case_types"] == ["false_positive"]
        assert "false_negative" in exported_cases[1]["case_types"]
        assert "low_confidence" in exported_cases[1]["case_types"]

        report = (output_dir / "error_analysis.md").read_text(encoding="utf-8")
        assert "Total samples: 5" in report
        assert "Auto Observation Template" in report


def test_analyze_errors_respects_max_cases() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        predictions_path = root / "predictions.jsonl"
        rows = [
            {
                "sample_id": f"case-{index}",
                "ground_truth_answer": "normal",
                "prediction": _prediction(is_anomaly=True, confidence=0.9),
                "error": None,
            }
            for index in range(3)
        ]
        predictions_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

        summary = analyze_errors_file(
            predictions_path=predictions_path,
            output_dir=root / "analysis",
            max_cases=1,
        )

        assert summary["matched_case_count"] == 3
        assert summary["exported_case_count"] == 1
        assert summary["case_type_counts"]["false_positive"] == 3
