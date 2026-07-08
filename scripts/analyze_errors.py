"""Analyze risky or failed industrial inspection prediction records."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.schema import InspectionResult  # noqa: E402
from src.eval.metrics import map_binary_ground_truth  # noqa: E402

CASE_TYPES = (
    "false_positive",
    "false_negative",
    "parse_failed",
    "low_confidence",
    "error_record",
)
LOW_CONFIDENCE_THRESHOLD = 0.5


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Predictions file does not exist: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError("Prediction row must be a JSON object")
            except (json.JSONDecodeError, TypeError) as exc:
                row = {
                    "sample_id": f"line_{line_number:08d}",
                    "prediction": None,
                    "ground_truth_answer": "",
                    "task_type": "unknown",
                    "object_category": "unknown",
                    "image_path": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
    return rows


def _select_prediction(row: dict[str, Any]) -> Any:
    """Select final/prediction/global output in preferred analysis order."""

    if row.get("final_prediction") is not None:
        return row.get("final_prediction")
    if row.get("prediction") is not None:
        return row.get("prediction")
    if row.get("global_prediction") is not None:
        return row.get("global_prediction")
    return None


def _validated_prediction(value: Any) -> InspectionResult | None:
    if not isinstance(value, dict):
        return None
    try:
        return InspectionResult(**value)
    except (ValidationError, TypeError):
        return None


def _resolve_existing_image_path(image_path: Any) -> Path | None:
    """Resolve common Windows/WSL image path variants without changing data."""

    if not image_path:
        return None
    raw_path = str(image_path)
    candidates = [Path(raw_path)]

    wsl_match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", raw_path)
    if os.name == "nt" and wsl_match:
        drive = wsl_match.group(1).upper()
        remainder = wsl_match.group(2).replace("/", "\\")
        candidates.append(Path(f"{drive}:\\{remainder}"))

    windows_match = re.match(r"^([a-zA-Z]):[\\/](.*)$", raw_path)
    if os.name != "nt" and windows_match:
        drive = windows_match.group(1).lower()
        remainder = windows_match.group(2).replace("\\", "/")
        candidates.append(Path(f"/mnt/{drive}/{remainder}"))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _safe_stem(value: Any, fallback: str) -> str:
    text = str(value or fallback)
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("._")
    return text or fallback


def _copy_image_if_available(
    row: dict[str, Any],
    images_dir: Path,
    fallback_index: int,
) -> str | None:
    source = _resolve_existing_image_path(row.get("image_path"))
    if source is None:
        return None

    images_dir.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix or ".img"
    stem = _safe_stem(row.get("sample_id"), f"case_{fallback_index:05d}")
    destination = images_dir / f"{stem}{suffix}"
    counter = 1
    while destination.exists():
        destination = images_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    shutil.copy2(source, destination)
    return str(destination)


def _classify_row(row: dict[str, Any]) -> tuple[list[str], InspectionResult | None]:
    case_types: list[str] = []
    has_error = bool(row.get("error"))
    if has_error:
        case_types.append("error_record")

    raw_prediction = _select_prediction(row)
    prediction = _validated_prediction(raw_prediction)
    prediction_present = raw_prediction is not None

    if prediction is None:
        if prediction_present or not has_error:
            case_types.append("parse_failed")
        return case_types, None

    if prediction.parse_status == "failed":
        case_types.append("parse_failed")

    if prediction.confidence < LOW_CONFIDENCE_THRESHOLD:
        case_types.append("low_confidence")

    ground_truth = map_binary_ground_truth(row.get("ground_truth_answer"))
    if ground_truth is not None and prediction.parse_status != "failed":
        if prediction.is_anomaly and not ground_truth:
            case_types.append("false_positive")
        elif not prediction.is_anomaly and ground_truth:
            case_types.append("false_negative")

    return case_types, prediction


def _case_record(
    row: dict[str, Any],
    case_types: list[str],
    prediction: InspectionResult | None,
    copied_image_path: str | None,
) -> dict[str, Any]:
    prediction_dict = prediction.model_dump() if prediction is not None else None
    return {
        "case_types": case_types,
        "sample_id": row.get("sample_id"),
        "image_path": row.get("image_path"),
        "copied_image_path": copied_image_path,
        "ground_truth_answer": row.get("ground_truth_answer"),
        "predicted_is_anomaly": (
            prediction.is_anomaly if prediction is not None else None
        ),
        "confidence": prediction.confidence if prediction is not None else None,
        "parse_status": prediction.parse_status if prediction is not None else None,
        "error": row.get("error"),
        "task_type": row.get("task_type"),
        "object_category": row.get("object_category"),
        "prediction": prediction_dict,
        "record": row,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_markdown_report(
    *,
    path: Path,
    predictions_path: Path,
    summary: dict[str, Any],
    cases: list[dict[str, Any]],
) -> None:
    lines = [
        "# Error Analysis Report",
        "",
        f"- Predictions: `{predictions_path}`",
        f"- Total samples: {summary['total_samples']}",
        f"- Matched risk/error samples: {summary['matched_case_count']}",
        f"- Exported cases: {summary['exported_case_count']}",
        f"- Max cases: {summary['max_cases']}",
        "",
        "## Error Counts",
        "",
        "| case_type | count |",
        "|---|---:|",
    ]
    for case_type in CASE_TYPES:
        lines.append(f"| {case_type} | {summary['case_type_counts'][case_type]} |")

    lines.extend(
        [
            "",
            "## Typical Error Cases",
            "",
            "| sample_id | case_types | ground_truth | prediction | confidence | error | image |",
            "|---|---|---|---|---:|---|---|",
        ]
    )
    for case in cases[:20]:
        prediction = case.get("prediction") or {}
        predicted = case.get("predicted_is_anomaly")
        confidence = case.get("confidence")
        confidence_text = "" if confidence is None else f"{confidence:.3f}"
        error = str(case.get("error") or "")
        if len(error) > 80:
            error = error[:77] + "..."
        image = case.get("copied_image_path") or case.get("image_path") or ""
        lines.append(
            "| {sample_id} | {case_types} | {gt} | {predicted} | {conf} | {error} | {image} |".format(
                sample_id=case.get("sample_id") or "",
                case_types=", ".join(case.get("case_types") or []),
                gt=case.get("ground_truth_answer") or "",
                predicted=predicted,
                conf=confidence_text,
                error=error.replace("|", "\\|"),
                image=str(image).replace("|", "\\|"),
            )
        )

    lines.extend(
        [
            "",
            "## Auto Observation Template",
            "",
            "- This report is generated by deterministic rules over prediction records; it does not manually judge visual correctness.",
            "- False positive and false negative counts are only computed for samples whose `ground_truth_answer` can be mapped to normal/abnormal.",
            f"- Records with `confidence < {LOW_CONFIDENCE_THRESHOLD}` are flagged as low-confidence risk cases.",
            "- `error_record` means the pipeline wrote a non-empty `error` field for that sample.",
            "- If copied images are available, inspect them manually before making claims about model capability.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_errors_file(
    *,
    predictions_path: Path,
    output_dir: Path,
    max_cases: int = 50,
) -> dict[str, Any]:
    """Analyze prediction errors and write JSONL/JSON/Markdown artifacts."""

    predictions_path = predictions_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if max_cases < 0:
        raise ValueError("--max-cases must be zero or greater")

    rows = _read_jsonl(predictions_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"

    case_type_counts: Counter[str] = Counter({case_type: 0 for case_type in CASE_TYPES})
    exported_cases: list[dict[str, Any]] = []
    matched_case_count = 0
    binary_mappable_count = 0
    copied_image_count = 0

    for index, row in enumerate(rows, start=1):
        if map_binary_ground_truth(row.get("ground_truth_answer")) is not None:
            binary_mappable_count += 1

        case_types, prediction = _classify_row(row)
        if not case_types:
            continue

        matched_case_count += 1
        for case_type in set(case_types):
            case_type_counts[case_type] += 1

        if len(exported_cases) >= max_cases:
            continue

        copied_image_path = _copy_image_if_available(row, images_dir, index)
        if copied_image_path is not None:
            copied_image_count += 1
        exported_cases.append(
            _case_record(row, case_types, prediction, copied_image_path)
        )

    summary = {
        "predictions_path": str(predictions_path),
        "output_dir": str(output_dir),
        "total_samples": len(rows),
        "matched_case_count": matched_case_count,
        "exported_case_count": len(exported_cases),
        "max_cases": max_cases,
        "case_type_counts": dict(case_type_counts),
        "binary_mappable_count": binary_mappable_count,
        "copied_image_count": copied_image_count,
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
    }

    error_cases_path = output_dir / "error_cases.jsonl"
    error_summary_path = output_dir / "error_summary.json"
    error_analysis_path = output_dir / "error_analysis.md"
    _write_jsonl(error_cases_path, exported_cases)
    error_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown_report(
        path=error_analysis_path,
        predictions_path=predictions_path,
        summary=summary,
        cases=exported_cases,
    )

    summary["error_cases_path"] = str(error_cases_path)
    summary["error_summary_path"] = str(error_summary_path)
    summary["error_analysis_path"] = str(error_analysis_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze false positives, false negatives, parse failures, low-confidence records, and pipeline errors."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Input predictions JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where analysis files will be written.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=50,
        help="Maximum number of detailed cases to export.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = analyze_errors_file(
            predictions_path=args.predictions,
            output_dir=args.output_dir,
            max_cases=args.max_cases,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Error analysis written to: {args.output_dir.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
