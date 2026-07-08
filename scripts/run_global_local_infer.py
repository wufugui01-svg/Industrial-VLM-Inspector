"""Run global-to-local crop-based batch inference over a JSONL sample index."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.crop_agent import make_grid_crops  # noqa: E402
from src.agent.inspector_agent import InspectorAgent  # noqa: E402
from src.agent.prompt_builder import PROMPT_TYPES  # noqa: E402
from src.agent.schema import InspectionResult  # noqa: E402
from src.datasets.path_utils import with_resolved_image_path  # noqa: E402
from src.models.base_vlm import BaseVLM  # noqa: E402
from src.models.mock_vlm import MockVLM  # noqa: E402
from src.models.qwen3vl_transformers import (  # noqa: E402
    Qwen3VLTransformers,
    QwenDependencyError,
)
from src.utils.profiler import Timer  # noqa: E402


@dataclass
class GlobalLocalSummary:
    total: int = 0
    succeeded: int = 0
    failed: int = 0


_SEVERITY_RANK = {
    "none": 0,
    "low": 1,
    "unknown": 2,
    "medium": 3,
    "high": 4,
}


def _prediction_dict(result: InspectionResult) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return result.dict()


def _read_index(index_path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with index_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {index_path}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} of {index_path}"
                )
            samples.append(row)
    return samples


def _create_backend(
    backend: str,
    *,
    model_path: str | None,
    max_new_tokens: int,
) -> BaseVLM:
    if backend == "mock":
        return MockVLM()
    if backend == "qwen":
        if not model_path:
            raise ValueError("--model-path is required for --backend qwen")
        return Qwen3VLTransformers(
            model_name_or_path=model_path,
            max_new_tokens=max_new_tokens,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def _max_confidence(
    global_prediction: InspectionResult,
    crop_predictions: list[dict[str, Any]],
) -> float:
    values = [global_prediction.confidence]
    for crop in crop_predictions:
        prediction = crop.get("prediction")
        if isinstance(prediction, dict) and isinstance(
            prediction.get("confidence"), int | float
        ):
            values.append(float(prediction["confidence"]))
    return max(values)


def _highest_severity(
    global_prediction: InspectionResult,
    anomalous_crop_predictions: list[dict[str, Any]],
) -> str:
    severities = [global_prediction.severity]
    for crop in anomalous_crop_predictions:
        prediction = crop.get("prediction")
        if isinstance(prediction, dict):
            severity = prediction.get("severity")
            if isinstance(severity, str):
                severities.append(severity)
    return max(severities, key=lambda value: _SEVERITY_RANK.get(value, -1))


def aggregate_global_local(
    global_prediction: InspectionResult,
    crop_predictions: list[dict[str, Any]],
    *,
    sample_id: str | None,
) -> InspectionResult:
    """Aggregate global and crop predictions with a simple first-version rule."""

    anomalous_crops = [
        crop
        for crop in crop_predictions
        if isinstance(crop.get("prediction"), dict)
        and crop["prediction"].get("is_anomaly") is True
    ]
    crop_anomaly = bool(anomalous_crops)
    final_is_anomaly = global_prediction.is_anomaly or crop_anomaly

    if anomalous_crops:
        defect_location = ", ".join(
            str(crop.get("region_name") or "unknown") for crop in anomalous_crops
        )
        first_crop_prediction = anomalous_crops[0]["prediction"]
        defect_type = str(first_crop_prediction.get("defect_type") or "unknown")
    else:
        defect_location = global_prediction.defect_location
        defect_type = global_prediction.defect_type

    if not final_is_anomaly:
        defect_type = "none"

    return InspectionResult(
        is_anomaly=final_is_anomaly,
        defect_type=defect_type,  # type: ignore[arg-type]
        defect_location=defect_location,
        severity=_highest_severity(global_prediction, anomalous_crops),  # type: ignore[arg-type]
        reason=(
            "global-local aggregation: final decision combines whole-image "
            "inspection and grid crop inspection."
        ),
        confidence=_max_confidence(global_prediction, crop_predictions),
        raw_model_answer=None,
        sample_id=sample_id,
        parse_status="success",
    )


def _crop_prediction_record(
    *,
    crop: dict[str, Any],
    prediction: InspectionResult,
) -> dict[str, Any]:
    return {
        "crop_path": crop["crop_path"],
        "region_name": crop["region_name"],
        "box": crop["box"],
        "prediction": _prediction_dict(prediction),
    }


def _output_record(
    sample: dict[str, Any],
    *,
    global_prediction: dict[str, Any] | None,
    crop_predictions: list[dict[str, Any]],
    final_prediction: dict[str, Any] | None,
    error: str | None,
    fallback_sample_id: str | None = None,
    latency_sec: float | None = None,
) -> dict[str, Any]:
    return {
        "sample_id": sample.get("sample_id", fallback_sample_id),
        "image_path": sample.get("image_path", ""),
        "global_prediction": global_prediction,
        "crop_predictions": crop_predictions,
        "final_prediction": final_prediction,
        "ground_truth_answer": sample.get("answer", ""),
        "task_type": sample.get("task_type", "unknown"),
        "object_category": sample.get("object_category", "unknown"),
        "latency_sec": latency_sec,
        "error": error,
    }


def run_global_local_inference(
    *,
    index_path: Path,
    output_path: Path,
    backend: str = "mock",
    model_path: str | None = None,
    prompt_type: str = "strict_json",
    grid: str = "2x2",
    crop_dir: Path = Path("outputs/crops"),
    limit: int | None = None,
    max_new_tokens: int = 512,
    dataset_root: Path | None = None,
    show_progress: bool = True,
    vlm: BaseVLM | None = None,
) -> GlobalLocalSummary:
    """Run whole-image and grid-crop inference for each indexed sample."""

    index_path = index_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    crop_dir = crop_dir.expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"Index file does not exist: {index_path}")
    if index_path == output_path:
        raise ValueError("--index and --output must be different files")
    if limit is not None and limit < 0:
        raise ValueError("--limit must be zero or greater")
    if backend not in {"mock", "qwen"}:
        raise ValueError(f"Unsupported backend: {backend}")
    if prompt_type not in PROMPT_TYPES:
        choices = ", ".join(PROMPT_TYPES)
        raise ValueError(f"Unsupported prompt_type '{prompt_type}'. Choose from: {choices}")

    samples = _read_index(index_path)
    samples_to_process = samples[:limit] if limit is not None else samples
    if vlm is None:
        vlm = _create_backend(
            backend,
            model_path=model_path,
            max_new_tokens=max_new_tokens,
        )

    agent = InspectorAgent(vlm, prompt_type=prompt_type)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = GlobalLocalSummary()
    progress = tqdm(
        samples_to_process,
        desc=f"{backend.capitalize()} global-local inference",
        unit="sample",
        disable=not show_progress,
    )

    try:
        with output_path.open("w", encoding="utf-8", newline="\n") as destination:
            for line_index, original_sample in enumerate(progress, start=1):
                sample = with_resolved_image_path(dict(original_sample), dataset_root)
                crop_predictions: list[dict[str, Any]] = []

                with Timer(synchronize_cuda=True) as timer:
                    try:
                        sample_id = sample.get("sample_id")
                        normalized_sample_id = (
                            str(sample_id) if sample_id is not None else None
                        )

                        global_result = agent.inspect(sample)
                        sample_crop_dir = crop_dir / str(
                            sample.get("sample_id") or f"line_{line_index:08d}"
                        )
                        crops = make_grid_crops(
                            str(sample.get("image_path", "")),
                            grid,
                            str(sample_crop_dir),
                        )
                        for crop in crops:
                            crop_sample = {
                                **sample,
                                "image_path": crop["crop_path"],
                                "reference_image_path": None,
                                "crop_region_name": crop["region_name"],
                                "crop_box": crop["box"],
                            }
                            crop_result = agent.inspect(crop_sample)
                            crop_predictions.append(
                                _crop_prediction_record(
                                    crop=crop,
                                    prediction=crop_result,
                                )
                            )

                        final_result = aggregate_global_local(
                            global_result,
                            crop_predictions,
                            sample_id=normalized_sample_id,
                        )
                        global_prediction = _prediction_dict(global_result)
                        final_prediction = _prediction_dict(final_result)
                        error = None
                        fallback_sample_id = None
                        summary.succeeded += 1
                    except Exception as exc:
                        global_prediction = None
                        final_prediction = None
                        error = f"{type(exc).__name__}: {exc}"
                        fallback_sample_id = f"line_{line_index:08d}"
                        summary.failed += 1

                record = _output_record(
                    sample,
                    global_prediction=global_prediction,
                    crop_predictions=crop_predictions,
                    final_prediction=final_prediction,
                    error=error,
                    fallback_sample_id=fallback_sample_id,
                    latency_sec=timer.elapsed_sec,
                )
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                summary.total += 1
    finally:
        progress.close()

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run global-to-local industrial inspection over a JSONL index."
    )
    parser.add_argument("--index", type=Path, required=True, help="Input JSONL index.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file.")
    parser.add_argument(
        "--backend",
        choices=["mock", "qwen"],
        default="mock",
        help="Inference backend.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Required local model directory when --backend qwen is selected.",
    )
    parser.add_argument(
        "--prompt-type",
        choices=PROMPT_TYPES,
        default="strict_json",
        help="Inspection prompt strategy.",
    )
    parser.add_argument(
        "--grid",
        choices=["2x2", "3x3"],
        default="2x2",
        help="Grid crop layout.",
    )
    parser.add_argument(
        "--crop-dir",
        type=Path,
        default=Path("outputs/crops"),
        help="Directory where crop images are saved.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional dataset root used with image_relative_path for portability.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of samples to process.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens for the Qwen backend.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = run_global_local_inference(
            index_path=args.index,
            output_path=args.output,
            backend=args.backend,
            model_path=args.model_path,
            prompt_type=args.prompt_type,
            grid=args.grid,
            crop_dir=args.crop_dir,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
            dataset_root=args.dataset_root,
        )
    except (FileNotFoundError, ValueError, QwenDependencyError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(f"Processed samples: {summary.total}")
    print(f"Succeeded: {summary.succeeded}")
    print(f"Failed: {summary.failed}")
    print(f"Output: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
