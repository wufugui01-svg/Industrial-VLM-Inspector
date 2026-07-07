"""Run the inspection pipeline for one sample from a JSONL index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.inspector_agent import InspectorAgent  # noqa: E402
from src.agent.schema import InspectionResult  # noqa: E402
from src.datasets.path_utils import with_resolved_image_path  # noqa: E402
from src.models.mock_vlm import MockVLM  # noqa: E402
from src.models.qwen3vl_transformers import (  # noqa: E402
    Qwen3VLTransformers,
    QwenDependencyError,
)
from src.visualization.report_image import create_report_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one indexed sample through an inspection backend."
    )
    parser.add_argument(
        "--index",
        type=Path,
        required=True,
        help="Path to an MMAD JSONL index.",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Optional sample_id to select; defaults to the first sample.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional MMAD root used with image_relative_path for portability.",
    )
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
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens for the Qwen backend (default: 512).",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument(
        "--prompt-type",
        choices=["basic", "industrial", "strict_json"],
        default="basic",
        help="Inspection prompt strategy (default: basic).",
    )
    parser.add_argument(
        "--save-report",
        type=Path,
        default=None,
        help="Optional path for a PNG/JPEG visual inspection report.",
    )
    return parser.parse_args()


def load_sample(index_path: Path, sample_id: str | None) -> dict[str, Any]:
    """Load the requested sample, or the first non-empty JSONL row."""

    if not index_path.is_file():
        raise FileNotFoundError(f"Index file does not exist: {index_path}")

    with index_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {index_path}: {exc}"
                ) from exc
            if not isinstance(sample, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} of {index_path}"
                )
            if sample_id is None or str(sample.get("sample_id")) == sample_id:
                return sample

    if sample_id is None:
        raise ValueError(f"No samples found in index: {index_path}")
    raise ValueError(f"sample_id not found: {sample_id}")


def result_as_json(result: InspectionResult) -> str:
    """Serialize with either Pydantic v1 or v2."""

    if hasattr(result, "model_dump_json"):
        return result.model_dump_json(indent=2)
    return result.json(indent=2)


def main() -> int:
    args = parse_args()
    try:
        sample = load_sample(args.index.expanduser().resolve(), args.sample_id)
        sample = with_resolved_image_path(sample, args.dataset_root)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    if args.backend == "mock":
        vlm = MockVLM()
    else:
        if not args.model_path:
            raise SystemExit("error: --model-path is required for --backend qwen")
        try:
            vlm = Qwen3VLTransformers(
                model_name_or_path=args.model_path,
                max_new_tokens=args.max_new_tokens,
                device_map=args.device_map,
                torch_dtype=args.torch_dtype,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
            )
        except (FileNotFoundError, ValueError, QwenDependencyError) as exc:
            raise SystemExit(f"error: {exc}") from exc

    try:
        result = InspectorAgent(vlm, prompt_type=args.prompt_type).inspect(sample)
    except (FileNotFoundError, QwenDependencyError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(result_as_json(result))
    if args.save_report is not None:
        try:
            report_path = create_report_image(
                image_path=str(sample.get("image_path", "")),
                result=result,
                output_path=str(args.save_report),
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise SystemExit(f"error: could not create report image: {exc}") from exc
        print(f"Report image written to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
