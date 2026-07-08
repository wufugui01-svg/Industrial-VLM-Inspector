"""Run reference-based batch inference over a JSONL sample index."""

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

from src.agent.inspector_agent import InspectorAgent  # noqa: E402
from src.agent.prompt_builder import PROMPT_TYPES  # noqa: E402
from src.agent.reference_selector import (  # noqa: E402
    ReferenceSelector,
    ReferenceStrategy,
)
from src.agent.schema import InspectionResult  # noqa: E402
from src.models.base_vlm import BaseVLM  # noqa: E402
from src.models.mock_vlm import MockVLM  # noqa: E402
from src.models.qwen3vl_transformers import (  # noqa: E402
    Qwen3VLTransformers,
    QwenDependencyError,
)
from src.utils.profiler import Timer  # noqa: E402


@dataclass
class ReferenceInferenceSummary:
    total: int = 0
    succeeded: int = 0
    failed: int = 0


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


def _output_record(
    sample: dict[str, Any],
    *,
    prediction: dict[str, Any] | None,
    error: str | None,
    fallback_sample_id: str | None = None,
    latency_sec: float | None = None,
) -> dict[str, Any]:
    return {
        "sample_id": sample.get("sample_id", fallback_sample_id),
        "image_path": sample.get("image_path", ""),
        "reference_image_path": sample.get("reference_image_path"),
        "prediction": prediction,
        "ground_truth_answer": sample.get("answer", ""),
        "task_type": sample.get("task_type", "unknown"),
        "object_category": sample.get("object_category", "unknown"),
        "latency_sec": latency_sec,
        "error": error,
    }


def run_reference_inference(
    *,
    index_path: Path,
    output_path: Path,
    backend: str = "mock",
    model_path: str | None = None,
    prompt_type: str = "reference_strict",
    reference_strategy: ReferenceStrategy = "first",
    limit: int | None = None,
    max_new_tokens: int = 512,
    show_progress: bool = True,
    vlm: BaseVLM | None = None,
) -> ReferenceInferenceSummary:
    """Run reference selection and VLM inference for every JSONL sample."""

    index_path = index_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
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
    if limit is not None:
        samples_to_process = samples[:limit]
    else:
        samples_to_process = samples

    if vlm is None:
        vlm = _create_backend(
            backend,
            model_path=model_path,
            max_new_tokens=max_new_tokens,
        )

    selector = ReferenceSelector(reference_strategy)
    agent = InspectorAgent(vlm, prompt_type=prompt_type)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = ReferenceInferenceSummary()
    progress = tqdm(
        samples_to_process,
        desc=f"{backend.capitalize()} reference inference",
        unit="sample",
        disable=not show_progress,
    )

    try:
        with output_path.open("w", encoding="utf-8", newline="\n") as destination:
            for line_index, original_sample in enumerate(progress, start=1):
                sample: dict[str, Any] = dict(original_sample)
                reference_image_path = selector.select(sample, samples)
                sample["reference_image_path"] = reference_image_path

                with Timer(synchronize_cuda=True) as timer:
                    try:
                        result = agent.inspect(sample)
                        prediction = _prediction_dict(result)
                        error = None
                        fallback_sample_id = None
                        if result.parse_status == "failed":
                            summary.failed += 1
                        else:
                            summary.succeeded += 1
                    except Exception as exc:
                        prediction = None
                        error = f"{type(exc).__name__}: {exc}"
                        fallback_sample_id = f"line_{line_index:08d}"
                        summary.failed += 1

                record = _output_record(
                    sample,
                    prediction=prediction,
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
        description="Run reference-based industrial inspection over a JSONL index."
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
        default="reference_strict",
        help="Inspection prompt strategy.",
    )
    parser.add_argument(
        "--reference-strategy",
        choices=["first", "random", "similarity"],
        default="first",
        help="Reference image selection strategy.",
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
        summary = run_reference_inference(
            index_path=args.index,
            output_path=args.output,
            backend=args.backend,
            model_path=args.model_path,
            prompt_type=args.prompt_type,
            reference_strategy=args.reference_strategy,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
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
