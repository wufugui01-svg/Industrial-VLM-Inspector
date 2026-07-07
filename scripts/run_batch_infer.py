"""Run model-free batch inference over a JSONL sample index."""

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
from src.agent.schema import InspectionResult  # noqa: E402
from src.datasets.path_utils import with_resolved_image_path  # noqa: E402
from src.models.base_vlm import BaseVLM  # noqa: E402
from src.models.mock_vlm import MockVLM  # noqa: E402
from src.models.qwen3vl_transformers import (  # noqa: E402
    Qwen3VLTransformers,
    QwenDependencyError,
)
from src.utils.profiler import (  # noqa: E402
    Timer,
    get_gpu_memory_mb,
    reset_gpu_peak_memory,
)


@dataclass
class BatchSummary:
    total: int = 0
    succeeded: int = 0
    failed: int = 0


def _prediction_dict(result: InspectionResult) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return result.dict()


def _output_record(
    sample: dict[str, Any],
    *,
    prediction: dict[str, Any] | None,
    error: str | None,
    fallback_sample_id: str | None = None,
    latency_sec: float | None = None,
    gpu_memory_allocated_mb: float | None = None,
    gpu_memory_reserved_mb: float | None = None,
    gpu_peak_memory_allocated_mb: float | None = None,
) -> dict[str, Any]:
    return {
        "sample_id": sample.get("sample_id", fallback_sample_id),
        "prediction": prediction,
        "ground_truth_answer": sample.get("answer", ""),
        "task_type": sample.get("task_type", "unknown"),
        "object_category": sample.get("object_category", "unknown"),
        "image_path": sample.get("image_path", ""),
        "error": error,
        "latency_sec": latency_sec,
        "gpu_memory_allocated_mb": gpu_memory_allocated_mb,
        "gpu_memory_reserved_mb": gpu_memory_reserved_mb,
        "gpu_peak_memory_allocated_mb": gpu_peak_memory_allocated_mb,
    }


def run_batch_inference(
    *,
    index_path: Path,
    output_path: Path,
    backend: str = "mock",
    limit: int | None = None,
    show_progress: bool = True,
    model_path: str | None = None,
    max_new_tokens: int = 512,
    prompt_type: str = "basic",
    vlm: BaseVLM | None = None,
    dataset_root: Path | None = None,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> BatchSummary:
    """Process JSONL rows independently and write one output row per input."""

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

    if vlm is None:
        if backend == "mock":
            vlm = MockVLM()
        else:
            if not model_path:
                raise ValueError("model_path is required for the Qwen backend")
            vlm = Qwen3VLTransformers(
                model_name_or_path=model_path,
                max_new_tokens=max_new_tokens,
                device_map=device_map,
                torch_dtype=torch_dtype,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
    agent = InspectorAgent(vlm, prompt_type=prompt_type)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = BatchSummary()
    progress = tqdm(
        total=limit,
        desc=f"{backend.capitalize()} inference",
        unit="sample",
        disable=not show_progress,
    )

    try:
        with (
            index_path.open("r", encoding="utf-8") as source,
            output_path.open("w", encoding="utf-8", newline="\n") as destination,
        ):
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                if limit is not None and summary.total >= limit:
                    break

                sample: dict[str, Any] = {}
                reset_gpu_peak_memory()
                with Timer(synchronize_cuda=True) as timer:
                    try:
                        parsed = json.loads(line)
                        if not isinstance(parsed, dict):
                            raise TypeError("Index row must be a JSON object")
                        sample = with_resolved_image_path(parsed, dataset_root)
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
                        fallback_sample_id = f"line_{line_number:08d}"
                        summary.failed += 1

                gpu_memory = get_gpu_memory_mb()
                record = _output_record(
                    sample,
                    prediction=prediction,
                    error=error,
                    fallback_sample_id=fallback_sample_id,
                    latency_sec=timer.elapsed_sec,
                    **gpu_memory,
                )

                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                summary.total += 1
                progress.update(1)
    finally:
        progress.close()

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch industrial inspection with the mock backend."
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
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional MMAD root used with image_relative_path for portability.",
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
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of non-empty rows to process.",
    )
    parser.add_argument(
        "--prompt-type",
        choices=["basic", "industrial", "strict_json"],
        default="basic",
        help="Inspection prompt strategy (default: basic).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = run_batch_inference(
            index_path=args.index,
            output_path=args.output,
            backend=args.backend,
            limit=args.limit,
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            prompt_type=args.prompt_type,
            dataset_root=args.dataset_root,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
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
