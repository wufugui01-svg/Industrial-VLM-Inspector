"""Run the MMAD multiple-choice benchmark independently of InspectorAgent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.mmad_prompt_builder import (  # noqa: E402
    build_mmad_multiple_choice_prompt,
)
from src.datasets.path_utils import with_resolved_image_path  # noqa: E402
from src.eval.mmad_benchmark import (  # noqa: E402
    calculate_mmad_metrics,
    parse_mmad_answer,
)
from src.models.mock_mmad_vlm import MockMMADVLM  # noqa: E402
from src.models.qwen3vl_transformers import (  # noqa: E402
    Qwen3VLTransformers,
    QwenDependencyError,
)
from src.utils.profiler import (  # noqa: E402
    Timer,
    get_gpu_memory_mb,
    reset_gpu_peak_memory,
)


def _backend(
    name: str,
    model_path: str | None,
    max_new_tokens: int,
    device_map: str,
    torch_dtype: str,
    min_pixels: int | None,
    max_pixels: int | None,
) -> Any:
    if name == "mock":
        return MockMMADVLM()
    if not model_path:
        raise ValueError("--model-path is required for --backend qwen")
    return Qwen3VLTransformers(
        model_name_or_path=model_path,
        max_new_tokens=max_new_tokens,
        device_map=device_map,
        torch_dtype=torch_dtype,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )


def run_mmad_benchmark(
    *,
    index_path: Path,
    output_path: Path,
    metrics_path: Path,
    backend: str = "mock",
    model_path: str | None = None,
    max_new_tokens: int = 32,
    limit: int | None = None,
    show_progress: bool = True,
    dataset_root: Path | None = None,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> dict[str, Any]:
    """Run option prediction and save both row-level output and metrics."""

    index_path = index_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    metrics_path = metrics_path.expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"Index file does not exist: {index_path}")
    if limit is not None and limit < 0:
        raise ValueError("--limit must be zero or greater")
    vlm = _backend(
        backend,
        model_path,
        max_new_tokens,
        device_map,
        torch_dtype,
        min_pixels,
        max_pixels,
    )
    rows: list[dict[str, Any]] = []

    with index_path.open("r", encoding="utf-8") as source:
        iterator = tqdm(
            source,
            desc=f"{backend.capitalize()} MMAD benchmark",
            unit="sample",
            disable=not show_progress,
        )
        for line_number, line in enumerate(iterator, start=1):
            if not line.strip():
                continue
            if limit is not None and len(rows) >= limit:
                break
            sample: dict[str, Any] = {}
            raw_answer = ""
            error: str | None = None
            prediction_answer: str | None = None
            parse_status = "failed"
            reset_gpu_peak_memory()
            with Timer(synchronize_cuda=True) as timer:
                try:
                    sample = json.loads(line)
                    if not isinstance(sample, dict):
                        raise TypeError("Index row must be a JSON object")
                    sample = with_resolved_image_path(sample, dataset_root)
                    prompt = build_mmad_multiple_choice_prompt(sample)
                    image_path = sample.get("image_path")
                    images = [str(image_path)] if image_path else []
                    raw_answer = vlm.generate(images, prompt)
                    prediction_answer, parse_status = parse_mmad_answer(
                        raw_answer, sample.get("options")
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            memory = get_gpu_memory_mb()
            rows.append(
                {
                    "sample_id": sample.get(
                        "sample_id", f"line_{line_number:08d}"
                    ),
                    "prediction_answer": prediction_answer,
                    "ground_truth_answer": sample.get("answer", ""),
                    "parse_status": parse_status,
                    "raw_model_answer": raw_answer,
                    "task_type": sample.get("task_type", "unknown"),
                    "object_category": sample.get(
                        "object_category", "unknown"
                    ),
                    "image_path": sample.get("image_path", ""),
                    "error": error,
                    "latency_sec": timer.elapsed_sec,
                    **memory,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = calculate_mmad_metrics(rows)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--backend", choices=["mock", "qwen"], default="mock")
    parser.add_argument("--model-path")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metrics = run_mmad_benchmark(
            index_path=args.index,
            output_path=args.output,
            metrics_path=args.metrics_output,
            backend=args.backend,
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
            dataset_root=args.dataset_root,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
    except (FileNotFoundError, ValueError, QwenDependencyError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
