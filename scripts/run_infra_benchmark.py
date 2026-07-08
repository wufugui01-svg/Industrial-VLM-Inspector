"""Benchmark VLM inference latency and local resource usage."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.inspector_agent import InspectorAgent  # noqa: E402
from src.agent.prompt_builder import PROMPT_TYPES  # noqa: E402
from src.datasets.path_utils import with_resolved_image_path  # noqa: E402
from src.models.base_vlm import BaseVLM  # noqa: E402
from src.models.mock_vlm import MockVLM  # noqa: E402
from src.models.qwen3vl_transformers import (  # noqa: E402
    Qwen3VLTransformers,
    QwenDependencyError,
)
from src.utils.profiler import (  # noqa: E402
    Timer,
    get_cuda_memory_info,
    get_process_memory_mb,
    reset_gpu_peak_memory,
)

CSV_FIELDS = (
    "backend",
    "model_path",
    "max_new_tokens",
    "total_samples",
    "avg_latency_sec",
    "p50_latency_sec",
    "p95_latency_sec",
    "throughput_samples_per_sec",
    "max_gpu_allocated_mb",
    "max_gpu_reserved_mb",
    "process_memory_mb",
)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + (
        ordered[upper_index] - ordered[lower_index]
    ) * fraction


def _parse_max_new_tokens_list(value: str) -> list[int]:
    tokens: list[int] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            parsed = int(stripped)
        except ValueError as exc:
            raise ValueError(
                "--max-new-tokens-list must be comma-separated integers"
            ) from exc
        if parsed <= 0:
            raise ValueError("max_new_tokens values must be greater than zero")
        tokens.append(parsed)
    if not tokens:
        raise ValueError("--max-new-tokens-list cannot be empty")
    return tokens


def _read_samples(
    index_path: Path,
    *,
    limit: int | None,
    dataset_root: Path | None,
) -> list[dict[str, Any]]:
    if not index_path.is_file():
        raise FileNotFoundError(f"Index file does not exist: {index_path}")
    if limit is not None and limit < 0:
        raise ValueError("--limit must be zero or greater")

    samples: list[dict[str, Any]] = []
    with index_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if limit is not None and len(samples) >= limit:
                break
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError(f"Index row {line_number} is not a JSON object")
            samples.append(with_resolved_image_path(parsed, dataset_root))
    return samples


def _create_backend(
    *,
    backend: str,
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


def _cleanup_backend() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _benchmark_one_config(
    *,
    samples: list[dict[str, Any]],
    backend: str,
    model_path: str | None,
    max_new_tokens: int,
    prompt_type: str,
) -> dict[str, Any]:
    vlm = _create_backend(
        backend=backend,
        model_path=model_path,
        max_new_tokens=max_new_tokens,
    )
    agent = InspectorAgent(vlm, prompt_type=prompt_type)
    latencies: list[float] = []
    gpu_allocated_values: list[float] = []
    gpu_reserved_values: list[float] = []
    process_memory_values: list[float] = []

    with Timer(synchronize_cuda=True) as round_timer:
        for sample in samples:
            reset_gpu_peak_memory()
            with Timer(synchronize_cuda=True) as sample_timer:
                try:
                    agent.inspect(sample)
                except Exception:
                    pass

            if sample_timer.elapsed_sec is not None:
                latencies.append(sample_timer.elapsed_sec)

            cuda_info = get_cuda_memory_info()
            allocated = _numeric(cuda_info.get("gpu_peak_allocated_mb"))
            if allocated is None:
                allocated = _numeric(cuda_info.get("gpu_allocated_mb"))
            reserved = _numeric(cuda_info.get("gpu_peak_reserved_mb"))
            if reserved is None:
                reserved = _numeric(cuda_info.get("gpu_reserved_mb"))
            if allocated is not None:
                gpu_allocated_values.append(allocated)
            if reserved is not None:
                gpu_reserved_values.append(reserved)

            process_memory = get_process_memory_mb()
            if process_memory is not None:
                process_memory_values.append(process_memory)

    elapsed = round_timer.elapsed_sec or 0.0
    total_samples = len(samples)
    throughput = total_samples / elapsed if elapsed > 0 else None
    return {
        "backend": backend,
        "model_path": model_path or "",
        "max_new_tokens": max_new_tokens,
        "total_samples": total_samples,
        "avg_latency_sec": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "p50_latency_sec": _percentile(latencies, 0.50),
        "p95_latency_sec": _percentile(latencies, 0.95),
        "throughput_samples_per_sec": throughput,
        "max_gpu_allocated_mb": (
            max(gpu_allocated_values) if gpu_allocated_values else None
        ),
        "max_gpu_reserved_mb": (
            max(gpu_reserved_values) if gpu_reserved_values else None
        ),
        "process_memory_mb": (
            max(process_memory_values) if process_memory_values else get_process_memory_mb()
        ),
    }


def run_infra_benchmark(
    *,
    index_path: Path,
    output_path: Path,
    backend: str,
    model_path: str | None = None,
    limit: int | None = None,
    max_new_tokens_list: list[int] | None = None,
    prompt_type: str = "strict_json",
    dataset_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Run VLM inference benchmark and write a CSV summary."""

    if backend not in {"mock", "qwen"}:
        raise ValueError(f"Unsupported backend: {backend}")
    if prompt_type not in PROMPT_TYPES:
        choices = ", ".join(PROMPT_TYPES)
        raise ValueError(f"Unsupported prompt_type '{prompt_type}'. Choose from: {choices}")
    if max_new_tokens_list is None:
        max_new_tokens_list = [128]

    index_path = index_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    dataset_root = dataset_root.expanduser().resolve() if dataset_root else None
    samples = _read_samples(index_path, limit=limit, dataset_root=dataset_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for max_new_tokens in max_new_tokens_list:
        print(
            f"Running infra benchmark: backend={backend}, "
            f"max_new_tokens={max_new_tokens}, samples={len(samples)}"
        )
        rows.append(
            _benchmark_one_config(
                samples=samples,
                backend=backend,
                model_path=model_path,
                max_new_tokens=max_new_tokens,
                prompt_type=prompt_type,
            )
        )
        _cleanup_backend()

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark VLM inference latency, throughput, GPU memory, and process memory."
    )
    parser.add_argument("--index", type=Path, required=True, help="Input JSONL index.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV file.")
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
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit.",
    )
    parser.add_argument(
        "--max-new-tokens-list",
        default="64,128,256",
        help="Comma-separated max_new_tokens values, e.g. 64,128,256.",
    )
    parser.add_argument(
        "--prompt-type",
        choices=list(PROMPT_TYPES),
        default="strict_json",
        help="Prompt type used during inference.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional dataset root used with image_relative_path for portability.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        token_values = _parse_max_new_tokens_list(args.max_new_tokens_list)
        rows = run_infra_benchmark(
            index_path=args.index,
            output_path=args.output,
            backend=args.backend,
            model_path=args.model_path,
            limit=args.limit,
            max_new_tokens_list=token_values,
            prompt_type=args.prompt_type,
            dataset_root=args.dataset_root,
        )
    except (FileNotFoundError, ValueError, QwenDependencyError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(f"Benchmark rows: {len(rows)}")
    print(f"Output: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
