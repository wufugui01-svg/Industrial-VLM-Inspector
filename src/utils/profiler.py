"""Lightweight inference timing and CUDA memory helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


def _import_torch() -> Any:
    import torch

    return torch


@dataclass
class Timer:
    """Measure elapsed wall-clock time with ``with Timer()``."""

    synchronize_cuda: bool = False
    elapsed_sec: float | None = field(default=None, init=False)
    _started_at: float | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> "Timer":
        self.elapsed_sec = None
        if self.synchronize_cuda:
            synchronize_cuda_if_available()
        self._started_at = perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback
        if self._started_at is not None:
            if self.synchronize_cuda:
                synchronize_cuda_if_available()
            self.elapsed_sec = perf_counter() - self._started_at


def synchronize_cuda_if_available() -> None:
    """Synchronize CUDA when available; remain a no-op on CPU-only systems."""

    try:
        torch = _import_torch()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def reset_gpu_peak_memory() -> None:
    """Reset PyTorch's peak-memory counter for the active CUDA device."""

    try:
        torch = _import_torch()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def get_gpu_memory_mb() -> dict[str, float | None]:
    """Return current PyTorch CUDA memory, or null-compatible values."""

    empty = {
        "gpu_memory_allocated_mb": None,
        "gpu_memory_reserved_mb": None,
        "gpu_peak_memory_allocated_mb": None,
    }
    try:
        torch = _import_torch()
        if not torch.cuda.is_available():
            return empty
        return {
            "gpu_memory_allocated_mb": torch.cuda.memory_allocated()
            / (1024**2),
            "gpu_memory_reserved_mb": torch.cuda.memory_reserved()
            / (1024**2),
            "gpu_peak_memory_allocated_mb": torch.cuda.max_memory_allocated()
            / (1024**2),
        }
    except Exception:
        return empty


def get_cuda_info() -> dict[str, Any]:
    """Return CUDA availability details without requiring PyTorch."""

    info: dict[str, Any] = {
        "available": False,
        "device_count": 0,
        "device_name": None,
        "cuda_version": None,
    }
    try:
        torch = _import_torch()
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        available = bool(torch.cuda.is_available())
        info["available"] = available
        if available:
            info["device_count"] = int(torch.cuda.device_count())
            if info["device_count"] > 0:
                info["device_name"] = str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return info
