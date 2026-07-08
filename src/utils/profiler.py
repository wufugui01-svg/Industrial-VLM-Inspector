"""Lightweight inference timing and CUDA memory helpers."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - platform dependent
    resource = None  # type: ignore[assignment]


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


def get_cuda_memory_info() -> dict[str, Any]:
    """Return CUDA device and PyTorch memory details without nvidia-smi."""

    info: dict[str, Any] = {
        "available": False,
        "device_count": 0,
        "device_name": None,
        "cuda_version": None,
        "gpu_allocated_mb": None,
        "gpu_reserved_mb": None,
        "gpu_peak_allocated_mb": None,
        "gpu_peak_reserved_mb": None,
    }
    try:
        torch = _import_torch()
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        available = bool(torch.cuda.is_available())
        info["available"] = available
        if not available:
            return info

        info["device_count"] = int(torch.cuda.device_count())
        if info["device_count"] > 0:
            info["device_name"] = str(torch.cuda.get_device_name(0))
        info["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
        info["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)
        info["gpu_peak_allocated_mb"] = (
            torch.cuda.max_memory_allocated() / (1024**2)
        )
        if hasattr(torch.cuda, "max_memory_reserved"):
            info["gpu_peak_reserved_mb"] = (
                torch.cuda.max_memory_reserved() / (1024**2)
            )
        else:
            info["gpu_peak_reserved_mb"] = info["gpu_reserved_mb"]
    except Exception:
        pass
    return info


def _process_memory_from_proc_status() -> float | None:
    status_path = "/proc/self/status"
    if not os.path.exists(status_path):
        return None
    try:
        with open(status_path, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    except OSError:
        return None
    return None


def _process_memory_from_windows_api() -> float | None:
    if os.name != "nt":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(ProcessMemoryCounters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            handle,
            ctypes.pointer(counters),
            counters.cb,
        )
        if ok:
            return counters.WorkingSetSize / (1024**2)
    except Exception:
        return None
    return None


def get_process_memory_mb() -> float | None:
    """Return current process resident memory in MB when available."""

    proc_memory = _process_memory_from_proc_status()
    if proc_memory is not None:
        return proc_memory

    windows_memory = _process_memory_from_windows_api()
    if windows_memory is not None:
        return windows_memory

    try:
        if resource is None:
            return None
        usage = resource.getrusage(resource.RUSAGE_SELF)
        value = float(usage.ru_maxrss)
        if os.name == "posix":
            return value / 1024.0
        return value / (1024**2)
    except Exception:
        return None


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
