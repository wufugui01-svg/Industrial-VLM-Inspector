"""Tests for timing and optional CUDA profiling."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from src.utils.profiler import (
    Timer,
    get_cuda_info,
    get_cuda_memory_info,
    get_gpu_memory_mb,
    get_process_memory_mb,
)


def test_timer_records_elapsed_time() -> None:
    with Timer() as timer:
        sum(range(100))

    assert timer.elapsed_sec is not None
    assert timer.elapsed_sec >= 0.0


def test_missing_torch_returns_null_gpu_values() -> None:
    with patch(
        "src.utils.profiler._import_torch",
        side_effect=ImportError("torch unavailable"),
    ):
        memory = get_gpu_memory_mb()
        cuda_memory = get_cuda_memory_info()
        info = get_cuda_info()

    assert memory["gpu_memory_allocated_mb"] is None
    assert memory["gpu_memory_reserved_mb"] is None
    assert memory["gpu_peak_memory_allocated_mb"] is None
    assert cuda_memory["available"] is False
    assert cuda_memory["gpu_allocated_mb"] is None
    assert cuda_memory["gpu_reserved_mb"] is None
    assert info == {
        "available": False,
        "device_count": 0,
        "device_name": None,
        "cuda_version": None,
    }


def test_no_cuda_returns_null_memory_without_error() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        version=SimpleNamespace(cuda=None),
    )
    with patch("src.utils.profiler._import_torch", return_value=fake_torch):
        memory = get_gpu_memory_mb()
        cuda_memory = get_cuda_memory_info()
        info = get_cuda_info()

    assert memory["gpu_memory_allocated_mb"] is None
    assert memory["gpu_memory_reserved_mb"] is None
    assert cuda_memory["gpu_allocated_mb"] is None
    assert cuda_memory["gpu_reserved_mb"] is None
    assert info["available"] is False


def test_cuda_memory_and_device_info_are_reported_in_mb() -> None:
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        memory_allocated=lambda: 256 * 1024**2,
        memory_reserved=lambda: 512 * 1024**2,
        max_memory_allocated=lambda: 640 * 1024**2,
        max_memory_reserved=lambda: 768 * 1024**2,
        device_count=lambda: 1,
        get_device_name=lambda index: "Test GPU",
    )
    fake_torch = SimpleNamespace(
        cuda=fake_cuda,
        version=SimpleNamespace(cuda="12.8"),
    )
    with patch("src.utils.profiler._import_torch", return_value=fake_torch):
        memory = get_gpu_memory_mb()
        cuda_memory = get_cuda_memory_info()
        info = get_cuda_info()

    assert memory["gpu_memory_allocated_mb"] == 256.0
    assert memory["gpu_memory_reserved_mb"] == 512.0
    assert memory["gpu_peak_memory_allocated_mb"] == 640.0
    assert cuda_memory["available"] is True
    assert cuda_memory["device_count"] == 1
    assert cuda_memory["device_name"] == "Test GPU"
    assert cuda_memory["cuda_version"] == "12.8"
    assert cuda_memory["gpu_allocated_mb"] == 256.0
    assert cuda_memory["gpu_reserved_mb"] == 512.0
    assert cuda_memory["gpu_peak_allocated_mb"] == 640.0
    assert cuda_memory["gpu_peak_reserved_mb"] == 768.0
    assert info == {
        "available": True,
        "device_count": 1,
        "device_name": "Test GPU",
        "cuda_version": "12.8",
    }


def test_process_memory_uses_first_available_source() -> None:
    with (
        patch(
            "src.utils.profiler._process_memory_from_proc_status",
            return_value=123.0,
        ),
        patch(
            "src.utils.profiler._process_memory_from_windows_api",
            return_value=456.0,
        ),
    ):
        assert get_process_memory_mb() == 123.0


def test_process_memory_can_fallback_to_windows_api() -> None:
    with (
        patch(
            "src.utils.profiler._process_memory_from_proc_status",
            return_value=None,
        ),
        patch(
            "src.utils.profiler._process_memory_from_windows_api",
            return_value=456.0,
        ),
    ):
        assert get_process_memory_mb() == 456.0
