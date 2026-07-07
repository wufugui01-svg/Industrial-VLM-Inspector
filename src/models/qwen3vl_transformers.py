"""Local Qwen3-VL inference through Hugging Face Transformers."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.models.base_vlm import BaseVLM


class QwenDependencyError(RuntimeError):
    """Raised when the optional local Qwen inference stack is unavailable."""


def _load_qwen_dependencies() -> tuple[Any, Any, Any]:
    """Import heavy Qwen dependencies only when the backend is selected."""

    try:
        import torch
    except ImportError as exc:
        raise QwenDependencyError(
            "The Qwen backend requires PyTorch. Install a CUDA-compatible "
            "PyTorch build before using '--backend qwen'."
        ) from exc

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise QwenDependencyError(
            "The Qwen backend requires a Qwen3-VL-compatible Transformers "
            "installation. Install 'transformers>=4.57.0' and 'accelerate'."
        ) from exc

    return torch, AutoModelForImageTextToText, AutoProcessor


class Qwen3VLTransformers(BaseVLM):
    """Run Qwen3-VL from an explicitly supplied local model directory."""

    def __init__(
        self,
        model_name_or_path: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 512,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> None:
        if not model_name_or_path or not str(model_name_or_path).strip():
            raise ValueError("model_name_or_path must be a local model path")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")

        model_path = Path(model_name_or_path).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"Local Qwen model directory does not exist: {model_path}"
            )

        torch, model_class, processor_class = _load_qwen_dependencies()
        self._torch = torch
        self.model_name_or_path = str(model_path)
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        try:
            processor_kwargs: dict[str, Any] = {
                "local_files_only": True,
            }
            if min_pixels is not None:
                processor_kwargs["min_pixels"] = min_pixels
            if max_pixels is not None:
                processor_kwargs["max_pixels"] = max_pixels
            self.processor = processor_class.from_pretrained(
                self.model_name_or_path,
                **processor_kwargs,
            )
            self.model = model_class.from_pretrained(
                self.model_name_or_path,
                device_map=self.device_map,
                dtype=self.torch_dtype,
                local_files_only=True,
            )
        except ImportError as exc:
            raise QwenDependencyError(
                "Qwen model loading failed because an optional dependency is "
                "missing. Install 'transformers>=4.57.0', 'accelerate', "
                "'qwen-vl-utils', 'torchvision', and a compatible PyTorch build. "
                f"Original import error: {exc}"
            ) from exc
        self.model.eval()
        self._print_device_placement()

    def _input_device(self) -> Any:
        """Return a usable device for input tensors with sharded models."""

        device = getattr(self.model, "device", None)
        if device is not None and str(device) != "meta":
            return device
        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for mapped_device in device_map.values():
                if str(mapped_device) not in {"disk", "meta"}:
                    return mapped_device
        try:
            return next(self.model.parameters()).device
        except (StopIteration, AttributeError) as exc:
            raise RuntimeError("Could not determine Qwen input device") from exc

    def _print_device_placement(self) -> None:
        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, dict):
            devices = sorted({str(device) for device in device_map.values()})
        else:
            devices = [str(getattr(self.model, "device", "unknown"))]

        print(
            f"Qwen model device placement: {', '.join(devices)}",
            file=sys.stderr,
        )
        if self._torch.cuda.is_available():
            allocated_gb = self._torch.cuda.memory_allocated() / (1024**3)
            reserved_gb = self._torch.cuda.memory_reserved() / (1024**3)
            print(
                "CUDA memory after model load: "
                f"allocated={allocated_gb:.2f} GiB, reserved={reserved_gb:.2f} GiB",
                file=sys.stderr,
            )

    @staticmethod
    def _validated_images(images: Sequence[str | None]) -> list[str]:
        validated: list[str] = []
        for image in images:
            if image is None:
                continue
            image_text = str(image).strip()
            if not image_text:
                continue
            image_path = Path(image_text).expanduser().resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"Input image does not exist: {image_path}")
            validated.append(str(image_path))
        return validated

    def generate(self, images: list[str], prompt: str) -> str:
        """Generate text for zero, one, or multiple local images."""

        validated_images = self._validated_images(images)
        content: list[dict[str, str]] = [
            {"type": "image", "image": image_path}
            for image_path in validated_images
        ]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except ImportError as exc:
            raise QwenDependencyError(
                "Qwen input processing requires its optional vision "
                "dependencies. Install 'qwen-vl-utils', 'torchvision', and Pillow."
            ) from exc
        inputs = inputs.to(self._input_device())

        with self._torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        input_ids = inputs["input_ids"]
        generated_ids_trimmed = [
            output_ids[len(source_ids) :]
            for source_ids, output_ids in zip(input_ids, generated_ids)
        ]
        decoded = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip() if decoded else ""
