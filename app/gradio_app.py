"""Gradio demo for single-image industrial defect inspection."""

from __future__ import annotations

import gc
import json
import sys
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.inspector_agent import InspectorAgent  # noqa: E402
from src.agent.schema import InspectionResult  # noqa: E402
from src.models.base_vlm import BaseVLM  # noqa: E402
from src.models.mock_vlm import MockVLM  # noqa: E402
from src.models.qwen3vl_transformers import (  # noqa: E402
    Qwen3VLTransformers,
)
from src.visualization.report_image import create_report_image  # noqa: E402

_MOCK_VLM = MockVLM()
_INFERENCE_LOCK = Lock()
_REPORT_DIR = Path(tempfile.gettempdir()) / "industrial_vlm_inspector_reports"
_QWEN_CACHE: tuple[str, Qwen3VLTransformers] | None = None
_MAX_TEMP_REPORTS = 50


def _load_qwen_backend(model_path: str) -> Qwen3VLTransformers:
    """Cache one local Qwen model so repeated clicks reuse loaded weights."""

    global _QWEN_CACHE
    if _QWEN_CACHE is not None and _QWEN_CACHE[0] == model_path:
        return _QWEN_CACHE[1]
    _QWEN_CACHE = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
    backend = Qwen3VLTransformers(model_name_or_path=model_path)
    _QWEN_CACHE = (model_path, backend)
    return backend


def _cleanup_temp_reports() -> None:
    """Bound temporary report growth while leaving recent UI results intact."""

    if not _REPORT_DIR.is_dir():
        return
    reports = sorted(
        _REPORT_DIR.glob("report_*.png"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for stale_report in reports[_MAX_TEMP_REPORTS:]:
        try:
            stale_report.unlink()
        except OSError:
            continue


def _select_backend(backend: str, model_path: str) -> BaseVLM:
    if backend == "mock":
        return _MOCK_VLM
    if backend == "qwen":
        normalized_path = model_path.strip()
        if not normalized_path:
            raise ValueError("Model path is required when backend is qwen.")
        return _load_qwen_backend(normalized_path)
    raise ValueError(f"Unsupported backend: {backend}")


def _raw_answer_for_display(result: InspectionResult) -> str:
    raw_answer = result.raw_model_answer or ""
    try:
        parsed: Any = json.loads(raw_answer)
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, TypeError):
        return raw_answer


def inspect_uploaded_image(
    image_path: str | None,
    model_path: str,
    backend: str,
    prompt_type: str,
    report_output_path: str | None = None,
) -> tuple[str, str, str, str, float, str, str, str]:
    """Run one uploaded image through the shared InspectorAgent pipeline."""

    if not image_path:
        raise ValueError("Please upload an image before running inspection.")
    resolved_image = Path(image_path).expanduser().resolve()
    if not resolved_image.is_file():
        raise FileNotFoundError(f"Uploaded image does not exist: {resolved_image}")

    vlm = _select_backend(backend, model_path)
    sample = {
        "sample_id": "gradio-upload",
        "image_path": str(resolved_image),
        "question": "Inspect this image for industrial defects.",
        "options": [],
        "task_type": "Anomaly Detection",
        "object_category": "unknown",
    }
    with _INFERENCE_LOCK:
        result = InspectorAgent(vlm, prompt_type=prompt_type).inspect(sample)

    if report_output_path:
        report_path = Path(report_output_path).expanduser().resolve()
    else:
        _REPORT_DIR.mkdir(parents=True, exist_ok=True)
        _cleanup_temp_reports()
        report_path = _REPORT_DIR / f"report_{uuid4().hex}.png"
    saved_report = create_report_image(
        str(resolved_image),
        result,
        str(report_path),
    )

    return (
        "Yes" if result.is_anomaly else "No",
        result.defect_type,
        result.defect_location,
        result.severity,
        result.confidence,
        result.reason,
        _raw_answer_for_display(result),
        saved_report,
    )


def _run_with_ui_error(
    image_path: str | None,
    model_path: str,
    backend: str,
    prompt_type: str,
) -> tuple[str, str, str, str, float, str, str, str]:
    import gradio as gr

    try:
        return inspect_uploaded_image(
            image_path,
            model_path,
            backend,
            prompt_type,
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def build_demo() -> Any:
    """Construct the Gradio Blocks application."""

    import gradio as gr

    with gr.Blocks(title="Industrial VLM Inspector") as demo:
        gr.Markdown(
            "# Industrial VLM Inspector\n"
            "Upload an industrial image and run a structured defect inspection."
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="Inspection Image",
                    type="filepath",
                )
                model_path_input = gr.Textbox(
                    label="Local Qwen Model Path",
                    placeholder="/path/to/Qwen3-VL-4B-Instruct",
                    info="Required only when backend is qwen.",
                )
                backend_input = gr.Dropdown(
                    choices=["mock", "qwen"],
                    value="mock",
                    label="Backend",
                )
                prompt_type_input = gr.Dropdown(
                    choices=["basic", "industrial", "strict_json"],
                    value="strict_json",
                    label="Prompt Type",
                )
                run_button = gr.Button("Run Inspection", variant="primary")

            with gr.Column():
                anomaly_output = gr.Textbox(label="Anomaly")
                defect_type_output = gr.Textbox(label="Defect Type")
                defect_location_output = gr.Textbox(label="Defect Location")
                severity_output = gr.Textbox(label="Severity")
                confidence_output = gr.Number(label="Confidence")
                reason_output = gr.Textbox(label="Reason", lines=3)
                raw_json_output = gr.Textbox(
                    label="Raw Model JSON",
                    lines=12,
                    max_lines=20,
                )
                report_image_output = gr.Image(
                    label="Inspection Report",
                    interactive=False,
                )

        run_button.click(
            fn=_run_with_ui_error,
            inputs=[
                image_input,
                model_path_input,
                backend_input,
                prompt_type_input,
            ],
            outputs=[
                anomaly_output,
                defect_type_output,
                defect_location_output,
                severity_output,
                confidence_output,
                reason_output,
                raw_json_output,
                report_image_output,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue().launch()
