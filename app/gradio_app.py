"""Gradio demo for multi-mode industrial defect inspection."""

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
from src.agent.crop_agent import make_grid_crops  # noqa: E402
from src.models.base_vlm import BaseVLM  # noqa: E402
from src.models.mock_vlm import MockVLM  # noqa: E402
from src.models.qwen3vl_transformers import (  # noqa: E402
    Qwen3VLTransformers,
)
from src.utils.profiler import Timer  # noqa: E402
from src.visualization.report_image import create_report_image  # noqa: E402

_MOCK_VLM = MockVLM()
_INFERENCE_LOCK = Lock()
_REPORT_DIR = Path(tempfile.gettempdir()) / "industrial_vlm_inspector_reports"
_CROP_DIR = Path(tempfile.gettempdir()) / "industrial_vlm_inspector_crops"
_QWEN_CACHE: tuple[str, int, Qwen3VLTransformers] | None = None
_MAX_TEMP_REPORTS = 50


def _load_qwen_backend(model_path: str, max_new_tokens: int) -> Qwen3VLTransformers:
    """Cache one local Qwen model so repeated clicks reuse loaded weights."""

    global _QWEN_CACHE
    if (
        _QWEN_CACHE is not None
        and _QWEN_CACHE[0] == model_path
        and _QWEN_CACHE[1] == max_new_tokens
    ):
        return _QWEN_CACHE[2]
    _QWEN_CACHE = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
    backend = Qwen3VLTransformers(
        model_name_or_path=model_path,
        max_new_tokens=max_new_tokens,
    )
    _QWEN_CACHE = (model_path, max_new_tokens, backend)
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


def _select_backend(backend: str, model_path: str, max_new_tokens: int) -> BaseVLM:
    if backend == "mock":
        return _MOCK_VLM
    if backend == "qwen":
        normalized_path = model_path.strip()
        if not normalized_path:
            raise ValueError("Model path is required when backend is qwen.")
        return _load_qwen_backend(normalized_path, max_new_tokens)
    raise ValueError(f"Unsupported backend: {backend}")


def _raw_answer_for_display(result: InspectionResult) -> str:
    raw_answer = result.raw_model_answer or ""
    try:
        parsed: Any = json.loads(raw_answer)
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, TypeError):
        return raw_answer


def _result_to_dict(result: InspectionResult) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    return result.dict()


def _format_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _max_confidence(results: list[InspectionResult]) -> float:
    return max((result.confidence for result in results), default=0.0)


def _aggregate_global_local_result(
    global_result: InspectionResult,
    crop_results: list[dict[str, Any]],
) -> InspectionResult:
    anomalous_crops = [
        crop
        for crop in crop_results
        if isinstance(crop.get("result"), InspectionResult)
        and crop["result"].is_anomaly
    ]
    all_results = [global_result] + [
        crop["result"]
        for crop in crop_results
        if isinstance(crop.get("result"), InspectionResult)
    ]
    if anomalous_crops:
        first_result = anomalous_crops[0]["result"]
        defect_location = ", ".join(
            str(crop.get("region_name") or "unknown") for crop in anomalous_crops
        )
        defect_type = first_result.defect_type
        severity = first_result.severity
        is_anomaly = True
    else:
        defect_location = global_result.defect_location
        defect_type = global_result.defect_type
        severity = global_result.severity
        is_anomaly = global_result.is_anomaly

    if not is_anomaly:
        defect_type = "none"
        severity = "none"

    return InspectionResult(
        is_anomaly=is_anomaly,
        defect_type=defect_type,
        defect_location=defect_location,
        severity=severity,
        reason=(
            "global-local aggregation: final decision combines whole-image "
            "inspection and grid crop inspection."
        ),
        confidence=_max_confidence(all_results),
        raw_model_answer=None,
        sample_id="gradio-upload",
        parse_status="success",
    )


def _build_sample(
    *,
    image_path: Path,
    reference_image_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "sample_id": "gradio-upload",
        "image_path": str(image_path),
        "reference_image_path": str(reference_image_path) if reference_image_path else None,
        "question": "Inspect this image for industrial defects.",
        "options": [],
        "task_type": "Anomaly Detection",
        "object_category": "unknown",
    }


def _run_single_or_reference(
    *,
    vlm: BaseVLM,
    image_path: Path,
    reference_image_path: Path | None,
    prompt_type: str,
    detection_mode: str,
) -> tuple[InspectionResult, str]:
    effective_prompt_type = prompt_type
    warning = ""
    if detection_mode == "reference":
        effective_prompt_type = "reference_strict"
        if reference_image_path is None:
            warning = (
                "Reference image is missing; ran single-image fallback with "
                "reference_strict prompt downgraded by the prompt builder."
            )

    sample = _build_sample(
        image_path=image_path,
        reference_image_path=reference_image_path,
    )
    result = InspectorAgent(vlm, prompt_type=effective_prompt_type).inspect(sample)
    raw_json = _raw_answer_for_display(result)
    if warning:
        raw_json = _format_json(
            {
                "warning": warning,
                "raw_model_answer": result.raw_model_answer,
                "parsed_result": _result_to_dict(result),
            }
        )
    return result, raw_json


def _run_global_local(
    *,
    vlm: BaseVLM,
    image_path: Path,
    prompt_type: str,
    grid: str,
) -> tuple[InspectionResult, str]:
    agent = InspectorAgent(vlm, prompt_type=prompt_type)
    global_result = agent.inspect(_build_sample(image_path=image_path))
    crop_output_dir = _CROP_DIR / uuid4().hex
    crops = make_grid_crops(str(image_path), grid, str(crop_output_dir))
    crop_records: list[dict[str, Any]] = []

    for crop in crops:
        crop_result = agent.inspect(
            _build_sample(image_path=Path(str(crop["crop_path"])))
        )
        crop_records.append(
            {
                "crop_path": crop["crop_path"],
                "region_name": crop["region_name"],
                "box": crop["box"],
                "result": crop_result,
            }
        )

    final_result = _aggregate_global_local_result(global_result, crop_records)
    raw_json = _format_json(
        {
            "final_prediction": _result_to_dict(final_result),
            "global_prediction": _result_to_dict(global_result),
            "crop_predictions": [
                {
                    "crop_path": crop["crop_path"],
                    "region_name": crop["region_name"],
                    "box": crop["box"],
                    "prediction": _result_to_dict(crop["result"]),
                }
                for crop in crop_records
            ],
        }
    )
    return final_result, raw_json


def inspect_uploaded_image(
    image_path: str | None,
    model_path: str = "",
    backend: str = "mock",
    prompt_type: str = "strict_json",
    reference_image_path: str | None = None,
    detection_mode: str = "single",
    grid: str = "2x2",
    max_new_tokens: int = 512,
    report_output_path: str | None = None,
) -> tuple[str, str, str, str, float, str, str, float, str]:
    """Run one uploaded image through the shared InspectorAgent pipeline."""

    if not image_path:
        raise ValueError("Please upload an image before running inspection.")
    if detection_mode not in {"single", "reference", "global_local"}:
        raise ValueError("detection_mode must be one of: single, reference, global_local")
    max_new_tokens = int(max_new_tokens)
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")
    resolved_image = Path(image_path).expanduser().resolve()
    if not resolved_image.is_file():
        raise FileNotFoundError(f"Uploaded image does not exist: {resolved_image}")
    resolved_reference = None
    if reference_image_path:
        resolved_reference = Path(reference_image_path).expanduser().resolve()
        if not resolved_reference.is_file():
            resolved_reference = None

    vlm = _select_backend(backend, model_path, max_new_tokens)
    with _INFERENCE_LOCK:
        with Timer(synchronize_cuda=True) as timer:
            if detection_mode == "global_local":
                result, raw_json = _run_global_local(
                    vlm=vlm,
                    image_path=resolved_image,
                    prompt_type=prompt_type,
                    grid=grid,
                )
            else:
                result, raw_json = _run_single_or_reference(
                    vlm=vlm,
                    image_path=resolved_image,
                    reference_image_path=resolved_reference,
                    prompt_type=prompt_type,
                    detection_mode=detection_mode,
                )
    latency_sec = timer.elapsed_sec or 0.0

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
        raw_json,
        latency_sec,
        saved_report,
    )


def _run_with_ui_error(
    image_path: str | None,
    reference_image_path: str | None,
    model_path: str,
    backend: str,
    detection_mode: str,
    prompt_type: str,
    grid: str,
    max_new_tokens: int,
) -> tuple[str, str, str, str, float, str, str, float, str]:
    import gradio as gr

    try:
        return inspect_uploaded_image(
            image_path=image_path,
            reference_image_path=reference_image_path,
            model_path=model_path,
            backend=backend,
            detection_mode=detection_mode,
            prompt_type=prompt_type,
            grid=grid,
            max_new_tokens=max_new_tokens,
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
                    label="Test Image",
                    type="filepath",
                )
                reference_image_input = gr.Image(
                    label="Reference Image (optional)",
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
                detection_mode_input = gr.Dropdown(
                    choices=["single", "reference", "global_local"],
                    value="single",
                    label="Detection Mode",
                    info="reference uses the optional reference image; global_local runs whole-image plus grid crops.",
                )
                prompt_type_input = gr.Dropdown(
                    choices=["basic", "industrial", "strict_json", "reference_strict"],
                    value="strict_json",
                    label="Prompt Type",
                )
                grid_input = gr.Dropdown(
                    choices=["2x2", "3x3"],
                    value="2x2",
                    label="Grid",
                    info="Only used in global_local mode.",
                )
                max_new_tokens_input = gr.Number(
                    label="Max New Tokens",
                    value=512,
                    precision=0,
                    minimum=1,
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
                latency_output = gr.Number(label="Latency (sec)")
                report_image_output = gr.Image(
                    label="Inspection Report",
                    interactive=False,
                )

        run_button.click(
            fn=_run_with_ui_error,
            inputs=[
                image_input,
                reference_image_input,
                model_path_input,
                backend_input,
                detection_mode_input,
                prompt_type_input,
                grid_input,
                max_new_tokens_input,
            ],
            outputs=[
                anomaly_output,
                defect_type_output,
                defect_location_output,
                severity_output,
                confidence_output,
                reason_output,
                raw_json_output,
                latency_output,
                report_image_output,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue().launch()
