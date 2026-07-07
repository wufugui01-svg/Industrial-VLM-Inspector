# Industrial-VLM-Inspector

## 1. Project Overview

Industrial-VLM-Inspector is a training-free industrial visual-inspection
research prototype built around MMAD, a local Qwen3-VL model, structured JSON
output, and reproducible command-line workflows.

The repository contains code and tests only. MMAD data, model weights,
checkpoints, generated indexes, and experiment outputs are intentionally not
committed.

## 2. Features

- Lightweight MMAD directory inspection and JSONL indexing.
- Portable indexes containing absolute and dataset-relative image paths.
- Mock backend that runs without a GPU.
- Local-only Qwen3-VL inference; models are never downloaded automatically.
- Open-ended industrial inspection with Pydantic validation and JSON repair.
- Independent MMAD multiple-choice runner with exact option accuracy.
- Batch error isolation, synchronized latency, and peak CUDA-memory metrics.
- Reproducible prompt ablation with stratified sampling and a manifest.
- Gradio demo and report-image generation without fabricated bounding boxes.

## 3. Architecture

```text
MMAD annotations ──> JSONL index ──┬─> MMAD choice prompt ─> A/B/C/D metrics
                                   │
                                   └─> inspection prompt ──> Mock/Qwen VLM
                                                            │
                                                            v
                                               parser -> InspectionResult
                                                  │              │
                                                  v              v
                                              metrics       report/Gradio
```

The open-ended inspection pipeline and MMAD multiple-choice benchmark are
separate because their outputs and evaluation semantics are different.

## 4. Installation

```bash
cd /mnt/d/AIWork/projects/Industrial-VLM-Inspector
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

For Qwen, first install a PyTorch build compatible with the installed NVIDIA
driver by following the PyTorch selector. Then install:

```bash
python -m pip install -r requirements-qwen.txt
```

Check CUDA before loading the model:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 5. Dataset Preparation

MMAD is external and must be obtained and used under its upstream terms:

```bash
export MMAD_ROOT=/mnt/d/AIWork/datasets/MMAD
```

Expected files include `mmad.json`, `metadata.csv`, `domain_knowledge.json`,
and the extracted dataset directories `DS-MVTec`, `GoodsAD`, `MVTec-AD`,
`MVTec-LOCO`, and `VisA`.

Inspect file names and extensions without reading image contents:

```bash
python scripts/inspect_mmad_structure.py \
  --mmad-root "$MMAD_ROOT" \
  --max-files 20
```

The result is saved to `docs/mmad_structure_report.md`.

## 6. Build MMAD Index

```bash
python scripts/build_mmad_index.py \
  --mmad-root "$MMAD_ROOT" \
  --output data/processed/mmad_index_200.jsonl \
  --limit 200
```

Each row contains the question, options, answer label, answer text, absolute
image path, and `image_relative_path`. When an index moves between Windows and
WSL, pass `--dataset-root "$MMAD_ROOT"` to inference commands.

## 7. Single Image Inference

Mock smoke test:

```bash
python scripts/run_single_sample.py \
  --index data/processed/mmad_index_200.jsonl \
  --dataset-root "$MMAD_ROOT" \
  --sample-id "mmad_00000001" \
  --backend mock \
  --prompt-type strict_json \
  --save-report outputs/reports/mock_report.png
```

Qwen inference uses a generic local model path:

```bash
export QWEN_MODEL=/path/to/Qwen3-VL-4B-Instruct

python scripts/run_single_sample.py \
  --index data/processed/mmad_index_200.jsonl \
  --dataset-root "$MMAD_ROOT" \
  --sample-id "mmad_00000001" \
  --backend qwen \
  --model-path "$QWEN_MODEL" \
  --prompt-type strict_json \
  --max-new-tokens 128 \
  --max-pixels 1003520 \
  --save-report outputs/reports/qwen_report.png
```

`--device-map`, `--torch-dtype`, `--min-pixels`, and `--max-pixels` can be
used to control placement, precision, and vision-token cost.

## 8. Batch Inference

```bash
python scripts/run_batch_infer.py \
  --index data/processed/mmad_index_200.jsonl \
  --dataset-root "$MMAD_ROOT" \
  --output outputs/predictions/mock_predictions.jsonl \
  --backend mock \
  --prompt-type strict_json \
  --limit 50
```

Replace `mock` with `qwen` and supply `--model-path "$QWEN_MODEL"` for real
inference. Each row records pipeline latency, current memory, peak allocated
CUDA memory, parse status, and any per-sample error.

## 9. Evaluation

Evaluate open-ended inspection output:

```bash
python scripts/evaluate_predictions.py \
  --predictions outputs/predictions/mock_predictions.jsonl \
  --output outputs/metrics/mock_metrics.json
```

`parse_status=failed` is not counted as a successful inference or included in
binary accuracy. Schema validity, raw parse success, repair, and pipeline
errors are reported separately.

Run the actual MMAD option benchmark:

```bash
python scripts/run_mmad_benchmark.py \
  --index data/processed/mmad_index_200.jsonl \
  --dataset-root "$MMAD_ROOT" \
  --output outputs/predictions/mmad_qwen.jsonl \
  --metrics-output outputs/metrics/mmad_qwen.json \
  --backend qwen \
  --model-path "$QWEN_MODEL" \
  --max-new-tokens 32 \
  --limit 200
```

This runner compares the predicted option label directly with the annotated
label and reports overall coverage/accuracy plus task/category accuracy.

## 10. Prompt Ablation

```bash
python scripts/run_prompt_ablation.py \
  --index data/processed/mmad_index_200.jsonl \
  --dataset-root "$MMAD_ROOT" \
  --output-dir outputs/ablation/prompt_qwen \
  --backend qwen \
  --model-path "$QWEN_MODEL" \
  --max-new-tokens 128 \
  --seed 42 \
  --limit 50
```

The same stratified sample set is used for all three prompts. The output
directory contains predictions, metrics, `selected_samples.jsonl`,
`experiment_manifest.json`, and `prompt_ablation_summary.csv`.

## 11. Gradio Demo

```bash
python app/gradio_app.py
```

Open `http://127.0.0.1:7860`. Test `mock` first; for Qwen, select `qwen` and
enter a local model path. The demo caches one model and bounds temporary report
retention. It is a local research interface, not a public production service.

## 12. Example Results

```json
{
  "is_anomaly": true,
  "defect_type": "crack",
  "defect_location": "rim",
  "severity": "medium",
  "reason": "Visible discontinuity near the object rim.",
  "confidence": 0.82,
  "parse_status": "success"
}
```

This illustrates the schema only. It is not a measured result.

| Experiment | Samples | Accuracy | JSON parse rate | Status |
| --- | ---: | ---: | ---: | --- |
| Qwen3-VL MMAD benchmark | TODO | TODO | N/A | Not recorded |
| Inspection prompt ablation | TODO | N/A | TODO | Not recorded |
| Cross-dataset evaluation | TODO | TODO | TODO | Not implemented |

Ignored local smoke outputs are not treated as published experiments because
they lack a committed manifest and reproducible environment record.

## 13. Limitations

- The system is training-free VLM inspection, not a trained anomaly detector.
- Localization is textual, not pixel-level segmentation.
- Small or low-contrast defects may be missed.
- Output depends on prompt wording and model capability.
- Model confidence is not statistically calibrated.
- MMAD option accuracy and open-ended inspection quality are different tasks.
- The Gradio interface is not hardened for public deployment.

## 14. Roadmap

- [x] Separate MMAD option evaluation from open-ended inspection.
- [x] Add portable dataset-relative paths and reproducible prompt sampling.
- [ ] Validate the MMAD runner on the full official split and compare upstream.
- [ ] Add normal-reference and controlled multi-image experiments.
- [ ] Add calibrated confidence and systematic failure analysis.
- [ ] Add additional local VLM backends.
- [ ] Publish reproducible experiment manifests and measured tables.

## 15. Citation

Upstream resources:

- MMAD: <https://github.com/jam-cc/MMAD>
- MMAD paper: <https://arxiv.org/abs/2410.09453>
- Qwen3-VL: <https://github.com/QwenLM/Qwen3-VL>

Use the current citation and license text supplied by each upstream project.
This repository's source code is provided under the MIT License; datasets and
model weights retain their own terms.
