# Industrial-VLM-Inspector

## Project Overview

Industrial-VLM-Inspector is a research-oriented benchmark and agent system for
industrial anomaly inspection with vision-language models. It focuses on
training-free VLM inspection workflows over MMAD-style industrial anomaly
datasets, with structured JSON outputs, prompt variants, reference-image
comparison, global-to-local crop inspection, baseline methods, evaluation, error
analysis, and latency / memory benchmarking.

The repository contains source code, tests, and documentation only. The
following artifacts are intentionally not committed:

- MMAD dataset files
- local VLM model weights
- generated `outputs/`
- generated JSONL indexes under `data/processed/`
- checkpoints or tensor weight files

Resume-friendly description:

> Built an industrial anomaly detection benchmark and agent framework using
> Qwen3-VL-style vision-language models, supporting single-image inspection,
> reference-based comparison, global-to-local crop reasoning, baseline
> comparison, prompt ablation, JSON reliability evaluation, latency / memory
> profiling, Gradio visualization, and structured error analysis.

## Motivation

Industrial visual inspection systems need more than a simple demo page. A useful
VLM inspection project should answer several practical questions:

- Can the model reliably return machine-parseable JSON?
- How does single-image inspection compare with reference-image comparison?
- Does a global-to-local crop strategy expose defects missed by whole-image
  prompting?
- How do prompt variants affect JSON validity, latency, and failure modes?
- How do VLM methods compare with simple random / majority baselines?
- What are the model's common error cases?

This project is designed around those benchmark questions while keeping the
pipeline lightweight and reproducible.

## Key Features

- Single-image VLM inspection.
- Reference-based VLM inspection with optional normal reference image.
- Global-to-local VLM inspection with `2x2` or `3x3` grid crops.
- Mock backend for CPU-only smoke tests.
- Local Qwen3-VL backend through Transformers; model weights are never
  downloaded automatically.
- Random and majority baselines.
- Prompt ablation over `basic`, `industrial`, `strict_json`, and
  `reference_strict` prompts.
- Pydantic `InspectionResult` schema with JSON parsing and fallback handling.
- JSON reliability evaluation including valid-rate and parse-status counts.
- Binary metrics when labels can be mapped to normal / abnormal.
- Latency, process memory, and optional CUDA memory benchmark.
- Error-case analysis for false positives, false negatives, parse failures,
  low-confidence samples, and pipeline errors.
- Gradio interface for local qualitative inspection.
- Markdown benchmark report generation from existing artifacts.

## Method Overview

```text
MMAD / industrial dataset
        |
        v
JSONL sample index
        |
        +--> random / majority baselines
        |
        +--> single-image VLM agent
        |
        +--> reference-based VLM agent
        |
        +--> global-to-local VLM agent
                  |
                  +--> whole image prediction
                  +--> grid crop predictions
                  +--> simple final aggregation

All VLM outputs
        |
        v
JSON parser + Pydantic schema
        |
        +--> metrics
        +--> prompt ablation
        +--> infra benchmark
        +--> error analysis
        +--> Gradio / report image
```

The project has two related but distinct evaluation styles:

- Open-ended inspection output, normalized into `InspectionResult`.
- MMAD multiple-choice option prediction, evaluated directly against answer
  labels.

These should not be mixed as if they were the same metric.

## Project Structure

```text
Industrial-VLM-Inspector/
├── app/
│   └── gradio_app.py
├── configs/
│   └── default.yaml
├── data/
│   ├── processed/
│   └── samples/
├── docs/
│   ├── benchmark_report.md
│   └── mmad_structure_report.md
├── scripts/
│   ├── analyze_errors.py
│   ├── build_mmad_index.py
│   ├── evaluate_predictions.py
│   ├── generate_benchmark_report.py
│   ├── inspect_mmad_structure.py
│   ├── run_baselines.py
│   ├── run_batch_infer.py
│   ├── run_full_benchmark.py
│   ├── run_global_local_infer.py
│   ├── run_infra_benchmark.py
│   ├── run_mmad_benchmark.py
│   ├── run_prompt_ablation.py
│   ├── run_reference_infer.py
│   ├── run_single_sample.py
│   └── run_single_vs_reference.py
├── src/
│   ├── agent/
│   ├── baselines/
│   ├── datasets/
│   ├── eval/
│   ├── models/
│   ├── utils/
│   └── visualization/
└── tests/
```

## Installation

Create an environment from the project root:

```bash
cd /mnt/d/AIWork/projects/Industrial-VLM-Inspector

python -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m compileall src scripts app
pytest -q
```

For Qwen inference, install the optional dependencies after installing a PyTorch
build compatible with your NVIDIA driver:

```bash
python -m pip install -r requirements-qwen.txt
```

Check CUDA:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Dataset Preparation

MMAD is external. Download and extract it outside the repository:

```bash
export MMAD_ROOT=/mnt/d/AIWork/datasets/MMAD
```

The dataset is not committed to this repository. Inspect the dataset structure:

```bash
python scripts/inspect_mmad_structure.py \
  --mmad-root "$MMAD_ROOT" \
  --max-files 200
```

Build a JSONL index:

```bash
python scripts/build_mmad_index.py \
  --mmad-root "$MMAD_ROOT" \
  --output data/processed/mmad_index_200.jsonl \
  --limit 200
```

If an index contains Windows paths but you run in WSL, pass
`--dataset-root "$MMAD_ROOT"` to inference scripts so `image_relative_path` can
be resolved correctly.

## Quick Start

Run a mock smoke test without GPU or model weights:

```bash
python scripts/run_single_sample.py \
  --index data/processed/mmad_index_200.jsonl \
  --dataset-root "$MMAD_ROOT" \
  --backend mock \
  --prompt-type strict_json
```

Run a small mock batch:

```bash
python scripts/run_batch_infer.py \
  --index data/processed/mmad_index_200.jsonl \
  --dataset-root "$MMAD_ROOT" \
  --output outputs/predictions/mock_predictions.jsonl \
  --backend mock \
  --prompt-type strict_json \
  --limit 20
```

Evaluate the batch:

```bash
python scripts/evaluate_predictions.py \
  --predictions outputs/predictions/mock_predictions.jsonl \
  --output outputs/metrics/mock_metrics.json
```

## Single-image Inference

Set a local model path when using Qwen:

```bash
export QWEN_MODEL=/mnt/d/AIWork/projects/fiber_vlm_agent_project_v3_refactored_20260528/models/vlm/Qwen/Qwen3-VL-4B-Instruct
```

Mock:

```bash
python scripts/run_single_sample.py \
  --index data/processed/mmad_index_200.jsonl \
  --dataset-root "$MMAD_ROOT" \
  --sample-id "mmad_00000001" \
  --backend mock \
  --prompt-type strict_json \
  --save-report outputs/reports/mock_single.png
```

Qwen:

```bash
python scripts/run_single_sample.py \
  --index data/processed/mmad_index_200.jsonl \
  --dataset-root "$MMAD_ROOT" \
  --sample-id "mmad_00000001" \
  --backend qwen \
  --model-path "$QWEN_MODEL" \
  --prompt-type strict_json \
  --max-new-tokens 128 \
  --save-report outputs/reports/qwen_single.png
```

## Reference-based Inference

Reference-based inference selects a normal / good sample from the same category
when possible, writes `reference_image_path`, and runs the VLM agent with a
reference-aware prompt.

Mock:

```bash
python scripts/run_reference_infer.py \
  --index data/processed/mmad_index_200.jsonl \
  --output outputs/predictions/reference_mock.jsonl \
  --backend mock \
  --prompt-type reference_strict \
  --reference-strategy first \
  --limit 20
```

Qwen:

```bash
python scripts/run_reference_infer.py \
  --index data/processed/mmad_index_200.jsonl \
  --output outputs/predictions/reference_qwen.jsonl \
  --backend qwen \
  --model-path "$QWEN_MODEL" \
  --prompt-type reference_strict \
  --reference-strategy first \
  --max-new-tokens 128 \
  --limit 20
```

## Global-to-local Inference

Global-to-local inference first inspects the whole image, then crops the image
into a grid and inspects each crop. The final prediction is a simple aggregation
over whole-image and crop predictions. It is not pixel-level segmentation.

Mock:

```bash
python scripts/run_global_local_infer.py \
  --index data/processed/mmad_index_200.jsonl \
  --dataset-root "$MMAD_ROOT" \
  --output outputs/predictions/global_local_mock.jsonl \
  --backend mock \
  --prompt-type strict_json \
  --grid 2x2 \
  --crop-dir outputs/crops \
  --limit 10
```

Qwen:

```bash
python scripts/run_global_local_infer.py \
  --index data/processed/mmad_index_200.jsonl \
  --dataset-root "$MMAD_ROOT" \
  --output outputs/predictions/global_local_qwen.jsonl \
  --backend qwen \
  --model-path "$QWEN_MODEL" \
  --prompt-type strict_json \
  --grid 2x2 \
  --crop-dir outputs/crops \
  --max-new-tokens 128 \
  --limit 10
```

## Full Benchmark

The full benchmark runs:

- random baseline
- majority baseline
- single-image VLM
- reference-based VLM
- global-to-local VLM

Mock:

```bash
python scripts/run_full_benchmark.py \
  --index data/processed/mmad_index_200.jsonl \
  --output-dir outputs/benchmark_mock \
  --backend mock \
  --limit 50 \
  --grid 2x2
```

Qwen:

```bash
python scripts/run_full_benchmark.py \
  --index data/processed/mmad_index_200.jsonl \
  --output-dir outputs/benchmark_qwen \
  --backend qwen \
  --model-path "$QWEN_MODEL" \
  --limit 50 \
  --max-new-tokens 128 \
  --grid 2x2
```

The script writes per-method predictions, metrics, and
`benchmark_summary.csv`.

## Prompt Ablation

Prompt ablation runs:

- `basic` in single-image mode
- `industrial` in single-image mode
- `strict_json` in single-image mode
- `reference_strict` in reference-based mode

Mock:

```bash
python scripts/run_prompt_ablation.py \
  --index data/processed/mmad_index_200.jsonl \
  --dataset-root "$MMAD_ROOT" \
  --output-dir outputs/ablation/prompt_mock \
  --backend mock \
  --limit 50 \
  --max-new-tokens 128
```

Qwen:

```bash
python scripts/run_prompt_ablation.py \
  --index data/processed/mmad_index_200.jsonl \
  --dataset-root "$MMAD_ROOT" \
  --output-dir outputs/ablation/prompt_qwen \
  --backend qwen \
  --model-path "$QWEN_MODEL" \
  --limit 50 \
  --max-new-tokens 128
```

The output directory contains per-prompt predictions, metrics,
`selected_samples.jsonl`, `experiment_manifest.json`, and
`prompt_ablation_summary.csv`.

## Inference Benchmark

Infrastructure benchmarking measures latency, throughput, process memory, and
optional CUDA memory across `max_new_tokens` configurations.

Mock:

```bash
python scripts/run_infra_benchmark.py \
  --index data/processed/mmad_index_200.jsonl \
  --output outputs/infra_benchmark/mock_infra.csv \
  --backend mock \
  --limit 10 \
  --max-new-tokens-list 64,128,256 \
  --prompt-type strict_json
```

Qwen:

```bash
python scripts/run_infra_benchmark.py \
  --index data/processed/mmad_index_200.jsonl \
  --dataset-root "$MMAD_ROOT" \
  --output outputs/infra_benchmark/qwen_infra.csv \
  --backend qwen \
  --model-path "$QWEN_MODEL" \
  --limit 10 \
  --max-new-tokens-list 64,128,256 \
  --prompt-type strict_json
```

This script does not require `nvidia-smi`. If CUDA is unavailable, GPU memory
fields are left empty.

## Error Analysis

Analyze failed or risky prediction records:

```bash
python scripts/analyze_errors.py \
  --predictions outputs/predictions/qwen_predictions.jsonl \
  --output-dir outputs/error_analysis/qwen \
  --max-cases 50
```

The script exports:

- `error_cases.jsonl`
- `error_summary.json`
- `error_analysis.md`
- copied images under `images/` when paths exist

It flags false positives, false negatives, parse failures, low-confidence
records, and records with an `error` field. False positive / false negative
labels are only as reliable as the available normal / abnormal mapping.

## Gradio Demo

Start the local demo:

```bash
python app/gradio_app.py
```

Open:

```text
http://127.0.0.1:7860
```

Inputs:

- test image
- optional reference image
- backend: `mock` or `qwen`
- local `model_path`
- detection mode: `single`, `reference`, or `global_local`
- prompt type: `basic`, `industrial`, `strict_json`, or `reference_strict`
- grid: `2x2` or `3x3`
- max new tokens

The Qwen model is loaded only after clicking **Run Inspection** with
`backend=qwen` and a valid `model_path`. It is not loaded when the page starts.

## Results

This repository does not include datasets, model weights, or generated outputs.
Therefore, committed README numbers are intentionally limited.

Current result tables should be generated from local artifacts:

```bash
python scripts/generate_benchmark_report.py \
  --benchmark-summary outputs/benchmark_qwen/metrics/benchmark_summary.csv \
  --prompt-ablation-summary outputs/ablation/prompt_qwen/prompt_ablation_summary.csv \
  --infra-summary outputs/infra_benchmark/qwen_infra.csv \
  --error-analysis outputs/error_analysis/qwen/error_analysis.md \
  --output docs/benchmark_report.md
```

If any input file is missing, the report generator writes `TODO` instead of
inventing numbers.

Measured Qwen benchmark results: TODO.

Measured full-dataset MMAD option benchmark results: TODO.

## Limitations

- The system is training-free; it does not train or fine-tune an anomaly
  detector.
- Global-to-local output is grid-level aggregation, not pixel-level
  segmentation.
- Textual `defect_location` should not be interpreted as a bounding box or mask.
- Small, subtle, or low-contrast defects may be missed.
- Defect type classification depends heavily on VLM capability and prompt
  wording.
- Confidence is model-reported and not calibrated.
- Binary metrics can be misleading when dataset answer options change order.
- Reference selection currently uses simple metadata heuristics, not learned
  visual similarity.
- The Gradio app is a local research interface, not a production service.

## Roadmap

- [x] MMAD structure inspection and index building.
- [x] Mock backend for no-GPU pipeline testing.
- [x] Qwen3-VL local backend.
- [x] Structured JSON schema and parser repair path.
- [x] Single-image, reference-based, and global-to-local inference scripts.
- [x] Random and majority baselines.
- [x] Prompt ablation runner.
- [x] Latency / memory benchmark.
- [x] Error analysis and report generation.
- [ ] Improve MMAD normal / abnormal label mapping using `answer_text` and
  `options`.
- [ ] Add stronger reference selection beyond simple metadata heuristics.
- [ ] Add more VLM backends.
- [ ] Add calibrated confidence analysis.
- [ ] Run and document larger reproducible Qwen experiments with manifests.

