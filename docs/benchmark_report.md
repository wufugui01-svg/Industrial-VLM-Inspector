# Benchmark Report

## Project Overview

Industrial-VLM-Inspector is a training-free industrial visual inspection pipeline built around structured VLM inference, prompt variants, reference-based comparison, global-to-local crops, baselines, and evaluation utilities.

## Dataset

The project expects MMAD-style indexed samples. Dataset files and model weights are external artifacts and are not stored in this repository.

## Compared Methods

The report may include random/majority baselines, single-image VLM, reference-based VLM, and global-local VLM depending on the supplied benchmark summary.

## Benchmark Results

TODO: benchmark summary CSV is missing.

## Prompt Ablation

Source: `/mnt/d/AIWork/projects/Industrial-VLM-Inspector/outputs/ablation/final_prompt_mock_50/prompt_ablation_summary.csv`

| prompt_type | mode | total_samples | json_valid_rate | binary_accuracy | avg_latency_sec | p95_latency_sec | avg_confidence | error_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| basic | single | 50 | 1 | 0.4667 | 4.316e-05 | 7.408e-05 | 0.5 | 0 |
| industrial | single | 50 | 1 | 0.4667 | 3.747e-05 | 6.564e-05 | 0.5 | 0 |
| strict_json | single | 50 | 1 | 0.4667 | 3.679e-05 | 6.476e-05 | 0.5 | 0 |
| reference_strict | reference | 50 | 1 | 0.4667 | 2.208e-05 | 3.152e-05 | 0.5 | 0 |

## Inference Performance

Source: `/mnt/d/AIWork/projects/Industrial-VLM-Inspector/outputs/metrics/final_infra_benchmark_mock.csv`

| backend | model_path | max_new_tokens | total_samples | avg_latency_sec | p50_latency_sec | p95_latency_sec | throughput_samples_per_sec | max_gpu_allocated_mb | max_gpu_reserved_mb | process_memory_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mock |  | 64 | 50 | 2.946e-05 | 2.483e-05 | 5.237e-05 | 1274 | 0 | 0 | 656.1484 |
| mock |  | 128 | 50 | 3.249e-05 | 2.622e-05 | 6.827e-05 | 3944 | 0 | 0 | 656.1836 |
| mock |  | 256 | 50 | 2.861e-05 | 2.366e-05 | 5.856e-05 | 4196 | 0 | 0 | 656.1953 |

## Error Analysis

TODO: error analysis file was not provided or does not exist.

## Observations

- Prompt ablation summary includes 4 prompt configuration row(s).
- Inference performance summary includes 3 max_new_tokens configuration row(s).

These observations only summarize existing artifact structure and values; they do not invent model performance.

## Limitations

- Report contents are generated from existing CSV/JSON/MD artifacts only.
- Missing inputs are marked as TODO instead of being inferred.
- Binary metrics depend on the label-mapping logic used when the input artifacts were generated.
- Global-local mode reports grid-level aggregation, not pixel-level segmentation.
- Prompt and model behavior may vary across model versions, decoding settings, and image resolution settings.
