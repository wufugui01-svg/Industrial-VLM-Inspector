"""Tests for benchmark report generation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts.generate_benchmark_report import generate_benchmark_report


def test_generate_benchmark_report_uses_existing_artifacts() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        benchmark = root / "benchmark_summary.csv"
        prompt = root / "prompt_ablation_summary.csv"
        infra = root / "infra.csv"
        errors = root / "error_summary.json"
        output = root / "benchmark_report.md"

        benchmark.write_text(
            "method,total_samples,json_valid_rate,error_count\n"
            "single_vlm,2,1.0,0\n",
            encoding="utf-8",
        )
        prompt.write_text(
            "prompt_type,mode,total_samples,json_valid_rate\n"
            "strict_json,single,2,1.0\n",
            encoding="utf-8",
        )
        infra.write_text(
            "backend,max_new_tokens,total_samples,avg_latency_sec\n"
            "mock,128,2,0.01\n",
            encoding="utf-8",
        )
        errors.write_text(
            json.dumps(
                {
                    "total_samples": 2,
                    "matched_case_count": 1,
                    "case_type_counts": {"false_negative": 1},
                }
            ),
            encoding="utf-8",
        )

        returned_path = generate_benchmark_report(
            benchmark_summary=benchmark,
            prompt_ablation_summary=prompt,
            infra_summary=infra,
            error_analysis=errors,
            output_path=output,
        )

        report = output.read_text(encoding="utf-8")

    assert returned_path == output.resolve()
    assert "## Project Overview" in report
    assert "## Dataset" in report
    assert "## Compared Methods" in report
    assert "## Benchmark Results" in report
    assert "| method | total_samples | json_valid_rate | error_count |" in report
    assert "| single_vlm | 2 | 1 | 0 |" in report
    assert "## Prompt Ablation" in report
    assert "| strict_json | single | 2 | 1 |" in report
    assert "## Inference Performance" in report
    assert "| mock | 128 | 2 | 0.01 |" in report
    assert "## Error Analysis" in report
    assert "`{\"false_negative\": 1}`" in report
    assert "## Observations" in report
    assert "## Limitations" in report


def test_generate_benchmark_report_marks_missing_inputs_as_todo() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        output = root / "benchmark_report.md"

        generate_benchmark_report(
            benchmark_summary=root / "missing_benchmark.csv",
            prompt_ablation_summary=root / "missing_prompt.csv",
            infra_summary=root / "missing_infra.csv",
            error_analysis=root / "missing_error.md",
            output_path=output,
        )
        report = output.read_text(encoding="utf-8")

    assert "TODO: benchmark summary CSV is missing." in report
    assert "TODO: prompt ablation summary was not provided or does not exist." in report
    assert "TODO: inference infrastructure summary was not provided or does not exist." in report
    assert "TODO: error analysis file was not provided or does not exist." in report
