"""Generate a Markdown benchmark report from existing experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text.replace("|", "\\|").replace("\n", "<br>")
    if number != number:  # NaN
        return ""
    if abs(number) >= 1000 or (0 < abs(number) < 0.0001):
        return f"{number:.4g}"
    return f"{number:.4f}".rstrip("0").rstrip(".")


def _csv_to_markdown(path: Path) -> str:
    rows = _read_csv(path)
    if not rows:
        return "_No rows found in the provided CSV._"

    fieldnames = list(rows[0].keys())
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_format_value(row.get(field)) for field in fieldnames)
            + " |"
        )
    return "\n".join(lines)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_to_markdown(path: Path) -> str:
    data = _load_json(path)
    if not isinstance(data, dict):
        return "```json\n" + json.dumps(data, ensure_ascii=False, indent=2) + "\n```"

    preferred_keys = [
        "total_samples",
        "matched_case_count",
        "exported_case_count",
        "case_type_counts",
        "success_count",
        "error_count",
        "json_valid_rate",
        "binary_accuracy",
        "avg_latency_sec",
        "p95_latency_sec",
        "avg_confidence",
    ]
    rows = [
        (key, data[key])
        for key in preferred_keys
        if key in data
    ]
    if not rows:
        rows = list(data.items())[:20]

    lines = ["| field | value |", "|---|---|"]
    for key, value in rows:
        if isinstance(value, (dict, list)):
            rendered = "`" + json.dumps(value, ensure_ascii=False) + "`"
        else:
            rendered = _format_value(value)
        lines.append(f"| {key} | {rendered} |")
    return "\n".join(lines)


def _existing_file(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    return resolved if resolved.is_file() else None


def _section_from_optional_file(
    *,
    title: str,
    path: Path | None,
    missing_message: str,
) -> list[str]:
    lines = [f"## {title}", ""]
    existing = _existing_file(path)
    if existing is None:
        lines.extend([f"TODO: {missing_message}", ""])
        return lines

    lines.extend([f"Source: `{existing}`", ""])
    suffix = existing.suffix.lower()
    if suffix == ".csv":
        lines.extend([_csv_to_markdown(existing), ""])
    elif suffix == ".json":
        lines.extend([_json_to_markdown(existing), ""])
    elif suffix == ".md":
        lines.extend([existing.read_text(encoding="utf-8").strip(), ""])
    else:
        lines.extend(
            [
                "```text",
                existing.read_text(encoding="utf-8").strip(),
                "```",
                "",
            ]
        )
    return lines


def _observations(
    *,
    benchmark_summary: Path | None,
    prompt_ablation_summary: Path | None,
    infra_summary: Path | None,
    error_analysis: Path | None,
) -> list[str]:
    lines = ["## Observations", ""]
    observations: list[str] = []

    benchmark = _existing_file(benchmark_summary)
    if benchmark is not None and benchmark.suffix.lower() == ".csv":
        rows = _read_csv(benchmark)
        observations.append(
            f"Benchmark summary includes {len(rows)} compared method row(s)."
        )

    prompt = _existing_file(prompt_ablation_summary)
    if prompt is not None and prompt.suffix.lower() == ".csv":
        rows = _read_csv(prompt)
        observations.append(
            f"Prompt ablation summary includes {len(rows)} prompt configuration row(s)."
        )

    infra = _existing_file(infra_summary)
    if infra is not None and infra.suffix.lower() == ".csv":
        rows = _read_csv(infra)
        observations.append(
            f"Inference performance summary includes {len(rows)} max_new_tokens configuration row(s)."
        )

    errors = _existing_file(error_analysis)
    if errors is not None:
        observations.append(
            "Error analysis artifact is available; inspect exported cases before making qualitative claims."
        )

    if not observations:
        observations.append(
            "TODO: No completed experiment artifacts were provided for automatic observations."
        )

    lines.extend(f"- {observation}" for observation in observations)
    lines.append("")
    lines.append(
        "These observations only summarize existing artifact structure and values; they do not invent model performance."
    )
    lines.append("")
    return lines


def generate_benchmark_report(
    *,
    benchmark_summary: Path,
    output_path: Path,
    prompt_ablation_summary: Path | None = None,
    infra_summary: Path | None = None,
    error_analysis: Path | None = None,
) -> Path:
    """Generate a Markdown report from benchmark-related files."""

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Benchmark Report",
        "",
        "## Project Overview",
        "",
        "Industrial-VLM-Inspector is a training-free industrial visual inspection pipeline built around structured VLM inference, prompt variants, reference-based comparison, global-to-local crops, baselines, and evaluation utilities.",
        "",
        "## Dataset",
        "",
        "The project expects MMAD-style indexed samples. Dataset files and model weights are external artifacts and are not stored in this repository.",
        "",
        "## Compared Methods",
        "",
        "The report may include random/majority baselines, single-image VLM, reference-based VLM, and global-local VLM depending on the supplied benchmark summary.",
        "",
    ]

    lines.extend(
        _section_from_optional_file(
            title="Benchmark Results",
            path=benchmark_summary,
            missing_message="benchmark summary CSV is missing.",
        )
    )
    lines.extend(
        _section_from_optional_file(
            title="Prompt Ablation",
            path=prompt_ablation_summary,
            missing_message="prompt ablation summary was not provided or does not exist.",
        )
    )
    lines.extend(
        _section_from_optional_file(
            title="Inference Performance",
            path=infra_summary,
            missing_message="inference infrastructure summary was not provided or does not exist.",
        )
    )
    lines.extend(
        _section_from_optional_file(
            title="Error Analysis",
            path=error_analysis,
            missing_message="error analysis file was not provided or does not exist.",
        )
    )
    lines.extend(
        _observations(
            benchmark_summary=benchmark_summary,
            prompt_ablation_summary=prompt_ablation_summary,
            infra_summary=infra_summary,
            error_analysis=error_analysis,
        )
    )
    lines.extend(
        [
            "## Limitations",
            "",
            "- Report contents are generated from existing CSV/JSON/MD artifacts only.",
            "- Missing inputs are marked as TODO instead of being inferred.",
            "- Binary metrics depend on the label-mapping logic used when the input artifacts were generated.",
            "- Global-local mode reports grid-level aggregation, not pixel-level segmentation.",
            "- Prompt and model behavior may vary across model versions, decoding settings, and image resolution settings.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown benchmark report from existing artifacts."
    )
    parser.add_argument(
        "--benchmark-summary",
        type=Path,
        required=True,
        help="Benchmark summary CSV path.",
    )
    parser.add_argument(
        "--prompt-ablation-summary",
        type=Path,
        default=None,
        help="Optional prompt ablation summary CSV path.",
    )
    parser.add_argument(
        "--infra-summary",
        type=Path,
        default=None,
        help="Optional inference performance CSV/JSON path.",
    )
    parser.add_argument(
        "--error-analysis",
        type=Path,
        default=None,
        help="Optional error analysis JSON/Markdown path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmark_report.md"),
        help="Output Markdown report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = generate_benchmark_report(
        benchmark_summary=args.benchmark_summary,
        prompt_ablation_summary=args.prompt_ablation_summary,
        infra_summary=args.infra_summary,
        error_analysis=args.error_analysis,
        output_path=args.output,
    )
    print(f"Benchmark report written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
