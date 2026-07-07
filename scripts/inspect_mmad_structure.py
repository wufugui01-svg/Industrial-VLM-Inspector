"""Inspect an MMAD directory without reading or modifying dataset contents."""

from __future__ import annotations

import argparse
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
ANNOTATION_EXTENSIONS = {".json", ".jsonl", ".csv", ".txt"}
RANDOM_SEED = 42


@dataclass
class ScanResult:
    """Lightweight metadata collected from directory entries."""

    root: Path
    first_level_dirs: list[str] = field(default_factory=list)
    second_level_dirs: dict[str, list[str]] = field(default_factory=dict)
    files_by_extension: Counter[str] = field(default_factory=Counter)
    image_count: int = 0
    annotation_count: int = 0
    total_file_count: int = 0
    image_samples: list[str] = field(default_factory=list)
    annotation_samples: list[str] = field(default_factory=list)
    scan_errors: list[str] = field(default_factory=list)


def _relative_posix(path: Path, root: Path) -> str:
    """Return a stable, Markdown-friendly path relative to the dataset root."""

    return path.relative_to(root).as_posix()


def _reservoir_add(
    samples: list[str],
    item: str,
    seen_count: int,
    limit: int,
    rng: random.Random,
) -> None:
    """Uniformly retain at most ``limit`` examples without storing every path."""

    if limit == 0:
        return
    if len(samples) < limit:
        samples.append(item)
        return

    replacement_index = rng.randrange(seen_count)
    if replacement_index < limit:
        samples[replacement_index] = item


def _inspect_directory_levels(root: Path, result: ScanResult) -> None:
    """Collect directory names from only the first two levels."""

    try:
        first_level = sorted(
            (entry for entry in root.iterdir() if entry.is_dir()),
            key=lambda path: path.name.casefold(),
        )
    except OSError as exc:
        result.scan_errors.append(f"Cannot inspect dataset root: {exc}")
        return

    result.first_level_dirs = [path.name for path in first_level]
    for directory in first_level:
        try:
            children = sorted(
                (entry.name for entry in directory.iterdir() if entry.is_dir()),
                key=str.casefold,
            )
        except OSError as exc:
            result.scan_errors.append(
                f"Cannot inspect second-level directories under "
                f"{_relative_posix(directory, root)}: {exc}"
            )
            children = []
        result.second_level_dirs[directory.name] = children


def inspect_mmad(root: Path, max_files: int) -> ScanResult:
    """Recursively count files using names and extensions only.

    ``max_files`` bounds the number of randomized example paths retained for
    each supported category. It does not truncate the recursive file counts.
    """

    if max_files < 0:
        raise ValueError("--max-files must be zero or greater")

    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"MMAD root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"MMAD root is not a directory: {root}")

    result = ScanResult(root=root)
    _inspect_directory_levels(root, result)
    rng = random.Random(RANDOM_SEED)
    image_seen = 0
    annotation_seen = 0

    def record_walk_error(exc: OSError) -> None:
        result.scan_errors.append(str(exc))

    for current_dir, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False, onerror=record_walk_error
    ):
        directory_names.sort(key=str.casefold)
        file_names.sort(key=str.casefold)
        current_path = Path(current_dir)

        for file_name in file_names:
            path = current_path / file_name
            extension = path.suffix.casefold()
            result.total_file_count += 1
            result.files_by_extension[extension or "[no extension]"] += 1
            relative_path = _relative_posix(path, root)

            if extension in IMAGE_EXTENSIONS:
                image_seen += 1
                result.image_count += 1
                _reservoir_add(
                    result.image_samples,
                    relative_path,
                    image_seen,
                    max_files,
                    rng,
                )

            if extension in ANNOTATION_EXTENSIONS:
                annotation_seen += 1
                result.annotation_count += 1
                _reservoir_add(
                    result.annotation_samples,
                    relative_path,
                    annotation_seen,
                    max_files,
                    rng,
                )

    rng.shuffle(result.image_samples)
    rng.shuffle(result.annotation_samples)
    return result


def _markdown_path_list(paths: list[str]) -> list[str]:
    if not paths:
        return ["- None found."]
    return [f"- `{path}`" for path in paths]


def build_markdown_report(result: ScanResult, max_files: int) -> str:
    """Render scan metadata as a Markdown report."""

    lines = [
        "# MMAD Dataset Structure Report",
        "",
        f"- Dataset root: `{result.root}`",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        "- Scan mode: file names and extensions only; file contents were not read.",
        f"- Maximum retained example paths per category: `{max_files}`",
        f"- Total files discovered: `{result.total_file_count}`",
        f"- Supported image files: `{result.image_count}`",
        f"- Supported annotation/text files: `{result.annotation_count}`",
        "",
        "## First- and second-level directory structure",
        "",
    ]

    if not result.first_level_dirs:
        lines.append("- No first-level directories found.")
    else:
        for first_level in result.first_level_dirs:
            lines.append(f"- `{first_level}/`")
            children = result.second_level_dirs.get(first_level, [])
            if children:
                lines.extend(f"  - `{child}/`" for child in children)
            else:
                lines.append("  - _(no second-level directories)_")

    lines.extend(
        [
            "",
            "## File counts by supported extension",
            "",
            "| Extension | Count |",
            "| --- | ---: |",
        ]
    )
    for extension in sorted(IMAGE_EXTENSIONS | ANNOTATION_EXTENSIONS):
        lines.append(f"| `{extension}` | {result.files_by_extension[extension]} |")

    lines.extend(
        [
            "",
            "## All discovered extensions",
            "",
            "| Extension | Count |",
            "| --- | ---: |",
        ]
    )
    for extension, count in sorted(
        result.files_by_extension.items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(f"| `{extension}` | {count} |")

    lines.extend(["", "## Random image path examples", ""])
    lines.extend(_markdown_path_list(result.image_samples))
    lines.extend(["", "## Random annotation/text path examples", ""])
    lines.extend(_markdown_path_list(result.annotation_samples))

    lines.extend(["", "## Scan warnings", ""])
    if result.scan_errors:
        lines.extend(f"- {error}" for error in result.scan_errors)
    else:
        lines.append("- None.")

    if result.image_count == 0 and result.files_by_extension[".zip"] > 0:
        lines.extend(
            [
                "",
                "## Observation",
                "",
                "No supported image files were found, while ZIP archives are present. "
                "The dataset appears to be downloaded but not extracted "
                "under this root.",
            ]
        )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect an MMAD dataset directory without reading file contents."
    )
    parser.add_argument(
        "--mmad-root",
        type=Path,
        required=True,
        help="Path to the local MMAD dataset root.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=200,
        help="Maximum randomized example paths retained per category (default: 200).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = inspect_mmad(args.mmad_root, args.max_files)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    project_root = Path(__file__).resolve().parents[1]
    report_path = project_root / "docs" / "mmad_structure_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_markdown_report(result, args.max_files), encoding="utf-8"
    )

    print(f"Scanned: {result.root}")
    print(f"Total files: {result.total_file_count}")
    print(f"Image files: {result.image_count}")
    print(f"Annotation/text files: {result.annotation_count}")
    print("Random image examples:")
    for path in result.image_samples:
        print(f"  {path}")
    print("Random annotation/text examples:")
    for path in result.annotation_samples:
        print(f"  {path}")
    print(f"Report written to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
