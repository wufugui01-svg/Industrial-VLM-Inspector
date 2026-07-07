"""Build a flat JSONL index from MMAD annotations."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

DEFAULT_QUESTION = ""
DEFAULT_ANSWER = ""
DEFAULT_TASK_TYPE = "unknown"
DEFAULT_OBJECT_CATEGORY = "unknown"


@dataclass
class IndexSummary:
    """Statistics produced while writing an MMAD index."""

    total_samples: int = 0
    missing_image_count: int = 0
    missing_image_sample_count: int = 0
    task_type_counts: Counter[str] = field(default_factory=Counter)
    first_samples: list[dict[str, Any]] = field(default_factory=list)


def _as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def normalize_options(value: Any) -> list[str]:
    """Normalize MMAD option dictionaries or strings to a JSON list."""

    if value is None:
        return []
    if isinstance(value, dict):
        return [f"{key}: {_as_text(text)}" for key, text in value.items()]
    if isinstance(value, (list, tuple)):
        return [_as_text(item) for item in value if _as_text(item)]
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    return [_as_text(value)] if _as_text(value) else []


def infer_object_category(image_reference: str) -> str:
    """Infer the object category from ``dataset/category/...`` paths."""

    normalized = image_reference.replace("\\", "/").strip("/")
    parts = [part for part in normalized.split("/") if part]
    return parts[1] if len(parts) >= 2 else DEFAULT_OBJECT_CATEGORY


def absolute_image_path(mmad_root: Path, image_reference: str) -> Path:
    """Resolve an annotation image reference against the MMAD root."""

    raw_path = Path(image_reference).expanduser()
    candidate = raw_path if raw_path.is_absolute() else mmad_root / raw_path
    return candidate.resolve(strict=False)


def _build_sample(
    *,
    sample_id: str,
    image_reference: str,
    annotation: dict[str, Any],
    mmad_root: Path,
    object_category: str | None = None,
    conversation_index: int | None = None,
) -> dict[str, Any]:
    image_path = absolute_image_path(mmad_root, image_reference)
    category = _as_text(object_category) or infer_object_category(image_reference)
    options = normalize_options(
        _first_present(annotation, "Options", "options")
    )
    answer = _as_text(
        _first_present(annotation, "Answer", "answer"),
        DEFAULT_ANSWER,
    )
    answer_text = ""
    for option in options:
        label, separator, text = option.partition(":")
        if separator and label.strip().casefold() == answer.casefold():
            answer_text = text.strip()
            break
    return {
        "sample_id": sample_id,
        "image_path": str(image_path),
        "image_relative_path": Path(image_reference).as_posix(),
        "image_exists": image_path.is_file(),
        "question": _as_text(
            _first_present(annotation, "Question", "question"),
            DEFAULT_QUESTION,
        ),
        "answer": answer,
        "answer_text": answer_text,
        "options": options,
        "task_type": _as_text(
            _first_present(annotation, "type", "task_type"),
            DEFAULT_TASK_TYPE,
        ),
        "object_category": category or DEFAULT_OBJECT_CATEGORY,
        "conversation_index": conversation_index,
    }


def iter_mmad_json(mmad_root: Path, annotation_path: Path) -> Iterator[dict[str, Any]]:
    """Yield one flat sample for each MMAD conversation item."""

    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {annotation_path}")

    sample_number = 0
    for image_reference, image_record in payload.items():
        if not isinstance(image_record, dict):
            continue

        conversations = _first_present(
            image_record, "conversation", "conversations", "questions"
        )
        if not isinstance(conversations, list):
            conversations = [image_record]

        category = _as_text(
            _first_present(image_record, "object_category", "category")
        )
        for conversation_index, conversation in enumerate(conversations):
            if not isinstance(conversation, dict):
                conversation = {}
            sample_number += 1
            yield _build_sample(
                sample_id=f"mmad_{sample_number:08d}",
                image_reference=_as_text(image_reference),
                annotation=conversation,
                mmad_root=mmad_root,
                object_category=category,
                conversation_index=conversation_index,
            )


def iter_metadata_csv(
    mmad_root: Path, annotation_path: Path
) -> Iterator[dict[str, Any]]:
    """Yield samples from the Hugging Face metadata CSV fallback."""

    with annotation_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=1):
            image_reference = _as_text(
                _first_present(row, "query_image", "image_path", "image")
            )
            yield _build_sample(
                sample_id=f"mmad_{row_number:08d}",
                image_reference=image_reference,
                annotation=row,
                mmad_root=mmad_root,
                object_category=_as_text(
                    _first_present(row, "object_category", "category")
                ),
                conversation_index=0,
            )


def iter_mmad_samples(mmad_root: Path) -> Iterator[dict[str, Any]]:
    """Read the preferred MMAD annotation source available under ``mmad_root``."""

    json_path = mmad_root / "mmad.json"
    csv_path = mmad_root / "metadata.csv"
    if json_path.is_file():
        yield from iter_mmad_json(mmad_root, json_path)
        return
    if csv_path.is_file():
        yield from iter_metadata_csv(mmad_root, csv_path)
        return
    raise FileNotFoundError(
        f"No supported MMAD annotation file found under {mmad_root}. "
        "Expected 'mmad.json' or 'metadata.csv'."
    )


def _limited(samples: Iterable[dict[str, Any]], limit: int | None) -> Iterable:
    if limit is None:
        yield from samples
        return
    if limit < 0:
        raise ValueError("--limit must be zero or greater")
    for index, sample in enumerate(samples):
        if index >= limit:
            break
        yield sample


def build_mmad_index(
    *,
    mmad_root: Path,
    output_path: Path,
    limit: int | None = None,
) -> IndexSummary:
    """Write an MMAD JSONL index and return build statistics."""

    mmad_root = mmad_root.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not mmad_root.is_dir():
        raise NotADirectoryError(f"MMAD root is not a directory: {mmad_root}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = IndexSummary()
    missing_paths: set[str] = set()

    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for sample in _limited(iter_mmad_samples(mmad_root), limit):
            output.write(json.dumps(sample, ensure_ascii=False) + "\n")
            summary.total_samples += 1
            summary.task_type_counts[sample["task_type"]] += 1
            if len(summary.first_samples) < 3:
                summary.first_samples.append(sample)
            if not sample["image_exists"]:
                summary.missing_image_sample_count += 1
                missing_paths.add(sample["image_path"])

    summary.missing_image_count = len(missing_paths)
    return summary
