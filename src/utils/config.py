"""Project YAML configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_project_config(path: Path | None = None) -> dict[str, Any]:
    """Load a YAML mapping from the project default or an explicit path."""

    config_path = (path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return payload


def configured_random_seed(default: int = 42) -> int:
    """Read the configured seed with a safe integer fallback."""

    value = load_project_config().get("project", {}).get("random_seed", default)
    return int(value)
