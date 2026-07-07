"""Tests for project configuration loading."""

from src.utils.config import configured_random_seed, load_project_config


def test_default_config_is_loadable() -> None:
    config = load_project_config()

    assert config["project"]["name"] == "Industrial-VLM-Inspector"
    assert configured_random_seed() == 42
