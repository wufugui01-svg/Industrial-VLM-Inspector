"""Tests for reference image selection."""

from __future__ import annotations

import pytest

from src.agent.reference_selector import (
    ReferenceSelector,
    is_normal_sample,
    same_category,
)


def test_first_strategy_prefers_first_normal_same_category_candidate() -> None:
    current = {
        "sample_id": "current",
        "image_path": "/data/bottle/bad.png",
        "object_category": "bottle",
        "label": "abnormal",
    }
    samples = [
        current,
        {
            "sample_id": "wrong-category",
            "image_path": "/data/cable/good.png",
            "object_category": "cable",
            "label": "good",
        },
        {
            "sample_id": "bad-same-category",
            "image_path": "/data/bottle/crack.png",
            "object_category": "bottle",
            "label": "abnormal",
        },
        {
            "sample_id": "first-normal-bottle",
            "image_path": "/data/bottle/good-001.png",
            "object_category": "bottle",
            "label": "good",
        },
        {
            "sample_id": "second-normal-bottle",
            "image_path": "/data/bottle/good-002.png",
            "object_category": "bottle",
            "label": "normal",
        },
    ]

    selected = ReferenceSelector("first").select(current, samples)

    assert selected == "/data/bottle/good-001.png"


def test_random_strategy_selects_one_valid_normal_candidate() -> None:
    current = {
        "sample_id": "current",
        "image_path": "/data/capsule/bad.png",
        "object_category": "capsule",
    }
    samples = [
        current,
        {
            "sample_id": "normal-1",
            "image_path": "/data/capsule/good-001.png",
            "object_category": "capsule",
            "answer": "normal",
        },
        {
            "sample_id": "normal-2",
            "image_path": "/data/capsule/good-002.png",
            "object_category": "capsule",
            "answer": "good",
        },
    ]

    selected = ReferenceSelector("random", seed=7).select(current, samples)

    assert selected in {
        "/data/capsule/good-001.png",
        "/data/capsule/good-002.png",
    }
    assert selected != current["image_path"]


def test_similarity_strategy_falls_back_to_first_candidate() -> None:
    current = {
        "sample_id": "current",
        "image_path": "/data/grid/bad.png",
        "category": "grid",
    }
    samples = [
        current,
        {
            "sample_id": "normal-1",
            "image_path": "/data/grid/good-001.png",
            "category": "grid",
            "defect_type": "none",
        },
        {
            "sample_id": "normal-2",
            "image_path": "/data/grid/good-002.png",
            "category": "grid",
            "defect_type": "none",
        },
    ]

    selected = ReferenceSelector("similarity").select(current, samples)

    assert selected == "/data/grid/good-001.png"


def test_no_reference_candidate_returns_none() -> None:
    current = {
        "sample_id": "current",
        "image_path": "/data/bottle/bad.png",
        "object_category": "bottle",
    }
    samples = [
        current,
        {
            "sample_id": "abnormal",
            "image_path": "/data/bottle/crack.png",
            "object_category": "bottle",
            "label": "abnormal",
        },
    ]

    assert ReferenceSelector("first").select(current, samples) is None


def test_selector_does_not_select_current_sample_itself() -> None:
    current = {
        "sample_id": "current",
        "image_path": "/data/bottle/good-current.png",
        "object_category": "bottle",
        "label": "good",
    }
    samples = [
        current,
        {
            "sample_id": "other-normal",
            "image_path": "/data/bottle/good-other.png",
            "object_category": "bottle",
            "label": "good",
        },
    ]

    selected = ReferenceSelector("first").select(current, samples)

    assert selected == "/data/bottle/good-other.png"


def test_selector_returns_none_when_only_candidate_is_current_sample() -> None:
    current = {
        "sample_id": "current",
        "image_path": "/data/bottle/good-current.png",
        "object_category": "bottle",
        "label": "good",
    }

    assert ReferenceSelector("first").select(current, [current]) is None


@pytest.mark.parametrize(
    "sample",
    [
        {"answer": "normal"},
        {"answer": "No defect"},
        {"label": "good"},
        {"label": "ok"},
        {"defect_type": "none"},
        {"defect_type": "normal"},
        {"label": False},
        {"label": 0},
    ],
)
def test_is_normal_sample_accepts_normal_and_good_labels(sample: dict) -> None:
    assert is_normal_sample(sample) is True


@pytest.mark.parametrize(
    "sample",
    [
        {"answer": "A"},
        {"label": "abnormal"},
        {"defect_type": "crack"},
        {"label": True},
        {},
    ],
)
def test_is_normal_sample_rejects_unknown_or_abnormal_labels(sample: dict) -> None:
    assert is_normal_sample(sample) is False


def test_same_category_supports_object_category_and_category_fields() -> None:
    assert same_category(
        {"object_category": "Bottle"},
        {"category": "bottle"},
    )
    assert not same_category(
        {"object_category": "bottle"},
        {"category": "cable"},
    )


def test_unsupported_strategy_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unsupported reference strategy"):
        ReferenceSelector("nearest")  # type: ignore[arg-type]
