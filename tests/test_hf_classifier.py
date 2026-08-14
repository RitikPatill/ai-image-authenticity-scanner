"""Unit tests for ai_image_scanner.detectors.hf_classifier.

All network calls are mocked — no model weights are downloaded during testing.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from PIL import Image

from ai_image_scanner.detectors.hf_classifier import HFClassifierDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_image(size: int = 64, value: int = 128) -> Image.Image:
    arr = np.full((size, size, 3), value, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _make_mock_pipe(label: str, confidence: float) -> MagicMock:
    """Return a mock pipeline that yields a fixed classification result."""
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"label": label, "score": confidence}]
    return mock_pipe


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_score_ai_label_high_score() -> None:
    """label_1 (AI) with confidence 0.9 → score ≈ 0.9."""
    with patch("ai_image_scanner.detectors.hf_classifier.pipeline") as mock_pipeline:
        mock_pipeline.return_value = _make_mock_pipe("label_1", 0.9)
        det = HFClassifierDetector()
        result = det.score(_solid_image())

    assert result["signal"] == "hf_classifier"
    assert abs(result["score"] - 0.9) < 1e-6


def test_score_real_label_low_score() -> None:
    """label_0 (real) with confidence 0.85 → fallback 0.5 (unknown label)."""
    with patch("ai_image_scanner.detectors.hf_classifier.pipeline") as mock_pipeline:
        mock_pipeline.return_value = _make_mock_pipe("label_0", 0.85)
        det = HFClassifierDetector()
        result = det.score(_solid_image())

    assert result["signal"] == "hf_classifier"
    # label_0 is not in _AI_LABELS → fallback to 0.5
    assert abs(result["score"] - 0.5) < 1e-6


def test_score_artificial_label() -> None:
    """'artificial' label (Organika/sdxl-detector) maps to AI score."""
    with patch("ai_image_scanner.detectors.hf_classifier.pipeline") as mock_pipeline:
        mock_pipeline.return_value = _make_mock_pipe("artificial", 0.8)
        det = HFClassifierDetector()
        result = det.score(_solid_image())

    assert result["signal"] == "hf_classifier"
    assert abs(result["score"] - 0.8) < 1e-6


def test_score_in_range() -> None:
    """score() returns a float in [0, 1] for various mock outputs."""
    cases = [
        ("label_1", 0.0),
        ("label_1", 1.0),
        ("label_1", 0.5),
        ("artificial", 0.0),
        ("artificial", 1.0),
    ]
    for label, confidence in cases:
        with patch("ai_image_scanner.detectors.hf_classifier.pipeline") as mock_pipeline:
            mock_pipeline.return_value = _make_mock_pipe(label, confidence)
            det = HFClassifierDetector()
            s = det.score(_solid_image())["score"]

        assert isinstance(s, float), f"Expected float, got {type(s)}"
        assert 0.0 <= s <= 1.0, f"Score out of range for ({label}, {confidence}): {s}"


def test_score_returns_dict() -> None:
    """score() returns a dict with 'score' and 'signal' keys."""
    with patch("ai_image_scanner.detectors.hf_classifier.pipeline") as mock_pipeline:
        mock_pipeline.return_value = _make_mock_pipe("label_1", 0.7)
        det = HFClassifierDetector()
        result = det.score(_solid_image())

    assert isinstance(result, dict)
    assert "score" in result
    assert "signal" in result


def test_pipeline_loaded_lazily() -> None:
    """Pipeline is NOT created at __init__ time, only on first score() call."""
    with patch("ai_image_scanner.detectors.hf_classifier.pipeline") as mock_pipeline:
        mock_pipeline.return_value = _make_mock_pipe("label_1", 0.6)
        det = HFClassifierDetector()
        mock_pipeline.assert_not_called()  # not yet loaded
        det.score(_solid_image())
        mock_pipeline.assert_called_once()  # loaded on first score()


def test_pipeline_reused_across_calls() -> None:
    """Pipeline is instantiated only once across multiple score() calls."""
    with patch("ai_image_scanner.detectors.hf_classifier.pipeline") as mock_pipeline:
        mock_pipeline.return_value = _make_mock_pipe("label_1", 0.6)
        det = HFClassifierDetector()
        det.score(_solid_image())
        det.score(_solid_image(value=200))
        mock_pipeline.assert_called_once()
