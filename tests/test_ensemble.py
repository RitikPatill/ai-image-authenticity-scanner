"""Unit tests for ai_image_scanner.ensemble.

All three detectors are mocked — no model weights or calibration files needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from ai_image_scanner import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_image(size: int = 64, value: int = 128) -> Image.Image:
    arr = np.full((size, size, 3), value, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _make_detectors(freq_score: float, clip_score: float, hf_score: float) -> dict:
    """Build a mock detectors dict returning fixed scores."""
    freq = MagicMock()
    freq.score.return_value = freq_score  # FrequencyDetector returns bare float

    clip = MagicMock()
    clip.score.return_value = {"score": clip_score, "signal": "clip_drift"}

    hf = MagicMock()
    hf.score.return_value = {"score": hf_score, "signal": "hf_classifier"}

    return {"frequency": freq, "clip_drift": clip, "hf_classifier": hf}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_score_image_returns_required_keys() -> None:
    """score_image() result has 'score', 'verdict', and 'signals' keys."""
    from ai_image_scanner.ensemble import score_image

    dets = _make_detectors(0.4, 0.5, 0.6)
    result = score_image(_solid_image(), detectors=dets)

    assert "score" in result
    assert "verdict" in result
    assert "signals" in result


def test_signals_contains_all_three() -> None:
    """'signals' sub-dict contains frequency, clip_drift, hf_classifier."""
    from ai_image_scanner.ensemble import score_image

    dets = _make_detectors(0.3, 0.6, 0.7)
    result = score_image(_solid_image(), detectors=dets)

    assert "frequency" in result["signals"]
    assert "clip_drift" in result["signals"]
    assert "hf_classifier" in result["signals"]


def test_weighted_average_math() -> None:
    """Weighted average is computed correctly using config weights."""
    from ai_image_scanner.ensemble import score_image

    freq_s, clip_s, hf_s = 0.2, 0.6, 0.8
    dets = _make_detectors(freq_s, clip_s, hf_s)
    result = score_image(_solid_image(), detectors=dets)

    expected = (
        config.WEIGHT_FREQUENCY * freq_s
        + config.WEIGHT_CLIP * clip_s
        + config.WEIGHT_CLASSIFIER * hf_s
    )
    assert abs(result["score"] - expected) < 1e-6, (
        f"Expected {expected:.6f}, got {result['score']:.6f}"
    )


def test_verdict_ai_when_score_above_threshold() -> None:
    """verdict is 'AI' when weighted score >= 0.5."""
    from ai_image_scanner.ensemble import score_image

    # All signals at 1.0 → score = 1.0 → AI
    dets = _make_detectors(1.0, 1.0, 1.0)
    result = score_image(_solid_image(), detectors=dets)
    assert result["verdict"] == "AI"


def test_verdict_real_when_score_below_threshold() -> None:
    """verdict is 'REAL' when weighted score < 0.5."""
    from ai_image_scanner.ensemble import score_image

    # All signals at 0.0 → score = 0.0 → REAL
    dets = _make_detectors(0.0, 0.0, 0.0)
    result = score_image(_solid_image(), detectors=dets)
    assert result["verdict"] == "REAL"


def test_score_range() -> None:
    """Combined score is always in [0, 1]."""
    from ai_image_scanner.ensemble import score_image

    for vals in [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.3, 0.7, 0.5)]:
        dets = _make_detectors(*vals)
        s = score_image(_solid_image(), detectors=dets)["score"]
        assert 0.0 <= s <= 1.0, f"Score out of range for {vals}: {s}"


def test_individual_signal_scores_preserved() -> None:
    """Each signal's raw score is preserved in the 'signals' dict."""
    from ai_image_scanner.ensemble import score_image

    freq_s, clip_s, hf_s = 0.1, 0.4, 0.9
    dets = _make_detectors(freq_s, clip_s, hf_s)
    result = score_image(_solid_image(), detectors=dets)

    assert abs(result["signals"]["frequency"]["score"] - freq_s) < 1e-6
    assert abs(result["signals"]["clip_drift"]["score"] - clip_s) < 1e-6
    assert abs(result["signals"]["hf_classifier"]["score"] - hf_s) < 1e-6


def test_frequency_bare_float_wrapped() -> None:
    """FrequencyDetector returning a bare float is handled correctly."""
    from ai_image_scanner.ensemble import score_image

    # freq returns bare float (not a dict) — ensemble must wrap it
    dets = _make_detectors(0.5, 0.5, 0.5)
    # Verify no TypeError is raised
    result = score_image(_solid_image(), detectors=dets)
    assert isinstance(result["signals"]["frequency"]["score"], float)
