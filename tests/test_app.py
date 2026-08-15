"""Smoke tests for app.py.

All tests mock ai_image_scanner.ensemble.score_image so no model weights
or calibration files are needed. build_ui() is never called.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import matplotlib.figure
import pytest
from PIL import Image

# Import only the public helpers — this must NOT trigger build_ui()
from app import _make_bar_chart, predict

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_FAKE_RESULT = {
    "score": 0.82,
    "verdict": "AI",
    "signals": {
        "frequency": {"score": 0.75},
        "clip_drift": {"score": 0.88},
        "hf_classifier": {"score": 0.83},
    },
}

_FAKE_RESULT_REAL = {
    "score": 0.29,
    "verdict": "REAL",
    "signals": {
        "frequency": {"score": 0.20},
        "clip_drift": {"score": 0.35},
        "hf_classifier": {"score": 0.31},
    },
}


def _solid_image(size: int = 64, color: int = 128) -> Image.Image:
    return Image.new("RGB", (size, size), (color, color, color))


# ---------------------------------------------------------------------------
# Test 1 — predict with uploaded image
# ---------------------------------------------------------------------------

def test_predict_with_image():
    img = _solid_image()
    with patch("ai_image_scanner.ensemble.score_image", return_value=_FAKE_RESULT) as mock_score:
        label, fig, result = predict(image=img, url="")

    mock_score.assert_called_once()
    assert "AI" in label or "REAL" in label
    assert isinstance(fig, matplotlib.figure.Figure)
    assert set(result.keys()) >= {"score", "verdict", "signals"}


# ---------------------------------------------------------------------------
# Test 2 — URL takes priority over uploaded image
# ---------------------------------------------------------------------------

def test_predict_prefers_url_over_image():
    img = _solid_image()

    # Build minimal PNG bytes to return from the fake HTTP response
    buf = io.BytesIO()
    _solid_image(color=200).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    mock_response = MagicMock()
    mock_response.content = png_bytes
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.get", return_value=mock_response) as mock_get, \
         patch("ai_image_scanner.ensemble.score_image", return_value=_FAKE_RESULT) as mock_score:
        label, fig, result = predict(image=img, url="https://example.com/img.png")

    mock_get.assert_called_once()
    mock_score.assert_called_once()
    # The image passed to score_image should come from URL bytes (200,200,200),
    # not from the uploaded solid-128 image.
    called_img: Image.Image = mock_score.call_args[0][0]
    pixel = called_img.getpixel((0, 0))
    assert pixel == (200, 200, 200), f"Expected URL-sourced pixel (200,200,200), got {pixel}"


# ---------------------------------------------------------------------------
# Test 3 — no input raises an error
# ---------------------------------------------------------------------------

def test_predict_no_input_raises():
    import gradio as gr

    with pytest.raises((gr.Error, ValueError)):
        predict(image=None, url="")


# ---------------------------------------------------------------------------
# Test 4 — bar chart returns a Figure
# ---------------------------------------------------------------------------

def test_bar_chart_colors():
    signals = {
        "frequency": {"score": 0.8},
        "clip_drift": {"score": 0.3},
        "hf_classifier": {"score": 0.6},
    }
    fig = _make_bar_chart(signals)
    assert isinstance(fig, matplotlib.figure.Figure)
