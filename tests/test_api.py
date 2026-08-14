"""Integration tests for the FastAPI endpoints.

All detectors are mocked so no calibration files or model weights are needed.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Fixed return value used by all mocked score_image calls
_MOCK_RESULT = {
    "score": 0.72,
    "verdict": "AI",
    "signals": {
        "frequency": {"score": 0.65},
        "clip_drift": {"score": 0.80},
        "hf_classifier": {"score": 0.70},
    },
}


def _png_bytes(size: int = 64) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def client():
    """TestClient with score_image and _get_detectors mocked."""
    with patch("ai_image_scanner.api.score_image", return_value=_MOCK_RESULT.copy()), \
         patch("ai_image_scanner.ensemble._get_detectors", return_value={}):
        from ai_image_scanner.api import app
        yield TestClient(app)


# ---------------------------------------------------------------------------
# POST /detect
# ---------------------------------------------------------------------------


def test_post_detect_returns_expected_keys(client):
    response = client.post(
        "/detect",
        files={"file": ("test.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert {"score", "verdict", "signals", "processing_time_ms"} <= data.keys()


def test_post_detect_score_range(client):
    response = client.post(
        "/detect",
        files={"file": ("test.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    score = response.json()["score"]
    assert 0.0 <= score <= 1.0


def test_post_detect_signals_structure(client):
    response = client.post(
        "/detect",
        files={"file": ("test.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    signals = response.json()["signals"]
    for key in ("frequency", "clip_drift", "hf_classifier"):
        assert key in signals
        assert "score" in signals[key]
        assert isinstance(signals[key]["score"], float)


def test_post_detect_bad_file(client):
    response = client.post(
        "/detect",
        files={"file": ("not_an_image.bin", b"this is not an image", "application/octet-stream")},
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# GET /detect?url=...
# ---------------------------------------------------------------------------


def test_get_detect_url_success(client):
    mock_response = MagicMock()
    mock_response.content = _png_bytes()
    mock_response.raise_for_status = MagicMock()

    with patch("ai_image_scanner.api.httpx.get", return_value=mock_response):
        response = client.get("/detect", params={"url": "http://example.com/image.png"})

    assert response.status_code == 200
    data = response.json()
    assert {"score", "verdict", "signals", "processing_time_ms"} <= data.keys()


def test_get_detect_url_bad_status(client):
    import httpx as _httpx

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = _httpx.HTTPStatusError(
        "404", request=MagicMock(), response=MagicMock()
    )

    with patch("ai_image_scanner.api.httpx.get", return_value=mock_response):
        response = client.get("/detect", params={"url": "http://example.com/missing.png"})

    assert response.status_code == 400
