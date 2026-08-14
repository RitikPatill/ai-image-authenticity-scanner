"""FastAPI application for AI Image Authenticity Scanner.

Start with:
    uvicorn ai_image_scanner.api:app --reload
"""

from __future__ import annotations

import io
import time

import httpx
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from ai_image_scanner.ensemble import score_image

app = FastAPI(title="AI Image Authenticity Scanner", version="0.4.0")


@app.on_event("startup")
def _warm_up() -> None:
    """Pre-load detector weights so the first request is not slow."""
    from ai_image_scanner.ensemble import _get_detectors

    _get_detectors()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_image_from_bytes(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Cannot decode image bytes")


def _build_response(result: dict, t0: float) -> dict:
    result["processing_time_ms"] = round((time.time() - t0) * 1000, 1)
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/detect")
def detect_upload(file: UploadFile = File(...)) -> dict:
    """Score an uploaded image file."""
    t0 = time.time()
    data = file.file.read()
    img = _load_image_from_bytes(data)
    result = score_image(img)
    return _build_response(result, t0)


@app.get("/detect")
def detect_url(url: str = Query(..., description="Public image URL")) -> dict:
    """Score an image fetched from a public URL."""
    t0 = time.time()
    try:
        response = httpx.get(url, timeout=10, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {exc}")
    img = _load_image_from_bytes(response.content)
    result = score_image(img)
    return _build_response(result, t0)
