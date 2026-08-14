#!/usr/bin/env python3
"""Build CLIP centroids from synthetic images.

Usage:
    python scripts/build_clip_calibration.py

Generates synthetic PIL images (band-limited noise for "real", sin-grid for "AI"),
embeds them with openai/clip-vit-base-patch32, and saves averaged embeddings as:
    calibration/clip_centroid_real.npy   shape (512,)
    calibration/clip_centroid_ai.npy     shape (512,)

No internet required after the CLIP model weights are cached by HuggingFace.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image

REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ai_image_scanner import config  # noqa: E402

N_SYNTHETIC = 20  # images per class
IMG_SIZE = 256


# ---------------------------------------------------------------------------
# Synthetic image generators (mirrors build_frequency_calibration.py)
# ---------------------------------------------------------------------------

def _make_real_synthetic(rng: np.random.Generator) -> Image.Image:
    """Band-limited 1/f noise — mimics natural camera-sensor noise."""
    from scipy.fft import fft2, ifft2

    noise = rng.standard_normal((IMG_SIZE, IMG_SIZE)).astype(np.float32)
    F = fft2(noise)
    rows = np.fft.fftfreq(IMG_SIZE)
    cols = np.fft.fftfreq(IMG_SIZE)
    col_g, row_g = np.meshgrid(cols, rows)
    freq = np.hypot(row_g, col_g)
    freq[0, 0] = 1.0
    rolloff = 1.0 / (freq ** 0.5)
    rolloff[0, 0] = 0.0
    filtered = np.real(ifft2(F * rolloff))
    lo, hi = filtered.min(), filtered.max()
    if hi > lo:
        filtered = (filtered - lo) / (hi - lo) * 255.0
    arr = np.stack([filtered] * 3, axis=-1).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _make_ai_synthetic(rng: np.random.Generator, idx: int) -> Image.Image:
    """Periodic sin-grid — mimics structured artifacts in diffusion models."""
    x = np.linspace(0, 2 * np.pi * (2 + idx % 4), IMG_SIZE, dtype=np.float32)
    y = np.linspace(0, 2 * np.pi * (3 + idx % 3), IMG_SIZE, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    grid = np.sin(X) * np.sin(Y)
    grid += rng.standard_normal(grid.shape).astype(np.float32) * 0.05
    lo, hi = grid.min(), grid.max()
    if hi > lo:
        grid = (grid - lo) / (hi - lo) * 255.0
    arr = np.stack([grid] * 3, axis=-1).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_centroids(
    out_real: pathlib.Path,
    out_ai: pathlib.Path,
    n_synthetic: int = N_SYNTHETIC,
) -> None:
    """Compute and save CLIP centroids to *out_real* and *out_ai*."""
    import torch
    import torch.nn.functional as F
    import open_clip

    print("Loading CLIP model (openai/ViT-B-32)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval()

    rng = np.random.default_rng(42)

    def embed_images(images: list[Image.Image]) -> np.ndarray:
        embeddings = []
        for img in images:
            tensor = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                feat = model.encode_image(tensor)
            feat = F.normalize(feat, dim=-1)
            embeddings.append(feat.squeeze(0).numpy().astype(np.float32))
        return np.stack(embeddings)  # (N, 512)

    print(f"Generating {n_synthetic} synthetic real images...")
    real_images = [_make_real_synthetic(rng) for _ in range(n_synthetic)]
    real_embeddings = embed_images(real_images)
    real_centroid = real_embeddings.mean(axis=0)

    print(f"Generating {n_synthetic} synthetic AI images...")
    ai_images = [_make_ai_synthetic(rng, i) for i in range(n_synthetic)]
    ai_embeddings = embed_images(ai_images)
    ai_centroid = ai_embeddings.mean(axis=0)

    out_real.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_real, real_centroid)
    np.save(out_ai, ai_centroid)

    print(f"Saved real centroid ({real_centroid.shape}) -> {out_real}")
    print(f"Saved AI centroid   ({ai_centroid.shape}) -> {out_ai}")


def main() -> None:
    build_centroids(
        out_real=config.CLIP_CENTROID_REAL,
        out_ai=config.CLIP_CENTROID_AI,
    )


if __name__ == "__main__":
    main()
