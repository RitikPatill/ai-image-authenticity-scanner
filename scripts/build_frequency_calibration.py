#!/usr/bin/env python3
"""Build frequency_calibration.npz from sample images.

Usage:
    python scripts/build_frequency_calibration.py

Reads images from:
    calibration/samples/real/  — real-photo stand-ins (JPEGs or PNGs)
    calibration/samples/ai/    — AI-image stand-ins

If either directory is empty, synthetic images are generated automatically
(band-limited noise for real, sin-grid for AI) — enough for a demo-quality
calibration without any external downloads.

Saves:
    calibration/frequency_calibration.npz
    Keys: real_mean, real_std, ai_mean, ai_std  — each shape (N_RINGS,)
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image

# Allow running as a script from the repo root
REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ai_image_scanner import config  # noqa: E402 — after sys.path patch
from ai_image_scanner.detectors.frequency import FrequencyDetector, N_RINGS, IMG_SIZE  # noqa: E402

SAMPLES_DIR = config.CALIBRATION_DIR / "samples"
REAL_DIR = SAMPLES_DIR / "real"
AI_DIR = SAMPLES_DIR / "ai"
N_SYNTHETIC = 10  # synthetic images per class when real files are absent


# ---------------------------------------------------------------------------
# Synthetic image generators
# ---------------------------------------------------------------------------

def _make_real_synthetic(rng: np.random.Generator) -> Image.Image:
    """Band-limited 1/f noise — mimics natural camera-sensor noise."""
    noise = rng.standard_normal((IMG_SIZE, IMG_SIZE)).astype(np.float32)
    # Apply 1/f roll-off in frequency domain
    from scipy.fft import fft2, ifft2
    F = fft2(noise)
    rows = np.fft.fftfreq(IMG_SIZE)
    cols = np.fft.fftfreq(IMG_SIZE)
    col_g, row_g = np.meshgrid(cols, rows)
    freq = np.hypot(row_g, col_g)
    freq[0, 0] = 1.0  # avoid div-by-zero at DC
    rolloff = 1.0 / (freq ** 0.5)
    rolloff[0, 0] = 0.0  # zero DC component
    F_filtered = F * rolloff
    filtered = np.real(ifft2(F_filtered))
    # Normalize to [0, 255]
    lo, hi = filtered.min(), filtered.max()
    if hi > lo:
        filtered = (filtered - lo) / (hi - lo) * 255.0
    return Image.fromarray(filtered.astype(np.uint8), mode="L")


def _make_ai_synthetic(rng: np.random.Generator, idx: int) -> Image.Image:
    """Periodic sin-grid — mimics structured DCT artifacts in diffusion models."""
    x = np.linspace(0, 2 * np.pi * (2 + idx % 4), IMG_SIZE, dtype=np.float32)
    y = np.linspace(0, 2 * np.pi * (3 + idx % 3), IMG_SIZE, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    grid = np.sin(X) * np.sin(Y)
    # Add a small amount of noise so images aren't identical
    grid += rng.standard_normal(grid.shape).astype(np.float32) * 0.05
    lo, hi = grid.min(), grid.max()
    if hi > lo:
        grid = (grid - lo) / (hi - lo) * 255.0
    return Image.fromarray(grid.astype(np.uint8), mode="L")


# ---------------------------------------------------------------------------
# Load or synthesize images
# ---------------------------------------------------------------------------

def _load_images(directory: pathlib.Path) -> list[Image.Image]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in exts)
    return [Image.open(p) for p in paths]


def _get_images(directory: pathlib.Path, kind: str) -> list[Image.Image]:
    images = _load_images(directory)
    if images:
        print(f"  Loaded {len(images)} {kind} images from {directory}")
        return images

    print(f"  No images found in {directory} — generating {N_SYNTHETIC} synthetic {kind} images.")
    rng = np.random.default_rng(42)
    if kind == "real":
        return [_make_real_synthetic(rng) for _ in range(N_SYNTHETIC)]
    else:
        return [_make_ai_synthetic(rng, i) for i in range(N_SYNTHETIC)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    AI_DIR.mkdir(parents=True, exist_ok=True)

    print("Building frequency calibration...")

    # Instantiate detector without loading calibration (it doesn't exist yet)
    detector = FrequencyDetector.__new__(FrequencyDetector)

    real_images = _get_images(REAL_DIR, "real")
    ai_images = _get_images(AI_DIR, "ai")

    print(f"  Computing spectra for {len(real_images)} real images...")
    real_spectra = np.stack([detector.extract_ring_spectrum(img) for img in real_images])

    print(f"  Computing spectra for {len(ai_images)} AI images...")
    ai_spectra = np.stack([detector.extract_ring_spectrum(img) for img in ai_images])

    out_path = config.FREQUENCY_CALIBRATION
    np.savez(
        out_path,
        real_mean=real_spectra.mean(axis=0).astype(np.float32),
        real_std=real_spectra.std(axis=0).astype(np.float32),
        ai_mean=ai_spectra.mean(axis=0).astype(np.float32),
        ai_std=ai_spectra.std(axis=0).astype(np.float32),
    )
    print(f"Saved calibration to {out_path}")
    print(f"  real_mean shape: {real_spectra.mean(axis=0).shape}")
    print(f"  N_RINGS: {N_RINGS}")


if __name__ == "__main__":
    main()
