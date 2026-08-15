#!/usr/bin/env python3
"""Generate synthetic sample images for offline demos.

Writes 6 PNG files to assets/samples/ (or --out-dir):
  real_01.png, real_02.png, real_03.png  — band-limited 1/f noise
  ai_01.png,   ai_02.png,   ai_03.png   — periodic sin-grid

Usage:
    python scripts/generate_samples.py
    python scripts/generate_samples.py --out-dir /tmp/samples
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
from PIL import Image

IMG_SIZE = 256


def _1f_noise_image(seed: int) -> Image.Image:
    """Band-limited 1/f noise image (mimics natural camera-sensor noise)."""
    from scipy.fft import fft2, ifft2

    rng = np.random.default_rng(seed)
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
    return Image.fromarray(filtered.astype(np.uint8), mode="L").convert("RGB")


def _grid_image(freq: float, phase: float) -> Image.Image:
    """Periodic sin-grid image (mimics structured DCT artifacts in diffusion models)."""
    rng = np.random.default_rng(int(freq * 100 + phase * 10))
    x = np.linspace(0, 2 * np.pi * freq, IMG_SIZE, dtype=np.float32)
    y = np.linspace(0, 2 * np.pi * freq, IMG_SIZE, dtype=np.float32)
    X, Y = np.meshgrid(x + phase, y + phase)
    grid = np.sin(X) * np.sin(Y)
    grid += rng.standard_normal(grid.shape).astype(np.float32) * 0.05
    lo, hi = grid.min(), grid.max()
    if hi > lo:
        grid = (grid - lo) / (hi - lo) * 255.0
    return Image.fromarray(grid.astype(np.uint8), mode="L").convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic sample images.")
    parser.add_argument(
        "--out-dir",
        default="assets/samples",
        help="Output directory (default: assets/samples)",
    )
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Real samples — seeds 42, 43, 44
    for i, seed in enumerate([42, 43, 44], start=1):
        img = _1f_noise_image(seed)
        path = out_dir / f"real_{i:02d}.png"
        img.save(path)
        print(f"Saved {path}")

    # AI samples — distinct frequency/phase combos
    ai_params = [
        (8, 0.0),
        (12, 0.5),
        (6, 1.0),
    ]
    for i, (freq, phase) in enumerate(ai_params, start=1):
        img = _grid_image(freq, phase)
        path = out_dir / f"ai_{i:02d}.png"
        img.save(path)
        print(f"Saved {path}")

    print(f"\nDone — 6 sample images written to {out_dir}/")


if __name__ == "__main__":
    main()
