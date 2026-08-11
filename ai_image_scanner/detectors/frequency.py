"""Frequency-domain detector for AI-generated images.

Computes a 2D DCT on the grayscale image, extracts a normalized
ring-power spectrum (radial energy bins) as a feature vector, and
scores the image by comparing it to a precomputed calibration.
"""

from __future__ import annotations

import pathlib

import numpy as np
from PIL import Image
from scipy.fft import dctn

from ai_image_scanner import config

N_RINGS = 32
IMG_SIZE = 256  # resize target (square)
FREQUENCY_CALIBRATION = config.FREQUENCY_CALIBRATION


class FrequencyDetector:
    """Score images via DCT ring-power spectrum deviation from real-image centroid."""

    def __init__(self, calibration_path: pathlib.Path = FREQUENCY_CALIBRATION):
        if not pathlib.Path(calibration_path).exists():
            raise FileNotFoundError(
                f"Calibration file not found: {calibration_path}\n"
                "Run scripts/build_frequency_calibration.py first"
            )
        data = np.load(calibration_path)
        self.real_mean: np.ndarray = data["real_mean"].astype(np.float32)
        self.real_std: np.ndarray = data["real_std"].astype(np.float32)
        self.ai_mean: np.ndarray = data["ai_mean"].astype(np.float32)
        self.ai_std: np.ndarray = data["ai_std"].astype(np.float32)

    def extract_ring_spectrum(self, img: Image.Image) -> np.ndarray:
        """Return normalized ring-power spectrum of shape (N_RINGS,).

        Steps:
        1. Convert to grayscale, resize to IMG_SIZE x IMG_SIZE.
        2. Compute 2D DCT (norm='ortho'). Energy concentrates at DC corner [0,0].
        3. Build radial distance map from the DC corner.
        4. Bin squared coefficients into N_RINGS rings, normalize by total power.
        Ring 0 (DC component) is excluded to avoid gain-normalization artifacts.
        """
        gray = img.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = np.array(gray, dtype=np.float32)

        dct2d = dctn(arr, norm="ortho")
        power = dct2d ** 2

        # Radial distance from DC corner (0,0)
        H, W = arr.shape
        rows = np.arange(H)
        cols = np.arange(W)
        col_grid, row_grid = np.meshgrid(cols, rows)
        dist = np.hypot(row_grid, col_grid)  # shape (H, W)

        max_dist = dist.max()
        # N_RINGS+1 bin edges so we have N_RINGS bins
        bin_edges = np.linspace(0, max_dist, N_RINGS + 2)[1:]  # skip edge at 0

        ring_power = np.zeros(N_RINGS, dtype=np.float32)
        for i in range(N_RINGS):
            low = 0.0 if i == 0 else bin_edges[i - 1]
            high = bin_edges[i]
            mask = (dist > low) & (dist <= high)
            ring_power[i] = power[mask].sum()

        total = ring_power.sum()
        if total > 0:
            ring_power /= total

        return ring_power

    def score(self, img: Image.Image) -> float:
        """Return AI-probability score in [0.0, 1.0]. Higher = more likely AI.

        Computes average per-ring deviation from the real-image mean,
        measured in standard deviations, then maps through a sigmoid.
        """
        spectrum = self.extract_ring_spectrum(img)
        raw = float(np.mean(np.abs(spectrum - self.real_mean) / (self.real_std + 1e-8)))
        # Sigmoid centred at raw=1 (one avg std-dev from real centroid → 0.5 score)
        score = float(1.0 / (1.0 + np.exp(-4.0 * (raw - 1.0))))
        return max(0.0, min(1.0, score))


def score_image(img: Image.Image, detector: FrequencyDetector | None = None) -> float:
    """Module-level convenience: score *img* with an optional pre-built detector."""
    if detector is None:
        detector = FrequencyDetector()
    return detector.score(img)
