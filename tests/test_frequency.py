"""Unit tests for ai_image_scanner.detectors.frequency.

All tests use synthetic data — no real images or network access required.
Calibration is built in-memory via a pytest fixture.
"""

from __future__ import annotations

import io
import pathlib

import numpy as np
import pytest
from PIL import Image

from ai_image_scanner.detectors.frequency import (
    N_RINGS,
    FrequencyDetector,
    score_image,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _white_image(size: int = 256) -> Image.Image:
    return Image.fromarray(np.full((size, size), 255, dtype=np.uint8), mode="L")


def _noise_image(size: int = 256, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = (rng.random((size, size)) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _grid_image(size: int = 256, freq: float = 8.0) -> Image.Image:
    """Periodic sin-grid — should look more AI-like to the detector."""
    x = np.linspace(0, 2 * np.pi * freq, size, dtype=np.float32)
    y = np.linspace(0, 2 * np.pi * freq, size, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    grid = np.sin(X) * np.sin(Y)
    lo, hi = grid.min(), grid.max()
    grid = (grid - lo) / (hi - lo) * 255.0
    return Image.fromarray(grid.astype(np.uint8), mode="L")


def _build_calibration_npz(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write a synthetic frequency_calibration.npz to tmp_path and return its path."""
    rng = np.random.default_rng(99)

    # Real: small random values across all rings (flat-ish spectrum after normalisation)
    real_mean = np.full(N_RINGS, 1.0 / N_RINGS, dtype=np.float32)
    real_std = (rng.random(N_RINGS) * 0.005 + 0.001).astype(np.float32)

    # AI: energy concentrated in mid-rings
    ai_mean = np.zeros(N_RINGS, dtype=np.float32)
    ai_mean[N_RINGS // 4 : N_RINGS // 2] = 2.0 / N_RINGS
    ai_std = (rng.random(N_RINGS) * 0.005 + 0.001).astype(np.float32)

    out = tmp_path / "frequency_calibration.npz"
    np.savez(out, real_mean=real_mean, real_std=real_std, ai_mean=ai_mean, ai_std=ai_std)
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def detector(tmp_path: pathlib.Path) -> FrequencyDetector:
    """FrequencyDetector loaded from an in-memory synthetic calibration."""
    cal_path = _build_calibration_npz(tmp_path)
    return FrequencyDetector(calibration_path=cal_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ring_spectrum_shape(detector: FrequencyDetector) -> None:
    """extract_ring_spectrum returns an array of shape (N_RINGS,)."""
    img = _white_image()
    spectrum = detector.extract_ring_spectrum(img)
    assert spectrum.shape == (N_RINGS,), f"Expected ({N_RINGS},), got {spectrum.shape}"
    assert spectrum.dtype == np.float32


def test_ring_spectrum_sums_to_one(detector: FrequencyDetector) -> None:
    """Ring spectrum of a random noise image should sum to ~1.0 (normalized)."""
    img = _noise_image()
    spectrum = detector.extract_ring_spectrum(img)
    total = spectrum.sum()
    assert abs(total - 1.0) < 1e-4, f"Expected sum~1.0, got {total}"


def test_score_range(detector: FrequencyDetector) -> None:
    """score() must return a float in [0.0, 1.0]."""
    for make_img in [_white_image, _noise_image, _grid_image]:
        s = detector.score(make_img())
        assert isinstance(s, float), f"score() returned {type(s)}"
        assert 0.0 <= s <= 1.0, f"score() out of range: {s}"


def test_score_noise_vs_grid(tmp_path: pathlib.Path) -> None:
    """Grid image (AI-like) should score higher than noise image (real-like).

    We build a calibration whose real_mean matches noise spectra closely,
    so noise gets a low score and the structured grid gets a high score.
    """
    det_tmp = FrequencyDetector.__new__(FrequencyDetector)

    # Compute actual spectra for noise and grid
    noise_spectrum = det_tmp.extract_ring_spectrum(_noise_image())
    grid_spectrum = det_tmp.extract_ring_spectrum(_grid_image())

    # Build calibration centred on noise spectrum
    cal_path = tmp_path / "cal.npz"
    np.savez(
        cal_path,
        real_mean=noise_spectrum.astype(np.float32),
        real_std=np.full(N_RINGS, 1e-3, dtype=np.float32),
        ai_mean=grid_spectrum.astype(np.float32),
        ai_std=np.full(N_RINGS, 1e-3, dtype=np.float32),
    )

    det = FrequencyDetector(calibration_path=cal_path)
    noise_score = det.score(_noise_image())
    grid_score = det.score(_grid_image())

    assert noise_score < grid_score, (
        f"Expected noise_score ({noise_score:.4f}) < grid_score ({grid_score:.4f})"
    )


def test_score_image_convenience(detector: FrequencyDetector) -> None:
    """score_image() module-level function returns same result as detector.score()."""
    img = _noise_image(seed=7)
    expected = detector.score(img)
    result = score_image(img, detector=detector)
    assert result == expected, f"score_image mismatch: {result} != {expected}"


def test_missing_calibration_raises(tmp_path: pathlib.Path) -> None:
    """FrequencyDetector raises FileNotFoundError when calibration is missing."""
    missing = tmp_path / "does_not_exist.npz"
    with pytest.raises(FileNotFoundError, match="build_frequency_calibration"):
        FrequencyDetector(calibration_path=missing)
