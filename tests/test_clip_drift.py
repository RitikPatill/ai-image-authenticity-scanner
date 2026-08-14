"""Unit tests for ai_image_scanner.detectors.clip_drift.

Tests use CLIP embeddings on synthetic PIL images.
CLIP weights (~350 MB) are downloaded once and cached by HuggingFace.
Mark with @pytest.mark.slow or guard with importorskip to skip in CI.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("open_clip", reason="open-clip-torch not installed")

from ai_image_scanner.detectors.clip_drift import CLIPDriftDetector  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _noise_image(size: int = 64, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = (rng.random((size, size, 3)) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _grid_image(size: int = 64, freq: float = 4.0) -> Image.Image:
    x = np.linspace(0, 2 * np.pi * freq, size, dtype=np.float32)
    y = np.linspace(0, 2 * np.pi * freq, size, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    grid = np.sin(X) * np.sin(Y)
    lo, hi = grid.min(), grid.max()
    grid = ((grid - lo) / (hi - lo) * 255).astype(np.uint8)
    arr = np.stack([grid] * 3, axis=-1)
    return Image.fromarray(arr, mode="RGB")


def _build_centroids(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Build tiny CLIP centroids from synthetic images and save to tmp_path."""
    import torch
    import torch.nn.functional as F
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval()

    def embed(img: Image.Image) -> np.ndarray:
        t = preprocess(img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            feat = model.encode_image(t)
        return F.normalize(feat, dim=-1).squeeze(0).numpy().astype(np.float32)

    rng = np.random.default_rng(7)
    real_embs = np.stack([embed(_noise_image(seed=i)) for i in range(5)])
    ai_embs = np.stack([embed(_grid_image(freq=2 + i)) for i in range(5)])

    real_path = tmp_path / "clip_centroid_real.npy"
    ai_path = tmp_path / "clip_centroid_ai.npy"
    np.save(real_path, real_embs.mean(axis=0))
    np.save(ai_path, ai_embs.mean(axis=0))
    return real_path, ai_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def centroid_paths(tmp_path_factory: pytest.TempPathFactory):
    tmp = tmp_path_factory.mktemp("clip_cal")
    return _build_centroids(tmp)


@pytest.fixture(scope="module")
def detector(centroid_paths):
    real_path, ai_path = centroid_paths
    return CLIPDriftDetector(real_centroid_path=real_path, ai_centroid_path=ai_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_embed_shape(detector: CLIPDriftDetector) -> None:
    """embed() returns shape (512,)."""
    img = _noise_image()
    emb = detector.embed(img)
    assert emb.shape == (512,), f"Expected (512,), got {emb.shape}"


@pytest.mark.slow
def test_embed_l2_normalised(detector: CLIPDriftDetector) -> None:
    """embed() output is L2-normalised (norm ≈ 1.0)."""
    img = _noise_image(seed=3)
    emb = detector.embed(img)
    norm = float(np.linalg.norm(emb))
    assert abs(norm - 1.0) < 1e-4, f"Expected norm≈1.0, got {norm}"


@pytest.mark.slow
def test_score_returns_dict(detector: CLIPDriftDetector) -> None:
    """score() returns a dict with 'score' and 'signal' keys."""
    result = detector.score(_noise_image())
    assert isinstance(result, dict)
    assert "score" in result
    assert result["signal"] == "clip_drift"


@pytest.mark.slow
def test_score_range(detector: CLIPDriftDetector) -> None:
    """score() returns a float in [0, 1] for various images."""
    for img in [_noise_image(seed=1), _grid_image(), _noise_image(seed=42)]:
        s = detector.score(img)["score"]
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0, f"Score out of range: {s}"


@pytest.mark.slow
def test_missing_real_centroid_raises(tmp_path: pathlib.Path, centroid_paths) -> None:
    """CLIPDriftDetector raises FileNotFoundError when real centroid is absent."""
    _, ai_path = centroid_paths
    with pytest.raises(FileNotFoundError, match="real centroid"):
        CLIPDriftDetector(
            real_centroid_path=tmp_path / "missing.npy",
            ai_centroid_path=ai_path,
        )


@pytest.mark.slow
def test_missing_ai_centroid_raises(tmp_path: pathlib.Path, centroid_paths) -> None:
    """CLIPDriftDetector raises FileNotFoundError when AI centroid is absent."""
    real_path, _ = centroid_paths
    with pytest.raises(FileNotFoundError, match="AI centroid"):
        CLIPDriftDetector(
            real_centroid_path=real_path,
            ai_centroid_path=tmp_path / "missing.npy",
        )
