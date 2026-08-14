"""CLIP embedding drift detector for AI-generated images.

Loads openai/clip-vit-base-patch32 via open-clip-torch.
Compares image CLIP embeddings to precomputed centroids for real and AI images.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import open_clip
except ImportError as exc:  # pragma: no cover
    raise ImportError("open-clip-torch is required: pip install open-clip-torch") from exc


def _sigmoid(x: float, k: float = 4.0) -> float:
    """Sigmoid with steepness k, centred at 0."""
    return float(1.0 / (1.0 + np.exp(-k * x)))


class CLIPDriftDetector:
    """Score images by cosine distance to real/AI CLIP embedding centroids."""

    def __init__(
        self,
        real_centroid_path: pathlib.Path,
        ai_centroid_path: pathlib.Path,
        device: str = "cpu",
    ) -> None:
        self.device = device

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model.eval()
        model.to(device)
        self._model = model
        self._preprocess = preprocess

        real_centroid_path = pathlib.Path(real_centroid_path)
        ai_centroid_path = pathlib.Path(ai_centroid_path)

        if not real_centroid_path.exists():
            raise FileNotFoundError(
                f"CLIP real centroid not found: {real_centroid_path}\n"
                "Run scripts/build_clip_calibration.py first."
            )
        if not ai_centroid_path.exists():
            raise FileNotFoundError(
                f"CLIP AI centroid not found: {ai_centroid_path}\n"
                "Run scripts/build_clip_calibration.py first."
            )

        real = np.load(real_centroid_path).astype(np.float32)
        ai = np.load(ai_centroid_path).astype(np.float32)

        # L2-normalise centroids so cosine sim is a dot product
        self._real_centroid = torch.from_numpy(real / (np.linalg.norm(real) + 1e-8)).to(device)
        self._ai_centroid = torch.from_numpy(ai / (np.linalg.norm(ai) + 1e-8)).to(device)

    def embed(self, img: Image.Image) -> np.ndarray:
        """Return L2-normalised 512-dim CLIP embedding for *img*."""
        tensor = self._preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self._model.encode_image(tensor)
        features = F.normalize(features, dim=-1)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def score(self, img: Image.Image) -> dict:
        """Return AI-probability score in [0, 1]. Higher = more likely AI.

        Formula:
            raw = (cos_sim(emb, ai_centroid) - cos_sim(emb, real_centroid)) / 2
            score = sigmoid(raw * 4)
        """
        emb = torch.from_numpy(self.embed(img)).to(self.device)
        cos_real = float(torch.dot(emb, self._real_centroid))
        cos_ai = float(torch.dot(emb, self._ai_centroid))

        raw = (cos_ai - cos_real) / 2.0
        s = _sigmoid(raw, k=4.0)
        return {"score": max(0.0, min(1.0, s)), "signal": "clip_drift"}
