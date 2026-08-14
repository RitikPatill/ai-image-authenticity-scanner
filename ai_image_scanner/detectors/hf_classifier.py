"""HuggingFace image-classification detector for AI-generated images.

Wraps Organika/sdxl-detector (or any binary image classifier) via
transformers.pipeline. Loaded lazily on first call to avoid startup cost.
"""

from __future__ import annotations

from PIL import Image
from transformers import pipeline

from ai_image_scanner import config

HF_CLASSIFIER_MODEL: str = config.HF_CLASSIFIER_MODEL

# Known label strings that indicate the AI-generated class (lowercased).
# Organika/sdxl-detector uses "artificial"; generic HF classifiers use "label_1".
_AI_LABELS = {"label_1", "artificial", "ai"}


class HFClassifierDetector:
    """Score images using a HuggingFace binary image classifier."""

    def __init__(
        self,
        model_id: str = HF_CLASSIFIER_MODEL,
        device: str = "cpu",
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._pipe = None  # lazy

    def _get_pipe(self):
        if self._pipe is None:
            self._pipe = pipeline(
                "image-classification",
                model=self._model_id,
                device=self._device,
            )
        return self._pipe

    def score(self, img: Image.Image) -> dict:
        """Return AI-probability score in [0, 1]. Higher = more likely AI.

        Looks up known AI-positive label strings (case-insensitive). Falls back
        to 0.5 for unknown binary classifiers.
        """
        pipe = self._get_pipe()
        results: list[dict] = pipe(img.convert("RGB"))

        # Lowercase all label keys for case-insensitive matching
        label_scores = {r["label"].lower(): r["score"] for r in results}

        ai_score = None
        for lbl in label_scores:
            if lbl in _AI_LABELS:
                ai_score = float(label_scores[lbl])
                break

        if ai_score is None:
            # Generic fallback for unknown binary classifiers
            ai_score = 0.5

        return {"score": max(0.0, min(1.0, ai_score)), "signal": "hf_classifier"}
