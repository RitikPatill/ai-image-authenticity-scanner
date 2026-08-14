"""Ensemble scorer combining frequency, CLIP drift, and HF classifier signals.

Usage:
    from ai_image_scanner.ensemble import score_image
    result = score_image(img)
    # {"score": 0.72, "verdict": "AI", "signals": {...}}
"""

from __future__ import annotations

from PIL import Image

from ai_image_scanner import config

# Module-level detector cache — built once on first call
_detectors: dict | None = None


def _build_detectors() -> dict:
    """Instantiate all three detectors. Called at most once."""
    from ai_image_scanner.detectors.frequency import FrequencyDetector
    from ai_image_scanner.detectors.clip_drift import CLIPDriftDetector
    from ai_image_scanner.detectors.hf_classifier import HFClassifierDetector

    return {
        "frequency": FrequencyDetector(calibration_path=config.FREQUENCY_CALIBRATION),
        "clip_drift": CLIPDriftDetector(
            real_centroid_path=config.CLIP_CENTROID_REAL,
            ai_centroid_path=config.CLIP_CENTROID_AI,
        ),
        "hf_classifier": HFClassifierDetector(model_id=config.HF_CLASSIFIER_MODEL),
    }


def _get_detectors() -> dict:
    global _detectors
    if _detectors is None:
        _detectors = _build_detectors()
    return _detectors


def score_image(img: Image.Image, detectors: dict | None = None) -> dict:
    """Score *img* with all three detectors and return a combined result.

    Returns:
        {
          "score": float,            # weighted average in [0, 1]
          "verdict": "AI" | "REAL",  # threshold 0.5
          "signals": {
            "frequency":     {"score": float},
            "clip_drift":    {"score": float},
            "hf_classifier": {"score": float},
          }
        }
    """
    if detectors is None:
        detectors = _get_detectors()

    freq_det = detectors["frequency"]
    clip_det = detectors["clip_drift"]
    hf_det = detectors["hf_classifier"]

    # FrequencyDetector.score() returns a bare float — wrap it
    freq_score = freq_det.score(img)
    if isinstance(freq_score, dict):
        freq_val = float(freq_score["score"])
    else:
        freq_val = float(freq_score)

    clip_result = clip_det.score(img)
    clip_val = float(clip_result["score"])

    hf_result = hf_det.score(img)
    hf_val = float(hf_result["score"])

    combined = (
        config.WEIGHT_FREQUENCY * freq_val
        + config.WEIGHT_CLIP * clip_val
        + config.WEIGHT_CLASSIFIER * hf_val
    )
    combined = max(0.0, min(1.0, combined))

    return {
        "score": combined,
        "verdict": "AI" if combined >= 0.5 else "REAL",
        "signals": {
            "frequency": {"score": freq_val},
            "clip_drift": {"score": clip_val},
            "hf_classifier": {"score": hf_val},
        },
    }
