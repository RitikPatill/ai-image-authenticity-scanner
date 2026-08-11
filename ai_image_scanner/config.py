import pathlib

# Detector weights — must sum to 1.0
# assert WEIGHT_FREQUENCY + WEIGHT_CLIP + WEIGHT_CLASSIFIER == 1.0
WEIGHT_FREQUENCY: float = 0.30
WEIGHT_CLIP: float = 0.35
WEIGHT_CLASSIFIER: float = 0.35

# Path to bundled calibration centroids
CALIBRATION_DIR = pathlib.Path(__file__).parent.parent / "calibration"
CLIP_CENTROID_REAL = CALIBRATION_DIR / "clip_centroid_real.npy"
CLIP_CENTROID_AI   = CALIBRATION_DIR / "clip_centroid_ai.npy"

# HuggingFace model id for the pre-trained classifier
HF_CLASSIFIER_MODEL = "Organika/sdxl-detector"
