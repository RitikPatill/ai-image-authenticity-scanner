# Calibration Sample Sources

> **Note:** The images in `real/` and `ai/` are **synthetically generated**
> by `scripts/build_frequency_calibration.py` using NumPy, rather than
> real photographs or DiffusionDB crops. This keeps the repo dependency-free
> and avoids licensing concerns. The calibration math is identical; biological
> validity is lower but sufficient for a portfolio/demo detector.

## Synthetic generation strategy

| Folder | Generation method | Rationale |
|--------|-------------------|-----------|
| `real/` | Band-limited noise: random Gaussian noise in DCT space with 1/f roll-off | Mimics the natural high-frequency falloff of camera sensor noise |
| `ai/`  | Structured grid: `sin(x) * sin(y)` tiled at multiple spatial frequencies | Mimics periodic DCT artifacts common in diffusion-model outputs |

## Replacing with real images

To use actual images, place 10 (or more) representative JPEGs in each
folder and re-run:

```
python scripts/build_frequency_calibration.py
```

Suggested real sources:
- **Real photos:** Unsplash (https://unsplash.com, CC0 license)
- **AI images:** DiffusionDB (https://huggingface.co/datasets/poloclub/diffusiondb, CC-BY 4.0)
