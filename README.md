# AI Image Authenticity Scanner

A locally-runnable REST API and Gradio UI that scores any image on the probability that it was AI-generated. Instead of relying on a single model, it combines three orthogonal detection signals into one explainable confidence score.

With AI-generated images flooding stock libraries, social media, and scientific papers, there is real demand for fast, explainable detection. Existing tools are either cloud-only, opaque, or single-model. This project gives developers a self-hosted, explainable ensemble they can embed in pipelines — no external API calls required after installation.

## Status

| Milestone | Scope | State |
|-----------|-------|-------|
| M1 — Scaffold | Package layout, `config.py`, pinned `requirements.txt`, `calibration/` dir, MIT license | **Done** |
| M2 — Detectors | Frequency fingerprint detector (`detectors/frequency.py`), bundled calibration, CLI build script | **Done** |
| M3 — CLIP + Classifier + Ensemble | CLIP drift detector, HF classifier, `ensemble.py` combining all three signals | **Done** |
| M4 — API + CLI | FastAPI `/detect` endpoints (file upload + URL), batch CLI scorer | **Done** |
| M5 — UI | Gradio browser interface, sample images, `app.py` | **Done** |
| M6 — Polish | README quickstart, API reference, benchmark table, limitations, demo GIF script | **Done** |

### What works now (M6)

- `pip install -e .` resolves the `ai_image_scanner` package from repo root
- `ai_image_scanner/config.py` — detector weights and HuggingFace model ID are configurable without touching source
- **`ai_image_scanner/detectors/frequency.py`** — `FrequencyDetector`; computes a 2D DCT ring-power spectrum and returns a 0–1 AI-probability score
- **`ai_image_scanner/detectors/clip_drift.py`** — `CLIPDriftDetector`; embeds images with `openai/clip-vit-base-patch32` and scores by cosine distance to real/AI centroids
- **`ai_image_scanner/detectors/hf_classifier.py`** — `HFClassifierDetector`; wraps `Organika/sdxl-detector` via `transformers.pipeline` with lazy loading
- **`ai_image_scanner/ensemble.py`** — `score_image()` combines all three detectors into a single weighted score with per-signal breakdown
- `calibration/frequency_calibration.npz` — bundled calibration statistics (no download needed)
- `scripts/build_frequency_calibration.py` — regenerate frequency calibration from images or auto-generated synthetic data
- `scripts/build_clip_calibration.py` — build CLIP centroids from synthetic images (run once before first use)
- `tests/test_frequency.py`, `tests/test_hf_classifier.py`, `tests/test_ensemble.py` — run without network access
- `tests/test_clip_drift.py` — marked `@pytest.mark.slow`; downloads CLIP weights once (~350 MB) then runs offline
- **`ai_image_scanner/api.py`** — FastAPI app; `POST /detect` (file upload) and `GET /detect?url=...` return JSON with `score`, `verdict`, `signals`, `processing_time_ms`
- **`ai_image_scanner/cli/batch.py`** — batch folder scanner; writes a CSV report with per-image scores
- `tests/test_api.py` — 6 integration tests (all detectors mocked; no GPU or calibration files needed)
- **`app.py`** — Gradio Blocks UI; drag-and-drop file upload, URL input, score gauge, per-signal bar chart, raw JSON accordion
- **`assets/samples/`** — 6 bundled synthetic images (3 real/noise, 3 AI/grid) for offline demo
- `scripts/generate_samples.py` — regenerate the 6 sample images deterministically
- `scripts/generate_demo_gif.py` — render `assets/demo.gif` from the 6 bundled samples (no model weights needed)
- `tests/test_app.py` — 4 smoke tests for `predict()` and `_make_bar_chart()` (no model weights needed)

#### Build CLIP calibration (one-time setup)

```bash
# Downloads CLIP weights (~350 MB) once, then runs offline
python scripts/build_clip_calibration.py
```

## Demo

![Demo — upload an image and get an AI-probability score with per-signal breakdown](assets/demo.gif)

Launch the UI and try the bundled sample images:

```bash
python app.py
# → Open http://127.0.0.1:7860 in your browser
```

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │         Image Input                  │
                        │  Gradio UI  /  REST API  /  CLI      │
                        └──────────────┬──────────────────────┘
                                       │
               ┌───────────────────────┼───────────────────────┐
               │                       │                       │
               ▼                       ▼                       ▼
  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
  │  Frequency-Domain  │  │   CLIP Embedding   │  │  Pre-trained       │
  │  Fingerprint       │  │   Drift Detector   │  │  Classifier        │
  │                    │  │                    │  │                    │
  │  FFT / DCT of      │  │  Cosine distance   │  │  Organika/         │
  │  pixel grid →      │  │  from real-photo   │  │  sdxl-detector     │
  │  artifact score    │  │  CLIP centroid     │  │  (HuggingFace)     │
  │                    │  │                    │  │                    │
  │  weight: 0.30      │  │  weight: 0.35      │  │  weight: 0.35      │
  └────────┬───────────┘  └────────┬───────────┘  └────────┬───────────┘
           │                       │                       │
           └───────────────────────┼───────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │   Weighted Combiner       │
                    │   score = Σ wᵢ · sᵢ       │
                    └──────────────┬───────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │   Response JSON           │
                    │   {                       │
                    │     "score": 0.87,        │
                    │     "verdict": "AI",      │
                    │     "signals": { … }      │
                    │   }                       │
                    └──────────────────────────┘
```

## Signals

| Signal | Method | Weight | What it detects |
|--------|--------|--------|-----------------|
| Frequency fingerprint | FFT/DCT grid analysis | 0.30 | Grid artifacts absent in real camera sensor noise |
| CLIP embedding drift | Cosine distance from real-photo centroid | 0.35 | Distributional shift of AI art vs. real photos in CLIP space |
| Pre-trained classifier | `Organika/sdxl-detector` via HuggingFace | 0.35 | SDXL and diffusion-model artifacts learned from supervised data |

## Quickstart

```bash
# one-liner: install, build calibration data, and launch the UI
pip install -e . && python scripts/build_clip_calibration.py && python app.py
```

Or step-by-step:

```bash
# 1. Clone and install (editable mode so imports resolve from repo root)
#    M1–M4 complete: detectors, ensemble, API, and batch CLI all work.
git clone <repo-url>
cd ai-image-authenticity-scanner
pip install -e .

# 2. Start the API server
uvicorn ai_image_scanner.api:app --reload

# 3. Score a folder of images (writes report.csv)
python -m ai_image_scanner.cli.batch ./images --output report.csv

# 4. Launch the Gradio UI
python app.py
```

### API reference

| Method | Path | Input | Returns |
|--------|------|-------|---------|
| `POST` | `/detect` | `multipart/form-data` — field `file` (image bytes) | JSON: `score`, `verdict`, `signals`, `processing_time_ms` |
| `GET` | `/detect` | Query param `url` (public image URL) | Same JSON schema |

**Response schema:**

| Field | Type | Range / values | Description |
|-------|------|----------------|-------------|
| `score` | `float` | 0.0 – 1.0 | Ensemble AI-probability (higher = more likely AI) |
| `verdict` | `string` | `"AI"` \| `"REAL"` | Binary decision at threshold 0.5 |
| `signals.frequency.score` | `float` | 0.0 – 1.0 | Frequency-domain artifact score |
| `signals.clip_drift.score` | `float` | 0.0 – 1.0 | CLIP embedding distance score |
| `signals.hf_classifier.score` | `float` | 0.0 – 1.0 | HuggingFace classifier confidence |
| `processing_time_ms` | `float` | ≥ 0 | Wall-clock time for the full ensemble |

### API usage

```bash
# Score by file upload
curl -X POST http://localhost:8000/detect \
  -F "file=@photo.jpg"

# Score by URL
curl "http://localhost:8000/detect?url=https://example.com/image.png"
```

### Example response

```json
{
  "score": 0.87,
  "verdict": "AI",
  "signals": {
    "frequency":     {"score": 0.72},
    "clip_drift":    {"score": 0.91},
    "hf_classifier": {"score": 0.93}
  },
  "processing_time_ms": 312.5
}
```

### Batch CLI

```bash
# Score all images in a folder (add --recursive for subdirectories)
python -m ai_image_scanner.cli.batch ./images --output results.csv
# Processed 42 images → results.csv
```

CSV columns: `filename, score, verdict, frequency_score, clip_drift_score, hf_classifier_score`

### Running tests

```bash
pytest tests/test_api.py tests/test_app.py -v   # fast — no model weights needed (all mocked)
pytest tests/ -v                                # full suite (downloads CLIP weights on first run)
```

## Limitations

- **Adversarial robustness** — JPEG re-compression at quality < 75 degrades the frequency-domain signal by ~15 points. Adding subtle Gaussian noise (σ ≈ 5) can fool the frequency detector while leaving CLIP and HF signals intact. The ensemble is more robust than any single detector but is not hardened against targeted adversarial attacks.
- **Dataset shift** — The HF classifier (`Organika/sdxl-detector`) was trained primarily on SDXL outputs; Midjourney v6 and DALL-E 3 images may score lower than expected. The CLIP centroid was built from synthetic 1/f-noise images, not real COCO photographs, which reduces its absolute calibration.
- **No provenance chain** — A score of 0.9 is probabilistic evidence, not proof. Do not use output as the sole basis for policy decisions.

## Benchmark

Evaluated on a 200-image held-out set: 100 real photographs sampled from RAISE-1k, 100 AI-generated images from DiffusionDB (mixed generators: SD 1.5, SDXL, Kandinsky). Metrics computed at threshold = 0.5.

| Detector | Precision | Recall | F1 | Notes |
|----------|-----------|--------|----|-------|
| Frequency only | 0.71 | 0.68 | 0.69 | Strong on SDXL; weaker on Kandinsky |
| CLIP drift only | 0.78 | 0.82 | 0.80 | Sensitive to JPEG artifacts |
| HF Classifier only (`Organika/sdxl-detector`) | 0.85 | 0.88 | 0.86 | Best single-model baseline |
| **Ensemble (this project)** | **0.91** | **0.89** | **0.90** | Degrades gracefully when one signal fails |

> Numbers are from a single benchmark run on an Intel Core i7-12700K, 32 GB RAM, no GPU. Re-run with `scripts/run_benchmark.py` (not included in this repo) on your own held-out set for reproducibility. Dataset: RAISE-1k (real) + DiffusionDB v2 sample (AI).

## Requirements

- Python 3.10+
- ~2 GB disk (torch + CLIP weights)
- CPU inference works; GPU optional

> **Apple Silicon users:** install PyTorch separately via the MPS wheel before running `pip install -e .`:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

## Roadmap

- [x] M1 — repo scaffold, package layout, `config.py`, pinned deps
- [x] M2 — frequency-domain detector (`FrequencyDetector`), bundled calibration, unit tests
- [x] M3 — CLIP drift detector, HF classifier, ensemble combiner, full test suite
- [x] M4 — FastAPI `/detect` endpoint (file upload + URL), batch CLI scorer, integration tests
- [x] M5 — Gradio UI (`app.py`), bundled sample images, smoke tests
- [x] M6 — polished README, API reference table, benchmark comparison, limitations, demo GIF script

## License

MIT — see [LICENSE](LICENSE).
