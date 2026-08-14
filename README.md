# AI Image Authenticity Scanner

A locally-runnable REST API and Gradio UI that scores any image on the probability that it was AI-generated. Instead of relying on a single model, it combines three orthogonal detection signals into one explainable confidence score.

With AI-generated images flooding stock libraries, social media, and scientific papers, there is real demand for fast, explainable detection. Existing tools are either cloud-only, opaque, or single-model. This project gives developers a self-hosted, explainable ensemble they can embed in pipelines — no external API calls required after installation.

## Status

| Milestone | Scope | State |
|-----------|-------|-------|
| M1 — Scaffold | Package layout, `config.py`, pinned `requirements.txt`, `calibration/` dir, MIT license | **Done** |
| M2 — Detectors | Frequency fingerprint detector (`detectors/frequency.py`), bundled calibration, CLI build script | **Done** |
| M3 — CLIP + Classifier + Ensemble | CLIP drift detector, HF classifier, `ensemble.py` combining all three signals | **Done** |
| M4 — API | FastAPI `/score` endpoint, file-upload and URL modes | Planned |
| M5 — CLI & UI | Batch CLI scorer, Gradio browser interface | Planned |

### What works now (M3)

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

#### Build CLIP calibration (one-time setup)

```bash
# Downloads CLIP weights (~350 MB) once, then runs offline
python scripts/build_clip_calibration.py
```

## Planned features

- **REST API + Gradio UI** — integrate via HTTP or use the browser interface
- **Batch CLI** — score an entire folder in one command
- **Self-hosted** — runs entirely on your machine; no data leaves your network

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │         Image Input                  │
                        │   (file upload  /  URL  /  CLI)      │
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
# 1. Clone and install (editable mode so imports resolve from repo root)
#    Works today — M1 and M2 are complete.
git clone <repo-url>
cd ai-image-authenticity-scanner
pip install -e .

# 2. Start the API server  [requires M4]
uvicorn app:app --reload

# 3. Launch the Gradio UI  [requires M5]
python -m ai_image_scanner.ui

# 4. Run the CLI on a folder  [requires M5]
python -m ai_image_scanner.cli score --folder ./images/
```

### API usage

```bash
# Score by file upload
curl -X POST http://localhost:8000/score \
  -F "file=@photo.jpg"

# Score by URL
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/image.png"}'
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
  }
}
```

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
- [ ] M4 — FastAPI server with `/score` endpoint (file upload + URL)
- [ ] M5 — Gradio UI and batch CLI scorer

<!-- TODO: add benchmark table (precision / recall on RAISE-1k + DiffusionDB sample) once benchmark run is complete -->

## License

MIT — see [LICENSE](LICENSE).
