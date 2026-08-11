# AI Image Authenticity Scanner

A locally-runnable REST API and Gradio UI that scores any image on the probability that it was AI-generated. Instead of relying on a single model, it combines three orthogonal detection signals into one explainable confidence score.

With AI-generated images flooding stock libraries, social media, and scientific papers, there is real demand for fast, explainable detection. Existing tools are either cloud-only, opaque, or single-model. This project gives developers a self-hosted, explainable ensemble they can embed in pipelines — no external API calls required after installation.

## Status

| Milestone | Scope | State |
|-----------|-------|-------|
| M1 — Scaffold | Package layout, `config.py`, pinned `requirements.txt`, `calibration/` dir, MIT license | **Done** |
| M2 — Detectors | Frequency fingerprint detector (`detectors/frequency.py`), bundled calibration, CLI build script | **Done** |
| M3 — API | FastAPI `/score` endpoint, file-upload and URL modes | Planned |
| M4 — CLI & UI | Batch CLI scorer, Gradio browser interface | Planned |
| M5 — Calibration | Generate and bundle `clip_centroid_real.npy` / `clip_centroid_ai.npy` | Planned |

### What works now (M2)

- `pip install -e .` resolves the `ai_image_scanner` package from repo root
- `ai_image_scanner/config.py` — detector weights and HuggingFace model ID are configurable without touching source
- **`ai_image_scanner/detectors/frequency.py`** — `FrequencyDetector` class and `score_image()` convenience function; computes a 2D DCT ring-power spectrum and returns a 0–1 AI-probability score
- `calibration/frequency_calibration.npz` — bundled calibration statistics (no download needed)
- `scripts/build_frequency_calibration.py` — regenerate calibration from images in `calibration/samples/{real,ai}/`; auto-generates synthetic stand-ins when the folders are empty
- `tests/test_frequency.py` — 6 unit tests; run with `pytest tests/test_frequency.py -v`

## Planned features

- **Ensemble of three detectors** — frequency fingerprint, CLIP embedding drift, and a pre-trained HuggingFace classifier
- **Per-signal breakdown** — the response JSON shows exactly why the verdict was reached
- **REST API + Gradio UI** — integrate via HTTP or use the browser interface
- **Batch CLI** — score an entire folder in one command
- **Pre-bundled calibration centroids** — no internet required at runtime after `pip install`
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

# 2. Start the API server  [requires M3]
uvicorn app:app --reload

# 3. Launch the Gradio UI  [requires M4]
python -m ai_image_scanner.ui

# 4. Run the CLI on a folder  [requires M4]
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
    "frequency": 0.72,
    "clip_drift": 0.91,
    "classifier": 0.93
  },
  "model_version": "0.1.0"
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
- [ ] M3 — FastAPI server with `/score` endpoint (file upload + URL)
- [ ] M4 — Gradio UI and batch CLI scorer
- [ ] M5 — generate and bundle CLIP calibration centroids; offline inference end-to-end

<!-- TODO: add benchmark table (precision / recall on RAISE-1k + DiffusionDB sample) once M2 is done -->

## License

MIT — see [LICENSE](LICENSE).
