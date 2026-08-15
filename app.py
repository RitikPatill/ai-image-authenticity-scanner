"""Gradio Blocks UI for the AI Image Authenticity Scanner.

Launch:
    python app.py

Public API (also consumed by tests):
    predict(image, url) -> (label_str, fig, result_dict)
    _make_bar_chart(signals) -> matplotlib.figure.Figure
"""

from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.figure

import httpx
from PIL import Image

import gradio as gr


# ---------------------------------------------------------------------------
# Core helpers (importable without triggering Gradio server logic)
# ---------------------------------------------------------------------------

def _make_bar_chart(signals: dict) -> matplotlib.figure.Figure:
    """Return a horizontal bar chart of per-signal AI-probability scores."""
    labels = ["Frequency", "CLIP Drift", "HF Classifier"]
    values = [
        signals["frequency"]["score"],
        signals["clip_drift"]["score"],
        signals["hf_classifier"]["score"],
    ]
    colors = ["#e74c3c" if v >= 0.5 else "#2ecc71" for v in values]

    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.barh(labels, values, color=colors)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("AI probability score")
    fig.tight_layout()
    return fig


def predict(
    image: Image.Image | None,
    url: str,
) -> tuple[str, matplotlib.figure.Figure, dict]:
    """Score an image and return (label, bar_chart_fig, raw_result).

    Priority: URL > uploaded image.
    Raises gr.Error on bad input or failed URL fetch.
    """
    from ai_image_scanner.ensemble import score_image

    img: Image.Image | None = None

    if url and url.strip():
        try:
            resp = httpx.get(url.strip(), timeout=10, follow_redirects=True)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except httpx.HTTPError as exc:
            raise gr.Error(f"Failed to fetch URL: {exc}") from exc
        except Exception as exc:
            raise gr.Error(f"Could not decode image from URL: {exc}") from exc
    elif image is not None:
        img = image.convert("RGB") if image.mode != "RGB" else image
    else:
        raise gr.Error("Please upload an image or paste a public image URL.")

    result = score_image(img)

    label_str = f"{result['verdict']} ({result['score'] * 100:.0f}%)"
    fig = _make_bar_chart(result["signals"])

    return label_str, fig, result


# ---------------------------------------------------------------------------
# Gradio layout — only constructed when the module is the entry point or
# explicitly requested; never called at import time.
# ---------------------------------------------------------------------------

_SAMPLE_DIR = "assets/samples"
_EXAMPLES = [
    [f"{_SAMPLE_DIR}/real_01.png", ""],
    [f"{_SAMPLE_DIR}/real_02.png", ""],
    [f"{_SAMPLE_DIR}/real_03.png", ""],
    [f"{_SAMPLE_DIR}/ai_01.png", ""],
    [f"{_SAMPLE_DIR}/ai_02.png", ""],
    [f"{_SAMPLE_DIR}/ai_03.png", ""],
]


def build_ui() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""
    with gr.Blocks(theme=gr.themes.Soft(), title="AI Image Authenticity Scanner") as demo:
        gr.Markdown(
            "# AI Image Authenticity Scanner\n"
            "Upload an image **or** paste a public URL to score its AI-generation probability."
        )

        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload image")
            url_input = gr.Textbox(label="…or paste a public image URL", placeholder="https://")

        analyse_btn = gr.Button("Analyse", variant="primary")

        with gr.Row():
            verdict_out = gr.Label(label="Verdict")
            chart_out = gr.Plot(label="Signal breakdown")
            json_out = gr.JSON(label="Raw result")

        analyse_btn.click(
            fn=predict,
            inputs=[image_input, url_input],
            outputs=[verdict_out, chart_out, json_out],
        )

        gr.Examples(
            examples=_EXAMPLES,
            inputs=[image_input, url_input],
            outputs=[verdict_out, chart_out, json_out],
            fn=predict,
            label="Sample images (3 real · 3 AI-synthetic)",
        )

    return demo


if __name__ == "__main__":
    build_ui().launch()
