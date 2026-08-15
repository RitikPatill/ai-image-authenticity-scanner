"""
Generate assets/demo.gif from the 6 bundled sample images.

Uses hard-coded plausible scores — no model weights or calibration files needed.
Requires only Pillow and matplotlib (both in requirements.txt).

Usage:
    python scripts/generate_demo_gif.py
    python scripts/generate_demo_gif.py --out assets/demo.gif --fps 1
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# ---------------------------------------------------------------------------
# Hard-coded sample specs — no detectors are called at runtime
# ---------------------------------------------------------------------------

SAMPLES_DIR = Path(__file__).parent.parent / "assets" / "samples"

SAMPLE_SPECS: list[tuple[str, str, dict[str, float]]] = [
    # (filename, label_with_score, {signal: score})
    ("real_01.png", "REAL (12%)", {"frequency": 0.08, "clip_drift": 0.10, "hf_classifier": 0.17}),
    ("real_02.png", "REAL (09%)", {"frequency": 0.05, "clip_drift": 0.12, "hf_classifier": 0.11}),
    ("real_03.png", "REAL (15%)", {"frequency": 0.11, "clip_drift": 0.14, "hf_classifier": 0.20}),
    ("ai_01.png",   "AI  (91%)", {"frequency": 0.88, "clip_drift": 0.93, "hf_classifier": 0.92}),
    ("ai_02.png",   "AI  (87%)", {"frequency": 0.79, "clip_drift": 0.91, "hf_classifier": 0.90}),
    ("ai_03.png",   "AI  (94%)", {"frequency": 0.92, "clip_drift": 0.95, "hf_classifier": 0.95}),
]

SIGNAL_LABELS = ["frequency", "clip_drift", "hf_classifier"]
SIGNAL_DISPLAY = ["Frequency", "CLIP drift", "HF Classifier"]
FRAME_W, FRAME_H = 640, 320
DURATION_MS = 1500


def _bar_color(score: float) -> str:
    """Green for low AI probability, red for high."""
    return "#e74c3c" if score >= 0.5 else "#2ecc71"


def make_frame(img_path: Path, label: str, scores: dict[str, float]) -> Image.Image:
    """Render one 640×320 RGB composite frame."""
    fig, (ax_img, ax_bar) = plt.subplots(
        1, 2,
        figsize=(FRAME_W / 100, FRAME_H / 100),
        dpi=100,
        gridspec_kw={"width_ratios": [1, 1.4]},
    )
    fig.patch.set_facecolor("#1a1a2e")

    # --- Left: image thumbnail ---
    try:
        pil_img = Image.open(img_path).convert("RGB")
    except FileNotFoundError:
        # If sample image is missing, use a placeholder
        pil_img = Image.new("RGB", (128, 128), color=(80, 80, 100))

    ax_img.imshow(pil_img)
    ax_img.axis("off")

    # Verdict text above the image
    is_ai = label.strip().startswith("AI")
    verdict_color = "#e74c3c" if is_ai else "#2ecc71"
    ax_img.set_title(
        label,
        color=verdict_color,
        fontsize=14,
        fontweight="bold",
        pad=6,
    )

    # --- Right: horizontal bar chart ---
    ax_bar.set_facecolor("#16213e")
    signal_scores = [scores[s] for s in SIGNAL_LABELS]
    colors = [_bar_color(v) for v in signal_scores]
    y_pos = range(len(SIGNAL_DISPLAY))

    bars = ax_bar.barh(list(y_pos), signal_scores, color=colors, height=0.5, edgecolor="none")

    # Score labels at end of each bar
    for bar, val in zip(bars, signal_scores):
        ax_bar.text(
            min(val + 0.03, 0.97),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0%}",
            va="center",
            ha="left",
            color="white",
            fontsize=10,
        )

    ax_bar.set_xlim(0, 1.0)
    ax_bar.set_yticks(list(y_pos))
    ax_bar.set_yticklabels(SIGNAL_DISPLAY, color="white", fontsize=10)
    ax_bar.set_xlabel("AI probability", color="#aaaaaa", fontsize=9)
    ax_bar.tick_params(colors="#aaaaaa")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax_bar.spines[spine].set_color("#444466")
    ax_bar.axvline(0.5, color="#888899", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_bar.set_title("Signal breakdown", color="#ccccdd", fontsize=11, pad=6)

    plt.tight_layout(pad=1.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    frame = Image.open(buf).convert("RGB").resize((FRAME_W, FRAME_H), Image.LANCZOS)
    return frame


def main(out_path: str = "assets/demo.gif", fps: int = 1) -> None:
    """Write animated GIF from SAMPLE_SPECS list."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    frames: list[Image.Image] = []
    for filename, label, scores in SAMPLE_SPECS:
        img_path = SAMPLES_DIR / filename
        frame = make_frame(img_path, label, scores)
        frames.append(frame)
        print(f"  rendered frame: {label}")

    duration = int(1000 / fps)  # ms per frame
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=DURATION_MS,
        optimize=False,
    )
    print(f"Saved {len(frames)}-frame GIF -> {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate assets/demo.gif from bundled sample images")
    parser.add_argument("--out", default="assets/demo.gif", help="Output GIF path")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second (ignored; duration is fixed at 1500 ms)")
    args = parser.parse_args()
    main(out_path=args.out, fps=args.fps)
