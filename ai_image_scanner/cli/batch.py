"""Batch folder scanner for AI Image Authenticity Scanner.

Usage:
    python -m ai_image_scanner.cli.batch ./images
    python -m ai_image_scanner.cli.batch ./images --output report.csv --recursive
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from ai_image_scanner.ensemble import score_image

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _iter_images(folder: Path, recursive: bool):
    if recursive:
        candidates = folder.rglob("*")
    else:
        candidates = folder.iterdir()
    for path in sorted(candidates):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            yield path


def main(folder: str, output: str = "report.csv", recursive: bool = False) -> None:
    folder_path = Path(folder)
    if not folder_path.is_dir():
        print(f"Error: {folder!r} is not a directory", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    for image_path in _iter_images(folder_path, recursive):
        try:
            img = Image.open(image_path).convert("RGB")
        except UnidentifiedImageError:
            print(f"Warning: skipping unreadable file {image_path}", file=sys.stderr)
            continue

        result = score_image(img)
        signals = result["signals"]
        rows.append(
            {
                "filename": str(image_path),
                "score": result["score"],
                "verdict": result["verdict"],
                "frequency_score": signals["frequency"]["score"],
                "clip_drift_score": signals["clip_drift"]["score"],
                "hf_classifier_score": signals["hf_classifier"]["score"],
            }
        )

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "score",
                "verdict",
                "frequency_score",
                "clip_drift_score",
                "hf_classifier_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed {len(rows)} images → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch-score a folder of images for AI generation probability."
    )
    parser.add_argument("folder", help="Path to folder containing images")
    parser.add_argument(
        "--output", default="report.csv", help="Output CSV path (default: report.csv)"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Descend into subdirectories",
    )
    args = parser.parse_args()
    main(args.folder, args.output, args.recursive)
