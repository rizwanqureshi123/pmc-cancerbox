"""
Build a random sample for manual quality assessment of masks and
image-caption pairs. Produces a CSV manifest with blank columns to fill
in during review, and copies the corresponding overlay PNGs into a
review folder so you can open images and the spreadsheet side by side.

Usage:
    python build_manual_review_sample.py --masks-dir pmc_outputs_v2/masks --overlays-dir pmc_outputs_v2/overlays --sample-size 250
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a manual review sample.")
    parser.add_argument("--masks-dir", type=str, default="pmc_outputs_v2/masks")
    parser.add_argument("--overlays-dir", type=str, default="pmc_outputs_v2/overlays")
    parser.add_argument("--sample-size", type=int, default=250)
    parser.add_argument("--output-dir", type=str, default="manual_review")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_dir = Path(args.masks_dir)
    overlay_dir = Path(args.overlays_dir)
    out_dir = Path(args.output_dir)
    images_out = out_dir / "overlays"
    images_out.mkdir(parents=True, exist_ok=True)

    files = sorted(mask_dir.glob("*.npz"))
    if not files:
        print(f"No .npz files found in {mask_dir}")
        return

    random.seed(args.seed)
    sample = random.sample(files, min(args.sample_size, len(files)))

    rows = []
    for f in sample:
        data = np.load(f, allow_pickle=True)
        stem = f.stem
        overlay_src = overlay_dir / f"{stem}.png"
        overlay_dest = images_out / f"{stem}.png"
        has_overlay = overlay_src.exists()
        if has_overlay:
            shutil.copy(overlay_src, overlay_dest)

        rows.append(
            {
                "id": stem,
                "modality": str(data["modality"]) if "modality" in data else "",
                "bbox_source": str(data["bbox_source"]) if "bbox_source" in data else "",
                "predicted_iou": float(data["iou"]),
                "caption": str(data["caption"]) if "caption" in data else "",
                "url": str(data["url"]) if "url" in data else "",
                "overlay_file": f"{stem}.png" if has_overlay else "",
                "mask_quality_1to5": "",
                "caption_matches_image_yn": "",
                "notes": "",
            }
        )

    csv_path = out_dir / "manual_review_sample.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Sampled {len(rows)} entries.")
    print(f"CSV manifest: {csv_path}")
    print(f"Overlay images copied to: {images_out}")
    print(
        "\nOpen the CSV in Excel and, while viewing each overlay image, fill in:\n"
        "  mask_quality_1to5        (1 = bad, 5 = excellent segmentation)\n"
        "  caption_matches_image_yn (y/n -- does the caption describe what's shown)\n"
        "  notes                    (anything worth flagging)"
    )


if __name__ == "__main__":
    main()