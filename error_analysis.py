"""
Error analysis for MCP-MedSAM outputs.

Finds likely failure cases using automated heuristics:
  - Low IoU:            predicted_iou below a threshold (model itself is unsure)
  - Very small tumor:   mask area is a tiny fraction of the image
  - Multiple lesions:   mask has multiple disconnected components
  - Low contrast:       image has low standard deviation in pixel intensity
  - Empty mask:         model predicted nothing at all

Copies a representative sample of each failure category's overlay images
into labeled subfolders for easy visual inspection, and writes a CSV log.

Usage:
    python error_analysis.py --masks-dir pmc_outputs_v2/masks --overlays-dir pmc_outputs_v2/overlays --cache-dir pmc_image_cache --output-dir error_analysis --samples-per-category 5
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


def count_components(mask: np.ndarray) -> int:
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(mask_u8)
    return max(0, num_labels - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated error analysis of MCP-MedSAM outputs.")
    parser.add_argument("--masks-dir", type=str, default="pmc_outputs_v2/masks")
    parser.add_argument("--overlays-dir", type=str, default="pmc_outputs_v2/overlays")
    parser.add_argument("--cache-dir", type=str, default="pmc_image_cache")
    parser.add_argument("--output-dir", type=str, default="error_analysis")
    parser.add_argument("--low-iou-threshold", type=float, default=0.4)
    parser.add_argument("--small-tumor-max-fraction", type=float, default=0.005)
    parser.add_argument("--low-contrast-std-threshold", type=float, default=20.0)
    parser.add_argument("--samples-per-category", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_dir = Path(args.masks_dir)
    overlay_dir = Path(args.overlays_dir)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(mask_dir.glob("*.npz"))
    if not files:
        print(f"No .npz files found in {mask_dir}")
        return

    categories = {
        "low_iou": [],
        "very_small_tumor": [],
        "multiple_lesions": [],
        "low_contrast": [],
        "empty_mask": [],
    }
    rows = []

    for f in files:
        data = np.load(f, allow_pickle=True)
        stem = f.stem
        mask = data["mask"]
        iou = float(data["iou"])
        modality = str(data["modality"]) if "modality" in data else ""
        caption = str(data["caption"]) if "caption" in data else ""

        mask_area = mask.sum()
        total_area = mask.shape[0] * mask.shape[1]
        area_fraction = mask_area / total_area if total_area else 0
        n_components = count_components(mask) if mask_area > 0 else 0

        image_std = None
        image_path_field = str(data["image_path"]) if "image_path" in data else ""
        candidate = None
        if image_path_field and Path(image_path_field).exists():
            candidate = Path(image_path_field)
        else:
            url = str(data["url"]) if "url" in data else ""
            if url:
                guess = cache_dir / Path(url).name
                if guess.exists():
                    candidate = guess
        if candidate is not None:
            img = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                image_std = float(img.std())

        reasons = []
        if mask_area == 0:
            categories["empty_mask"].append(stem)
            reasons.append("empty_mask")
        if iou < args.low_iou_threshold:
            categories["low_iou"].append(stem)
            reasons.append("low_iou")
        if 0 < area_fraction < args.small_tumor_max_fraction:
            categories["very_small_tumor"].append(stem)
            reasons.append("very_small_tumor")
        if n_components >= 2:
            categories["multiple_lesions"].append(stem)
            reasons.append("multiple_lesions")
        if image_std is not None and image_std < args.low_contrast_std_threshold:
            categories["low_contrast"].append(stem)
            reasons.append("low_contrast")

        if reasons:
            rows.append(
                {
                    "id": stem,
                    "modality": modality,
                    "iou": iou,
                    "mask_area_fraction": round(area_fraction, 5),
                    "n_components": n_components,
                    "image_std": round(image_std, 2) if image_std is not None else "",
                    "reasons": ";".join(reasons),
                    "caption": caption[:200],
                }
            )

    if rows:
        csv_path = out_dir / "flagged_cases.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Flagged {len(rows)} cases total (may overlap categories). CSV: {csv_path}")

    print("\nFailure category counts and samples copied:")
    for category, stems in categories.items():
        cat_dir = out_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        sample = stems[: args.samples_per_category]
        copied = 0
        for stem in sample:
            src = overlay_dir / f"{stem}.png"
            if src.exists():
                shutil.copy(src, cat_dir / f"{stem}.png")
                copied += 1
        print(f"  {category:20s} total={len(stems):5d}  sampled={copied}  -> {cat_dir}")

    print(
        f"\nReview the sampled overlays in {out_dir}/<category>/ and write 1-2 sentences per "
        "category on WHY that failure mode likely occurs."
    )


if __name__ == "__main__":
    main()