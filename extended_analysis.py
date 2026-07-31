"""
Extended MCP-MedSAM output analysis.

Adds to the base analyze_results.py:
  - Standard deviation of predicted IoU
  - % of masks with IoU >= 0.5, >= 0.6, >= 0.7
  - Modality distribution (bar chart)
  - Image resolution statistics (derived from mask shape, which matches
    the original image resolution used for inference)
  - Disease distribution: breast cancer vs. brain tumor vs. other
    (classified from caption keywords -- a simple heuristic, not a
    clinical diagnosis; documented as such in your report)

Usage:
    python extended_analysis.py --masks-dir pmc_outputs_v2/masks --output-dir eval_charts
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def classify_disease(caption: str) -> str:
    c = (caption or "").lower()
    breast = any(k in c for k in ["breast", "mammary", "mammograph"])
    brain = any(k in c for k in ["brain", "cerebral", "glioma", "glioblastoma", "intracranial"])
    if breast and brain:
        return "both/ambiguous"
    if breast:
        return "breast cancer"
    if brain:
        return "brain tumor"
    return "other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extended analysis of MCP-MedSAM outputs.")
    parser.add_argument("--masks-dir", type=str, default="pmc_outputs_v2/masks")
    parser.add_argument("--output-dir", type=str, default="eval_charts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_dir = Path(args.masks_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(mask_dir.glob("*.npz"))
    if not files:
        print(f"No .npz files found in {mask_dir}")
        return

    ious = []
    modalities: Counter = Counter()
    diseases: Counter = Counter()
    resolutions = []

    for f in files:
        data = np.load(f, allow_pickle=True)
        iou = float(data["iou"])
        modality = str(data["modality"]) if "modality" in data else "unknown"
        caption = str(data["caption"]) if "caption" in data else ""
        mask = data["mask"]

        ious.append(iou)
        modalities[modality] += 1
        diseases[classify_disease(caption)] += 1
        resolutions.append(mask.shape[:2])  # (H, W)

    ious_arr = np.array(ious)
    heights = np.array([r[0] for r in resolutions])
    widths = np.array([r[1] for r in resolutions])

    print(f"Total masks: {len(files)}")
    print(f"Mean IoU:    {ious_arr.mean():.4f}")
    print(f"Std IoU:     {ious_arr.std():.4f}")
    print(f"Median IoU:  {np.median(ious_arr):.4f}")
    print()
    for t in (0.5, 0.6, 0.7):
        pct = 100 * (ious_arr >= t).sum() / len(ious_arr)
        print(f"% masks with IoU >= {t}: {pct:.2f}%")
    print()
    print("Image resolution statistics (H x W, from mask shape):")
    print(f"  Height: mean={heights.mean():.1f} median={np.median(heights):.1f} min={heights.min()} max={heights.max()}")
    print(f"  Width:  mean={widths.mean():.1f} median={np.median(widths):.1f} min={widths.min()} max={widths.max()}")
    print()
    print("Disease distribution (caption-keyword heuristic, not clinical ground truth):")
    for k, v in diseases.most_common():
        print(f"  {k:20s} {v} ({100 * v / len(files):.1f}%)")

    # --- Charts ---
    mods, mod_counts = zip(*modalities.most_common())
    plt.figure(figsize=(8, 5))
    plt.bar(mods, mod_counts, color="#4C72B0")
    plt.title("Image Modality Distribution")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "modality_distribution.png", dpi=150)
    plt.close()

    dis, dis_counts = zip(*diseases.most_common())
    plt.figure(figsize=(6, 5))
    plt.bar(dis, dis_counts, color="#DD8452")
    plt.title("Disease Distribution (caption-keyword based)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "disease_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(ious_arr, bins=30, color="#55A868")
    plt.title("Predicted IoU Distribution")
    plt.xlabel("IoU")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "iou_distribution.png", dpi=150)
    plt.close()

    print(f"\nCharts saved to {out_dir}/")


if __name__ == "__main__":
    main()