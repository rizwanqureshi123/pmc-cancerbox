"""
Dataset bias / imbalance analysis.

Goes beyond raw counts to compute imbalance ratios and flag
overrepresented categories, so the report can discuss *why* this matters
for downstream model training (a model trained on an imbalanced dataset
will tend to perform best on the majority classes and worst on rare ones).

Usage:
    python bias_analysis.py --masks-dir pmc_outputs_v2/masks --output-dir eval_charts
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RADIOLOGY_MODALITIES = {"CT", "MR", "PET", "X-Ray", "Mammography", "US", "Fundus", "OCT"}
HISTOPATHOLOGY_MODALITIES = {"Microscopy"}
# Everything else (Endoscopy, etc.) is bucketed as "other"


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


def image_category(modality: str) -> str:
    if modality in RADIOLOGY_MODALITIES:
        return "radiology"
    if modality in HISTOPATHOLOGY_MODALITIES:
        return "histopathology"
    return "other"


def print_ratio_table(counter: Counter, title: str) -> None:
    total = sum(counter.values())
    print(f"\n{title}")
    items = counter.most_common()
    top_count = items[0][1] if items else 0
    for name, count in items:
        pct = 100 * count / total if total else 0
        ratio_to_top = f"1 : {top_count / count:.1f}" if count else "n/a"
        flag = "  <-- majority class" if count == top_count else ""
        print(f"  {name:20s} n={count:6d}  ({pct:5.1f}%)  ratio to majority = {ratio_to_top}{flag}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze dataset bias/imbalance.")
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

    modality_counts: Counter = Counter()
    disease_counts: Counter = Counter()
    category_counts: Counter = Counter()
    # Cross-tab: modality -> disease -> count
    cross_tab: dict[str, Counter] = {}

    for f in files:
        data = np.load(f, allow_pickle=True)
        modality = str(data["modality"]) if "modality" in data else "unknown"
        caption = str(data["caption"]) if "caption" in data else ""
        disease = classify_disease(caption)
        category = image_category(modality)

        modality_counts[modality] += 1
        disease_counts[disease] += 1
        category_counts[category] += 1
        cross_tab.setdefault(modality, Counter())[disease] += 1

    print(f"Total masks analyzed: {len(files)}")

    print_ratio_table(modality_counts, "MODALITY DISTRIBUTION (imbalance ratios)")
    print_ratio_table(disease_counts, "DISEASE DISTRIBUTION (caption-keyword heuristic)")
    print_ratio_table(category_counts, "RADIOLOGY vs HISTOPATHOLOGY vs OTHER")

    print("\nMODALITY x DISEASE cross-tab (counts):")
    all_diseases = sorted(disease_counts.keys())
    header = "  " + f"{'modality':20s}" + "".join(f"{d:>16s}" for d in all_diseases)
    print(header)
    for modality, _ in modality_counts.most_common():
        row = cross_tab.get(modality, Counter())
        line = f"  {modality:20s}" + "".join(f"{row.get(d, 0):16d}" for d in all_diseases)
        print(line)

    # --- Interpretive flags ---
    print("\nAUTOMATED IMBALANCE FLAGS (thresholds: >50% = dominant, <5% = underrepresented):")
    total = len(files)
    for name, count in modality_counts.items():
        pct = 100 * count / total
        if pct > 50:
            print(f"  ⚠ '{name}' is DOMINANT: {pct:.1f}% of the dataset.")
        elif pct < 5:
            print(f"  ⚠ '{name}' is UNDERREPRESENTED: only {pct:.1f}% of the dataset.")

    # --- Charts ---
    mods, mod_counts = zip(*modality_counts.most_common())
    plt.figure(figsize=(8, 5))
    plt.bar(mods, mod_counts, color="#4C72B0")
    plt.title("Modality Distribution (Bias Check)")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "bias_modality_distribution.png", dpi=150)
    plt.close()

    cats, cat_counts = zip(*category_counts.most_common())
    plt.figure(figsize=(6, 5))
    plt.bar(cats, cat_counts, color="#C44E52")
    plt.title("Radiology vs Histopathology vs Other")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "bias_radiology_vs_histopathology.png", dpi=150)
    plt.close()

    print(f"\nCharts saved to {out_dir}/")


if __name__ == "__main__":
    main()