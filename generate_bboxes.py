"""
Auto-generate bounding boxes for relevant PMC images using Grounding DINO
(zero-shot, caption-guided object detection), and write them back into the
labeled JSON files so MCP-MedSAM inference picks them up automatically.

This writes a NEW file per input (default suffix "_with_bbox") rather than
overwriting your originals, so nothing is destroyed if something goes wrong.

Usage:
    python generate_bboxes.py --json ..\\labeled-dataset-1783958094047.json ..\\labeled-dataset1-1778366141328.json --cache-dir pmc_image_cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import pipeline

from pmc_labeled_dataset import infer_modality, is_relevant

# Modality -> candidate anatomical/pathology text prompts for Grounding DINO.
# These are tried in addition to any keyword found directly in the caption.
MODALITY_PROMPTS = {
    "MR": ["tumor", "lesion", "mass", "brain lesion", "abnormality"],
    "CT": ["tumor", "lesion", "mass", "nodule", "abnormality"],
    "PET": ["tumor", "hot spot", "uptake area", "lesion"],
    "Mammography": ["breast mass", "calcification", "tumor", "lesion"],
    "US": ["mass", "nodule", "lesion", "cyst"],
    "X-Ray": ["abnormality", "opacity", "lesion", "fracture"],
    "Endoscopy": ["lesion", "polyp", "abnormal tissue"],
    "Dermoscopy": ["skin lesion", "mole", "abnormal skin area"],
    "OCT": ["retinal abnormality", "lesion"],
    "Fundus": ["optic disc", "retinal lesion", "hemorrhage"],
    "Microscopy": ["abnormal cells", "tumor cells", "tissue abnormality"],
}
GENERIC_PROMPTS = ["tumor", "lesion", "mass", "abnormality"]

CAPTION_KEYWORDS = [
    "tumor", "tumour", "mass", "lesion", "nodule", "calcification", "cyst",
    "carcinoma", "metastasis", "metastases", "polyp", "hemorrhage", "haemorrhage",
    "fracture", "opacity", "aneurysm", "abscess", "infarct",
]


def build_prompts(caption: str, modality: str) -> list[str]:
    """Turn a (possibly long, messy) caption into a short list of candidate
    text prompts for Grounding DINO, prioritizing terms literally present
    in the caption, then falling back to modality-typical terms."""
    caption_lower = (caption or "").lower()
    found = [kw for kw in CAPTION_KEYWORDS if kw in caption_lower]
    modality_terms = MODALITY_PROMPTS.get(modality, GENERIC_PROMPTS)
    prompts: list[str] = []
    for term in found + modality_terms:
        if term not in prompts:
            prompts.append(term)
    return prompts[:6]  # cap to keep each inference call fast


def load_image_rgb(url: str, cache_dir: Path) -> np.ndarray | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(url).name or hashlib.sha1(url.encode("utf-8")).hexdigest() + ".jpg"
    cached_path = cache_dir / filename
    if not cached_path.exists():
        with urlopen(url, timeout=60) as response:
            cached_path.write_bytes(response.read())
    image = cv2.imread(str(cached_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=-1)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-generate bboxes via Grounding DINO.")
    parser.add_argument("--json", nargs="+", required=True, help="One or more labeled JSON files.")
    parser.add_argument("--cache-dir", type=str, default="pmc_image_cache")
    parser.add_argument("--output-suffix", type=str, default="_with_bbox")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Save progress to the output file every N processed images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)

    device = 0 if args.device.startswith("cuda") and torch.cuda.is_available() else -1
    print(f"Loading Grounding DINO (device={'cuda:0' if device == 0 else 'cpu'})...")
    detector = pipeline(
        model="IDEA-Research/grounding-dino-tiny",
        task="zero-shot-object-detection",
        device=device,
    )

    for json_path_str in args.json:
        json_path = Path(json_path_str)
        output_path = json_path.with_name(json_path.stem + args.output_suffix + json_path.suffix)

        # Resume support: if a partial output file already exists from a
        # previous interrupted run, continue from it instead of the original.
        source_path = output_path if output_path.exists() else json_path
        if source_path is output_path:
            print(f"Resuming from existing partial output: {output_path.name}")

        with source_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Flatten to a list of relevant image dicts that still need processing.
        pending: list[dict] = []
        total = 0
        skipped_existing = 0
        for entry in data:
            images = entry["images"] if "images" in entry else [entry]
            for image_item in images:
                if not is_relevant(image_item):
                    continue
                total += 1
                if len(image_item.get("bbox", []) or []) == 4 or image_item.get("regions"):
                    skipped_existing += 1
                    continue
                pending.append(image_item)

        updated = 0
        skipped_no_detection = 0
        skipped_error = 0
        processed_since_save = 0

        def save_progress() -> None:
            with output_path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)

        progress = tqdm(pending, desc=json_path.name, unit="img")
        for image_item in progress:
            url = image_item.get("url")
            if not url:
                skipped_error += 1
                continue

            try:
                image_rgb = load_image_rgb(url, cache_dir)
            except Exception as exc:
                print(f"  Failed to load {url}: {exc}")
                skipped_error += 1
                continue
            if image_rgb is None:
                skipped_error += 1
                continue

            caption = image_item.get("caption", "")
            modality = infer_modality(caption, image_item.get("modality"))
            prompts = build_prompts(caption, modality)
            pil_image = Image.fromarray(image_rgb)

            try:
                results = detector(
                    pil_image, candidate_labels=prompts, threshold=args.confidence_threshold
                )
            except Exception as exc:
                print(f"  Detection failed for {url}: {exc}")
                skipped_error += 1
                continue

            if not results:
                skipped_no_detection += 1
                continue

            best = max(results, key=lambda r: r["score"])
            box = best["box"]
            bbox = [int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])]

            image_item["bbox"] = bbox
            image_item["bbox_source"] = "grounding_dino"
            image_item["bbox_confidence"] = float(best["score"])
            image_item["bbox_label"] = best["label"]
            updated += 1

            processed_since_save += 1
            if processed_since_save >= args.checkpoint_every:
                save_progress()
                processed_since_save = 0

        save_progress()  # final save, catches any remainder

        print(
            f"{json_path.name}: {total} relevant images | "
            f"{updated} new bboxes this run | {skipped_existing} already had bbox/regions | "
            f"{skipped_no_detection} no detection above threshold | {skipped_error} errors"
        )
        print(f"  -> saved to {output_path}")


if __name__ == "__main__":
    main()