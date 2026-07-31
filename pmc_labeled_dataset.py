"""
Load PMC relevance-labeled JSON files, filter relevant images in code,
extract bounding boxes from annotation regions, and infer modality prompts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import cv2
import numpy as np

RELEVANT_LABELS = {"relevant"}
NON_RELEVANT_LABELS = {"not_relevant", "irrelevant", "not relevant"}

CATEGORIES_MAP = {
    "CT": 0,
    "MR": 1,
    "Endoscopy": 2,
    "XRay": 3,
    "X-Ray": 3,
    "PET": 4,
    "Dermoscopy": 5,
    "Mammography": 6,
    "Mammo": 6,
    "US": 7,
    "OCT": 8,
    "Fundus": 9,
    "Microscopy": 10,
    "Microscope": 10,
}

MODALITY_KEYWORDS: list[tuple[str, list[str]]] = [
    ("MR", [r"\bmri\b", r"\bmr\b", r"magnetic resonance", r"t1-weighted", r"t2-weighted", r"flair"]),
    ("CT", [r"\bct\b", r"computed tomography", r"ct scan"]),
    ("PET", [r"\bpet\b", r"positron emission", r"micropet", r"scintimammograph"]),
    ("Mammography", [r"mammograph", r"mammo\b", r"medio-lateral"]),
    ("US", [r"ultrasound", r"sonograph", r"\bus\b"]),
    ("X-Ray", [r"x-ray", r"xray", r"radiograph", r"chest x"]),
    ("Endoscopy", [r"endoscop"]),
    ("Dermoscopy", [r"dermoscop"]),
    ("OCT", [r"optical coherence", r"\boct\b"]),
    ("Fundus", [r"fundus", r"retinal"]),
    ("Microscopy", [r"microscop", r"histolog", r"\bh&e\b", r"prussian blue"]),
]


def is_relevant(item: dict[str, Any]) -> bool:
    if "is_relevant" in item:
        return bool(item["is_relevant"])

    label = str(item.get("label", "")).strip().lower()
    if label in RELEVANT_LABELS:
        return True
    if label in NON_RELEVANT_LABELS:
        return False
    return False


def infer_modality(caption: str, explicit: str | None = None) -> str:
    if explicit:
        modality = explicit.strip()
        if modality in CATEGORIES_MAP:
            return "X-Ray" if modality == "XRay" else modality
        if modality.lower() == "mri":
            return "MR"
        if modality.lower() in {"ultrasound", "ultrasonography"}:
            return "US"

    text = (caption or "").lower()
    for modality, patterns in MODALITY_KEYWORDS:
        if any(re.search(pattern, text) for pattern in patterns):
            return modality
    return "CT"


def normalize_bbox(
    bbox: list[int],
    height: int,
    width: int,
    min_size: int = 2,
) -> list[int] | None:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    if x1 <= x0:
        x1 = min(width, x0 + min_size)
    if y1 <= y0:
        y1 = min(height, y0 + min_size)
    if x1 - x0 < min_size or y1 - y0 < min_size:
        return None
    return [x0, y0, x1, y1]


def bbox_from_points(points: np.ndarray) -> list[int]:
    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    return [int(x), int(y), int(x + w), int(y + h)]


def bbox_from_region(region: dict[str, Any]) -> list[int] | None:
    if "bbox" in region and len(region["bbox"]) == 4:
        x_min, y_min, x_max, y_max = [int(v) for v in region["bbox"]]
        return [x_min, y_min, x_max, y_max]

    if all(key in region for key in ("x", "y", "width", "height")):
        x = int(region["x"])
        y = int(region["y"])
        w = int(region["width"])
        h = int(region["height"])
        return [x, y, x + w, y + h]

    if "points" in region and region["points"]:
        points = np.array(region["points"], dtype=np.float32)
        if points.ndim == 1 and points.size == 4:
            x, y, w, h = points
            return [int(x), int(y), int(x + w), int(y + h)]
        if points.ndim == 2 and points.shape[1] == 2:
            return bbox_from_points(points)
    return None


def extract_bboxes(item: dict[str, Any], full_image_size: tuple[int, int] | None = None) -> list[list[int]]:
    if "bbox" in item and len(item["bbox"]) == 4:
        return [[int(v) for v in item["bbox"]]]

    bboxes: list[list[int]] = []
    for region in item.get("regions", []):
        bbox = bbox_from_region(region)
        if bbox is not None:
            bboxes.append(bbox)

    if not bboxes and full_image_size is not None:
        height, width = full_image_size
        if height > 0 and width > 0:
            margin = max(1, min(height, width) // 20)
            bboxes.append([margin, margin, width - margin, height - margin])

    return bboxes


def iter_json_records(data: list[Any]) -> Iterator[dict[str, Any]]:
    for entry in data:
        if "images" in entry:
            article_meta = {
                "pmcid": entry.get("pmcid"),
                "article_url": entry.get("article_url"),
            }
            for image in entry["images"]:
                yield {**article_meta, **image}
        else:
            yield entry


def resolve_image_path(
    item: dict[str, Any],
    image_dir: Path | None = None,
) -> Path | None:
    if item.get("file_path"):
        path = Path(item["file_path"])
        if path.exists():
            return path

    if image_dir is None:
        return None

    candidates: list[Path] = []
    if item.get("local_filename"):
        candidates.append(image_dir / item["local_filename"])

    url = item.get("url")
    if url:
        candidates.append(image_dir / Path(urlparse(url).path).name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_relevant_scans(
    json_path: str | Path,
    image_dir: str | Path | None = None,
    require_bbox: bool = True,
    allow_full_image_bbox: bool = False,
    default_modality: str = "CT",
) -> list[dict[str, Any]]:
    json_path = Path(json_path)
    image_root = Path(image_dir) if image_dir is not None else None

    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    valid_inputs: list[dict[str, Any]] = []

    for item in iter_json_records(data):
        if not is_relevant(item):
            continue

        image_path = resolve_image_path(item, image_root)
        caption = item.get("caption", "")
        modality = infer_modality(caption, item.get("modality"))
        if modality not in CATEGORIES_MAP and default_modality in CATEGORIES_MAP:
            modality = default_modality

        image_size = None
        if image_path is not None:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is not None:
                image_size = image.shape[:2]

        bboxes = extract_bboxes(
            item,
            full_image_size=image_size if allow_full_image_bbox else None,
        )
        if require_bbox and not bboxes:
            continue

        for bbox in bboxes or [None]:
            if bbox is not None and image_size is not None:
                height, width = image_size
                bbox = normalize_bbox(bbox, height, width)
                if bbox is None:
                    continue
            valid_inputs.append(
                {
                    "image_path": str(image_path) if image_path is not None else None,
                    "url": item.get("url"),
                    "bbox": bbox,
                    "modality": modality,
                    "modality_text": f"{modality} Image",
                    "category_idx": CATEGORIES_MAP[modality],
                    "caption": caption,
                    "pmcid": item.get("pmcid"),
                    "article_url": item.get("article_url"),
                    "source_json": str(json_path),
                    "bbox_source": item.get("bbox_source"),
                    "bbox_confidence": item.get("bbox_confidence"),
                }
            )

    return valid_inputs


def export_relevant_manifest(
    json_paths: list[str | Path],
    output_path: str | Path,
    image_dir: str | Path | None = None,
) -> dict[str, int]:
    output_path = Path(output_path)
    merged: list[dict[str, Any]] = []
    stats = {"files": 0, "relevant": 0, "with_bbox": 0, "with_local_image": 0}

    for json_path in json_paths:
        stats["files"] += 1
        scans = load_relevant_scans(
            json_path,
            image_dir=image_dir,
            require_bbox=False,
            allow_full_image_bbox=False,
        )
        for scan in scans:
            stats["relevant"] += 1
            if scan["bbox"] is not None:
                stats["with_bbox"] += 1
            if scan["image_path"] is not None:
                stats["with_local_image"] += 1
            merged.append(scan)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2)

    return stats


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Export relevant PMC images from labeled JSON files."
    )
    parser.add_argument("--json", nargs="+", required=True, help="One or more labeled JSON files.")
    parser.add_argument("--output", required=True, help="Path for the merged manifest JSON.")
    parser.add_argument(
        "--image-dir",
        type=str,
        default="",
        help="Optional directory containing downloaded PMC images.",
    )
    args = parser.parse_args()

    image_dir = args.image_dir or None
    stats = export_relevant_manifest(args.json, args.output, image_dir=image_dir)
    print(
        "Exported manifest: "
        f"{stats['relevant']} relevant scans from {stats['files']} file(s). "
        f"{stats['with_bbox']} have bounding boxes, "
        f"{stats['with_local_image']} have local images."
    )


if __name__ == "__main__":
    main()