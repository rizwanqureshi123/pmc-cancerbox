"""
Repair bbox/regions entries in PMC labeled JSON files.

Fixes:
- negative coordinates (which break NumPy slicing)
- point boxes with zero width/height
- out-of-bounds coordinates
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

import cv2

from pmc_labeled_dataset import is_relevant, normalize_bbox


def load_image_shape(
    item: dict,
    image_dir: Path | None,
    cache_dir: Path,
) -> tuple[int, int] | None:
    candidates: list[Path] = []
    if item.get("file_path"):
        candidates.append(Path(item["file_path"]))
    if image_dir and item.get("url"):
        candidates.append(image_dir / Path(item["url"]).name)

    for candidate in candidates:
        if candidate.exists():
            image = cv2.imread(str(candidate), cv2.IMREAD_UNCHANGED)
            if image is not None:
                height, width = image.shape[:2]
                return height, width

    url = item.get("url")
    if not url:
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_path = cache_dir / Path(url).name
    if not cached_path.exists():
        try:
            with urlopen(url, timeout=60) as response:
                cached_path.write_bytes(response.read())
        except OSError:
            return None

    image = cv2.imread(str(cached_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    height, width = image.shape[:2]
    return height, width


def apply_normalized_bbox(item: dict, bbox: list[int]) -> None:
    x0, y0, x1, y1 = bbox
    item["bbox"] = [x0, y0, x1, y1]
    item["regions"] = [
        {
            "x": x0,
            "y": y0,
            "width": x1 - x0,
            "height": y1 - y0,
        }
    ]


def repair_json(
    json_path: Path,
    image_dir: Path | None,
    cache_dir: Path,
    remove_invalid: bool,
) -> dict[str, int]:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    stats = {"checked": 0, "fixed": 0, "removed": 0, "invalid": 0}
    changed = False

    for entry in data:
        images = [entry] if "images" not in entry else entry["images"]
        for item in images:
            if not is_relevant(item) or not item.get("bbox"):
                continue

            stats["checked"] += 1
            image_size = load_image_shape(item, image_dir, cache_dir)
            if image_size is None:
                stats["invalid"] += 1
                continue

            height, width = image_size
            normalized = normalize_bbox(item["bbox"], height, width)
            if normalized is None:
                stats["invalid"] += 1
                if remove_invalid:
                    item.pop("bbox", None)
                    item.pop("regions", None)
                    stats["removed"] += 1
                    changed = True
                continue

            if normalized != [int(v) for v in item["bbox"]]:
                apply_normalized_bbox(item, normalized)
                stats["fixed"] += 1
                changed = True

    if changed:
        backup_path = json_path.with_suffix(json_path.suffix + ".bak")
        if not backup_path.exists():
            backup_path.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair bbox coordinates in PMC labeled JSON files.")
    parser.add_argument("--json", nargs="+", required=True, help="JSON files to repair.")
    parser.add_argument("--image-dir", type=str, default="", help="Optional local image directory.")
    parser.add_argument("--cache-dir", type=str, default="pmc_image_cache", help="Image download cache.")
    parser.add_argument(
        "--remove-invalid",
        action="store_true",
        help="Remove bbox/regions when they cannot be normalized.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir) if args.image_dir else None
    cache_dir = Path(args.cache_dir)

    total = {"checked": 0, "fixed": 0, "removed": 0, "invalid": 0}
    for json_path in args.json:
        stats = repair_json(Path(json_path), image_dir, cache_dir, args.remove_invalid)
        print(
            f"{json_path}: checked={stats['checked']} fixed={stats['fixed']} "
            f"removed={stats['removed']} invalid={stats['invalid']}"
        )
        for key in total:
            total[key] += stats[key]

    print(
        f"Total: checked={total['checked']} fixed={total['fixed']} "
        f"removed={total['removed']} invalid={total['invalid']}"
    )


if __name__ == "__main__":
    main()
