"""
Interactive bounding-box annotator for PMC labeled JSON files.

Draw a rectangle on each relevant image and save it back into the source JSON.
Use this before running inference if you want region-specific segmentation.

Controls:
  drag mouse  - draw bounding box
  s / Enter   - save bbox and go to next image
  r           - reset current box
  k           - skip image (no bbox saved)
  b           - go back to previous image
  q / Esc     - quit and save all changes
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

from pmc_labeled_dataset import is_relevant, iter_json_records, normalize_bbox


class BboxAnnotator:
    def __init__(self, window_name: str = "PMC BBox Annotator"):
        self.window_name = window_name
        self.drawing = False
        self.start_point: tuple[int, int] | None = None
        self.end_point: tuple[int, int] | None = None
        self.current_box: list[int] | None = None

    def _mouse_callback(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            if self.start_point is not None and self.end_point is not None:
                x0, y0 = self.start_point
                x1, y1 = self.end_point
                self.current_box = [
                    min(x0, x1),
                    min(y0, y1),
                    max(x0, x1),
                    max(y0, y1),
                ]

    def annotate(self, image_rgb: np.ndarray, title: str) -> str | list[int] | None:
        self.current_box = None
        self.start_point = None
        self.end_point = None
        canvas = image_rgb.copy()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        while True:
            display = canvas.copy()
            if self.start_point and self.end_point:
                cv2.rectangle(display, self.start_point, self.end_point, (0, 255, 0), 2)
            if self.current_box is not None:
                x0, y0, x1, y1 = self.current_box
                cv2.rectangle(display, (x0, y0), (x1, y1), (0, 0, 255), 2)

            cv2.imshow(self.window_name, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(20) & 0xFF

            if key in (13, ord("s")):
                if self.current_box is None:
                    continue
                x0, y0, x1, y1 = self.current_box
                if x1 - x0 < 2 or y1 - y0 < 2:
                    continue
                cv2.destroyWindow(self.window_name)
                return self.current_box
            if key == ord("r"):
                self.current_box = None
                self.start_point = None
                self.end_point = None
            if key == ord("k"):
                cv2.destroyWindow(self.window_name)
                return "skip"
            if key == ord("b"):
                cv2.destroyWindow(self.window_name)
                return "back"
            if key in (27, ord("q")):
                cv2.destroyWindow(self.window_name)
                return "quit"

    def close(self):
        cv2.destroyAllWindows()


def load_image_bgr(source: str | None, url: str | None, cache_dir: Path) -> np.ndarray | None:
    if source:
        path = Path(source)
        if path.exists():
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                return image

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

    return cv2.imread(str(cached_path), cv2.IMREAD_COLOR)


def image_needs_bbox(item: dict) -> bool:
    if not is_relevant(item):
        return False
    if item.get("bbox"):
        return False
    if item.get("regions"):
        return False
    return True


def apply_bbox_to_item(item: dict, bbox: list[int], image_shape: tuple[int, int]) -> bool:
    height, width = image_shape
    normalized = normalize_bbox(bbox, height, width)
    if normalized is None:
        return False
    x0, y0, x1, y1 = normalized
    item["bbox"] = [x0, y0, x1, y1]
    item["regions"] = [
        {
            "x": x0,
            "y": y0,
            "width": x1 - x0,
            "height": y1 - y0,
        }
    ]
    return True


def collect_tasks(
    json_paths: list[Path],
    only_missing: bool,
    start_index: int,
) -> list[dict]:
    tasks: list[dict] = []
    for json_path in json_paths:
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        for record_index, item in enumerate(iter_json_records(data)):
            if only_missing and not image_needs_bbox(item):
                continue
            if not is_relevant(item):
                continue
            tasks.append(
                {
                    "json_path": json_path,
                    "record_index": record_index,
                    "url": item.get("url"),
                    "caption": item.get("caption", ""),
                    "pmcid": item.get("pmcid"),
                }
            )

    if start_index > 0:
        tasks = tasks[start_index:]
    return tasks


def save_json(json_path: Path, data) -> None:
    backup_path = json_path.with_suffix(json_path.suffix + ".bak")
    if not backup_path.exists():
        backup_path.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_record(data, record_index: int) -> tuple[dict, dict | None]:
    current = 0
    for entry in data:
        if "images" in entry:
            for image in entry["images"]:
                if current == record_index:
                    return entry, image
                current += 1
        else:
            if current == record_index:
                return entry, None
            current += 1
    raise IndexError(f"Record index {record_index} not found")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate bounding boxes in PMC labeled JSON files.")
    parser.add_argument("--json", nargs="+", required=True, help="Labeled JSON files to update.")
    parser.add_argument("--image-dir", type=str, default="", help="Optional local PMC image directory.")
    parser.add_argument("--cache-dir", type=str, default="pmc_image_cache", help="Download cache for URL images.")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        default=True,
        help="Only annotate relevant images that do not already have bbox/regions.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start from this task index.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on images to annotate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_paths = [Path(path) for path in args.json]
    image_dir = Path(args.image_dir) if args.image_dir else None
    cache_dir = Path(args.cache_dir)

    tasks = collect_tasks(json_paths, args.only_missing, args.start_index)
    if args.limit > 0:
        tasks = tasks[: args.limit]

    if not tasks:
        print("No relevant images need bounding boxes.")
        return

    annotator = BboxAnnotator()
    loaded_data: dict[Path, list] = {}
    saved = 0
    skipped = 0
    index = 0

    print(f"Annotating {len(tasks)} images. Controls: drag=box, s=save, k=skip, b=back, q=quit")

    while index < len(tasks):
        task = tasks[index]
        json_path = task["json_path"]
        if json_path not in loaded_data:
            with json_path.open("r", encoding="utf-8") as handle:
                loaded_data[json_path] = json.load(handle)

        data = loaded_data[json_path]
        _parent, item = load_record(data, task["record_index"])

        local_path = None
        if image_dir and task.get("url"):
            candidate = image_dir / Path(task["url"]).name
            if candidate.exists():
                local_path = str(candidate)

        image_bgr = load_image_bgr(local_path, task.get("url"), cache_dir)
        if image_bgr is None:
            print(f"[{index + 1}/{len(tasks)}] Failed to load image, skipping: {task.get('url')}")
            skipped += 1
            index += 1
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        title = f"{task.get('pmcid', 'unknown')} | {task.get('caption', '')[:80]}"
        print(f"[{index + 1}/{len(tasks)}] {title}")

        result = annotator.annotate(image_rgb, title)
        if result == "quit":
            break
        if result == "back":
            index = max(0, index - 1)
            continue
        if result == "skip":
            skipped += 1
            index += 1
            continue

        if not apply_bbox_to_item(item, result, image_bgr.shape[:2]):
            print("Rejected zero-size bbox. Draw a larger box or press k to skip.")
            continue

        save_json(json_path, data)
        saved += 1
        index += 1

    annotator.close()
    print(f"Saved {saved} bounding boxes. Skipped {skipped} images.")


if __name__ == "__main__":
    main()
