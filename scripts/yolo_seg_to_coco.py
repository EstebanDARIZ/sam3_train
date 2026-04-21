"""
Convert YOLO segmentation format to COCO JSON format for SAM3 training.

YOLO seg format (per line): class_id x1 y1 x2 y2 ... (normalized coords)
Output: COCO JSON with polygon segmentation masks

Usage:
    python scripts/yolo_seg_to_coco.py \
        --images-dir /path/to/images \
        --labels-dir /path/to/labels \
        --output-dir /path/to/output \
        --class-names Squid Sardine Ray Sunfish "Pilot Fish" Shark Jellyfish
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image


CLASS_NAMES_DEFAULT = [
    "Squid",
    "Sardine",
    "Ray",
    "Sunfish",
    "Pilot Fish",
    "Shark",
    "Jellyfish",
]


def polygon_area(xs: list[float], ys: list[float]) -> float:
    """Shoelace formula."""
    n = len(xs)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    return abs(area) / 2.0


def convert_split(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    class_names: list[str],
    split_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in image_extensions
    )

    categories = [
        {"id": i + 1, "name": name, "supercategory": "object"}
        for i, name in enumerate(class_names)
    ]

    coco = {
        "info": {"description": f"SAM3 fine-tuning dataset — {split_name}"},
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": [],
    }

    ann_id = 1

    for img_id, img_path in enumerate(image_files, start=1):
        # Get image dimensions
        with Image.open(img_path) as img:
            width, height = img.size

        coco["images"].append(
            {
                "id": img_id,
                "file_name": img_path.name,
                "width": width,
                "height": height,
            }
        )

        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) < 7:  # need at least class + 3 points (6 coords)
                continue

            class_id = int(parts[0])
            if class_id >= len(class_names):
                print(f"  Warning: class_id {class_id} out of range in {label_path}")
                continue

            coords = [float(v) for v in parts[1:]]
            # coords are normalized: x1,y1,x2,y2,...
            xs_norm = coords[0::2]
            ys_norm = coords[1::2]

            xs_px = [x * width for x in xs_norm]
            ys_px = [y * height for y in ys_norm]

            # Flatten polygon for COCO: [x1, y1, x2, y2, ...]
            segmentation = []
            for x, y in zip(xs_px, ys_px):
                segmentation.extend([round(x, 2), round(y, 2)])

            x_min = min(xs_px)
            y_min = min(ys_px)
            x_max = max(xs_px)
            y_max = max(ys_px)
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
            area = polygon_area(xs_px, ys_px)

            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_id + 1,  # COCO is 1-indexed
                    "segmentation": [segmentation],
                    "bbox": [round(x_min, 2), round(y_min, 2), round(bbox_w, 2), round(bbox_h, 2)],
                    "area": round(area, 2),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    ann_file = output_dir / "_annotations.coco.json"
    with open(ann_file, "w") as f:
        json.dump(coco, f)

    print(
        f"[{split_name}] {len(coco['images'])} images, "
        f"{len(coco['annotations'])} annotations → {ann_file}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--val-images", type=Path, required=True)
    parser.add_argument("--val-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-names", nargs="+", default=CLASS_NAMES_DEFAULT)
    args = parser.parse_args()

    print(f"Classes: {args.class_names}")

    # Copy images + generate COCO JSON for each split
    # SAM3 expects: <split>/images/ + <split>/_annotations.coco.json
    for split, img_dir, lbl_dir in [
        ("train", args.train_images, args.train_labels),
        ("test", args.val_images, args.val_labels),
    ]:
        out_split = args.output_dir / split
        # Symlink or copy images
        out_images = out_split / "images"
        out_images.mkdir(parents=True, exist_ok=True)

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        for img in img_dir.iterdir():
            if img.suffix.lower() in image_extensions:
                dst = out_images / img.name
                if not dst.exists():
                    os.symlink(img.resolve(), dst)

        convert_split(
            images_dir=img_dir,
            labels_dir=lbl_dir,
            output_dir=out_split,
            class_names=args.class_names,
            split_name=split,
        )

    print(f"\nDone. SAM3 dataset ready at: {args.output_dir}")
    print(
        f"Set in your config:\n"
        f"  paths.dataset_root: {args.output_dir.resolve()}\n"
        f"  img_folder: ${{paths.dataset_root}}/train/images/\n"
        f"  ann_file:   ${{paths.dataset_root}}/train/_annotations.coco.json"
    )


if __name__ == "__main__":
    main()
