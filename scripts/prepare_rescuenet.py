"""Build class folders from the RescueNet image-classification CSVs.

The public release labels whole UAV frames by neighbourhood damage severity:
    0 Superficial Damage, 1 Medium Damage, 2 Major Damage
and ships train/val/test CSVs of (Image_ID, Neighborhood_ID). The frames themselves come from
the main RescueNet download. This script copies the frames of one split into one folder per
class, which is the layout the pipeline ingests.

    python scripts/prepare_rescuenet.py \
        data/RescueNet-ImageClassification/RescueNet-classification-train.csv \
        <folder containing the frames> data/rescuenet_train

Use the train split as the forge's ground truth. Keep val and test for the classifier.
"""

import argparse
import csv
import shutil
from pathlib import Path

LABELS = {"0": "Superficial_Damage", "1": "Medium_Damage", "2": "Major_Damage"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("frames", type=Path, help="folder holding the frames; searched recursively")
    ap.add_argument("out", type=Path)
    ap.add_argument("--link", action="store_true", help="hard-link instead of copying (same drive only)")
    ap.add_argument("--names", choices=["public", "bda"], default="public",
                    help="folder names: public (Superficial/Medium/Major_Damage) or bda (Intact/Damaged/Collapsed)")
    args = ap.parse_args()

    names = dict(LABELS)
    if args.names == "bda":
        names = {"0": "Intact", "1": "Damaged", "2": "Collapsed"}

    index = {}
    for p in args.frames.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            index.setdefault(p.name.lower(), p)
    if not index:
        raise SystemExit(f"no images under {args.frames}")

    counts = {v: 0 for v in names.values()}
    missing = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_id = row["Image_ID"].strip()
            label = row["Neighborhood_ID"].strip()
            src = index.get(image_id.lower())
            if src is None:
                missing.append(image_id)
                continue
            class_name = names[label]
            dst = args.out / class_name / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                continue
            if args.link:
                try:
                    dst.hardlink_to(src)
                except OSError:
                    shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)
            counts[class_name] += 1

    for class_name, n in counts.items():
        print(f"{class_name}: {n}")
    if missing:
        print(f"{len(missing)} listed frames not found, e.g. {missing[:3]}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
