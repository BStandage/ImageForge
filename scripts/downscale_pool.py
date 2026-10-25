"""Make a smaller copy of a class-folder dataset for upload to a remote machine.

The pipeline never looks at more than about 1024 px of a frame: the captioner and Redux both
resize their inputs, and CLIP sees 224 px. Shrinking 4000x3000 JPEGs to a 1024 px long side cuts
the RescueNet train split from 18 GB to about 1 GB with no effect on the output.

    python scripts/downscale_pool.py D:/datasets/rescuenet_train D:/datasets/rescuenet_train_1024 --long-side 1024
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def shrink(src, dst, long_side, quality):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return 0
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((long_side, long_side), Image.LANCZOS)
        im.save(dst.with_suffix(".jpg"), "JPEG", quality=quality, optimize=True)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--long-side", type=int, default=1024)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    jobs = []
    for p in sorted(args.src.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTS:
            jobs.append((p, args.dst / p.relative_to(args.src)))
    if not jobs:
        raise SystemExit(f"no images under {args.src}")
    with ThreadPoolExecutor(args.workers) as ex:
        done = sum(ex.map(lambda j: shrink(j[0], j[1], args.long_side, args.quality), jobs))
    print(f"{done} written, {len(jobs) - done} already present -> {args.dst}")


if __name__ == "__main__":
    main()
