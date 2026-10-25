"""Download a small public image classification set and lay it out as class folders.

Uses Imagenette (10 easy ImageNet classes, CC BY 4.0 via fast.ai), the 160px variant, about 100 MB.

    python scripts/fetch_sample_dataset.py data/sample --classes 3 --per-class 20
"""

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"

# WordNet ids used by Imagenette, mapped to readable folder names.
NAMES = {
    "n01440764": "tench",
    "n02102040": "english_springer",
    "n02979186": "cassette_player",
    "n03000684": "chain_saw",
    "n03028079": "church",
    "n03394916": "french_horn",
    "n03417042": "garbage_truck",
    "n03425413": "gas_pump",
    "n03445777": "golf_ball",
    "n03888257": "parachute",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path, help="folder to create with one subfolder per class")
    ap.add_argument("--classes", type=int, default=3)
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--cache", type=Path, default=Path("data/.downloads"))
    args = ap.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    archive = args.cache / "imagenette2-160.tgz"
    if not archive.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, archive)

    extracted = args.cache / "imagenette2-160"
    if not extracted.exists():
        print("extracting")
        with tarfile.open(archive) as tar:
            tar.extractall(args.cache, filter="data")

    train = extracted / "train"
    chosen = list(NAMES.items())[: args.classes]
    if args.out.exists():
        shutil.rmtree(args.out)
    for wnid, name in chosen:
        src = sorted((train / wnid).glob("*.JPEG"))[: args.per_class]
        dst = args.out / name
        dst.mkdir(parents=True)
        for p in src:
            shutil.copy2(p, dst / p.name.lower())
        print(f"{name}: {len(src)} images")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
