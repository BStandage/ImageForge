"""Ground-truth ingest. Expects one folder per class."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class GroundTruthImage:
    path: str
    class_name: str
    width: int
    height: int


@dataclass
class GroundTruth:
    root: str
    classes: list[str]
    images: list[GroundTruthImage]

    def by_class(self):
        out = {c: [] for c in self.classes}
        for img in self.images:
            out[img.class_name].append(img)
        return out

    def counts(self):
        return {c: len(v) for c, v in self.by_class().items()}


def scan(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"ground truth folder not found: {root}")

    class_dirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not class_dirs:
        raise ValueError(f"{root} has no class subfolders")

    images = []
    for cdir in class_dirs:
        for p in sorted(cdir.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except OSError:
                continue
            images.append(GroundTruthImage(str(p.resolve()), cdir.name, w, h))

    classes = [c.name for c in class_dirs]
    present = {i.class_name for i in images}
    empty = [c for c in classes if c not in present]
    if empty:
        raise ValueError(f"class folders with no readable images: {empty}")
    return GroundTruth(str(root.resolve()), classes, images)


def save(gt, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"root": gt.root, "classes": gt.classes, "images": [asdict(i) for i in gt.images]}
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return GroundTruth(
        data["root"], data["classes"], [GroundTruthImage(**i) for i in data["images"]]
    )


def pretty_class_name(folder_name):
    """Turn a folder name like golden_retriever-01 into 'golden retriever 01'."""
    return folder_name.replace("_", " ").replace("-", " ").strip()
