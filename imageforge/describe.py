"""Stage 2: caption the real images. Resumable; already-captioned paths are skipped."""

import random
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from rich.progress import track

from .io import append_jsonl, read_jsonl


def _load(item):
    with Image.open(item.path) as im:
        return im.convert("RGB")


def run(cfg, gt):
    done = {row["path"] for row in read_jsonl(cfg.captions_path)}
    rng = random.Random(cfg.seed)

    todo = []
    for class_name, imgs in gt.by_class().items():
        imgs = list(imgs)
        if cfg.caption.max_images_per_class is not None and len(imgs) > cfg.caption.max_images_per_class:
            rng.shuffle(imgs)
            imgs = imgs[: cfg.caption.max_images_per_class]
        todo.extend(i for i in imgs if i.path not in done)

    if not todo:
        return 0

    from .captioning import load_captioner

    captioner = load_captioner(cfg.caption)
    batch = max(1, cfg.caption.batch_size)
    batches = [todo[i : i + batch] for i in range(0, len(todo), batch)]
    written = 0
    # Decode the next batch's images on threads while the GPU works on the current one.
    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = pool.submit(lambda b: list(pool.map(_load, b)), batches[0])
        for idx in track(range(len(batches)), description="captioning"):
            images = pending.result()
            if idx + 1 < len(batches):
                pending = pool.submit(lambda b: list(pool.map(_load, b)), batches[idx + 1])
            captions = captioner.caption(images)
            append_jsonl(
                cfg.captions_path,
                (
                    {"path": item.path, "class_name": item.class_name, "caption": cap}
                    for item, cap in zip(batches[idx], captions)
                ),
            )
            written += len(batches[idx])
    return written
