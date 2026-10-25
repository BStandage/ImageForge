"""Stage 5: reject synthetic images that drifted off class, then export the rest as class folders.

Each real class gets a CLIP centroid. A synthetic image is kept if it is close enough to its own
class centroid and, optionally, closer to it than to any other class.
"""

import shutil
from pathlib import Path

from PIL import Image
from rich.progress import track

from .dataset import GroundTruthImage
from .io import read_jsonl, write_jsonl

BATCH = 16


class ClipScorer:
    def __init__(self, model_id, device):
        from transformers import CLIPModel, CLIPProcessor

        from .devices import resolve_device

        self.device = resolve_device(device)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def embed(self, paths, label, normalize=True):
        """CLIP image features. Unit-normalised by default for cosine comparisons; raw projection
        features (normalize=False) for distribution metrics, matching the usual CLIP-FID/KID practice."""
        import torch

        chunks = []
        for start in track(range(0, len(paths), BATCH), description=label):
            images = []
            for p in paths[start : start + BATCH]:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                # Explicit tower + projection: get_image_features changed its return type
                # between transformers 4 and 5.
                pooled = self.model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
                feats = self.model.visual_projection(pooled).float()
            if normalize:
                feats = torch.nn.functional.normalize(feats, dim=-1)
            chunks.append(feats.cpu())
        return torch.cat(chunks) if chunks else torch.empty(0)


def local_path(path, class_name, roots):
    """Resolve a manifest path written on another machine to this machine's copy.

    Tries the path as given, then each root's class folder by file name, with a .jpg
    fallback for frames that were downscaled on the way.
    """
    p = Path(path)
    if p.exists():
        return str(p)
    for root in roots:
        for cand in (root / class_name / p.name, root / class_name / (p.stem + ".jpg")):
            if cand.exists():
                return str(cand)
    return str(p)


def reference_images(cfg, gt):
    """Real images the pipeline is allowed to learn from: the ones that were captioned.

    When describe capped images per class, the rest of the ground truth was never used for
    prompts or references, and the filter must not use it either. Falls back to everything
    only when no captions exist. Paths are resolved locally so a run pulled back from another
    machine still finds its frames.
    """
    used = {row["path"] for row in read_jsonl(cfg.captions_path)}
    by_class = gt.by_class()
    if not used:
        selected = by_class
    else:
        selected = {c: [i for i in imgs if i.path in used] for c, imgs in by_class.items()}
        missing = [c for c, imgs in selected.items() if not imgs]
        if missing:
            raise ValueError(f"no captioned real images for classes {missing}; run describe first")
    out = {}
    for c, imgs in selected.items():
        out[c] = [GroundTruthImage(local_path(i.path, c, [cfg.ground_truth]), c, i.width, i.height) for i in imgs]
    return out


def _balance(rows, seed):
    """Mark accepted rows beyond the smallest class's count as rejected, chosen at random."""
    import random

    rng = random.Random(seed)
    accepted = {}
    for r in rows:
        if r["accepted"]:
            accepted.setdefault(r["class_name"], []).append(r)
    if not accepted:
        return
    target = min(len(v) for v in accepted.values())
    for class_name, items in accepted.items():
        rng.shuffle(items)
        for r in items[target:]:
            r["accepted"] = False
            r["reason"] = f"balanced out ({class_name} subsampled to {target})"


def _export(cfg, rows):
    if cfg.dataset_dir.exists():
        shutil.rmtree(cfg.dataset_dir)
    cfg.dataset_dir.mkdir(parents=True)
    size = cfg.filter.export_size
    fmt = cfg.filter.export_format
    for row in rows:
        if not row["accepted"]:
            continue
        src = Path(row["image"])
        dst = cfg.dataset_dir / row["class_name"] / f"{src.stem}.{fmt}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if size is None and fmt == src.suffix.lstrip("."):
            shutil.copy2(src, dst)
            continue
        with Image.open(src) as im:
            im = im.convert("RGB")
            if size is not None and im.size != (size, size):
                im = im.resize((size, size), Image.LANCZOS)
            im.save(dst, quality=95) if fmt == "jpg" else im.save(dst)

    # The real images this run learned from, so whoever trains on the export can keep them
    # out of validation and test.
    used = sorted({row["path"] for row in read_jsonl(cfg.captions_path)})
    (cfg.dataset_dir / "real_images_used.txt").write_text("\n".join(used) + "\n", encoding="utf-8")


def run(cfg, gt):
    generated = list(read_jsonl(cfg.generated_path))
    if not generated:
        raise FileNotFoundError(f"nothing generated at {cfg.generated_path}; run generate first")
    # A run pulled back from another machine carries that machine's paths.
    for g in generated:
        g["image"] = local_path(g["image"], g["class_name"], [cfg.images_dir])

    results = []
    if not cfg.filter.enabled:
        results = [{**g, "accepted": True, "similarity": None, "nearest_class": None, "reason": "filter disabled"} for g in generated]
    else:
        import torch

        scorer = ClipScorer(cfg.filter.model_id, cfg.filter.device)
        classes = gt.classes
        centroids = []
        for class_name, imgs in reference_images(cfg, gt).items():
            emb = scorer.embed([i.path for i in imgs], f"real/{class_name}")
            centroids.append(torch.nn.functional.normalize(emb.mean(dim=0, keepdim=True), dim=-1))
        centroids = torch.cat(centroids)  # [classes, dim]

        syn = scorer.embed([g["image"] for g in generated], "synthetic")
        sims = syn @ centroids.T  # [n, classes]
        for g, row in zip(generated, sims):
            own = classes.index(g["class_name"])
            own_sim = float(row[own])
            nearest = classes[int(row.argmax())]
            nearest_sim = float(row.max())
            reasons = []
            if own_sim < cfg.filter.min_similarity:
                reasons.append(f"similarity {own_sim:.3f} < {cfg.filter.min_similarity}")
            if (
                cfg.filter.require_nearest_class
                and nearest != g["class_name"]
                and nearest_sim - own_sim > cfg.filter.margin_for(g["class_name"])
            ):
                reasons.append(f"closer to {nearest} by {nearest_sim - own_sim:.3f}")
            results.append(
                {
                    **g,
                    "accepted": not reasons,
                    "similarity": round(own_sim, 4),
                    "nearest_class": nearest,
                    "nearest_margin": round(nearest_sim - own_sim, 4),
                    "reason": "; ".join(reasons) or "ok",
                }
            )

    if cfg.filter.balance:
        _balance(results, cfg.seed)

    write_jsonl(cfg.filtered_path, results)
    _export(cfg, results)

    summary = {c: {"accepted": 0, "rejected": 0} for c in gt.classes}
    for r in results:
        summary[r["class_name"]]["accepted" if r["accepted"] else "rejected"] += 1
    return summary
