"""Stage 3: decide how many images per class and write one job per image."""

import random
from dataclasses import asdict, dataclass

from .captioning.base import clean_caption
from .io import read_jsonl, write_jsonl
from .prompts import Caption, PromptBank


@dataclass
class Job:
    id: str
    class_name: str
    prompt: str
    seed: int
    width: int
    height: int
    reference: str | None  # path of a real image used for style conditioning
    source_caption: str  # path of the real image whose caption was used, or "synthetic"
    caption_source: str = "real"


def split_counts(total, classes, overrides=None):
    """Even split of `total` across classes. Overrides fix a class; the rest share what remains.

    Any remainder goes to the earliest classes in sorted order so the result is deterministic.
    """
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(classes)
    if unknown:
        raise ValueError(f"per_class names not in ground truth: {sorted(unknown)}")

    fixed = sum(overrides.values())
    if fixed > total:
        raise ValueError(f"per_class counts ({fixed}) exceed total_images ({total})")

    free = [c for c in sorted(classes) if c not in overrides]
    counts = dict(overrides)
    if free:
        remaining = total - fixed
        base, extra = divmod(remaining, len(free))
        for i, c in enumerate(free):
            counts[c] = base + (1 if i < extra else 0)
    return {c: counts[c] for c in classes}


def build_jobs(cfg, bank, classes):
    counts = split_counts(cfg.total_images, classes, cfg.per_class)
    rng = random.Random(cfg.seed)
    jobs = []
    for class_name in classes:
        n = counts[class_name]
        if n == 0:
            continue
        caps = bank.sample_captions(class_name, n, rng)
        for i, cap in enumerate(caps):
            seed = rng.randrange(0, 2**31 - 1)
            reference = None
            if cfg.backend.use_reference:
                reference = bank.pick_reference(class_name, cap, cfg.backend.reference_pairing, rng)
            jobs.append(
                Job(
                    id=f"{class_name}_{i:05d}",
                    class_name=class_name,
                    prompt=bank.render(class_name, cap, rng),
                    seed=seed,
                    width=cfg.backend.width,
                    height=cfg.backend.height,
                    reference=reference,
                    source_caption=cap.path or "synthetic",
                    caption_source=cap.source,
                )
            )
    return jobs


def load_bank(cfg):
    # Cleaned again on load so captions written before a cleaner improvement still benefit.
    real = [Caption(r["path"], r["class_name"], clean_caption(r["caption"]), "real") for r in read_jsonl(cfg.captions_path)]
    if not real:
        raise FileNotFoundError(f"no captions at {cfg.captions_path}; run describe first")
    references = {}
    for c in real:
        references.setdefault(c.class_name, []).append(c.path)

    source = cfg.prompt.source
    if source == "real":
        pool = real
    else:
        synthetic = [
            Caption(None, r["class_name"], r["caption"], "synthetic")
            for r in read_jsonl(cfg.synthetic_captions_path)
        ]
        if not synthetic:
            raise FileNotFoundError(
                f"prompt.source is {source!r} but there is nothing at {cfg.synthetic_captions_path}; run synthesize first"
            )
        pool = synthetic if source == "synthetic" else real + synthetic
    return PromptBank(cfg.prompt, pool, references)


def run(cfg, classes):
    bank = load_bank(cfg)
    jobs = build_jobs(cfg, bank, classes)
    write_jsonl(cfg.jobs_path, (asdict(j) for j in jobs))
    return jobs


def load_jobs(cfg):
    return [Job(**row) for row in read_jsonl(cfg.jobs_path)]
