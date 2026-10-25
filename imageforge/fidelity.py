"""How close is the synthetic set to the real set, per class, in CLIP embedding space.

Reports three numbers per class, each against a baseline measured between two random halves of
the real pool, so "close" has a reference point:

- fd: Frechet distance between the embedding distributions. Lower is closer. Synthetic-to-real
  near real-to-real means the sets are statistically alike. Biased upward at small n, hence
  the size-matched baseline.
- kid: Kernel Inception Distance on the same features. Unbiased, so it is the number to trust
  at small sample sizes. Zero means indistinguishable; the real-vs-real baseline scatters
  around zero.
- nn: mean cosine similarity from each synthetic image to its nearest real image. Much higher
  than the real baseline means the forge is copying references rather than varying them.
- div: mean pairwise distance within the set. Much lower than real means mode collapse.
"""

import json
import random

import numpy as np

from .filter import ClipScorer, local_path, reference_images
from .io import read_jsonl


def frechet_distance(a, b):
    """Frechet distance between Gaussians fitted to two sets of row vectors."""
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    # tr(sqrt(ca cb)) via the symmetric form sqrt(ca)^T cb sqrt(ca)
    w, v = np.linalg.eigh(ca)
    sqrt_ca = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    m = sqrt_ca @ cb @ sqrt_ca
    eig = np.linalg.eigvalsh((m + m.T) / 2)
    tr_sqrt = np.sqrt(np.clip(eig, 0, None)).sum()
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(ca) + np.trace(cb) - 2 * tr_sqrt)


def kernel_distance(a, b, degree=3):
    """Kernel Inception Distance: unbiased squared MMD with a polynomial kernel.

    k(x, y) = (x.y / d + 1) ** degree, the kernel from Binkowski et al. 2018. Unlike the
    Frechet distance it has no Gaussian assumption and no small-sample bias, which matters
    when a class has only a few dozen synthetic images.
    """
    d = a.shape[1]
    m, n = len(a), len(b)
    k_aa = (a @ a.T / d + 1) ** degree
    k_bb = (b @ b.T / d + 1) ** degree
    k_ab = (a @ b.T / d + 1) ** degree
    term_aa = (k_aa.sum() - np.trace(k_aa)) / (m * (m - 1))
    term_bb = (k_bb.sum() - np.trace(k_bb)) / (n * (n - 1))
    term_ab = k_ab.mean()
    return float(term_aa + term_bb - 2 * term_ab)


def nearest_neighbour_similarity(query, pool, exclude_self=False):
    sims = query @ pool.T
    if exclude_self:
        np.fill_diagonal(sims, -np.inf)
    return float(sims.max(axis=1).mean())


def diversity(x):
    """Mean pairwise cosine distance within a set."""
    sims = x @ x.T
    n = len(x)
    if n < 2:
        return 0.0
    off = (sims.sum() - np.trace(sims)) / (n * (n - 1))
    return float(1.0 - off)


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def compare(real, synthetic, rng, repeats=5):
    """Synthetic against the real pool, and a baseline of a same-sized real subset against the rest.

    Inputs are raw CLIP projection features. Frechet and kernel distances use them as they are,
    which is the usual CLIP-FID/KID convention; nearest-neighbour and diversity use unit vectors
    so they are cosine measures. Frechet distance is strongly biased by sample size, so the
    baseline uses as many real images as there are synthetic ones, averaged over several draws.
    """
    # The held-out subset can be at most half the real pool, or the "rest" it is compared
    # against becomes a handful of images and the baseline is meaningless.
    n = max(2, min(len(synthetic), len(real) // 2))
    idx = list(range(len(real)))
    real_u, syn_u = _unit(real), _unit(synthetic)
    base = {"fd": 0.0, "kid": 0.0, "nn": 0.0, "div": 0.0}
    for _ in range(repeats):
        rng.shuffle(idx)
        a, b = idx[:n], idx[n:]
        base["fd"] += frechet_distance(real[a], real[b])
        base["kid"] += kernel_distance(real[a], real[b])
        base["nn"] += nearest_neighbour_similarity(real_u[a], real_u[b])
        base["div"] += diversity(real_u[a])
    return {
        "synthetic_vs_real": {
            "fd": frechet_distance(synthetic, real),
            "kid": kernel_distance(synthetic, real),
            "nn": nearest_neighbour_similarity(syn_u, real_u),
            "div": diversity(syn_u),
            "n": len(synthetic),
        },
        "real_vs_real": {k: v / repeats for k, v in base.items()} | {"n": n},
    }


def run(cfg, gt):
    rows = list(read_jsonl(cfg.filtered_path)) or list(read_jsonl(cfg.generated_path))
    rows = [r for r in rows if r.get("accepted", True)]
    if not rows:
        raise FileNotFoundError(f"nothing generated at {cfg.generated_path}; run generate first")

    scorer = ClipScorer(cfg.filter.model_id, cfg.filter.device)
    rng = random.Random(cfg.seed)
    report = {}
    all_real, all_syn = [], []
    for class_name, imgs in reference_images(cfg, gt).items():
        syn_paths = [local_path(r["image"], class_name, [cfg.images_dir]) for r in rows if r["class_name"] == class_name]
        if len(syn_paths) < 2 or len(imgs) < 4:
            continue
        real = scorer.embed([i.path for i in imgs], f"real/{class_name}", normalize=False).numpy()
        syn = scorer.embed(syn_paths, f"synthetic/{class_name}", normalize=False).numpy()
        report[class_name] = compare(real, syn, rng)
        all_real.append(real)
        all_syn.append(syn)

    # The whole synthetic set against the whole real pool, ignoring class.
    if len(all_real) > 1:
        report["all"] = compare(np.concatenate(all_real), np.concatenate(all_syn), rng)

    (cfg.output / "fidelity.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report
