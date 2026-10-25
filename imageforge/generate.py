"""Stage 4: run the job list through the backend. Resumable; finished jobs are skipped."""

import hashlib
from dataclasses import asdict
from pathlib import Path

from rich.progress import Progress

from .backends import GenRequest, load_backend
from .io import append_jsonl, read_jsonl
from .planner import load_jobs


def output_path(cfg, job):
    return cfg.images_dir / job.class_name / f"{job.id}.png"


def _batches(jobs, size):
    """Consecutive jobs, split whenever the batch would mix reference and no-reference jobs."""
    batch = []
    for job in jobs:
        if batch and (len(batch) >= size or (batch[0].reference is None) != (job.reference is None)):
            yield batch
            batch = []
        batch.append(job)
    if batch:
        yield batch


def run(cfg):
    jobs = load_jobs(cfg)
    if not jobs:
        raise FileNotFoundError(f"no jobs at {cfg.jobs_path}; run plan first")

    done = {row["id"] for row in read_jsonl(cfg.generated_path)}
    todo = [j for j in jobs if j.id not in done or not output_path(cfg, j).exists()]
    if not todo:
        return 0

    backend = load_backend(cfg.backend)
    if not backend.supports_reference and any(j.reference for j in todo):
        print(f"backend {cfg.backend.name} ignores reference images")

    written = 0
    with Progress() as progress:
        task = progress.add_task("generating", total=len(todo))
        for batch in _batches(todo, max(1, cfg.backend.batch_size)):
            requests = [
                GenRequest(
                    prompt=j.prompt,
                    seed=j.seed,
                    width=j.width,
                    height=j.height,
                    negative=cfg.prompt.negative,
                    reference=Path(j.reference) if j.reference else None,
                )
                for j in batch
            ]
            images = backend.generate(requests)
            rows = []
            for job, image in zip(batch, images):
                path = output_path(cfg, job)
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path, format="PNG")
                rows.append(
                    {
                        **asdict(job),
                        "image": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "backend": cfg.backend.name,
                        "model": cfg.backend.model_id,
                        "steps": cfg.backend.steps,
                        "guidance": cfg.backend.guidance,
                        "reference_strength": cfg.backend.reference_strength if job.reference else 0.0,
                    }
                )
            append_jsonl(cfg.generated_path, rows)
            written += len(rows)
            progress.advance(task, len(rows))
    return written
