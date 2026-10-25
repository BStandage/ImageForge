"""Exercises the generate stage with a fake backend so no model is needed."""

from pathlib import Path

from PIL import Image

from imageforge import generate
from imageforge.config import Config
from imageforge.io import append_jsonl, read_jsonl
from imageforge.planner import Job


class FakeBackend:
    supports_reference = True

    def __init__(self):
        self.calls = []

    def generate(self, requests):
        self.calls.append(list(requests))
        return [Image.new("RGB", (r.width, r.height), (r.seed % 255, 0, 0)) for r in requests]


def _jobs(n, with_ref=True):
    return [
        Job(
            id=f"cat_{i:05d}",
            class_name="cat",
            prompt=f"cat {i}",
            seed=i,
            width=16,
            height=16,
            reference=f"/real/{i}.png" if with_ref else None,
            source_caption=f"/real/{i}.png",
        )
        for i in range(n)
    ]


def test_batches_split_on_reference_presence():
    jobs = _jobs(3) + _jobs(2, with_ref=False) + _jobs(1)
    batches = list(generate._batches(jobs, size=4))
    assert [len(b) for b in batches] == [3, 2, 1]
    assert [len(b) for b in generate._batches(_jobs(5), size=2)] == [2, 2, 1]


def test_run_writes_images_and_manifest(tmp_path, monkeypatch):
    cfg = Config(ground_truth=tmp_path / "gt", output=tmp_path / "out")
    cfg.backend.batch_size = 2
    jobs = _jobs(3, with_ref=False)
    append_jsonl(cfg.jobs_path, [j.__dict__ for j in jobs])

    backend = FakeBackend()
    monkeypatch.setattr(generate, "load_backend", lambda _cfg: backend)

    assert generate.run(cfg) == 3
    assert [len(c) for c in backend.calls] == [2, 1]
    rows = list(read_jsonl(cfg.generated_path))
    assert [r["id"] for r in rows] == ["cat_00000", "cat_00001", "cat_00002"]
    for r in rows:
        assert Path(r["image"]).exists()
        assert len(r["sha256"]) == 64
        assert r["model"] == cfg.backend.model_id

    # Second run finds nothing to do and never loads a backend.
    monkeypatch.setattr(generate, "load_backend", lambda _cfg: (_ for _ in ()).throw(AssertionError("loaded")))
    assert generate.run(cfg) == 0


def test_run_redoes_jobs_whose_file_is_missing(tmp_path, monkeypatch):
    cfg = Config(ground_truth=tmp_path / "gt", output=tmp_path / "out")
    jobs = _jobs(2, with_ref=False)
    append_jsonl(cfg.jobs_path, [j.__dict__ for j in jobs])
    backend = FakeBackend()
    monkeypatch.setattr(generate, "load_backend", lambda _cfg: backend)
    generate.run(cfg)
    generate.output_path(cfg, jobs[1]).unlink()
    assert generate.run(cfg) == 1
    assert backend.calls[-1][0].seed == 1
