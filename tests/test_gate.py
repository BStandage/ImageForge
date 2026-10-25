"""The nearest-class gate's decision rule, exercised through filter.run with a stub scorer."""

from pathlib import Path

import torch
from PIL import Image

from imageforge import filter as stage
from imageforge.config import Config
from imageforge.dataset import GroundTruth, GroundTruthImage
from imageforge.io import append_jsonl, read_jsonl


class StubScorer:
    """Embeds by file name: real images sit on class axes, synthetic ones where the test says."""

    def __init__(self, *_args, **_kw):
        pass

    def embed(self, paths, label, normalize=True):
        vecs = []
        for p in paths:
            name = Path(p).stem
            if name.startswith("real_a"):
                vecs.append([1.0, 0.0])
            elif name.startswith("real_b"):
                vecs.append([0.0, 1.0])
            elif name == "syn_on_a":
                vecs.append([1.0, 0.0])
            elif name == "syn_near_b":
                vecs.append([0.70, 0.72])  # slightly closer to b
            elif name == "syn_far_b":
                vecs.append([0.3, 0.95])
        t = torch.tensor(vecs)
        return torch.nn.functional.normalize(t, dim=-1) if normalize else t


def _setup(tmp_path, margin):
    cfg = Config(ground_truth=tmp_path / "gt", output=tmp_path / "out")
    cfg.filter.min_similarity = 0.0
    cfg.filter.require_nearest_class = True
    cfg.filter.nearest_class_margin = margin
    for name in ("real_a1", "real_a2", "real_a3", "real_a4", "real_b1", "real_b2", "real_b3", "real_b4", "syn_on_a", "syn_near_b", "syn_far_b"):
        Image.new("RGB", (4, 4)).save(tmp_path / f"{name}.png")
    append_jsonl(
        cfg.captions_path,
        [{"path": str(tmp_path / f"real_{c}{i}.png"), "class_name": c, "caption": "x"} for c in "ab" for i in range(1, 5)],
    )
    append_jsonl(
        cfg.generated_path,
        [{"id": n, "class_name": "a", "image": str(tmp_path / f"{n}.png")} for n in ("syn_on_a", "syn_near_b", "syn_far_b")],
    )
    return cfg


def _gt(tmp_path):
    imgs = [GroundTruthImage(str(tmp_path / f"real_{c}{i}.png"), c, 4, 4) for c in "ab" for i in range(1, 5)]
    return GroundTruth(str(tmp_path), ["a", "b"], imgs)


def test_margin_zero_rejects_any_closer_class(tmp_path, monkeypatch):
    monkeypatch.setattr(stage, "ClipScorer", StubScorer)
    cfg = _setup(tmp_path, margin=0.0)
    stage.run(cfg, _gt(tmp_path))
    verdict = {r["id"]: r["accepted"] for r in read_jsonl(cfg.filtered_path)}
    assert verdict == {"syn_on_a": True, "syn_near_b": False, "syn_far_b": False}


def test_per_class_margin(tmp_path, monkeypatch):
    monkeypatch.setattr(stage, "ClipScorer", StubScorer)
    cfg = _setup(tmp_path, margin=0.0)
    cfg.filter.nearest_class_margin = {"a": 0.05}  # class a is lenient, others default to 0
    stage.run(cfg, _gt(tmp_path))
    rows = {r["id"]: r for r in read_jsonl(cfg.filtered_path)}
    assert rows["syn_near_b"]["accepted"] is True
    assert rows["syn_far_b"]["accepted"] is False
    assert cfg.filter.margin_for("b") == 0.0


def test_margin_spares_near_ties(tmp_path, monkeypatch):
    monkeypatch.setattr(stage, "ClipScorer", StubScorer)
    cfg = _setup(tmp_path, margin=0.05)
    stage.run(cfg, _gt(tmp_path))
    rows = {r["id"]: r for r in read_jsonl(cfg.filtered_path)}
    assert rows["syn_near_b"]["accepted"] is True
    assert rows["syn_far_b"]["accepted"] is False
    assert rows["syn_far_b"]["nearest_class"] == "b"
    assert rows["syn_far_b"]["nearest_margin"] > 0.05
