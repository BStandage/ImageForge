
import pytest

from imageforge import dataset
from imageforge.config import Config
from imageforge.filter import reference_images
from imageforge.io import append_jsonl


def test_reference_images_restricts_to_captioned(ground_truth_dir, tmp_path):
    gt = dataset.scan(ground_truth_dir)
    cfg = Config(ground_truth=ground_truth_dir, output=tmp_path / "out")
    by_class = gt.by_class()
    used = [by_class["cat"][0], by_class["dog"][1], by_class["rust_spot"][0]]
    append_jsonl(cfg.captions_path, [{"path": i.path, "class_name": i.class_name, "caption": "x"} for i in used])

    refs = reference_images(cfg, gt)
    assert {c: [i.path for i in v] for c, v in refs.items()} == {
        "cat": [by_class["cat"][0].path],
        "dog": [by_class["dog"][1].path],
        "rust_spot": [by_class["rust_spot"][0].path],
    }


def test_reference_images_falls_back_without_captions(ground_truth_dir, tmp_path):
    gt = dataset.scan(ground_truth_dir)
    cfg = Config(ground_truth=ground_truth_dir, output=tmp_path / "out")
    assert reference_images(cfg, gt) == gt.by_class()


def test_reference_images_rejects_uncaptioned_class(ground_truth_dir, tmp_path):
    gt = dataset.scan(ground_truth_dir)
    cfg = Config(ground_truth=ground_truth_dir, output=tmp_path / "out")
    cat = gt.by_class()["cat"][0]
    append_jsonl(cfg.captions_path, [{"path": cat.path, "class_name": "cat", "caption": "x"}])
    with pytest.raises(ValueError, match="dog"):
        reference_images(cfg, gt)
