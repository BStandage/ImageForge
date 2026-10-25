from pathlib import Path

import pytest

from imageforge import dataset


def test_scan_finds_classes_and_images(ground_truth_dir):
    gt = dataset.scan(ground_truth_dir)
    assert gt.classes == ["cat", "dog", "rust_spot"]
    assert gt.counts() == {"cat": 3, "dog": 2, "rust_spot": 1}
    first = gt.images[0]
    assert (first.width, first.height) == (32, 24)
    assert Path(first.path).is_absolute()


def test_scan_rejects_empty_class(ground_truth_dir):
    (ground_truth_dir / "empty").mkdir()
    with pytest.raises(ValueError, match="empty"):
        dataset.scan(ground_truth_dir)


def test_scan_rejects_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        dataset.scan(tmp_path / "nope")


def test_save_and_load_roundtrip(ground_truth_dir, tmp_path):
    gt = dataset.scan(ground_truth_dir)
    path = tmp_path / "out" / "gt.json"
    dataset.save(gt, path)
    loaded = dataset.load(path)
    assert loaded == gt


def test_pretty_class_name():
    assert dataset.pretty_class_name("golden_retriever-01") == "golden retriever 01"
