
from PIL import Image

from imageforge.config import Config
from imageforge.filter import _export
from imageforge.io import append_jsonl


def test_export_copies_accepted_and_lists_real_sources(tmp_path):
    cfg = Config(ground_truth=tmp_path / "gt", output=tmp_path / "out")
    img_a = tmp_path / "a.png"
    img_b = tmp_path / "b.png"
    Image.new("RGB", (8, 8)).save(img_a)
    Image.new("RGB", (8, 8)).save(img_b)
    append_jsonl(cfg.captions_path, [{"path": "/real/x.jpg", "class_name": "cat", "caption": "x"}])

    _export(
        cfg,
        [
            {"class_name": "cat", "image": str(img_a), "accepted": True},
            {"class_name": "cat", "image": str(img_b), "accepted": False},
        ],
    )

    assert (cfg.dataset_dir / "cat" / "a.png").exists()
    assert not (cfg.dataset_dir / "cat" / "b.png").exists()
    assert (cfg.dataset_dir / "real_images_used.txt").read_text() == "/real/x.jpg\n"


def test_balance_subsamples_to_smallest_class():
    from imageforge.filter import _balance

    rows = [{"class_name": "a", "accepted": True, "reason": "ok"} for _ in range(5)]
    rows += [{"class_name": "b", "accepted": True, "reason": "ok"} for _ in range(2)]
    rows += [{"class_name": "c", "accepted": False, "reason": "closer to a"} for _ in range(3)]
    _balance(rows, seed=1)
    kept = {c: sum(r["accepted"] for r in rows if r["class_name"] == c) for c in "abc"}
    assert kept == {"a": 2, "b": 2, "c": 0}
    assert sum("balanced out" in r["reason"] for r in rows) == 3


def test_export_resizes_and_converts(tmp_path):
    cfg = Config(ground_truth=tmp_path / "gt", output=tmp_path / "out")
    cfg.filter.export_size = 4
    cfg.filter.export_format = "jpg"
    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8), (200, 10, 10)).save(img)
    append_jsonl(cfg.captions_path, [{"path": "/real/x.jpg", "class_name": "cat", "caption": "x"}])

    _export(cfg, [{"class_name": "cat", "image": str(img), "accepted": True}])

    out = cfg.dataset_dir / "cat" / "a.jpg"
    assert out.exists()
    with Image.open(out) as im:
        assert im.size == (4, 4)
        assert im.format == "JPEG"
