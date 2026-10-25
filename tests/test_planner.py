
import pytest

from imageforge.config import Config
from imageforge.io import append_jsonl, read_jsonl
from imageforge.planner import build_jobs, load_jobs, run, split_counts
from imageforge.prompts import Caption, PromptBank

CLASSES = ["cat", "dog", "rust_spot"]


def test_split_even():
    assert split_counts(9, CLASSES) == {"cat": 3, "dog": 3, "rust_spot": 3}


def test_split_remainder_is_deterministic():
    assert split_counts(10, CLASSES) == {"cat": 4, "dog": 3, "rust_spot": 3}
    assert split_counts(11, CLASSES) == {"cat": 4, "dog": 4, "rust_spot": 3}


def test_split_preserves_class_order():
    assert list(split_counts(3, ["zebra", "apple"])) == ["zebra", "apple"]


def test_split_with_override():
    assert split_counts(10, CLASSES, {"rust_spot": 6}) == {"cat": 2, "dog": 2, "rust_spot": 6}


def test_split_zero_total():
    assert split_counts(0, CLASSES) == {"cat": 0, "dog": 0, "rust_spot": 0}


def test_split_rejects_unknown_class():
    with pytest.raises(ValueError, match="not in ground truth"):
        split_counts(10, CLASSES, {"bird": 1})


def test_split_rejects_overrides_over_total():
    with pytest.raises(ValueError, match="exceed"):
        split_counts(5, CLASSES, {"cat": 6})


def _bank():
    from imageforge.config import PromptConfig

    caps = [
        Caption("/r/cat/0.png", "cat", "a tabby on a sofa"),
        Caption("/r/cat/1.png", "cat", "a black cat outdoors"),
        Caption("/r/dog/0.png", "dog", "a dog on grass"),
        Caption("/r/rust_spot/0.png", "rust_spot", "rust on a steel beam"),
    ]
    return PromptBank(PromptConfig(template="{class_name}: {caption}"), caps)


def _config(tmp_path, **kw):
    return Config(ground_truth=tmp_path / "real", output=tmp_path / "out", total_images=7, seed=3, **kw)


def test_build_jobs_counts_and_fields(tmp_path):
    cfg = _config(tmp_path)
    jobs = build_jobs(cfg, _bank(), CLASSES)
    assert len(jobs) == 7
    per = {c: sum(j.class_name == c for j in jobs) for c in CLASSES}
    assert per == {"cat": 3, "dog": 2, "rust_spot": 2}
    assert len({j.id for j in jobs}) == 7
    assert len({j.seed for j in jobs}) == 7
    cat = [j for j in jobs if j.class_name == "cat"]
    assert all(j.prompt.startswith("cat: ") for j in cat)
    assert all(j.reference == j.source_caption for j in cat)
    assert all(j.width == 1024 and j.height == 1024 for j in jobs)


def test_build_jobs_cycles_captions_evenly(tmp_path):
    cfg = _config(tmp_path)
    cfg.total_images = 12
    jobs = build_jobs(cfg, _bank(), CLASSES)
    cat_sources = [j.source_caption for j in jobs if j.class_name == "cat"]
    assert cat_sources.count("/r/cat/0.png") == 2
    assert cat_sources.count("/r/cat/1.png") == 2


def test_build_jobs_is_deterministic(tmp_path):
    cfg = _config(tmp_path)
    a = build_jobs(cfg, _bank(), CLASSES)
    b = build_jobs(cfg, _bank(), CLASSES)
    assert a == b


def test_build_jobs_random_pairing_uses_other_image_of_same_class(tmp_path):
    cfg = _config(tmp_path)
    cfg.total_images = 12
    cfg.backend.reference_pairing = "random"
    jobs = build_jobs(cfg, _bank(), CLASSES)
    cat = [j for j in jobs if j.class_name == "cat"]
    assert all(j.reference != j.source_caption for j in cat)
    assert all(j.reference.startswith("/r/cat/") for j in cat)
    # classes with a single real image fall back to the caption's own image
    dog = [j for j in jobs if j.class_name == "dog"]
    assert all(j.reference == j.source_caption for j in dog)


def test_build_jobs_without_reference(tmp_path):
    cfg = _config(tmp_path)
    cfg.backend.use_reference = False
    jobs = build_jobs(cfg, _bank(), CLASSES)
    assert all(j.reference is None for j in jobs)


def test_run_writes_and_reloads_jobs(tmp_path):
    cfg = _config(tmp_path)
    append_jsonl(
        cfg.captions_path,
        [{"path": c.path, "class_name": c.class_name, "caption": c.caption} for c in _bank().by_class["cat"]]
        + [{"path": "/r/dog/0.png", "class_name": "dog", "caption": "a dog"}]
        + [{"path": "/r/rust_spot/0.png", "class_name": "rust_spot", "caption": "rust"}],
    )
    jobs = run(cfg, CLASSES)
    assert len(list(read_jsonl(cfg.jobs_path))) == 7
    assert load_jobs(cfg) == jobs


def _write_real_and_synthetic(cfg):
    append_jsonl(
        cfg.captions_path,
        [
            {"path": "/r/cat/0.png", "class_name": "cat", "caption": "real cat 0"},
            {"path": "/r/cat/1.png", "class_name": "cat", "caption": "real cat 1"},
            {"path": "/r/dog/0.png", "class_name": "dog", "caption": "real dog"},
            {"path": "/r/rust_spot/0.png", "class_name": "rust_spot", "caption": "real rust"},
        ],
    )
    append_jsonl(
        cfg.synthetic_captions_path,
        [
            {"class_name": "cat", "caption": "synthetic cat A", "source": "synthetic"},
            {"class_name": "dog", "caption": "synthetic dog A", "source": "synthetic"},
            {"class_name": "rust_spot", "caption": "synthetic rust A", "source": "synthetic"},
        ],
    )


def test_synthetic_source_uses_real_references(tmp_path):
    cfg = _config(tmp_path)
    cfg.total_images = 6
    cfg.prompt.source = "synthetic"
    _write_real_and_synthetic(cfg)
    jobs = run(cfg, CLASSES)
    assert all(j.caption_source == "synthetic" and j.source_caption == "synthetic" for j in jobs)
    assert all("synthetic" in j.prompt for j in jobs)
    cat = [j for j in jobs if j.class_name == "cat"]
    assert all(j.reference in ("/r/cat/0.png", "/r/cat/1.png") for j in cat)


def test_mixed_source_draws_from_both(tmp_path):
    cfg = _config(tmp_path)
    cfg.total_images = 9
    cfg.prompt.source = "mixed"
    _write_real_and_synthetic(cfg)
    jobs = run(cfg, CLASSES)
    cat = [j for j in jobs if j.class_name == "cat"]
    assert {j.caption_source for j in cat} == {"real", "synthetic"}
    assert all(j.reference for j in jobs)


def test_synthetic_source_requires_synthesize(tmp_path):
    cfg = _config(tmp_path)
    cfg.prompt.source = "synthetic"
    append_jsonl(cfg.captions_path, [{"path": "/r/cat/0.png", "class_name": "cat", "caption": "x"}])
    with pytest.raises(FileNotFoundError, match="synthesize"):
        run(cfg, CLASSES)


def test_run_requires_captions(tmp_path):
    with pytest.raises(FileNotFoundError, match="describe"):
        run(_config(tmp_path), CLASSES)
