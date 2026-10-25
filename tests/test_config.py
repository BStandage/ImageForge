from pathlib import Path

import pytest
import yaml

from imageforge.config import Config, dump_config, load_config


def test_relative_paths_resolve_against_config_file(tmp_path):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "run.yaml"
    cfg_path.write_text(yaml.safe_dump({"ground_truth": "../data", "output": "../runs/x", "total_images": 5}))
    cfg = load_config(cfg_path)
    assert cfg.ground_truth == (tmp_path / "data").resolve()
    assert cfg.output == (tmp_path / "runs" / "x").resolve()
    assert cfg.total_images == 5


def test_overrides_replace_fields(tmp_path):
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(yaml.safe_dump({"ground_truth": "data", "total_images": 5}))
    cfg = load_config(cfg_path, total_images=50, output=None)
    assert cfg.total_images == 50
    assert cfg.output == (tmp_path / "runs" / "latest").resolve()


def test_home_paths_expand(tmp_path):
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(yaml.safe_dump({"ground_truth": "~/somewhere"}))
    cfg = load_config(cfg_path)
    assert "~" not in str(cfg.ground_truth)
    assert cfg.ground_truth.is_absolute()


def test_validation():
    with pytest.raises(ValueError):
        Config(ground_truth="x", total_images=-1)
    with pytest.raises(ValueError):
        Config(ground_truth="x", backend={"reference_strength": 2.0})
    with pytest.raises(ValueError):
        Config(ground_truth="x", backend={"name": "dalle"})


def test_dump_roundtrip(tmp_path):
    cfg = Config(ground_truth=tmp_path / "gt", output=tmp_path / "out", total_images=3)
    cfg.backend.quantization = "nf4"
    out = tmp_path / "nested" / "cfg.yaml"
    dump_config(cfg, out)
    back = load_config(out)
    assert back.total_images == 3
    assert back.backend.quantization == "nf4"
    assert back.ground_truth == (tmp_path / "gt").resolve()


def test_derived_paths():
    cfg = Config(ground_truth="gt", output=Path("/o"))
    assert cfg.jobs_path == Path("/o/jobs.jsonl")
    assert cfg.images_dir == Path("/o/images")
    assert cfg.dataset_dir == Path("/o/dataset")
