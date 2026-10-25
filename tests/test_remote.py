"""The remote runner's config hand-off, with ssh/scp/tar stubbed out."""

from pathlib import Path

import yaml

from imageforge import remote
from imageforge.config import Config


def test_remote_config_uses_posix_paths_and_drops_host(tmp_path, monkeypatch):
    cfg = Config(ground_truth=tmp_path / "gt", output=tmp_path / "out")
    cfg.remote.host = "root@1.2.3.4"
    cfg.remote.port = 4242
    cfg.remote.workdir = "/workspace/imageforge"
    (tmp_path / "gt").mkdir()

    sent = {}
    commands = []
    monkeypatch.setattr(remote, "push_dir", lambda *a, **k: None)
    monkeypatch.setattr(remote, "pull_dir", lambda *a, **k: None)
    monkeypatch.setattr(remote, "_ssh", lambda c, command, check=True: commands.append(command) or 0)

    def fake_scp(c, local, dest):
        sent["config"] = yaml.safe_load(Path(local).read_text(encoding="utf-8"))
        sent["dest"] = dest

    monkeypatch.setattr(remote, "_scp", fake_scp)
    monkeypatch.setattr(remote, "wait_for_status", lambda *a, **k: 0)

    remote.run(cfg, "myrun", sync_ground_truth=True, replan=True)

    data = sent["config"]
    assert data["ground_truth"] == "/workspace/imageforge/runs/myrun/ground_truth"
    assert data["output"] == "/workspace/imageforge/runs/myrun/output"
    assert "\\" not in data["ground_truth"]
    assert data["remote"]["host"] is None
    assert sent["dest"] == "/workspace/imageforge/runs/myrun/config.yaml"
    assert any("imageforge run --config /workspace/imageforge/runs/myrun/config.yaml --replan" in c for c in commands)
