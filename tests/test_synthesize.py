
from imageforge import synthesize
from imageforge.config import Config, SynthConfig
from imageforge.dataset import GroundTruth, GroundTruthImage
from imageforge.io import append_jsonl, read_jsonl


def test_parse_descriptions_handles_numbered_and_bulleted_lists():
    text = """Here you go:
1. A grid of single-storey houses with grey roofs and a straight road between them, light debris in the yards.
2) "Dense houses along a canal, several roofs stripped to decking, piles of timber by the curb."
- Short one.
3. Not a list item because it is fine and long enough to pass the word filter here.
random prose line that should be ignored entirely
"""
    out = synthesize.parse_descriptions(text)
    assert out[0].startswith("A grid of single-storey houses")
    assert out[1].startswith("Dense houses along a canal")
    assert out[2].startswith("Not a list item")
    assert len(out) == 3


def test_build_prompt_fills_fields():
    cfg = SynthConfig(instruction="{k} examples of {class_name}:\n{examples}\nwrite {n}", outputs_per_call=3)
    p = synthesize.build_prompt(cfg, "Major_Damage", ["one two", "three four"])
    assert p == "2 examples of Major Damage:\n1. one two\n2. three four\nwrite 3"


class FakeGenerator:
    def __init__(self):
        self.calls = 0

    def complete(self, user, seed):
        self.calls += 1
        return "\n".join(
            f"{i}. Synthetic description number {self.calls}-{i} with enough words to pass the filter."
            for i in range(1, 4)
        )


def test_run_writes_per_class_and_resumes(tmp_path, monkeypatch):
    cfg = Config(ground_truth=tmp_path / "gt", output=tmp_path / "out")
    cfg.synth.per_class = 5
    cfg.synth.outputs_per_call = 3
    gt = GroundTruth(str(tmp_path), ["a", "b"], [GroundTruthImage("/a/0.jpg", "a", 8, 8), GroundTruthImage("/b/0.jpg", "b", 8, 8)])
    append_jsonl(
        cfg.captions_path,
        [{"path": "/a/0.jpg", "class_name": "a", "caption": "real a"}, {"path": "/b/0.jpg", "class_name": "b", "caption": "real b"}],
    )
    monkeypatch.setattr(synthesize, "TextGenerator", lambda _cfg: FakeGenerator())

    assert synthesize.run(cfg, gt) == 10
    rows = list(read_jsonl(cfg.synthetic_captions_path))
    assert sum(r["class_name"] == "a" for r in rows) == 5
    assert sum(r["class_name"] == "b" for r in rows) == 5
    assert all(r["source"] == "synthetic" and r["examples"] for r in rows)

    # Already full: nothing new, and the generator is never built.
    monkeypatch.setattr(synthesize, "TextGenerator", lambda _cfg: (_ for _ in ()).throw(AssertionError("loaded")))
    assert synthesize.run(cfg, gt) == 0
