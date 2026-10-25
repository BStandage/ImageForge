import random

import pytest

from imageforge.config import PromptConfig
from imageforge.prompts import Caption, PromptBank

CAPS = [
    Caption("/a.png", "rust_spot", "orange corrosion on a bolt"),
    Caption("/b.png", "rust_spot", "flaking paint over rust"),
    Caption("/c.png", "clean", "a shiny bolt"),
]


def test_render_uses_template_and_pretty_name():
    bank = PromptBank(PromptConfig(template="photo of {class_name}, {caption}"), CAPS)
    out = bank.render("rust_spot", CAPS[0], random.Random(0))
    assert out == "photo of rust spot, orange corrosion on a bolt"


def test_render_class_template_override_and_style():
    cfg = PromptConfig(
        template="{caption}",
        class_templates={"clean": "pristine {class_name}: {caption} {style}"},
        styles=["studio light"],
    )
    bank = PromptBank(cfg, CAPS)
    assert bank.render("clean", CAPS[2], random.Random(0)) == "pristine clean: a shiny bolt studio light"
    assert bank.render("rust_spot", CAPS[0], random.Random(0)) == "orange corrosion on a bolt"


def test_render_collapses_whitespace_when_style_empty():
    bank = PromptBank(PromptConfig(template="{caption}  {style}"), CAPS)
    assert bank.render("clean", CAPS[2], random.Random(0)) == "a shiny bolt"


def test_sample_captions_cycles_before_repeating():
    bank = PromptBank(PromptConfig(), CAPS)
    picked = bank.sample_captions("rust_spot", 5, random.Random(1))
    paths = [c.path for c in picked]
    assert sorted(paths[:2]) == ["/a.png", "/b.png"]
    assert sorted(paths[2:4]) == ["/a.png", "/b.png"]
    assert len(picked) == 5


def test_sample_captions_unknown_class():
    bank = PromptBank(PromptConfig(), CAPS)
    with pytest.raises(ValueError, match="describe"):
        bank.sample_captions("bird", 1, random.Random(0))


def test_classes_sorted():
    assert PromptBank(PromptConfig(), CAPS).classes() == ["clean", "rust_spot"]
