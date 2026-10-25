"""Captioner interface. One method: a list of images in, a list of strings out."""

from typing import Protocol


class Captioner(Protocol):
    def caption(self, images): ...


def load_captioner(cfg):
    if cfg.model == "qwen2vl":
        from .qwen import QwenCaptioner

        return QwenCaptioner(cfg)
    if cfg.model == "florence2":
        from .florence import FlorenceCaptioner

        return FlorenceCaptioner(cfg)
    raise ValueError(f"unknown captioner: {cfg.model}")


import re

# "The aerial drone photograph shows ...", "This image depicts ...", "In the photo, ..."
_PREAMBLE = re.compile(
    r"^(?:(?:the|this)\s+(?:aerial\s+)?(?:drone\s+)?(?:photograph|photo|image|picture|view)\s+"
    r"(?:shows|depicts|features|captures|displays|presents)\s+"
    r"|in\s+(?:the|this)\s+(?:aerial\s+)?(?:drone\s+)?(?:photograph|photo|image|picture),?\s+)",
    re.IGNORECASE,
)


def clean_caption(text):
    """Strip chat-style framing that vision models like to add, and unfinished tails."""
    text = text.strip().strip('"').strip()
    m = _PREAMBLE.match(text)
    if m:
        text = text[m.end():]
        text = text[0].upper() + text[1:] if text else text
    text = " ".join(text.split())
    return drop_trailing_fragment(text)


def drop_trailing_fragment(text):
    """If generation hit the token limit mid-sentence, drop the unfinished sentence."""
    if not text or text[-1] in ".!?":
        return text
    cut = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
    if cut == -1:
        return text
    return text[: cut + 1]
