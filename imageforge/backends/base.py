"""Backend interface. A backend turns a batch of requests into a batch of PIL images.

Adding a model means one new module with a class that has `generate`, and one line in `load_backend`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class GenRequest:
    prompt: str
    seed: int
    width: int
    height: int
    negative: str | None = None
    # Real image to condition on. Backends that cannot use it ignore it.
    reference: Path | None = None


class ImageBackend(Protocol):
    supports_reference: bool

    def generate(self, requests): ...


def load_backend(cfg):
    if cfg.name == "flux":
        from .flux import FluxBackend

        return FluxBackend(cfg)
    if cfg.name == "sdxl":
        from .sdxl import SDXLBackend

        return SDXLBackend(cfg)
    raise ValueError(f"unknown backend: {cfg.name}")
