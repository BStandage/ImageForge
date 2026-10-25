"""Run configuration. Loaded from YAML, validated with pydantic."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class CaptionConfig(BaseModel):
    """How real images are described before prompting."""

    model: Literal["qwen2vl", "florence2"] = "qwen2vl"
    model_id: str | None = None
    # Cap how many real images per class get captioned. None means all of them.
    max_images_per_class: int | None = None
    quantization: Literal["none", "nf4"] = "none"
    device: str = "auto"
    max_new_tokens: int = 120
    # Images per generate call. 4 fits an 8 GB card; 16 keeps a 48 GB card busy.
    batch_size: int = 4
    instruction: str = (
        "Describe this photo in one detailed sentence for an image generation model. "
        "Cover the subject, its setting, lighting, viewpoint and photographic style. "
        "Do not guess at anything you cannot see."
    )


class SynthConfig(BaseModel):
    """Write new plausible descriptions per class from samples of the real captions."""

    enabled: bool = False
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    quantization: Literal["none", "nf4"] = "none"
    device: str = "auto"
    # New descriptions to write per class.
    per_class: int = 200
    # Real captions shown as examples in each call, and descriptions asked for per call.
    examples_per_call: int = 8
    outputs_per_call: int = 10
    temperature: float = 0.9
    top_p: float = 0.95
    max_new_tokens: int = 900
    # Fields: {k} {class_name} {examples} {n}
    instruction: str = (
        "Here are {k} descriptions of real aerial drone photographs, all from the category "
        "'{class_name}':\n\n{examples}\n\n"
        "Write {n} descriptions of different photographs that would plausibly belong to the same "
        "category, the same place and the same time. Vary the layout, building density, roads, "
        "vegetation, water, the amount and kind of debris, the viewing angle, altitude and lighting, "
        "but stay consistent with what the examples show and never contradict the category. "
        "Each description is two or three plain sentences of what is visible, with no interpretation "
        "and no commentary. Output a numbered list with one description per line and nothing else."
    )
    system_prompt: str = "You write descriptions of photographs for an image generation model."


class PromptConfig(BaseModel):
    """How captions become generation prompts."""

    # Where prompt text comes from: captions of real images, synthesised descriptions, or both.
    source: Literal["real", "synthetic", "mixed"] = "real"
    # Available fields: {class_name} {caption} {style}
    template: str = "{caption}"
    # Per-class override of `template`. Keys are class folder names.
    class_templates: dict[str, str] = Field(default_factory=dict)
    # Optional phrases sampled at random into {style}. Empty string if unset.
    styles: list[str] = Field(default_factory=list)
    # Flux ignores negatives. Kept for backends that honour them.
    negative: str | None = None


class BackendConfig(BaseModel):
    """Which model generates, and how it is loaded."""

    name: Literal["flux", "sdxl"] = "flux"
    model_id: str = "black-forest-labs/FLUX.1-dev"
    # Flux only. Image conditioning so output matches the look of the real set.
    redux_model_id: str = "black-forest-labs/FLUX.1-Redux-dev"
    use_reference: bool = True
    # same: the reference is the real image whose caption is in the prompt, so outputs track
    # single originals closely. random: a different real image of the same class, which
    # recombines scene descriptions and looks across the pool and gives a more varied set.
    reference_pairing: Literal["same", "random"] = "same"
    # Weight of the reference image embedding relative to the text prompt.
    # 0 is text only. 1 is the stock Redux behaviour, which mostly ignores text.
    reference_strength: float = 0.35
    width: int = 1024
    height: int = 1024
    steps: int = 28
    guidance: float = 3.5
    max_sequence_length: int = 512
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    # Flux only. A pre-quantized GGUF transformer, as a local path or a Hugging Face file URL.
    # Cuts the biggest download from 24 GB to about 7 GB at Q4. The rest of the model still
    # comes from model_id.
    transformer_gguf: str | None = None
    # Applies to the transformer (unless transformer_gguf is set) and the T5 text encoder.
    quantization: Literal["none", "nf4", "int8"] = "none"
    # model: swap whole components in and out of VRAM, needs the biggest component to fit.
    # group: stream transformer blocks in and out of VRAM; small footprint, modest slowdown.
    # sequential: layer by layer, smallest footprint, slowest.
    offload: Literal["none", "model", "group", "sequential"] = "none"
    device: str = "auto"
    batch_size: int = 1


class FilterConfig(BaseModel):
    """CLIP-based rejection of off-class or low-quality outputs, and how the survivors are exported."""

    # Side length of the exported images. None keeps the generated size. Generating at 512 and
    # exporting at 256 gives better small images than generating at 256.
    export_size: int | None = None
    export_format: Literal["png", "jpg"] = "png"
    # After the gate, subsample every class to the size of the smallest so the export is balanced.
    balance: bool = False
    enabled: bool = True
    model_id: str = "openai/clip-vit-base-patch32"
    # Cosine similarity to the class centroid of the real images.
    min_similarity: float = 0.55
    # Also reject if another class's centroid is closer than the intended one.
    require_nearest_class: bool = True
    # How much closer another class must be before the image is rejected. 0 rejects on any
    # tie-break. Real classes overlap in embedding space, so a margin of 0 produces a synthetic
    # class that is purer than the real one, which a classifier then learns. Set it so the kept
    # set leans toward neighbouring classes about as often as the real class does. A single
    # number applies to every class; a mapping sets it per class folder name.
    nearest_class_margin: float | dict[str, float] = 0.0

    def margin_for(self, class_name):
        if isinstance(self.nearest_class_margin, dict):
            return float(self.nearest_class_margin.get(class_name, 0.0))
        return float(self.nearest_class_margin)
    device: str = "auto"


class RemoteConfig(BaseModel):
    """Run the whole pipeline on another machine over SSH."""

    host: str | None = None  # e.g. root@1.2.3.4
    port: int = 22
    workdir: str = "~/imageforge"
    python: str = "python3"
    # Copy the local Hugging Face token to the remote at setup so gated weights download there.
    copy_hf_token: bool = True


class Config(BaseModel):
    ground_truth: Path
    output: Path = Path("runs/latest")
    total_images: int = 100
    # Explicit per-class counts. Classes not listed share the remainder evenly.
    per_class: dict[str, int] = Field(default_factory=dict)
    seed: int = 0

    caption: CaptionConfig = Field(default_factory=CaptionConfig)
    synth: SynthConfig = Field(default_factory=SynthConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    remote: RemoteConfig = Field(default_factory=RemoteConfig)

    @model_validator(mode="after")
    def _check(self):
        if self.total_images < 0:
            raise ValueError("total_images must be >= 0")
        if not 0.0 <= self.backend.reference_strength <= 1.0:
            raise ValueError("backend.reference_strength must be between 0 and 1")
        return self

    # Derived paths. Everything a run produces lives under `output`.
    @property
    def ground_truth_manifest(self):
        return self.output / "ground_truth.json"

    @property
    def captions_path(self):
        return self.output / "captions.jsonl"

    @property
    def synthetic_captions_path(self):
        return self.output / "synthetic_captions.jsonl"

    @property
    def jobs_path(self):
        return self.output / "jobs.jsonl"

    @property
    def generated_path(self):
        return self.output / "generated.jsonl"

    @property
    def images_dir(self):
        return self.output / "images"

    @property
    def filtered_path(self):
        return self.output / "filtered.jsonl"

    @property
    def dataset_dir(self):
        return self.output / "dataset"


def load_config(path, **overrides):
    """Read a YAML config. Keyword overrides replace top-level fields."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for key, value in overrides.items():
        if value is not None:
            data[key] = value
    cfg = Config.model_validate(data)
    # Relative paths resolve against the config file, not the shell.
    base = path.resolve().parent
    cfg.ground_truth = _resolve(cfg.ground_truth, base)
    cfg.output = _resolve(cfg.output, base)
    return cfg


def _resolve(p, base):
    p = Path(p).expanduser()
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def dump_config(cfg, path):
    data = cfg.model_dump(mode="json")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
