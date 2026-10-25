"""Stage 2b: write new plausible descriptions per class from samples of the real captions.

A text model sees a handful of real captions from one class and is asked for more descriptions
that would belong to the same set. The output widens the prompt pool beyond the real frames
while staying inside what they look like. Resumable: existing descriptions are kept and counted.
"""

import random
import re

from .captioning.base import clean_caption
from .dataset import pretty_class_name
from .io import append_jsonl, read_jsonl

LINE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s*(.+?)\s*$")
MIN_WORDS = 8


def parse_descriptions(text):
    """Pull the items out of a numbered or bulleted list, dropping anything too short."""
    out = []
    for line in text.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        item = m.group(1).strip().strip('"').strip()
        if len(item.split()) >= MIN_WORDS:
            out.append(" ".join(item.split()))
    return out


def build_prompt(cfg, class_name, examples):
    numbered = "\n".join(f"{i}. {e}" for i, e in enumerate(examples, start=1))
    return cfg.instruction.format(
        k=len(examples),
        class_name=pretty_class_name(class_name),
        examples=numbered,
        n=cfg.outputs_per_call,
    )


class TextGenerator:
    def __init__(self, cfg):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .devices import resolve_device

        self.cfg = cfg
        device = resolve_device(cfg.device)
        kwargs = {"dtype": torch.bfloat16 if device != "cpu" else torch.float32}
        if cfg.quantization == "nf4":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
            kwargs["device_map"] = "auto"
        elif device == "cuda":
            kwargs["device_map"] = "auto"
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
        if "device_map" not in kwargs:
            self.model.to(device)
        self.model.eval()

    def complete(self, user, seed):
        import torch

        messages = [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": user},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        torch.manual_seed(seed)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
            )
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def run(cfg, gt):
    real = {c: [] for c in gt.classes}
    for row in read_jsonl(cfg.captions_path):
        real.setdefault(row["class_name"], []).append({**row, "caption": clean_caption(row["caption"])})
    if not any(real.values()):
        raise FileNotFoundError(f"no captions at {cfg.captions_path}; run describe first")

    have = {c: [] for c in gt.classes}
    for row in read_jsonl(cfg.synthetic_captions_path):
        have.setdefault(row["class_name"], []).append(row["caption"])

    todo = {c: cfg.synth.per_class - len(have[c]) for c in gt.classes if cfg.synth.per_class > len(have[c])}
    if not todo:
        return 0

    generator = TextGenerator(cfg.synth)
    rng = random.Random(cfg.seed)
    written = 0
    for class_name, need in todo.items():
        examples_pool = real.get(class_name, [])
        if not examples_pool:
            raise ValueError(f"no real captions for class {class_name!r}")
        seen = {c.lower() for c in have[class_name]}
        got = 0
        calls = 0
        max_calls = max(4, (need // cfg.synth.outputs_per_call + 1) * 4)
        while got < need and calls < max_calls:
            calls += 1
            sample = rng.sample(examples_pool, min(cfg.synth.examples_per_call, len(examples_pool)))
            text = generator.complete(build_prompt(cfg.synth, class_name, [r["caption"] for r in sample]), rng.randrange(0, 2**31 - 1))
            fresh = [d for d in parse_descriptions(text) if d.lower() not in seen][: need - got]
            if not fresh:
                continue
            seen.update(d.lower() for d in fresh)
            append_jsonl(
                cfg.synthetic_captions_path,
                (
                    {
                        "class_name": class_name,
                        "caption": d,
                        "source": "synthetic",
                        "examples": [r["path"] for r in sample],
                    }
                    for d in fresh
                ),
            )
            got += len(fresh)
        print(f"{class_name}: {got} new descriptions in {calls} calls")
        written += got
    return written
