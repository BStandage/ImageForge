"""Qwen2.5-VL captioner. Default because it needs no remote code and gives detailed, grounded captions."""


from ..devices import resolve_device
from .base import clean_caption

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

# Keep inputs bounded so a 4000px photo does not turn into thousands of vision tokens.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 768 * 28 * 28


class QwenCaptioner:
    def __init__(self, cfg):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.cfg = cfg
        model_id = cfg.model_id or DEFAULT_MODEL
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

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        if "device_map" not in kwargs:
            self.model.to(device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )
        # Batched generation with a decoder-only model needs left padding, or the shorter
        # prompts in a batch produce garbage.
        self.processor.tokenizer.padding_side = "left"

    def caption(self, images):
        import torch

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": self.cfg.instruction}],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        rgb = [im.convert("RGB") for im in images]
        inputs = self.processor(text=[text] * len(rgb), images=rgb, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=self.cfg.max_new_tokens, do_sample=False
            )
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        decoded = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return [clean_caption(t) for t in decoded]
