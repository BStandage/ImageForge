"""Florence-2 captioner. Small and fast. Needs trust_remote_code and a transformers version its remote code supports."""


from ..devices import resolve_device
from .base import clean_caption

DEFAULT_MODEL = "microsoft/Florence-2-large"
TASK = "<MORE_DETAILED_CAPTION>"


class FlorenceCaptioner:
    def __init__(self, cfg):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.cfg = cfg
        model_id = cfg.model_id or DEFAULT_MODEL
        self.device = resolve_device(cfg.device)
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=self.dtype, trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def caption(self, images):
        import torch

        rgb = [im.convert("RGB") for im in images]
        inputs = self.processor(text=[TASK] * len(rgb), images=rgb, return_tensors="pt")
        inputs = inputs.to(self.device, self.dtype)
        with torch.inference_mode():
            ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.cfg.max_new_tokens,
                num_beams=3,
                do_sample=False,
            )
        texts = self.processor.batch_decode(ids, skip_special_tokens=False)
        out = []
        for t, im in zip(texts, rgb):
            parsed = self.processor.post_process_generation(t, task=TASK, image_size=im.size)
            out.append(clean_caption(parsed.get(TASK, t)))
        return out
