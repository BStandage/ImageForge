"""Stable Diffusion XL through diffusers. Text only; here to show the backend seam works.

Set backend.model_id to an SDXL checkpoint, for example stabilityai/stable-diffusion-xl-base-1.0.
"""


from ..devices import resolve_device, resolve_dtype


class SDXLBackend:
    supports_reference = False

    def __init__(self, cfg):
        from diffusers import StableDiffusionXLPipeline

        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.dtype = resolve_dtype(cfg.dtype)
        self.pipe = StableDiffusionXLPipeline.from_pretrained(cfg.model_id, torch_dtype=self.dtype)
        self.pipe.set_progress_bar_config(disable=True)
        if cfg.offload == "model":
            self.pipe.enable_model_cpu_offload(device=self.device)
        elif cfg.offload == "sequential":
            self.pipe.enable_sequential_cpu_offload(device=self.device)
        else:
            self.pipe.to(self.device)

    def generate(self, requests):
        import torch

        if not requests:
            return []
        sizes = {(r.width, r.height) for r in requests}
        if len(sizes) != 1:
            raise ValueError("all requests in a batch must share one size")
        width, height = sizes.pop()
        negatives = [r.negative or "" for r in requests]
        return self.pipe(
            prompt=[r.prompt for r in requests],
            negative_prompt=negatives if any(negatives) else None,
            height=height,
            width=width,
            guidance_scale=self.cfg.guidance,
            num_inference_steps=self.cfg.steps,
            generator=[torch.Generator("cpu").manual_seed(r.seed) for r in requests],
        ).images
