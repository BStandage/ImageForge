"""FLUX.1 through diffusers, with optional Redux image conditioning.

Stock Redux replaces the text prompt with the reference image almost entirely. Here the
reference embedding is scaled by `reference_strength` and appended to the text embedding, so
the prompt still steers content while the reference pulls the look toward the real data.
"""

from PIL import Image

from ..devices import resolve_device, resolve_dtype

# Token count of the T5 sequence Redux pads to when it runs without text encoders.
REDUX_TEXT_TOKENS = 512


class FluxBackend:
    supports_reference = True

    def __init__(self, cfg):
        from diffusers import FluxPipeline

        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.dtype = resolve_dtype(cfg.dtype)

        kwargs = {"torch_dtype": self.dtype}
        components = ["transformer", "text_encoder_2"]
        if cfg.transformer_gguf:
            kwargs["transformer"] = _load_gguf_transformer(cfg.transformer_gguf, self.dtype)
            components = ["text_encoder_2"]
        if cfg.quantization != "none":
            kwargs["quantization_config"] = _quant_config(cfg.quantization, self.dtype, components)

        self.pipe = FluxPipeline.from_pretrained(cfg.model_id, **kwargs)
        self.pipe.set_progress_bar_config(disable=True)

        if cfg.offload == "model":
            self.pipe.enable_model_cpu_offload(device=self.device)
        elif cfg.offload == "group":
            _enable_group_offload(self.pipe, self.device)
        elif cfg.offload == "sequential":
            self.pipe.enable_sequential_cpu_offload(device=self.device)
        else:
            self.pipe.to(self.device)

        self.redux = None
        if cfg.use_reference and cfg.reference_strength > 0:
            from diffusers import FluxPriorReduxPipeline

            # Loaded without text encoders: only the SigLIP encoder and the embedder, about 1 GB.
            self.redux = FluxPriorReduxPipeline.from_pretrained(cfg.redux_model_id, torch_dtype=self.dtype)
            self.redux.set_progress_bar_config(disable=True)
            self.redux.to(self.device)

    def generate(self, requests):
        import torch

        if not requests:
            return []
        sizes = {(r.width, r.height) for r in requests}
        if len(sizes) != 1:
            raise ValueError("all requests in a batch must share one size")
        width, height = sizes.pop()

        prompts = [r.prompt for r in requests]
        generators = [torch.Generator("cpu").manual_seed(r.seed) for r in requests]
        common = dict(
            height=height,
            width=width,
            guidance_scale=self.cfg.guidance,
            num_inference_steps=self.cfg.steps,
            max_sequence_length=self.cfg.max_sequence_length,
            generator=generators,
        )

        refs = [r.reference for r in requests]
        if self.redux is None or any(r is None for r in refs):
            return self.pipe(prompt=prompts, **common).images

        exec_device = self.pipe._execution_device
        with torch.inference_mode():
            prompt_embeds, pooled, _ = self.pipe.encode_prompt(
                prompt=prompts,
                prompt_2=None,
                device=exec_device,
                num_images_per_prompt=1,
                max_sequence_length=self.cfg.max_sequence_length,
            )
            # Without text encoders Redux returns [zeros(512 tokens) | image tokens]. Keep the image
            # part. One call per reference: given several images at once Redux blends them into a
            # single embedding, which is not what a batch of separate jobs wants.
            per_image = []
            for p in refs:
                with Image.open(p) as im:
                    out = self.redux(image=im.convert("RGB"))
                per_image.append(out.prompt_embeds[:, REDUX_TEXT_TOKENS:, :])
            image_embeds = torch.cat(per_image, dim=0).to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
            prompt_embeds = torch.cat([prompt_embeds, image_embeds * self.cfg.reference_strength], dim=1)

        return self.pipe(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled, **common).images


def _enable_group_offload(pipe, device):
    """Keep only the transformer blocks in use on the GPU; everything else streams from CPU RAM.

    The text encoders and VAE are small enough to move in and out whole, so they get
    model-level offload hooks. Quantized bitsandbytes modules cannot be moved after loading,
    so a 4-bit text encoder stays wherever the loader put it.
    """
    import torch
    from diffusers.hooks import apply_group_offloading

    onload = torch.device(device)
    offload = torch.device("cpu")
    pipe.transformer.enable_group_offload(
        onload_device=onload,
        offload_device=offload,
        offload_type="leaf_level",
        use_stream=True,
    )
    for name in ("text_encoder", "text_encoder_2", "vae"):
        component = getattr(pipe, name, None)
        if component is None or getattr(component, "is_loaded_in_4bit", False) or getattr(component, "is_loaded_in_8bit", False):
            continue
        apply_group_offloading(
            component,
            onload_device=onload,
            offload_device=offload,
            offload_type="block_level",
            num_blocks_per_group=1,
        )


def _load_gguf_transformer(source, compute_dtype):
    """Load a GGUF Flux transformer, for example a Q4_K_S file from city96/FLUX.1-dev-gguf."""
    from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig

    return FluxTransformer2DModel.from_single_file(
        source,
        quantization_config=GGUFQuantizationConfig(compute_dtype=compute_dtype),
        torch_dtype=compute_dtype,
    )


def _quant_config(kind, compute_dtype, components):
    from diffusers import PipelineQuantizationConfig

    if kind == "nf4":
        return PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": compute_dtype,
            },
            components_to_quantize=components,
        )
    if kind == "int8":
        return PipelineQuantizationConfig(
            quant_backend="bitsandbytes_8bit",
            quant_kwargs={"load_in_8bit": True},
            components_to_quantize=components,
        )
    raise ValueError(f"unknown quantization: {kind}")
