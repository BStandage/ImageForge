# ImageForge

Forge class-balanced synthetic training images from a small set of real ones, with local diffusion models, and measure how close the result sits to the real distribution.

You give it a folder of real images with one subfolder per class and an image budget. It describes each real image with a vision model, writes more descriptions in the same vein with a text model, plans one job per image, generates with FLUX.1-dev conditioned on both a description and a real reference frame, drops the images that landed in the wrong class, and exports class folders ready for training, along with the list of real images it learned from and a set of distribution metrics against a real-versus-real baseline.

Every stage is a file on disk and resumes from what is already there. The generator sits behind a one-method interface. The same config runs on an 8 GB gaming card, in 4-bit with streamed weights, or on a rented 48 GB card at full precision, driven over SSH.

## Results so far

The first target is building damage assessment from drone imagery, using the public RescueNet image-classification release: 3,595 training frames of neighbourhoods after Hurricane Michael, labelled Superficial, Medium or Major damage, with official validation and test splits of 449 and 450 frames.

A night on one rented L40S produced 6,500 images at 512 px from the training split alone. The classes differ by degree of damage on near-identical scenes, which is the hard case for this approach.

| | real train set | forged set, 6,500 generated, 4,074 kept |
|---|---|---|
| real frames the classifier saw | 3,595 | 0 |
| KID to the real pool, CLIP features | 0.000, the baseline | 0.060 |
| nearest-real similarity | 0.94, the baseline | 0.90, so nothing is a copy |
| ViT-base, same recipe, validation accuracy | 0.77 | 0.64 |

A classifier trained on forged images only reaches about 83% of the accuracy of one trained on the real set, on a benchmark whose labels disagree with the images about a quarter of the time. The forged runs were also cut short by early stopping at the backbone unfreeze, so that gap is an upper bound on the current pipeline rather than a floor. The remaining systematic error is that the generator renders damage milder than its label; the gate removes those images rather than preventing them. A LoRA stage trained on the real frames is the planned fix, and mixed real-plus-forged training is the next experiment.

## Install

Core package, enough to plan jobs and run the tests:

```
pip install -r requirements-dev.txt
```

On the machine with the GPU, install PyTorch from the CUDA index that matches your driver, then the model stack:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

FLUX.1-dev and FLUX.1-Redux-dev are gated on Hugging Face. Accept the licence on each model page and log in once with `huggingface-cli login`. FLUX.1-dev weights are licensed for non-commercial use; swap the backend before selling anything made with it.

## Quick start

Lay the real images out as one folder per class:

```
data/real/
  rust_spot/
    img001.jpg
  clean/
    img001.jpg
```

Copy [configs/example.yaml](configs/example.yaml), set `ground_truth` and `output`, then:

```
imageforge run --config configs/my_run.yaml --total 500
```

Without data of your own:

```
python scripts/fetch_sample_dataset.py data/sample --classes 3 --per-class 12
imageforge run --config configs/sample_local.yaml
```

That pulls three Imagenette classes and runs the whole pipeline at 512 px on an 8 GB card.

## The pipeline

Each stage is its own command, in this order, and `run` chains them.

| command | what it does | writes to `output/` |
|---|---|---|
| `ingest` | scan the class folders | `ground_truth.json` |
| `describe` | caption every real image with Qwen2.5-VL | `captions.jsonl` |
| `synthesize` | write new plausible descriptions per class from the real captions | `synthetic_captions.jsonl` |
| `plan` | split the budget across classes and write one job per image | `jobs.jsonl` |
| `generate` | run the jobs through the backend | `images/<class>/*.png`, `generated.jsonl` |
| `filter` | gate off-class images, balance, export | `filtered.jsonl`, `dataset/<class>/*` |
| `fidelity` | distribution metrics against the real pool | `fidelity.json` |

**Describe.** The caption instruction is part of the config and should be written for the domain: ask for what is visible and forbid interpretation, or the model parrots the label back. Captions are cleaned of chat preamble and of any sentence cut off by the token limit.

**Synthesize.** A text model reads eight real captions from one class and writes ten descriptions of other images from the same set, matched in length and content. It repeats until it has `synth.per_class` per class, deduplicated, each recording the real captions it was written from. `prompt.source` picks what the planner draws from: `real`, `synthetic` or `mixed`.

**Plan.** `total_images` is split evenly, or by `per_class` with the others sharing the remainder. Each job takes a caption, cycling through the pool before repeating; a real reference frame of the same class, the caption's own frame or a random one depending on `reference_pairing`; a per-class template; a style phrase; and a seed. The plan is written before any GPU time is spent, so a run can be inspected first and resumed later.

**Generate.** FLUX.1-dev through diffusers. The reference frame is encoded by Flux Redux and its image tokens are scaled by `reference_strength` and appended to the text embedding, so the prompt sets the content and the reference sets the camera, palette and texture. Stock Redux would let the reference override the prompt. Every output records its prompt, seed, reference, caption source, model and hash.

**Filter and export.** CLIP embeds the real frames the pipeline used and the forged images. A forged image whose nearest class centroid is another class by more than `nearest_class_margin` is rejected. Real classes overlap, so a margin of zero produces forged classes purer than the real ones, and a classifier trained on them mislabels the real class's messier examples; set the margin, per class if needed, so the kept set leans toward neighbouring classes about as often as the real class does. `balance` subsamples to the smallest class. Survivors are exported at `export_size`, with `real_images_used.txt` beside them so held-out splits stay clean. Rejected images stay under `images/` with the reason recorded, and re-exporting with different settings takes minutes.

**Fidelity.** Four numbers per class and pooled, each beside the same measurement between a random subset of real frames and the rest:

- `kid`, Kernel Inception Distance on raw CLIP ViT-B/32 features. Unbiased, so it is the number to trust for small sets. Two samples of one distribution read zero.
- `fd`, the Fréchet distance on the same features. Biased upward at small sample sizes, which the size-matched baseline accounts for.
- `nn`, mean cosine similarity of each forged image to its nearest real one. Far above the baseline means the generator is copying reference frames.
- `div`, mean pairwise distance within the set. Far below the baseline means collapse.

These describe the data, not what a classifier will do with it. That measurement belongs wherever the classifier is trained, and the export is built to make it a clean experiment.

## Hardware

[configs/local_8gb.yaml](configs/local_8gb.yaml) fits an RTX 3070 Ti: a pre-quantised GGUF transformer, the T5 encoder in 4-bit, and `offload: group`, which streams transformer blocks through VRAM. Peak VRAM is under 5 GB, a 512 px image takes about 36 seconds, and the weights take about 20 GB of disk. Avoid `offload: model` with a GGUF transformer on 8 GB: the dequantised weights spill into shared memory and each step takes minutes.

[configs/rescuenet_rented.yaml](configs/rescuenet_rented.yaml) is the same run on a 48 GB card: full precision, no offload, batches of four, about 4.6 seconds per image. Set `HF_HOME` to put the weight cache on a drive with room.

## Running on another machine

Set `remote.host` and `remote.port` to any SSH target, a rented pod included. Once:

```
imageforge remote setup --config configs/rescuenet_rented.yaml
```

That copies the repo, builds a venv on top of the host's PyTorch, and copies your Hugging Face token so gated weights download there. Then:

```
imageforge remote run --config configs/rescuenet_rented.yaml -n 100 --even
imageforge remote run --config configs/rescuenet_rented.yaml --no-sync --replan
```

The first is a check; the second is the full count. The pool is uploaded, the run launches detached on the host so a dropped connection cannot kill it, progress is polled, and the output folder is pulled back. Only `ssh` and `tar` are needed on both ends. Shrink large frames first with `scripts/downscale_pool.py`; nothing in the pipeline looks at more than about 1024 px. Revoke the token when the host is gone.

## Repository layout

```
imageforge/
  config.py        one YAML config per run, validated
  dataset.py       class-folder ingest
  describe.py      captioning stage
  synthesize.py    description synthesis stage
  prompts.py       caption pool and templating
  planner.py       budget split and job list
  generate.py      resumable generation loop
  filter.py        CLIP gate, balance, export
  fidelity.py      KID, Fréchet, nearest-real, diversity
  remote.py        SSH runner
  cli.py           commands
  captioning/      Qwen2.5-VL and Florence-2 captioners
  backends/        FLUX.1 with Redux; SDXL as the minimal second backend
configs/           worked examples: sample, 8 GB local, rented card, remote host
scripts/           sample dataset fetch, RescueNet preparation, downscaling, contact sheets
tests/             60 tests; no model downloads
```

To add a backend, implement a class with a `supports_reference` flag and a `generate(requests) -> list[Image]` method, as in [imageforge/backends/sdxl.py](imageforge/backends/sdxl.py), and register it in `load_backend`.

## Tests

```
pytest
```

## What it is for

The point is the small-data case: a team with forty photos of the thing they care about, a rare defect, a rare damage state, a rare finding, and no practical way to collect four thousand. The pipeline turns a small labelled set into a large labelled one that stays inside the real distribution, with numbers attached that say how far inside, so the synthetic data can be trusted or rejected on evidence rather than by eye.
