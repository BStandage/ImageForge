"""Load the backend from a config and generate one image. Checks the model path without a dataset.

    python scripts/smoke_backend.py configs/local_8gb.yaml "a rusted steel bolt on a workbench" [reference.jpg]
"""

import sys
import time
from pathlib import Path

from imageforge.backends import GenRequest, load_backend
from imageforge.config import load_config


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    cfg = load_config(Path(sys.argv[1]))
    prompt = sys.argv[2]
    reference = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    if reference is None:
        cfg.backend.use_reference = False

    t0 = time.time()
    backend = load_backend(cfg.backend)
    print(f"loaded {cfg.backend.name} in {time.time() - t0:.0f}s")
    pipe = getattr(backend, "pipe", None)
    if pipe is not None:
        pipe.set_progress_bar_config(disable=False)

    t0 = time.time()
    image = backend.generate(
        [GenRequest(prompt=prompt, seed=0, width=cfg.backend.width, height=cfg.backend.height, reference=reference)]
    )[0]
    print(f"generated in {time.time() - t0:.0f}s")

    out = Path("runs") / "smoke.png"
    out.parent.mkdir(exist_ok=True)
    image.save(out)
    print(f"wrote {out}")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
