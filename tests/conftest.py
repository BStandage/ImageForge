
import pytest
from PIL import Image


def make_image(path, size=(32, 24), color=(120, 40, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


@pytest.fixture
def ground_truth_dir(tmp_path):
    root = tmp_path / "real"
    for i in range(3):
        make_image(root / "cat" / f"cat_{i}.png")
    for i in range(2):
        make_image(root / "dog" / f"dog_{i}.jpg")
    make_image(root / "rust_spot" / "r0.webp")
    (root / "rust_spot" / "notes.txt").write_text("ignored")
    return root
