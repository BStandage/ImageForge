"""Side-by-side sheet of reference frame and generated image for a run, grouped by class.

    python scripts/contact_sheet.py runs/rescuenet [--per-class 6] [--tile 320]

Writes contact_sheet.png into the run folder. Pairs with no reference show the generated image alone.
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--tile", type=int, default=320)
    ap.add_argument("--gt", type=Path, default=None,
                    help="local ground-truth root used to find reference frames when the run came from another machine")
    args = ap.parse_args()

    def local_reference(path, class_name):
        p = Path(path)
        if p.exists():
            return p
        if args.gt is not None:
            for cand in (args.gt / class_name / p.name, args.gt / class_name / (p.stem + ".jpg")):
                if cand.exists():
                    return cand
        return None

    rows = [json.loads(l) for l in open(args.run / "generated.jsonl", encoding="utf-8") if l.strip()]
    by_class = {}
    for r in rows:
        by_class.setdefault(r["class_name"], []).append(r)

    tile = args.tile
    header = 18
    pair_w = tile * 2 + 4
    pair_h = tile + header
    cols = 2
    pairs = [(c, r) for c in sorted(by_class) for r in by_class[c][: args.per_class]]
    rows_n = (len(pairs) + cols - 1) // cols
    sheet = Image.new("RGB", (pair_w * cols, pair_h * rows_n), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (class_name, r) in enumerate(pairs):
        x = (i % cols) * pair_w
        y = (i // cols) * pair_h
        draw.text((x + 4, y + 2), f"{r['id']}   left: reference   right: generated", fill="black")
        ref_path = local_reference(r["reference"], class_name) if r.get("reference") else None
        if ref_path is not None:
            ref = Image.open(ref_path).convert("RGB")
            ref.thumbnail((tile, tile))
            sheet.paste(ref, (x, y + header))
        gen_path = Path(r["image"])
        if not gen_path.exists():  # manifest written on another machine
            gen_path = args.run / "images" / class_name / gen_path.name
        gen = Image.open(gen_path).convert("RGB")
        gen.thumbnail((tile, tile))
        sheet.paste(gen, (x + tile + 4, y + header))
    out = args.run / "contact_sheet.png"
    sheet.save(out)
    print(f"wrote {out} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
