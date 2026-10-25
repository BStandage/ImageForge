"""Command line entry point. Each stage is its own command; `run` chains them."""

from pathlib import Path

import click
from rich import print as rprint
from rich.table import Table

from . import dataset
from .config import dump_config, load_config

config_option = click.option(
    "--config", "-c", "config", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="YAML config file.",
)
total_option = click.option("--total", "-n", "total", type=int, default=None, help="Override total_images.")
output_option = click.option("--output", "-o", "output", type=click.Path(path_type=Path), default=None, help="Override output folder.")
even_option = click.option("--even", is_flag=True, help="Ignore per_class and split the total evenly.")
replan_option = click.option("--replan", is_flag=True, help="Discard an existing job list and plan again.")
skip_filter_option = click.option("--skip-filter", is_flag=True, help="Export everything without the CLIP gate.")


@click.group(help="Synthetic training images from a real image set.")
def app():
    pass


def _load(config, total=None, output=None, even=False):
    cfg = load_config(config, total_images=total, output=str(output) if output else None)
    if even:
        cfg.per_class = {}
    cfg.output.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, cfg.output / "config.resolved.yaml")
    return cfg


def _ground_truth(cfg):
    if cfg.ground_truth_manifest.exists():
        return dataset.load(cfg.ground_truth_manifest)
    gt = dataset.scan(cfg.ground_truth)
    dataset.save(gt, cfg.ground_truth_manifest)
    return gt


def _counts_table(title, rows):
    t = Table(title=title)
    t.add_column("class")
    t.add_column("images", justify="right")
    for k, v in rows.items():
        t.add_row(k, str(v))
    return t


@app.command(help="Scan the ground-truth folder and write its manifest.")
@config_option
@output_option
def ingest(config, output):
    cfg = _load(config, output=output)
    gt = dataset.scan(cfg.ground_truth)
    dataset.save(gt, cfg.ground_truth_manifest)
    rprint(_counts_table("ground truth", gt.counts()))


@app.command(help="Caption the real images with a local vision model.")
@config_option
@output_option
def describe(config, output):
    from . import describe as stage

    cfg = _load(config, output=output)
    n = stage.run(cfg, _ground_truth(cfg))
    rprint(f"captioned {n} new images -> {cfg.captions_path}")


@app.command(help="Write new plausible descriptions per class from samples of the real captions.")
@config_option
@output_option
def synthesize(config, output):
    from . import synthesize as stage

    cfg = _load(config, output=output)
    n = stage.run(cfg, _ground_truth(cfg))
    rprint(f"wrote {n} new descriptions -> {cfg.synthetic_captions_path}")


@app.command(help="Split the image budget across classes and write one job per image.")
@config_option
@total_option
@output_option
@even_option
def plan(config, total, output, even):
    from . import planner

    cfg = _load(config, total, output, even)
    gt = _ground_truth(cfg)
    jobs = planner.run(cfg, gt.classes)
    counts = {c: 0 for c in gt.classes}
    for j in jobs:
        counts[j.class_name] += 1
    rprint(_counts_table(f"planned ({len(jobs)} jobs)", counts))


@app.command(help="Run pending jobs through the configured backend.")
@config_option
@output_option
def generate(config, output):
    from . import generate as stage

    cfg = _load(config, output=output)
    n = stage.run(cfg)
    rprint(f"generated {n} images -> {cfg.images_dir}")


@app.command("filter", help="Drop off-class images with CLIP and export the rest as class folders.")
@config_option
@output_option
def filter_cmd(config, output):
    from . import filter as stage

    cfg = _load(config, output=output)
    summary = stage.run(cfg, _ground_truth(cfg))
    t = Table(title="filter")
    t.add_column("class")
    t.add_column("accepted", justify="right")
    t.add_column("rejected", justify="right")
    for c, s in summary.items():
        t.add_row(c, str(s["accepted"]), str(s["rejected"]))
    rprint(t)
    rprint(f"dataset -> {cfg.dataset_dir}")


@app.command(help="Compare the synthetic set to the real pool per class in CLIP space, against a real-vs-real baseline.")
@config_option
@output_option
def fidelity(config, output):
    from . import fidelity as stage

    cfg = _load(config, output=output)
    report = stage.run(cfg, _ground_truth(cfg))
    t = Table(title="fidelity (lower fd/kid = closer; nn far above baseline = copying; div far below = collapse)")
    for col in ("class", "set", "n", "fd", "kid", "nn", "div"):
        t.add_column(col, justify="right" if col not in ("class", "set") else "left")
    for class_name, pair in report.items():
        for name, m in pair.items():
            t.add_row(class_name, name, str(m["n"]), f"{m['fd']:.1f}", f"{m['kid']:.4f}", f"{m['nn']:.3f}", f"{m['div']:.3f}")
    rprint(t)
    rprint(f"written to {cfg.output / 'fidelity.json'}")


@app.command(help="Every stage in order: ingest, describe, synthesize if enabled, plan, generate, filter.")
@config_option
@total_option
@output_option
@skip_filter_option
@even_option
@replan_option
def run(config, total, output, skip_filter, even, replan):
    from . import describe as describe_stage
    from . import filter as filter_stage
    from . import generate as generate_stage
    from . import planner

    cfg = _load(config, total, output, even)
    if replan and cfg.jobs_path.exists():
        cfg.jobs_path.unlink()
    gt = dataset.scan(cfg.ground_truth)
    dataset.save(gt, cfg.ground_truth_manifest)
    rprint(_counts_table("ground truth", gt.counts()))

    n = describe_stage.run(cfg, gt)
    rprint(f"captioned {n} new images")

    if cfg.synth.enabled or cfg.prompt.source != "real":
        from . import synthesize as synth_stage

        n = synth_stage.run(cfg, gt)
        rprint(f"synthesised {n} new descriptions")

    if not cfg.jobs_path.exists():
        jobs = planner.run(cfg, gt.classes)
        rprint(f"planned {len(jobs)} jobs")
    else:
        rprint(f"reusing existing job list {cfg.jobs_path}")

    n = generate_stage.run(cfg)
    rprint(f"generated {n} images")

    if skip_filter:
        cfg.filter.enabled = False
    summary = filter_stage.run(cfg, gt)
    kept = sum(s["accepted"] for s in summary.values())
    rprint(f"exported {kept} images -> {cfg.dataset_dir}")


@app.group(help="Run on another machine over SSH.")
def remote():
    pass


@remote.command("setup", help="Copy this repo to remote.host and install it there.")
@config_option
@click.option("--extras", default="gen", help="pip extras to install on the remote, e.g. gen or gen,quant")
def remote_setup(config, extras):
    from . import remote as stage

    cfg = load_config(config)
    stage.setup(cfg, Path(__file__).resolve().parent.parent, extras)


@remote.command("run", help="Run every stage on remote.host and pull the output folder back here.")
@config_option
@total_option
@output_option
@click.option("--name", default=None, help="Run name on the remote. Defaults to the output folder name.")
@click.option("--no-sync", "no_sync", is_flag=True, help="Skip re-uploading the ground-truth folder.")
@skip_filter_option
@even_option
@replan_option
def remote_run(config, total, output, name, no_sync, skip_filter, even, replan):
    from . import remote as stage

    cfg = _load(config, total, output, even)
    out = stage.run(cfg, name or cfg.output.name, sync_ground_truth=not no_sync, skip_filter=skip_filter, replan=replan)
    rprint(f"results in {out}")


if __name__ == "__main__":
    app()
