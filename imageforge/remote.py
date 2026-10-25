"""Run a config on another machine over SSH.

Code and the ground-truth folder go over as tar streams, the run happens there with the same
CLI, and the output folder comes back. Only `ssh` and `tar` are needed on both ends.
"""

import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import yaml


REPO_EXCLUDES = [".git", ".venv", ".forge", "runs", "data", "__pycache__", "*.egg-info", ".pytest_cache"]


def _ssh_base(cfg):
    return ["ssh", "-p", str(cfg.remote.port), "-o", "StrictHostKeyChecking=accept-new", cfg.remote.host]


def _ssh(cfg, command, check=True):
    return subprocess.run(_ssh_base(cfg) + [command], check=check).returncode


def _scp(cfg, local, remote):
    subprocess.run(
        ["scp", "-P", str(cfg.remote.port), "-o", "StrictHostKeyChecking=accept-new", "-q", str(local), f"{cfg.remote.host}:{remote}"],
        check=True,
    )


def push_dir(cfg, local, remote, excludes=None):
    args = ["tar", "-C", str(local)]
    for e in excludes or []:
        args.append(f"--exclude={e}")
    args += ["-cf", "-", "."]
    producer = subprocess.Popen(args, stdout=subprocess.PIPE)
    try:
        subprocess.run(
            _ssh_base(cfg) + [f"mkdir -p {shlex.quote(remote)} && tar -xf - -C {shlex.quote(remote)}"],
            stdin=producer.stdout,
            check=True,
        )
    finally:
        producer.stdout.close()
        producer.wait()
    if producer.returncode != 0:
        raise RuntimeError(f"tar failed with code {producer.returncode}")


def pull_dir(cfg, remote, local):
    """Copy a remote folder here. Safe to call while the remote run is still writing: GNU tar
    returns 1 when a file changed as it was read, and that is a warning for a partial copy."""
    local.mkdir(parents=True, exist_ok=True)
    producer = subprocess.Popen(_ssh_base(cfg) + [f"tar -C {shlex.quote(remote)} -cf - ."], stdout=subprocess.PIPE)
    try:
        subprocess.run(["tar", "-xf", "-", "-C", str(local)], stdin=producer.stdout, check=True)
    finally:
        producer.stdout.close()
        producer.wait()
    if producer.returncode not in (0, 1):
        raise RuntimeError(f"remote tar failed with code {producer.returncode}")


def _require_host(cfg):
    if not cfg.remote.host or "PASTE" in cfg.remote.host:
        raise ValueError("remote.host is not set in the config")
    return cfg.remote.host


def local_hf_token():
    """The Hugging Face token this machine uses, wherever its cache lives."""
    candidates = []
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    if os.environ.get("HF_HOME"):
        candidates.append(Path(os.environ["HF_HOME"]) / "token")
    candidates.append(Path.home() / ".cache" / "huggingface" / "token")
    for p in candidates:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    return None


def setup(cfg, repo_root, extras="gen"):
    """Copy this repo to the remote and install it in a venv there."""
    _require_host(cfg)
    work = cfg.remote.workdir
    src = f"{work}/src"
    print(f"copying code to {cfg.remote.host}:{src}")
    push_dir(cfg, repo_root, src, REPO_EXCLUDES)
    py = cfg.remote.python
    install = (
        f"cd {work} && "
        f"([ -d .venv ] || {py} -m venv --system-site-packages .venv) && "
        f".venv/bin/python -m pip install --upgrade pip && "
        f".venv/bin/python -m pip install -e 'src[{extras}]'"
    )
    print("installing on remote")
    _ssh(cfg, install)
    if cfg.remote.copy_hf_token:
        token = local_hf_token()
        if token:
            print("copying Hugging Face token")
            _ssh(cfg, f"mkdir -p ~/.cache/huggingface && printf %s {shlex.quote(token)} > ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token")
        else:
            print("no local Hugging Face token found; gated models will not download on the remote")


def run(cfg, run_name, sync_ground_truth=True, skip_filter=False, replan=False):
    """Push ground truth, run every stage remotely, pull the output folder back."""
    _require_host(cfg)
    work = cfg.remote.workdir
    remote_run = f"{work}/runs/{run_name}"
    remote_gt = f"{remote_run}/ground_truth"
    remote_out = f"{remote_run}/output"

    if sync_ground_truth:
        print(f"copying ground truth to {cfg.remote.host}:{remote_gt}")
        push_dir(cfg, cfg.ground_truth, remote_gt)

    # Written as plain POSIX strings: a Path built on Windows would serialise with backslashes.
    data = cfg.model_dump(mode="json")
    data["ground_truth"] = remote_gt
    data["output"] = remote_out
    data["remote"]["host"] = None
    with tempfile.TemporaryDirectory() as tmp:
        local_cfg = Path(tmp) / "config.yaml"
        with open(local_cfg, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        _ssh(cfg, f"mkdir -p {shlex.quote(remote_run)}")
        _scp(cfg, local_cfg, f"{remote_run}/config.yaml")

    flags = (" --skip-filter" if skip_filter else "") + (" --replan" if replan else "")
    log = f"{remote_run}/run.log"
    status = f"{remote_run}/run.status"
    # Detached from this SSH session, so a dropped connection does not kill the run. The exit
    # code lands in a status file; progress is polled with short connections.
    # setsid plus closed stdin: otherwise the backgrounded run keeps the SSH session's file
    # descriptors open and the launch command never returns.
    launch = (
        f"cd {work} && rm -f {status} && "
        f"setsid nohup sh -c '.venv/bin/imageforge run --config {remote_run}/config.yaml{flags} "
        f"> {log} 2>&1; echo $? > {status}' < /dev/null > /dev/null 2>&1 &"
    )
    print(f"running on {cfg.remote.host} (log: {log})")
    _ssh(cfg, launch)
    code = wait_for_status(cfg, status, log)
    if code != 0:
        raise RuntimeError(f"remote run failed with exit code {code}; see {cfg.remote.host}:{log}")

    print(f"pulling results to {cfg.output}")
    pull_dir(cfg, remote_out, cfg.output)
    return cfg.output


def wait_for_status(cfg, status, log, poll_seconds=60):
    """Poll the remote until the status file appears, printing the log's last line meanwhile."""
    import time

    last = ""
    while True:
        out = subprocess.run(
            _ssh_base(cfg) + [f"cat {status} 2>/dev/null; echo ---; tail -c 300 {log} 2>/dev/null | tr '\\r' '\\n' | grep -v '^$' | tail -1"],
            capture_output=True,
            text=True,
        ).stdout
        code_part, _, tail_part = out.partition("---")
        line = tail_part.strip()
        if line and line != last:
            print(line[:120])
            last = line
        if code_part.strip():
            return int(code_part.strip())
        time.sleep(poll_seconds)
