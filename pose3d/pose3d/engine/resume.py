"""Resumable training: a checkpoint after every epoch, `--resume` to continue, a clean stop on SIGTERM.

The final checkpoint of train.py (form_checkpoint) is written only after the last epoch, and
--path_to_checkpoint evaluates weights before training from epoch 0, so a run that is stopped (a
watchdog, a lost session) used to lose everything. Here:

  * After every epoch, `save_resume` writes `<resume_dir>/<run_name>_resume.pth`: model, optimizer,
    scheduler, the EMA copy (--ema), the RNG states and `next_epoch`, the first epoch still to run.
    The write is atomic (a temporary file, then os.replace), so a kill during the write leaves the
    previous epoch's checkpoint intact.
  * `--resume PATH` loads such a checkpoint and continues the epoch loop at `next_epoch`. The LR
    schedule continues where it was (the scheduler's own state). A checkpoint with
    next_epoch == n_epochs skips training and goes straight to the final evaluation.
  * SIGTERM (`install_sigterm_exit`) ends the process at once with exit code RESUMABLE_EXIT (75,
    EX_TEMPFAIL), keeping the last epoch's checkpoint: the work of the interrupted epoch is lost, the
    rest is not. The runner notebooks' watchdog sends SIGTERM first and treats 75 as "stopped, can
    resume", not as a failure.

The resumed run is a new W&B run (offline runs cannot be continued); its epoch metrics carry on from
`next_epoch`. A checkpoint from a different configuration is refused unless the differences are only
in the run-control options of RESUME_FREE.
"""

import os
import random
import signal
import sys
from pathlib import Path

import numpy as np
import torch

RESUMABLE_EXIT = 75
# Options that may differ between the stopped run and its continuation (where to read and write, how
# long to wait, what to call it); every other option must match, or the checkpoint is refused.
RESUME_FREE = {("run", "resume"), ("run", "resume_dir"), ("run", "path_to_checkpoint"), ("run", "run_name"),
               ("run", "path_to_datasets"), ("data", "num_workers")}


def resume_path(cfg) -> Path:
    return Path(cfg.run.resume_dir) / f"{cfg.run.run_name or 'run'}_resume.pth"


def _rng_state():
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state.get("cuda") is not None and torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])


def _unwrap(model):
    return model.module if hasattr(model, "module") and isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def save_resume(cfg, model, optimizer, scheduler, ema, next_epoch: int) -> Path:
    """Atomically write the resume checkpoint for a run whose epochs < next_epoch are done."""
    path = resume_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "next_epoch": next_epoch,
        "n_epochs": cfg.train.n_epochs,
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "rng": _rng_state(),
        "config": cfg.to_dict(),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    return path


def _config_mismatch(saved: dict, current: dict):
    out = []
    for section, values in current.items():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if (section, key) in RESUME_FREE:
                continue
            old = saved.get(section, {}).get(key, value)
            if str(old) != str(value):
                out.append(f"{section}.{key}: checkpoint {old!r}, now {value!r}")
    return out


def load_resume(cfg, model, optimizer, scheduler, ema) -> int:
    """Restore a resume checkpoint into the freshly built objects; returns the first epoch to run."""
    state = torch.load(cfg.run.resume, map_location=cfg.device, weights_only=False)
    mismatch = _config_mismatch(state.get("config", {}), cfg.to_dict())
    if mismatch:
        raise ValueError("--resume: the checkpoint was trained with other options:\n  " + "\n  ".join(mismatch))
    _unwrap(model).load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if ema is not None:
        if state.get("ema") is None:
            raise ValueError("--resume: the run uses --ema but the checkpoint has no EMA copy")
        ema.load_state_dict(state["ema"])
    _set_rng_state(state["rng"])
    return int(state["next_epoch"])


def install_sigterm_exit():
    """On SIGTERM, exit at once with RESUMABLE_EXIT: the last epoch's resume checkpoint is complete on disk."""
    def _handler(signum, frame):
        print(f"SIGTERM: stopping; resume from the last completed epoch with --resume (exit code {RESUMABLE_EXIT})",
              flush=True)
        sys.exit(RESUMABLE_EXIT)
    signal.signal(signal.SIGTERM, _handler)
