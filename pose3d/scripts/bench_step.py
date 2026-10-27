"""Time one model's training step and evaluation on synthetic images, without any dataset.

Builds the model exactly as `python -m pose3d` does from the same flags, then times the trainer's own
functions on random 224 x 224 images with random rotations: `train_epoch` (optimizer, EMA, grad clip,
compiled modules and all) over --steps batches after --warmup batches, and `calculate_evaluation_metrics`
on --eval_images images with 1 and with TrainConfig.eval_samples draws. This is the measurement behind the
timing tables of USAGE.md; with the number of steps per epoch it gives a run's ETA before launching it.
The first (warm-up) batches include torch.compile's compilation, which is reported separately.

    python scripts/bench_step.py [--steps 50] [--warmup 5] [--eval_images 360] -- <pose3d flags>

e.g. `python scripts/bench_step.py -- --vector_field versor --no-ddp`. Prints one JSON line per measurement.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from anywhere: pose3d/ holds the package

from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn  # noqa: E402

from pose3d.config import parse_args  # noqa: E402
from pose3d.engine import distributed  # noqa: E402
from pose3d.engine.losses import build_criterion  # noqa: E402
from pose3d.engine.metrics import calculate_evaluation_metrics  # noqa: E402
from pose3d.engine.trainer import build_step, train_epoch  # noqa: E402
from pose3d.models import build_model  # noqa: E402
from pose3d.train import build_optimizer, make_algebra  # noqa: E402


def random_batches(n, batch_size, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        q = torch.linalg.qr(torch.randn(batch_size, 3, 3, generator=g))[0]
        rot = q * torch.linalg.det(q).sign().view(-1, 1, 1)
        out.append({"img": torch.rand(batch_size, 3, 224, 224, generator=g), "rot": rot,
                    "cls_eval": torch.randint(0, 12, (batch_size, 1), generator=g)})
    return out


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def report(**kv):
    print(json.dumps(kv), flush=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", type=int, default=50, help="timed training steps")
    parser.add_argument("--warmup", type=int, default=5, help="untimed steps first (compilation happens here)")
    parser.add_argument("--eval_images", type=int, default=360)
    args = parser.parse_args(argv[:split])
    cfg = parse_args(["--path_to_datasets", "/nonexistent", "--no-ddp", *argv[split + 1:]])

    torch.backends.cudnn.benchmark = True
    distributed.setup(cfg)
    distributed.seed_everything(cfg)
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, make_algebra(cfg.model.algebra_dim)).to(cfg.device)
    optimizer = build_optimizer(model, cfg)
    criterion = build_criterion(cfg)
    step = build_step(model, criterion, cfg)
    ema = (AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg.train.ema_decay), use_buffers=True)
           if cfg.features.ema else None)
    n_params = sum(p.numel() for p in model.parameters())
    field = getattr(model, "vector_field", None)
    report(what="model", device=str(cfg.device),
           gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
           vector_field=cfg.flow.vector_field, params=n_params,
           vector_field_params=sum(p.numel() for p in field.parameters()) if field is not None else None)

    bs = cfg.train.batch_size
    t0 = time.time()
    loss = train_epoch(model, random_batches(args.warmup, bs, seed=1), optimizer, criterion, cfg, step=step, ema=ema)
    sync()
    report(what="warmup", steps=args.warmup, seconds=round(time.time() - t0, 1), loss=loss)

    batches = random_batches(args.steps, bs, seed=2)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    sync()
    t0 = time.time()
    loss = train_epoch(model, batches, optimizer, criterion, cfg, step=step, ema=ema)
    sync()
    dt = time.time() - t0
    report(what="train_step", batch=bs, steps=args.steps, ms_per_step=round(1000 * dt / args.steps, 1),
           loss=loss, finite=bool(torch.isfinite(torch.tensor(loss))),
           peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None)

    eval_model = ema.module if ema is not None else model
    n_eval = max(1, args.eval_images // bs)
    eval_batches = random_batches(n_eval, bs, seed=3)
    for n_samples in sorted({1, cfg.eval_samples}):
        for attempt in ("first", "second"):   # the first pass includes compiling the eval-mode graphs
            sync()
            t0 = time.time()
            calculate_evaluation_metrics(eval_model, eval_batches, cfg, n_samples=n_samples)
            sync()
            report(what="eval", pass_=attempt, images=n_eval * bs, samples=n_samples,
                   seconds=round(time.time() - t0, 2))


if __name__ == "__main__":
    main()
