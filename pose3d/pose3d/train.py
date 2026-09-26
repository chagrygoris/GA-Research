"""Training entry point: `python -m pose3d --path_to_datasets=... [flags]`."""

from pathlib import Path

import torch
from clifford.algebra.cliffordalgebra import CliffordAlgebra

from pose3d.config import Config, parse_args
from pose3d.datasets import create_dataloaders
from pose3d.engine.checkpoint import form_checkpoint, get_available_device, load_checkpoint
from pose3d.engine.losses import build_criterion
from pose3d.engine.metrics import calculate_evaluation_metrics
from pose3d.engine.tracking import (
    wandb_create_run,
    wandb_finish_run,
    wandb_log_artifact,
    wandb_log_code,
)
from pose3d.engine.trainer import train
from pose3d.models import build_model


def make_algebra(algebra_dim: int = 3) -> CliffordAlgebra:
    algebra_dim = int(algebra_dim)
    if algebra_dim > 6:
        print(
            f"Warning: algebra_dim={algebra_dim} gives mv_dim={2 ** algebra_dim}. "
            "This can significantly increase memory usage and runtime."
        )
    return CliffordAlgebra(tuple([1] * algebra_dim))


def build_scheduler(optimizer, cfg: Config):
    """Linear warmup for `warmup_epochs`, then cosine decay to 5% of the base lr."""
    warmup_epochs = cfg.train.warmup_epochs
    cosine_epochs = cfg.train.n_epochs - warmup_epochs

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=cfg.train.lr * 0.05,
    )
    if warmup_epochs <= 0:
        return cosine

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def log_model_size(model, run):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    for name in ("condition_head", "vector_field"):
        if hasattr(model, name):
            print(f"  {name}: {sum(p.numel() for p in getattr(model, name).parameters()):,}")

    if run is not None:
        sizes = {"trainable_params": trainable}
        if getattr(model, "adapter", None) is not None:
            sizes["params_excluding_backbone"] = sum(
                p.numel() for name, p in model.named_parameters()
                if not name.startswith("adapter.backbone.")
            )
        run.config.update(sizes, allow_val_change=True)


def instantiate(cfg: Config):
    train_loader, val_loader = create_dataloaders(cfg)
    print("Created Tralaloaders")

    algebra = make_algebra(cfg.model.algebra_dim)
    model = build_model(cfg, algebra)

    cfg.device = get_available_device()
    model.to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = build_criterion(cfg)

    print(cfg)
    run = wandb_create_run(cfg)
    print("W&B logging set up completed")
    log_model_size(model, run)

    return train_loader, val_loader, model, optimizer, scheduler, criterion, run


def main(argv=None):
    cfg = parse_args(argv)

    torch.backends.cudnn.benchmark = True

    train_loader, val_loader, model, optimizer, scheduler, criterion, run = instantiate(cfg)

    path = cfg.run.path_to_checkpoint
    if path is not None:
        try:
            load_checkpoint(model, optimizer, scheduler, path, cfg.device)
            print("Checkpoint successfully loaded. Starting evaluation")
            torch.save(torch.tensor(calculate_evaluation_metrics(model, val_loader, cfg)), "res.pth")
        except Exception as e:
            print(f"Failed to load checkpoint. Starting from scratch. Error: {e}")

    wandb_log_code(run, Path("."))
    torch.cuda.empty_cache()
    train(model, train_loader, val_loader, optimizer, scheduler, criterion, run, cfg)

    if cfg.run.save_checkpoint and not cfg.run.sanity_check:
        checkpoint_path = form_checkpoint(model, optimizer, scheduler, cfg)
        wandb_log_artifact(run, checkpoint_path, artifact_type="checkpoint")
    wandb_finish_run(run)


if __name__ == "__main__":
    main()
