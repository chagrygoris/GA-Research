"""Training / validation loop."""

import inspect
import time

import numpy as np
import torch
from tqdm import tqdm

from pose3d.engine.distributed import (
    all_reduce_sum, is_main, sync_buffers, wrap_ddp,
)
from pose3d.engine.metrics import acc_at, calculate_evaluation_metrics
from pose3d.engine.tracking import log_offline_sync


def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def _supports_class_argument(method) -> bool:
    params = list(inspect.signature(method).parameters.values())
    return any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params) or len(params) >= 2


def _compute_loss(model, data, criterion, cfg):
    img = data["img"].to(cfg.device)
    targets = data["rot"].to(cfg.device)

    clas = None
    if "cls" in data:
        clas = data["cls"].to(cfg.device)

    if hasattr(model, "compute_loss") and callable(getattr(model, "compute_loss")):
        # Important: avoid a duplicate forward pass before model.compute_loss(),
        # which can skew BatchNorm running statistics during training.
        if clas is not None and "cls" in inspect.signature(model.compute_loss).parameters:
            return model.compute_loss(img, targets, criterion, cls=clas)
        return model.compute_loss(img, targets, criterion)

    if clas is not None and _supports_class_argument(model.forward):
        outputs = model(img, clas)
    else:
        outputs = model(img)

    if cfg.train.loss == "prob":
        idx = model.get_nearest_idx(targets).long().view(-1)
        return criterion(outputs, idx)
    return criterion(outputs, targets)


class TrainStep(torch.nn.Module):
    """Model + criterion as one module whose forward is the loss.

    The trainer calls `model.compute_loss`, not `model.forward`, and DDP only synchronises
    gradients for what goes through its own forward, so DDP wraps this instead of the model.
    """

    def __init__(self, model, criterion, cfg):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.cfg = cfg

    def forward(self, data):
        return _compute_loss(self.model, data, self.criterion, self.cfg)


def build_step(model, criterion, cfg):
    return wrap_ddp(TrainStep(model, criterion, cfg), cfg)


def train_epoch(model, loader, optimizer, criterion, cfg, step=None):
    total_loss = 0.0
    n_objects = 0
    device_type = "cuda" if cfg.device.type == "cuda" else "cpu"

    scaler = torch.amp.GradScaler(device_type) if device_type == "cuda" else None

    if step is None:
        step = build_step(model, criterion, cfg)
    step.train()
    for data in tqdm(loader, disable=not is_main() or cfg.run.platform == "kaggle"):
        optimizer.zero_grad(set_to_none=True)

        loss = step(data)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = data["img"].shape[0]
        total_loss += float(loss.detach().item()) * bs
        n_objects += bs

    total_loss, n_objects = all_reduce_sum([total_loss, n_objects], cfg)
    return total_loss / max(n_objects, 1)


@torch.no_grad()
def validate_epoch(model, loader, criterion, cfg):
    total_loss = 0.0
    n_objects = 0

    model.eval()
    for data in tqdm(loader, disable=not is_main() or cfg.run.platform == "kaggle"):
        loss = _compute_loss(model, data, criterion, cfg)

        bs = data["img"].shape[0]
        total_loss += float(loss.detach().item()) * bs
        n_objects += bs

    total_loss, n_objects = all_reduce_sum([total_loss, n_objects], cfg)
    return total_loss / max(n_objects, 1)


def train(model, train_loader, val_loader, optimizer, scheduler, criterion, run, cfg):
    n_epochs = cfg.train.n_epochs
    step = build_step(model, criterion, cfg)
    for i in range(n_epochs):
        t0 = time.time()
        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(i)
        train_loss = train_epoch(model, train_loader, optimizer, criterion, cfg, step=step)
        sync_buffers(model)   # rank 0's BatchNorm statistics, so every shard is scored alike
        val_loss = validate_epoch(model, val_loader, criterion, cfg)
        mre = np.median(calculate_evaluation_metrics(model, val_loader, cfg)).__float__()

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "median_rotation_error": mre,
            "learning_rate": scheduler.get_last_lr()[0],
            "gradient_norm": grad_norm(model),
            "epoch_time": time.time() - t0,
        }
        if run is not None:
            run.log(metrics)
        scheduler.step()
        if is_main():
            print(
                f"Training on {cfg.device} epoch {i + 1} / {n_epochs}. \n"
                f"Train loss {train_loss}, val loss {val_loss}\n"
                f"Median rotation error {mre} ({time.time() - t0:.0f}s)"
            )
            log_offline_sync(run, cfg, step=i, metrics=metrics)

    final_evaluation(model, val_loader, run, cfg)


def final_evaluation(model, val_loader, run, cfg):
    '''Re-score the trained model with multi-sample prediction.

    Per-epoch metrics use a single draw so they stay cheap; a generative model
    deserves a proper mode estimate once, at the end.
    '''
    n_samples = cfg.eval_samples
    if n_samples <= 1:
        return None

    err = calculate_evaluation_metrics(model, val_loader, cfg, n_samples=n_samples)
    metrics = {
        "final_median_rotation_error": float(np.median(err)),
        "final_acc@15": acc_at(err, 15),
        "final_acc@30": acc_at(err, 30),
        "final_eval_samples": n_samples,
    }

    if is_main():
        print(
            f"Final evaluation with {n_samples} samples per image: "
            f"median rotation error {metrics['final_median_rotation_error']}, "
            f"acc@15 {metrics['final_acc@15']}, acc@30 {metrics['final_acc@30']}"
        )
        log_offline_sync(run, cfg, step=cfg.train.n_epochs, metrics=metrics, final=True)
    if run is not None:
        run.summary.update(metrics)
    return metrics
