"""Device selection and checkpoint save / load."""

from pathlib import Path

import torch


def get_available_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def form_checkpoint(model, optimizer, scheduler, cfg):
    checkpoint = {
        "model": model.state_dict(),
        "scheduler": scheduler.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg.to_dict(),
    }
    path = Path(f"{cfg.run.run_name}.pth").resolve()
    torch.save(checkpoint, path.__str__())
    return path


def load_checkpoint(model, optimizer, scheduler, path, device):
    if path is None:
        return model, optimizer, scheduler

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    try:
        scheduler.load_state_dict(checkpoint["scheduler"])
    except Exception:
        pass
    return model, optimizer, scheduler
