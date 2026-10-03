"""Visualize the flow-matching trajectories of a trained CliffordFlow on SO(3): the path from noise to a pose.

For a few validation images the flow is integrated from several random rotations (the Euler loop of CliffordFlow.predict, but every
intermediate rotation is kept). Each rotation R_t is shown relative to the ground truth, R_gt^T R_t, as an axis-angle vector inside the
ball of radius pi: the true pose is the origin, the distance from the origin is the geodesic error, and a path that ends near the origin
has found the pose. Opposite points of the ball's surface are the same rotation, so a path can jump across it (drawn as a break).

    python -m pose3d.visualize_flow --checkpoint a.pth --label GATr --checkpoint b.pth --label MLP \
        --val_cache pascal_val.pt --out viz/
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from pose3d.engine.flow_viz import (angle_deg, animation, errors, figure_compare, figure_filmstrip, figure_paths,  # noqa: F401
                                    pick_images, trajectories)
from pose3d.evaluate import build_model


def load(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model, _ = build_model(ckpt, "cpu")
    return model.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", action="append", required=True); p.add_argument("--label", action="append", required=True)
    p.add_argument("--val_cache", required=True, help="pascal_val.pt of the Pascal3D RAM cache"); p.add_argument("--out", default="viz")
    p.add_argument("--n_pool", type=int, default=96, help="validation images scored to choose the examples")
    p.add_argument("--quantiles", type=float, nargs="+", default=[0.1, 0.5, 0.8, 0.95], help="error quantiles of the examples (of the first model)")
    p.add_argument("--k", type=int, default=24); p.add_argument("--steps", type=int, default=20); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); assert len(a.checkpoint) == len(a.label)
    torch.set_num_threads(4); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    blob = torch.load(a.val_cache, map_location="cpu", weights_only=True)
    imgs, rots = blob["imgs"].float() / 255, blob["targets"].float().numpy()
    models = {lab: load(ck) for ck, lab in zip(a.checkpoint, a.label)}
    first = next(iter(models.values()))
    picks, errs = pick_images(first, imgs, rots, a.n_pool, a.quantiles)
    print("examples (val index, medoid error of", next(iter(models)), "):", list(zip(picks, [round(e, 1) for e in errs])), flush=True)
    for lab, m in models.items():
        e_m = errors(m, imgs[picks], rots[picks])
        figure_paths(lab, m, imgs, rots, picks, e_m, a.k, a.steps, a.seed, out / f"paths_{lab}.png")
        figure_filmstrip(lab, m, imgs, rots, picks[1], a.steps, a.seed, out / f"filmstrip_{lab}.png")
        animation(lab, m, imgs, rots, picks[1], a.k, a.steps, a.seed, out / f"flow_{lab}.gif")
        print("done", lab, flush=True)
    if len(models) > 1:
        figure_compare(models, imgs, rots, picks, errs, a.k, a.steps, a.seed, out / "compare_convergence.png")


if __name__ == "__main__":
    main()
