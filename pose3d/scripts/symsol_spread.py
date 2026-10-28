"""Spread of a flow checkpoint on SYMSOL I: how closely its probability mass sits on the equivalent ground truths.

Spread (Murphy et al., *Implicit-PDF*, ICML 2021, supplementary Table S1) is the expected angular distance from the
predicted distribution to the nearest ground truth,

    spread(x) = E_{R ~ p(R | x)} [ min_g d(R, R* g) ],

with d the rotation angle and R* g the image's equivalent poses (all known on SYMSOL I). IPDF evaluates the
expectation on its grid; a flow is sampled instead: K independent ODE samples per image (no medoid), each scored by its
angle to the nearest equivalent, averaged over the K samples and then over the test views of each shape (degrees).

Run from pose3d/:
    python scripts/symsol_spread.py --checkpoint <run>.pth --path_to_datasets <dir holding symsol/> [--views 1000]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose3d.datasets.symsol import SHAPE_SETS, Symsol                    # noqa: E402
from pose3d.engine.metrics import nearest_equivalent_error, project_to_orthogonal_manifold   # noqa: E402
from pose3d.evaluate import build_model                                   # noqa: E402
from pose3d.geometry.flow import exp_map, rotor_multiply                  # noqa: E402
from pose3d.geometry.rotor import random_rotor, rotor_to_matrix            # noqa: E402


@torch.no_grad()
def flow_samples(model, img, k: int, steps: int) -> torch.Tensor:
    """K raw flow samples per image, (B, K, 3, 3): CliffordFlow.predict without the medoid."""
    b = img.shape[0]
    cond_mv, fisher_a = model._features(img, None)
    if fisher_a is not None:
        raise SystemExit("checkpoints with --fisher_prior are not supported here")
    cond_mv = cond_mv.repeat_interleave(k, dim=0)
    rotor = random_rotor(b * k).to(img.device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((rotor.shape[0],), i * dt, device=img.device)
        rotor = rotor_multiply(rotor, exp_map(dt * model.velocity(rotor, t, cond_mv)), model.algebra)
    return rotor_to_matrix(rotor, model.algebra).view(b, k, 3, 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--path_to_datasets", required=True, help="directory holding symsol/test/{rotations.npz,images/}")
    p.add_argument("--views", type=int, default=None, help="first N test views per shape (default: all)")
    p.add_argument("--shapes", default=None, help="comma-separated subset of tet,cube,icosa,cone,cyl (default: all)")
    p.add_argument("--samples", type=int, default=64, help="flow samples per image (K)")
    p.add_argument("--steps", type=int, default=20, help="Euler steps of the ODE (the trainer's default)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write the per-view spreads and the summary here (.json)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, saved = build_model(ckpt, device)
    model.eval()
    print(f"checkpoint {args.checkpoint}: cond_tokens={saved.get('cond_tokens')}, pose_tokens={saved.get('pose_tokens')}, "
          f"gatr_pose_harmonics={saved.get('gatr_pose_harmonics')}, nearest_coupling={saved.get('symsol_nearest_coupling')}, "
          f"device {device}", flush=True)

    data = Symsol(args.path_to_datasets, train=False, set_number=1, num_views=args.views)
    shapes = SHAPE_SETS[1]
    keep = [shapes.index(s) for s in args.shapes.split(",")] if args.shapes else list(range(len(shapes)))
    index = torch.nonzero(torch.isin(data.cls, torch.tensor(keep))).flatten().tolist()
    loader = DataLoader(Subset(data, index), batch_size=args.batch_size, shuffle=False, num_workers=2)
    spread, first = [], []
    for n, batch in enumerate(loader):
        img = batch["img"].to(device)
        samples = flow_samples(model, img, args.samples, args.steps)                     # (B, K, 3, 3)
        b, k = samples.shape[:2]
        rots_eq = batch["rots_eq"].to(device).repeat_interleave(k, dim=0)
        n_eq = batch["n_eq"].repeat_interleave(k)
        err = nearest_equivalent_error(project_to_orthogonal_manifold(samples.reshape(-1, 3, 3)), rots_eq, n_eq)
        err = err.reshape(b, k)
        spread.append(err.mean(1))
        first.append(err[:, 0])
        if n % 20 == 0:
            print(f"  {min((n + 1) * args.batch_size, len(index))}/{len(index)} views", flush=True)
    spread, first = np.concatenate(spread), np.concatenate(first)
    cls = data.cls[index].numpy()

    summary = {}
    print(f"\nspread (deg), K = {args.samples} samples per view, {args.steps} Euler steps")
    print(f"{'shape':>6} {'views':>6} {'spread':>8} {'median of per-view':>19} {'1-sample median err':>20}")
    for c in keep:
        s, m = shapes[c], cls == c
        summary[s] = dict(views=int(m.sum()), spread=float(spread[m].mean()), spread_median=float(np.median(spread[m])),
                          one_sample_median_error=float(np.median(first[m])))
        print(f"{s:>6} {m.sum():>6} {spread[m].mean():>8.2f} {np.median(spread[m]):>19.2f} {np.median(first[m]):>20.2f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(args=vars(args), summary=summary, spread_per_view=spread.tolist(),
                           cls=cls.tolist()), f)


if __name__ == "__main__":
    main()
