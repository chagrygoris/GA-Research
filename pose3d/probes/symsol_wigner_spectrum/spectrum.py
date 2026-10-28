"""Wigner-D spectrum of the SYMSOL I targets: how much of each shape's pose distribution lives in each degree l.

The Wigner-D regression head (`models/wignerd.py`) regresses `flat_wigner(R)`, the Wigner-D matrices D^l(R) of the
ground-truth rotation for l = 0..lmax (each block scaled by sqrt(2l+1)). With a symmetric shape the ground truth is
any of its |G| equivalent poses R* g, so the minimiser of that squared loss is their average

    c^l = (1 / |G|) sum_g sqrt(2l+1) D^l(R* g)        (a (2l+1) x (2l+1) block per degree l),

the expected target. This script computes it per test view from SYMSOL's own equivalent poses and reports

1. magnitude: |c^l|_F / (2l+1), the norm of the expected target in degree l as a fraction of a single pose's norm
   (2l+1). It is 1 for a shape without symmetry and sqrt(m_l / (2l+1)) for a finite group G, where m_l is the number
   of G-invariant directions in the degree-l irrep: a degree with magnitude 0 carries no information about the pose.
2. oracle decoding: with the exact target truncated at degree L, f_L(R) = sum_{l <= L} <c^l, sqrt(2l+1) D^l(R)>,
   the score the Wigner-D head's grid argmax decodes. Per L: the error from the argmax over an SO(3) HEALPix grid to
   the nearest equivalent pose, and the margin
       (min_g f_L(R* g) - max_{R: d(R, all R* g) > 15 deg} f_L(R)) / (max f_L - mean f_L),
   positive when every equivalent pose scores above every grid rotation more than 15 deg from all of them, i.e.
   degrees <= L are enough to single out the equivalent set at 15 deg resolution.

Run from pose3d/:  python -m probes.symsol_wigner_spectrum.spectrum --rotations <symsol>/test/rotations.npz
"""

import argparse
import json
import os

import numpy as np
import torch

SHAPES = ["tet", "cube", "icosa", "cone", "cyl"]


def angles(rots: torch.Tensor):
    """Euler angles of rotation matrices exactly as WignerD.target computes them (nearest rotation first)."""
    from e3nn import o3

    u, _, vh = torch.linalg.svd(rots.double())
    sign = torch.det(u @ vh)
    u = torch.cat([u[..., :2], u[..., 2:] * sign[..., None, None]], dim=-1)
    return o3.matrix_to_angles(u @ vh)


def wigner_block(l: int, a, b, g) -> torch.Tensor:
    """sqrt(2l+1) D^l, the degree-l block of image2sphere's flat_wigner, (..., 2l+1, 2l+1)."""
    from image2sphere import so3_utils

    return (2 * l + 1) ** 0.5 * so3_utils.wigner_D(l, a, b, g)


def geodesic_deg(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """Rotation angle between every r1[i] and r2[j], degrees, (len(r1), len(r2))."""
    tr = torch.einsum("aij,bij->ab", r1, r2)
    return torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rotations", required=True, help="SYMSOL test/rotations.npz (equivalent poses per view)")
    p.add_argument("--views", type=int, default=30, help="test views per shape")
    p.add_argument("--lmax", type=int, default=10)
    p.add_argument("--grid_level", type=int, default=3, help="SO(3) HEALPix level (3: 36,864 rotations, 7.5 deg)")
    p.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = p.parse_args()

    from e3nn import o3
    from image2sphere import so3_utils

    torch.set_default_dtype(torch.float64)
    data = np.load(args.rotations)
    ga, gb, gg = (t.double() for t in so3_utils.so3_healpix_grid(rec_level=args.grid_level))
    grid = o3.angles_to_matrix(ga, gb, gg)
    results = {}
    for shape in SHAPES:
        eq = torch.from_numpy(data[shape][: args.views]).double()           # (V, |G|, 3, 3)
        V, G = eq.shape[:2]
        ea, eb, eg = angles(eq.reshape(-1, 3, 3))
        far = torch.stack([geodesic_deg(grid, eq[v]).min(1).values for v in range(V)]) > 15.0   # (V, Q)
        cum_grid = torch.zeros(V, grid.shape[0])
        cum_eq = torch.zeros(V, G)
        mag, err, margin = [], [], []
        for l in range(args.lmax + 1):
            d_eq = wigner_block(l, ea, eb, eg).reshape(V, G, 2 * l + 1, 2 * l + 1)
            c = d_eq.mean(1)                                                      # expected target, (V, 2l+1, 2l+1)
            mag.append((torch.linalg.matrix_norm(c) / (2 * l + 1)).tolist())
            cum_eq += torch.einsum("vij,vgij->vg", c, d_eq)
            for i in range(0, grid.shape[0], 8192):
                d_grid = wigner_block(l, ga[i:i + 8192], gb[i:i + 8192], gg[i:i + 8192])
                cum_grid[:, i:i + 8192] += torch.einsum("vij,qij->vq", c, d_grid)
            best = grid[cum_grid.argmax(1)]                                       # (V, 3, 3)
            err.append([geodesic_deg(best[v:v + 1], eq[v]).min().item() for v in range(V)])
            span = (cum_grid.max(1).values - cum_grid.mean(1)).clamp_min(1e-12)
            far_max = torch.where(far, cum_grid, torch.tensor(-float("inf"))).max(1).values
            margin.append(((cum_eq.min(1).values - far_max) / span).tolist())
            print(f"{shape:5s} l={l:2d}  |c_l|/(2l+1)={np.mean(mag[-1]):.3f}  argmax err (L<={l}) median "
                  f"{np.median(err[-1]):6.2f} deg  margin median {np.median(margin[-1]):+.3f}", flush=True)
        results[shape] = {"n_equivalents": G, "magnitude": mag, "argmax_error_deg": err, "margin": margin}
    with open(os.path.join(args.out, "spectrum.json"), "w") as f:
        json.dump({"views": args.views, "lmax": args.lmax, "grid_level": args.grid_level, "shapes": results}, f)
    plot(results, args)


def plot(results, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {"tet": "tetrahedron (12)", "cube": "cube (24)", "icosa": "icosahedron (60)", "cone": "cone (360)",
             "cyl": "cylinder (720)"}
    ls = np.arange(args.lmax + 1)
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.2), sharey=True)
    for ax, s in zip(axes, SHAPES):
        m = np.array(results[s]["magnitude"])                                     # (L+1, V)
        ax.bar(ls, m.mean(1), color="#4c72b0")
        ax.set_title(names[s])
        ax.set_xlabel("degree l")
        ax.set_xticks(ls)
    axes[0].set_ylabel(r"$\|c^l\|_F\,/\,(2l+1)$")
    fig.suptitle("Expected Wigner-D target per degree (1 = a single pose, 0 = no information)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "magnitude_per_degree.png"), dpi=130)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    for s in SHAPES:
        e = np.array(results[s]["argmax_error_deg"])
        mg = np.array(results[s]["margin"])
        a1.plot(ls, np.median(e, 1), marker="o", label=names[s])
        a2.plot(ls, np.median(mg, 1), marker="o", label=names[s])
    a1.axhline(15, color="grey", lw=0.8, ls="--")
    a1.set(xlabel="truncation degree L", ylabel="median error of the argmax (deg)",
           title="Oracle decoding of the exact target truncated at L", xticks=ls)
    a2.axhline(0, color="grey", lw=0.8, ls="--")
    a2.set(xlabel="truncation degree L", ylabel="margin (> 0: equivalents above every far rotation)",
           title="Are degrees <= L enough to single out the equivalent poses?", xticks=ls)
    a1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "truncation.png"), dpi=130)


if __name__ == "__main__":
    main()
