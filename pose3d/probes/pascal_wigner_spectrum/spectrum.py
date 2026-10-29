"""Wigner-D degree analysis for Pascal3D+: what each degree l can express about a pose there.

Pascal3D+ labels one pose per image (no list of equivalent poses), so the SYMSOL analysis
(probes/symsol_wigner_spectrum) has two Pascal counterparts that need only the labels:

1. Single pose, truncated at degree L. The Wigner-D regression target of a pose R0 (models/wignerd.py,
   blocks sqrt(2l+1) D^l) scores a rotation R by f_L(R) = sum_{l <= L} (2l+1) chi_l(w), with w the angle between R
   and R0 and chi_l(w) = sin((2l+1) w / 2) / sin(w / 2) the character of degree l. The peak depends on w only, so
   the half width at half maximum of f_L (degrees) is the angular resolution degrees <= L allow.
2. Class pose prior. For each class, c^l = mean over its training poses of sqrt(2l+1) D^l(R): the expected Wigner-D
   target of an image of that class whose pose is unknown. Reported: its magnitude |c^l|_F / (2l+1) (1 = every
   training pose identical, 0 = degree l carries no class-level pose information), and the error of the truncated
   prior's argmax over an SO(3) HEALPix grid against each validation pose of the class (the image-free baseline).

Run from pose3d/:  python -m probes.pascal_wigner_spectrum.spectrum --labels <pascal_labels.npz>
where the .npz holds train_rots, train_cls, val_rots, val_cls (the RAM cache targets and the annotation classes).
"""

import argparse
import json
import os

import numpy as np
import torch

CLASSES = ("aeroplane", "bicycle", "boat", "bottle", "bus", "car", "chair", "diningtable", "motorbike", "sofa",
           "train", "tvmonitor")


def angles(rots: torch.Tensor):
    """Euler angles exactly as WignerD.target computes them (nearest rotation first)."""
    from e3nn import o3

    u, _, vh = torch.linalg.svd(rots.double())
    sign = torch.det(u @ vh)
    u = torch.cat([u[..., :2], u[..., 2:] * sign[..., None, None]], dim=-1)
    return o3.matrix_to_angles(u @ vh)


def wigner_block(l, a, b, g):
    from image2sphere import so3_utils

    return (2 * l + 1) ** 0.5 * so3_utils.wigner_D(l, a, b, g)


def geodesic_deg(r1, r2):
    tr = torch.einsum("aij,bij->ab", r1, r2)
    return torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))


def single_pose_profile(lmax: int):
    """f_L(w) / f_L(0) on a fine grid of w, and its half width at half maximum per L (degrees)."""
    w = np.radians(np.linspace(1e-4, 180, 36001))
    profiles, hwhm = {}, {}
    for L in range(1, lmax + 1):
        f = sum((2 * l + 1) * np.sin((2 * l + 1) * w / 2) / np.sin(w / 2) for l in range(L + 1))
        f = f / f[0]
        profiles[L] = f
        hwhm[L] = float(np.degrees(w[np.argmax(f < 0.5)]))
    return np.degrees(w), profiles, hwhm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--lmax", type=int, default=10)
    p.add_argument("--grid_level", type=int, default=3)
    p.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = p.parse_args()

    from e3nn import o3
    from image2sphere import so3_utils

    torch.set_default_dtype(torch.float64)
    d = np.load(args.labels)
    tr_r, tr_c = torch.from_numpy(d["train_rots"]).double(), d["train_cls"]
    va_r, va_c = torch.from_numpy(d["val_rots"]).double(), d["val_cls"]
    ga, gb, gg = (t.double() for t in so3_utils.so3_healpix_grid(rec_level=args.grid_level))
    grid = o3.angles_to_matrix(ga, gb, gg)

    w, profiles, hwhm = single_pose_profile(max(args.lmax, 20))
    print("single pose, half width at half maximum (deg):", {L: round(h, 2) for L, h in hwhm.items()})

    results = {}
    ta, tb, tg = angles(tr_r)
    for c, name in enumerate(CLASSES):
        m = tr_c == c
        score = torch.zeros(grid.shape[0])
        mag, err_median, acc30 = [], [], []
        val = va_r[va_c == c]
        for l in range(args.lmax + 1):
            cl = wigner_block(l, ta[m], tb[m], tg[m]).mean(0)                          # (2l+1, 2l+1)
            mag.append(float(torch.linalg.matrix_norm(cl) / (2 * l + 1)))
            for i in range(0, grid.shape[0], 8192):
                score[i:i + 8192] += torch.einsum("ij,qij->q", cl, wigner_block(l, ga[i:i + 8192], gb[i:i + 8192], gg[i:i + 8192]))
            best = grid[score.argmax()][None]
            e = geodesic_deg(best, val)[0].numpy() if len(val) else np.array([np.nan])
            err_median.append(float(np.median(e)))
            acc30.append(float((e < 30).mean()))
        results[name] = dict(train=int(m.sum()), val=int(len(val)), magnitude=mag, prior_argmax_median_error=err_median,
                             prior_argmax_acc30=acc30)
        print(f"{name:12s} train {m.sum():5d} val {len(val):4d}  |c_l|/(2l+1) " + " ".join(f"{x:.2f}" for x in mag)
              + f"  prior-argmax median err L=1: {err_median[1]:.1f}, L={args.lmax}: {err_median[-1]:.1f} deg", flush=True)

    with open(os.path.join(args.out, "spectrum.json"), "w") as f:
        json.dump(dict(lmax=args.lmax, grid_level=args.grid_level, single_pose_hwhm_deg=hwhm, classes=results), f)
    plot(w, profiles, hwhm, results, args)


def plot(w, profiles, hwhm, results, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    for L in (1, 2, 3, 4, 6, 8, 10, 15, 20):
        a1.plot(w, profiles[L], label=f"L = {L}")
    a1.axhline(0.5, color="grey", lw=0.8, ls="--")
    a1.set(xlim=(0, 120), xlabel="angle to the labelled pose (deg)", ylabel="f_L / f_L(0)",
           title="One labelled pose, Wigner-D target truncated at L")
    a1.legend(fontsize=8, ncol=2)
    Ls = sorted(hwhm)
    a2.plot(Ls, [hwhm[L] for L in Ls], marker="o")
    for y, txt in [(15, "15 deg"), (9, "~9 deg (best Pascal3D+ median)"), (5, "5 deg")]:
        a2.axhline(y, color="grey", lw=0.8, ls="--")
        a2.text(Ls[-1], y + 0.6, txt, ha="right", fontsize=8, color="grey")
    a2.set(xlabel="truncation degree L", ylabel="half width at half maximum (deg)", yscale="log",
           title="Angular resolution of degrees <= L", xticks=Ls[::2])
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "single_pose_resolution.png"), dpi=130)

    ls = np.arange(args.lmax + 1)
    fig, axes = plt.subplots(3, 4, figsize=(15, 8.5), sharex=True, sharey=True)
    for ax, (name, r) in zip(axes.flat, results.items()):
        ax.bar(ls, r["magnitude"], color="#4c72b0")
        ax.set_title(f"{name} ({r['train']} train)", fontsize=10)
        ax.set_xticks(ls)
    for ax in axes[-1]:
        ax.set_xlabel("degree l")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\|c^l\|_F\,/\,(2l+1)$")
    fig.suptitle("Class pose prior per Wigner degree (1 = all training poses identical, 0 = no class-level information)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "class_prior_magnitude.png"), dpi=130)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2))
    for name, r in results.items():
        a1.plot(ls, r["prior_argmax_median_error"], marker="o", ms=3, label=name)
        a2.plot(ls, r["prior_argmax_acc30"], marker="o", ms=3, label=name)
    a1.set(xlabel="truncation degree L", ylabel="median error on val (deg)",
           title="Image-free baseline: argmax of the class prior truncated at L", xticks=ls)
    a2.set(xlabel="truncation degree L", ylabel="acc@30 on val", title="Same, acc@30", xticks=ls)
    a2.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "class_prior_truncation.png"), dpi=130)


if __name__ == "__main__":
    main()
