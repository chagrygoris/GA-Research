"""Truncate a trained Image2Sphere (I2SReal) checkpoint's Wigner-D output at degree L and re-decode on Pascal3D+.

I2S predicts psi, the Wigner-D coefficients of the logit of its pose distribution, blocks l = 0..lmax:
log p(R | x) = sum_l <psi_l, sqrt(2l+1) D^l(R)> + const (models/i2s_real.py: grid_logits). This script

1. keeps only degrees <= L (psi_l = 0 above L) and decodes as I2S does, by the argmax over the SO(3) HEALPix grid,
   and reports the median rotation error, acc@15 and acc@30 per class and class-averaged for every L;
2. reports the energy per degree |psi_l|^2, which by the orthogonality of the Wigner-D functions on SO(3) is exactly
   the variance the degree-l part contributes to the predicted log-density (l = 0 is a constant).

Run from pose3d/:
    python -m probes.pascal_wigner_spectrum.i2s_truncation --checkpoint i2s.pth --val_cache pascal_val.pt \
        --labels pascal_labels.npz [--grid_level 5]
"""

import argparse
import json
import os

import numpy as np
import torch

from probes.pascal_wigner_spectrum.spectrum import CLASSES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val_cache", required=True, help="pascal_val.pt of the RAM cache (imgs, targets)")
    p.add_argument("--labels", required=True, help="npz with val_cls (from spectrum.py's label extraction)")
    p.add_argument("--grid_level", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = p.parse_args()

    from e3nn import o3
    from image2sphere import so3_utils
    from pose3d.evaluate import build_model

    torch.set_grad_enabled(False)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        # upstream image2sphere's own training script (train.py): its I2S module is I2SReal.net, and upstream feeds
        # the ResNet [0, 1] images without ImageNet normalisation, so the wrapper must not normalise either.
        from pose3d.models.i2s_real import I2SReal

        model = I2SReal(encoder_type="resnet101", pretrained_backbone=False, lmax=6, normalize_input=False)
        missing, unexpected = model.load_state_dict({"net." + k: v for k, v in ckpt["model_state_dict"].items()
                                                     if k not in ("eval_wigners", "eval_rotmats")}, strict=False)
        assert set(missing) <= {"mean", "std"} and not unexpected, (missing, unexpected)
    else:
        model, saved = build_model(ckpt, torch.device("cpu"))
    model.eval()
    lmax = model.lmax
    cache = torch.load(args.val_cache, map_location="cpu", weights_only=False)
    imgs, gt = cache["imgs"], cache["targets"].double()
    cls = np.load(args.labels)["val_cls"]
    assert len(cls) == len(gt), (len(cls), len(gt))

    psi = []
    for i in range(0, len(imgs), args.batch_size):
        x = imgs[i:i + args.batch_size].float() / (255.0 if imgs.dtype == torch.uint8 else 1.0)
        psi.append(model(x).squeeze(1).double())
        print(f"  coefficients {min(i + args.batch_size, len(imgs))}/{len(imgs)}", flush=True)
    psi = torch.cat(psi)                                                          # (N, M)
    sizes = [(2 * l + 1) ** 2 for l in range(lmax + 1)]
    starts = np.cumsum([0] + sizes)
    energy = torch.stack([psi[:, starts[l]:starts[l + 1]].square().sum(1) for l in range(lmax + 1)], 1).numpy()

    a, b, g = so3_utils.so3_healpix_grid(rec_level=args.grid_level)
    grid = o3.angles_to_matrix(a, b, g).double()
    q = a.numel()
    print(f"grid level {args.grid_level}: {q} rotations", flush=True)
    err = np.zeros((lmax + 1, len(gt)))
    chunk_q = 262144
    best_val = torch.full((lmax + 1, len(gt)), -float("inf"), dtype=torch.float64)
    best_idx = torch.zeros((lmax + 1, len(gt)), dtype=torch.long)
    for j in range(0, q, chunk_q):
        wig = so3_utils.flat_wigner(lmax, a[j:j + chunk_q], b[j:j + chunk_q], g[j:j + chunk_q]).double()   # (Qc, M)
        cum = torch.zeros(len(gt), wig.shape[0], dtype=torch.float64)
        for l in range(lmax + 1):
            cum += psi[:, starts[l]:starts[l + 1]] @ wig[:, starts[l]:starts[l + 1]].T
            v, i = cum.max(1)
            better = v > best_val[l]
            best_val[l] = torch.where(better, v, best_val[l])
            best_idx[l] = torch.where(better, i + j, best_idx[l])
        print(f"  grid {min(j + chunk_q, q)}/{q}", flush=True)
    for l in range(lmax + 1):
        pred = grid[best_idx[l]]
        cos = ((pred.transpose(-1, -2) @ gt).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
        err[l] = np.degrees(np.arccos(np.clip(cos.numpy(), -1, 1)))

    results = {"checkpoint": os.path.basename(args.checkpoint), "grid_level": args.grid_level, "lmax": lmax,
               "per_L": [], "energy_per_degree": {}}
    for L in range(lmax + 1):
        e = err[L]
        per_class = {CLASSES[c]: float(np.median(e[cls == c])) for c in range(len(CLASSES))}
        row = dict(L=L, median=float(np.median(e)), class_mean_median=float(np.mean(list(per_class.values()))),
                   acc15=float((e < 15).mean()), acc30=float((e < 30).mean()), per_class_median=per_class)
        results["per_L"].append(row)
        print(f"L <= {L}: median {row['median']:6.2f}  class-mean {row['class_mean_median']:6.2f}  "
              f"acc@15 {row['acc15']:.3f}  acc@30 {row['acc30']:.3f}", flush=True)
    for c, name in enumerate(CLASSES):
        m = energy[cls == c]
        results["energy_per_degree"][name] = (m.mean(0)).tolist()
    results["energy_per_degree"]["all"] = energy.mean(0).tolist()
    with open(os.path.join(args.out, "i2s_truncation.json"), "w") as f:
        json.dump(results, f)
    plot(results, args)


def plot(results, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ls = [r["L"] for r in results["per_L"]][1:]
    rows = results["per_L"][1:]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    ax = axes[0]
    for name in CLASSES:
        ax.plot(Ls, [r["per_class_median"][name] for r in rows], marker="o", ms=3, lw=1, label=name)
    ax.plot(Ls, [r["class_mean_median"] for r in rows], color="k", lw=2.5, marker="o", label="class mean")
    ax.set(xlabel="truncation degree L", ylabel="median rotation error (deg)", yscale="log", xticks=Ls,
           title="I2S output truncated at L: median error per class")
    ax.legend(fontsize=7, ncol=2)
    ax = axes[1]
    ax.plot(Ls, [r["acc15"] for r in rows], marker="o", label="acc@15")
    ax.plot(Ls, [r["acc30"] for r in rows], marker="o", label="acc@30")
    ax.set(xlabel="truncation degree L", ylabel="accuracy", xticks=Ls, ylim=(0, 1), title="All val images")
    ax.legend()
    ax = axes[2]
    lmax = results["lmax"]
    for name in CLASSES:
        e = np.array(results["energy_per_degree"][name][1:])
        ax.plot(range(1, lmax + 1), e / e.sum(), marker="o", ms=3, lw=1, label=name)
    e = np.array(results["energy_per_degree"]["all"][1:])
    ax.plot(range(1, lmax + 1), e / e.sum(), color="k", lw=2.5, marker="o", label="all")
    ax.set(xlabel="degree l", ylabel=r"share of $\sum_{l\geq1}\|\psi_l\|^2$", xticks=list(range(1, lmax + 1)),
           title="Energy of the predicted log-density per degree")
    fig.suptitle(f"Image2Sphere {results['checkpoint']} on Pascal3D+ val (1,309 images), grid level {results['grid_level']}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "i2s_truncation.png"), dpi=130)


if __name__ == "__main__":
    main()
