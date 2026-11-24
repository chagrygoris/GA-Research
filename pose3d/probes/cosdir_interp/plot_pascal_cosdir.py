"""Validation median error (all test images) and class-mean median error (mean of the 12 per-class medians) per epoch
of the cosine-attention GATr with the 14x14 map on Pascal3D+ (50 epochs, 4xL4, resumed at epoch 38 in a second Kaggle
session; dotted line). Dashed: the reference recipe's final class-mean 8.95 deg (100 epochs). Input
results/pascal_cosdir_e50.json; writes ../../reports/figures/pascal_cosdir_e50.png."""
import json, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker as mt
HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "results", "pascal_cosdir_e50.json")))
fig, ax = plt.subplots(figsize=(9, 4.6))
e = range(1, len(d["median_by_epoch"]) + 1)
ax.plot(e, d["median_by_epoch"], color="#2a78d6", lw=1.6, label=f"median, all images (final {d['final']['median']:.2f}°)")
ax.plot(range(1, len(d["class_mean_by_epoch"]) + 1), d["class_mean_by_epoch"], color="#eb6834", lw=1.6,
        label=f"class-mean median (final {d['final']['class_mean']:.2f}°)")
ax.axhline(8.95, color="#52514e", ls="--", lw=1); ax.text(1.5, 8.95 * 0.9, "reference recipe, 100 epochs: class-mean 8.95°", fontsize=8.5, color="#52514e")
ax.axvline(d["resumed_at"] - 0.5, color="#9a9890", ls=":", lw=1); ax.text(d["resumed_at"] + 0.3, 60, "resumed", fontsize=8.5, color="#52514e")
ax.set_yscale("log"); ax.set_ylim(5, 150); ax.set_yticks([5, 8, 10, 20, 50, 100])
ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}°")); ax.yaxis.set_minor_formatter(mt.NullFormatter())
ax.set_xlabel("epoch"); ax.set_ylabel("validation error (log scale)"); ax.grid(color="#e4e3df", lw=0.8)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_title("Pascal3D+: cosine-attention GATr, 14×14 map, ResNet-101, warp + synth, 50 epochs", loc="left", fontsize=11)
fig.tight_layout(); out = os.path.join(HERE, "../../reports/figures/pascal_cosdir_e50.png"); fig.savefig(out, dpi=200, facecolor="white"); print(out)
