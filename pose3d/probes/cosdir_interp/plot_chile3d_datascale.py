"""Validation median error against training images seen (epoch x N) for the 3D SYMSOL-CHILE 7-gon with N = 4,500
(2 seeds), 13,500 and 45,000 training views; same recipe otherwise. Dashed: the guessing level 90/7 = 12.9 deg.
Input results/chile3d_pyr7_datascale.json; writes ../../reports/figures/cosdir_chile3d_datascale.png."""
import json, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker as mt
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results", "chile3d_pyr7_datascale.json")))["runs"]
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.axhline(90 / 7, color="#52514e", ls="--", lw=1); ax.text(1.6e4, 90 / 7 * 1.07, "guessing level 12.9°", fontsize=8.5, color="#52514e")
for k, c in (("n4500_seed0", "#9fc3ec"), ("n4500_seed1", "#2a78d6"), ("n13500", "#eb6834"), ("n45000", "#1baf7a")):
    r = R[k]; x = [(i + 1) * r["n"] for i in range(len(r["mre"]))]
    lab = f"{r['n']:,} views" + (" (seed 1)" if k.endswith("seed1") else " (seed 0)" if r["n"] == 4500 else "") + f", final {r['final']:.2f}°"
    ax.plot(x, r["mre"], color=c, lw=1.4, label=lab)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(8, 90); ax.set_yticks([10, 20, 40, 80])
ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}°")); ax.yaxis.set_minor_formatter(mt.NullFormatter())
ax.set_xlabel("training images seen (log scale)"); ax.set_ylabel("validation median error (log scale)"); ax.grid(color="#e4e3df", lw=0.8)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_title("3D 7-gon: 3x and 10x more training views do not leave the guessing level", loc="left", fontsize=11)
fig.tight_layout(); out = os.path.join(HERE, "../../reports/figures/cosdir_chile3d_datascale.png"); fig.savefig(out, dpi=200, facecolor="white"); print(out)
