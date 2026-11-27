"""Validation median error by epoch for the 3D 7-gonal prism (D7; seed 0) and the 3D 7-gonal pyramid (C7; seeds 0 and 1),
both with the bottom-view recipe of the cosine-attention GATr (ImageNet ResNet-50, 14x14 map, 368 sphere points, start
s_h 100, constant learning rate, 4,500 views uniform on SO(3), 180 epochs, 2xT4). Dashed: the guessing level 90/7 = 12.9
deg, shared by both shapes (axis known, turn about it guessed).
Inputs results/chile3d_prism7.json and results/chile3d_pyr7_mre_by_epoch.json; writes
../../reports/figures/cosdir_chile3d_prism7.png."""
import json, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker as mt
HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, "results", "chile3d_prism7.json")))["runs"]["chile3d_prism7_seed0"]
Y = json.load(open(os.path.join(HERE, "results", "chile3d_pyr7_mre_by_epoch.json")))["runs"]
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.axhline(90 / 7, color="#52514e", ls="--", lw=1); ax.text(3, 90 / 7 * 0.90, "guessing level 12.9°", fontsize=8.5, color="#52514e")
for v, c, lab in ((Y["chile3d_seed0"]["mre"], "#9fc3ec", "pyramid, seed 0 (final 12.74°)"),
                  (Y["chile3d_seed1"]["mre"], "#2a78d6", "pyramid, seed 1 (final 13.11°)"),
                  (P["mre"], "#eb6834", f"prism, seed 0 (final {P['final_32sample_median']:.2f}°)")):
    ax.plot(range(1, len(v) + 1), v, color=c, lw=1.4, label=lab)
ax.set_yscale("log"); ax.set_ylim(8, 90); ax.set_yticks([10, 20, 40, 80])
ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}°")); ax.yaxis.set_minor_formatter(mt.NullFormatter())
ax.set_xlabel("epoch"); ax.set_ylabel("validation median error (log scale)"); ax.grid(color="#e4e3df", lw=0.8)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_title("3D 7-gonal prism: the axis is learned later than for the pyramid, the turn not at all", loc="left", fontsize=11)
fig.tight_layout(); out = os.path.join(HERE, "../../reports/figures/cosdir_chile3d_prism7.png"); fig.savefig(out, dpi=200, facecolor="white"); print(out)
