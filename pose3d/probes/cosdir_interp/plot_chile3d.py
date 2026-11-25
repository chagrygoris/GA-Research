"""Validation median error by epoch: the bottom-view recipe of the cosine-attention GATr (14x14 map, 368 sphere points,
start s_h 100, constant learning rate) on the 3D SYMSOL-CHILE 7-gon (poses uniform on SO(3); 2 seeds, 180 epochs; plus one run whose
encoder is the ResNet-101 of the Pascal3D+ run) and,
for reference, on the bottom view of the same pyramid (120 epochs). Dashed: the guessing level 90/7 = 12.9 deg.
Input results/chile3d_pyr7_mre_by_epoch.json; writes ../../reports/figures/cosdir_chile3d_pyr7.png."""
import json, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker as mt
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results", "chile3d_pyr7_mre_by_epoch.json")))["runs"]
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.axhline(90 / 7, color="#52514e", ls="--", lw=1); ax.text(122, 90 / 7 * 1.06, "guessing level 12.9°", fontsize=8.5, color="#52514e")
for k, c, lab in (("chile3d_seed0", "#2a78d6", "3D views, seed 0 (final 12.74°)"), ("chile3d_seed1", "#6250d6", "3D views, seed 1 (final 13.11°)"),
                  ("chile3d_pascalr101_seed0", "#1baf7a", "3D views, Pascal ResNet-101 init (final 13.31°)"),
                  ("bottom7_s100_reference", "#eda100", "bottom view, same recipe (final 0.46°)")):
    v = R[k]["mre"]; ax.plot(range(1, len(v) + 1), v, color=c, lw=1.4, label=lab)
ax.set_yscale("log"); ax.set_ylim(0.3, 90); ax.set_yticks([0.5, 1, 3, 10, 30])
ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}°")); ax.yaxis.set_minor_formatter(mt.NullFormatter())
ax.set_xlabel("epoch"); ax.set_ylabel("validation median error (log scale)"); ax.grid(color="#e4e3df", lw=0.8)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_title("7-gonal pyramid: the bottom-view recipe does not transfer to 3D poses", loc="left", fontsize=11)
fig.tight_layout(); out = os.path.join(HERE, "../../reports/figures/cosdir_chile3d_pyr7.png"); fig.savefig(out, dpi=200, facecolor="white"); print(out)
