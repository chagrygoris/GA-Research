"""Validation median error (1,000 uniform-SO(3) test views) per epoch of the curriculum run on the 3D 7-gon (tilt limit
rising from 15 to 180 deg by epoch 33, then uniform-SO(3) weights; right axis), against the two uniform-pool runs with
4,500 views (one epoch = 4,500 images there, 20,000 here: x is epochs, not images). Dashed: guessing level 12.9 deg.
Inputs results/chile3d_pyr7_curriculum.json, results/chile3d_pyr7_mre_by_epoch.json; writes
../../reports/figures/cosdir_chile3d_curriculum.png."""
import json, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker as mt
HERE = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(HERE, "results", "chile3d_pyr7_curriculum.json")))
U = json.load(open(os.path.join(HERE, "results", "chile3d_pyr7_mre_by_epoch.json")))["runs"]
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.axhline(90 / 7, color="#52514e", ls="--", lw=1); ax.text(45, 90 / 7 * 0.88, "guessing level 12.9°", fontsize=8.5, color="#52514e")
e = range(1, len(C["mre"]) + 1)
ax.plot(e, C["mre"], color="#e34948", lw=1.8, label=f"curriculum, 20k-view pool (final {C['final']:.2f}°)")
for k, c, lab in (("chile3d_seed0", "#9fc3ec", "uniform pool, 4,500 views, seed 0"), ("chile3d_seed1", "#2a78d6", "uniform pool, 4,500 views, seed 1")):
    v = U[k]["mre"][:len(C["mre"])]; ax.plot(range(1, len(v) + 1), v, color=c, lw=1.1, label=lab)
ax.set_yscale("log"); ax.set_ylim(8, 100); ax.set_yticks([10, 20, 40, 80])
ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}°")); ax.yaxis.set_minor_formatter(mt.NullFormatter())
ax.set_xlabel("epoch"); ax.set_ylabel("validation median error, uniform test views (log)"); ax.grid(color="#e4e3df", lw=0.8)
ax2 = ax.twinx(); ax2.plot(e, C["tilt_limit"], color="#9a9890", ls=":", lw=1.5); ax2.set_ylim(0, 190)
ax2.set_ylabel("curriculum tilt limit (deg, dotted)", color="#52514e")
for sp in ("top",): ax.spines[sp].set_visible(False); ax2.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="center right", bbox_to_anchor=(1.0, 0.62))
ax.set_title("3D 7-gon: a bottom-to-all-tilts curriculum also ends at the guessing level", loc="left", fontsize=11)
fig.tight_layout(); out = os.path.join(HERE, "../../reports/figures/cosdir_chile3d_curriculum.png"); fig.savefig(out, dpi=200, facecolor="white"); print(out)
