"""Per-epoch validation median error of the four 28x28-feature-map runs on the bottom views (13-gon seeds 0, 1, 2 at start
s_h 200; 10-gon seed 0 at start s_h 120; 240 epochs, constant learning rate). Each run was stopped by the Kaggle
watchdog after its first session (epoch 138 or 145) and continued from its per-epoch checkpoint; the dotted line marks
the resume. Seed 1 was not resumed (it had already learned). Input: g28_bottom_mre_by_epoch.json (written from the kernel
logs); writes ../../reports/figures/cosdir_g28_curves.png."""
import json, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker as mt
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "results", "g28_bottom_mre_by_epoch.json")))
STYLE = {"13gon_s200_seed0": ("#1baf7a", "-"), "13gon_s200_seed1": ("#0d6e4c", "-"), "13gon_s200_seed2": ("#7fd6b4", "-"),
         "10gon_s120_seed0": ("#eb6834", "-")}
INK2, GRID = "#52514e", "#e4e3df"
fig, ax = plt.subplots(figsize=(10, 5.2))
for n in (10, 13):
    ax.axhline(90 / n, color="#eb6834" if n == 10 else "#1baf7a", ls="--", lw=1, alpha=0.7)
    ax.text(3, 90 / n * 1.05, f"guessing level, {n}-gon ({90 / n:.1f}°)", fontsize=8.5, color=INK2)
for k, r in D["runs"].items():
    c, ls = STYLE[k]
    ax.plot(range(1, len(r["mre"]) + 1), r["mre"], color=c, lw=1.4, ls=ls, label=r["label"])
    if r.get("resumed_at"):
        ax.axvline(r["resumed_at"], color=c, ls=":", lw=1)
ax.set_yscale("log"); ax.set_ylim(0.3, 90)
ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}°")); ax.set_yticks([0.5, 1, 3, 10, 30])
ax.yaxis.set_minor_formatter(mt.NullFormatter())
ax.set_xlim(0, 242); ax.set_xlabel("epoch"); ax.set_ylabel("validation median error (log scale)")
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_title("28×28 feature map, bottom views: 13-gon learned in 2 of 3 seeds, 10-gon did not (dotted: resume)", loc="left", fontsize=11)
fig.tight_layout(); out = os.path.join(HERE, "../../reports/figures/cosdir_g28_curves.png")
fig.savefig(out, dpi=200, facecolor="white"); print(out)
