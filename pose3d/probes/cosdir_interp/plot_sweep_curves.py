"""Median error by epoch of the 180-epoch s_h runs on the 10- and 13-gon bottom views (results/bottom10_13_sweep_mre_by_epoch.json).
Writes ../../reports/figures/cosdir_bottom10_13_sweep_curves.png."""
import json, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker as mt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
d = json.load(open(f"{R}/bottom10_13_sweep_mre_by_epoch.json"))
C = {"10": "#eb6834", "13": "#1baf7a"}; INK2, GRID = "#52514e", "#e4e3df"
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for k, n in enumerate(("10", "13")):
    a = ax[k]
    for lab, y in d.items():
        if not lab.startswith(n): continue
        a.plot(range(1, len(y) + 1), y, color=C[n], lw=1.8, ls="-" if "s_h" in lab else ":", label=lab.split(", ")[1])
    a.axhline(90 / int(n), color=INK2, lw=1, ls="--"); a.text(3, 90 / int(n) * 1.06, f"guessing level {90 / int(n):.1f}°", fontsize=8.5, color=INK2)
    if n == "10": a.axvline(180, color=INK2, lw=0.8, ls=":"); a.text(183, 40, "resumed\nfrom epoch 180", fontsize=8, color=INK2)
    a.set_yscale("log"); a.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}°")); a.set_yticks([0.3, 1, 3, 10, 30, 70])
    a.yaxis.set_minor_formatter(mt.NullFormatter()); a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
    a.set_xlabel("epoch"); a.set_title(f"{n}-gon bottom view", loc="left", fontsize=11); a.legend(frameon=False, fontsize=9)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
ax[0].set_ylabel("median rotation error (log scale)")
fig.tight_layout()
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../reports/figures/cosdir_bottom10_13_sweep_curves.png"); fig.savefig(p, dpi=170, facecolor="#fcfcfb"); print(p)
