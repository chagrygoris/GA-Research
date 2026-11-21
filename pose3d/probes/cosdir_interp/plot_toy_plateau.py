"""Figure for the toy model of the plateau (toy_plateau.py, results/toy_plateau.json): example loss curves, plateau
length against the window pass a_10, and against the starting size of the learnable weights.
Writes ../../reports/figures/cosdir_toy_plateau.png."""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker as mt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
res = json.load(open(f"{R}/toy_plateau.json"))
C = {"gd": "#2a78d6", "adam": "#eb6834"}; LAB = {"gd": "gradient descent", "adam": "Adam"}
INK2, GRID = "#52514e", "#e4e3df"
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
seq = ["#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]
for k, kappa in enumerate((10, 12, 15, 18, 27)):
    r = next(x for x in res if x["opt"] == "gd" and x["kappa"] == kappa and x["eps"] == 1e-3 and x["seed"] == 0)
    st = np.arange(len(r["loss"])) * r["log_every"]
    ax[0].plot(st, r["loss"], color=seq[k], lw=1.8, label=f"κ {kappa}: a_10 = {r['a_n']:.3f}")
ax[0].axhline(0.5, color=INK2, lw=0.8, ls="--"); ax[0].text(50, 0.51, "turn-blind loss", fontsize=8, color=INK2)
ax[0].set_xlim(0, 4200); ax[0].set_xlabel("step"); ax[0].set_ylabel("loss (clean targets)")
ax[0].set_title("Loss curves (gradient descent): plateau, then a sudden drop", loc="left", fontsize=10)
ax[0].legend(fontsize=8, frameon=True, loc="lower right", facecolor="#fcfcfb", edgecolor="none")
for opt in ("gd", "adam"):
    rs = [x for x in res if x["opt"] == opt and x["eps"] == 1e-3 and x["escape_step"]]
    a = np.array([x["a_n"] for x in rs]); s = np.array([x["escape_step"] for x in rs])
    ax[1].scatter(a, s, color=C[opt], s=30, label=LAB[opt], zorder=3)
    sl, ic = np.polyfit(np.log(a), np.log(s), 1)
    xx = np.geomspace(a.min(), a.max()); ax[1].plot(xx, np.exp(ic) * xx ** sl, color=C[opt], lw=1, alpha=0.7)
    ax[1].text(xx[len(xx) // 2] * 1.15, np.exp(ic) * xx[len(xx) // 2] ** sl * 1.25, f"slope {sl:.2f}", color=C[opt], fontsize=9)
    rs = [x for x in res if x["opt"] == opt and x["kappa"] == 12 and x["escape_step"]]
    e = np.array([x["eps"] for x in rs]); s = np.array([x["escape_step"] for x in rs])
    ax[2].scatter(e, s, color=C[opt], s=30, label=LAB[opt], zorder=3)
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("a_10 (how much of degree 10 a window passes)")
ax[1].set_ylabel("steps until the loss halves"); ax[1].set_title("Plateau length vs window pass", loc="left", fontsize=10)
ax[2].set_xscale("log"); ax[2].set_xlabel("starting size of the learnable weights"); ax[2].set_ylabel("steps until the loss halves")
ax[2].set_title("Plateau length vs starting size (κ 12)", loc="left", fontsize=10)
for a_ in ax:
    a_.grid(color=GRID, lw=0.8); a_.set_axisbelow(True)
    for sp in ("top", "right"): a_.spines[sp].set_visible(False)
ax[1].legend(fontsize=8.5, frameon=False); ax[2].legend(fontsize=8.5, frameon=False)
fig.tight_layout()
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../reports/figures/cosdir_toy_plateau.png"); fig.savefig(p, dpi=160, facecolor="#fcfcfb"); print(p)
