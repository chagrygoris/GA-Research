"""One-figure summary of the s_h (start inverse temperature) experiments on the bottom views (14x14 map, 368 sphere
points, constant rate). Left: outcome of every run by pyramid and start s_h, with the runs scheduled for 2026-11-21.
Right: how much of degree n an attention window passes, a_n = i_n(kappa) / i_0(kappa) with kappa = 0.37..0.48 s_h
(the ratio fitted on the checkpoints), against the measured median a_n of the finished checkpoints.
Writes ../../reports/figures/cosdir_s_h_scaling.png."""
import os, numpy as np, matplotlib, matplotlib.ticker
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import ive
C = {7: "#2a78d6", 10: "#eb6834", 13: "#1baf7a"}            # categorical slots 1-3 (validated)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
# finished runs: (n, start s_h, learned, note)
RUNS = [(7, 30, False, ""), (7, 30, False, "log s_h 30x faster\n(60 epochs)"), (7, 60, True, "epoch 61"), (7, 60, False, "same setup"),
        (7, 60, False, "s_h frozen"), (7, 100, True, "epoch 65"), (10, 60, False, ""), (13, 60, False, "")]
PLAN = [(10, 120), (13, 200)]
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.3), gridspec_kw={"width_ratios": [1.15, 1]})
Y = {7: 2, 10: 1, 13: 0}
OFF = {(7, 30): [0.12, -0.12], (7, 60): [0.22, 0.0, -0.22], (7, 100): [0.22]}
seen = {}
for n, s, ok, note in RUNS:
    k = seen.get((n, s), 0); seen[(n, s)] = k + 1
    y = Y[n] + OFF.get((n, s), [0])[k]
    if ok: a.scatter(s, y, s=70, color=C[n], edgecolor="white", linewidth=1.5, zorder=3)
    else: a.scatter(s, y, s=55, marker="x", color=C[n], linewidth=2, zorder=3)
    if note:
        left = s == 30
        a.text(s / 1.09 if left else s * 1.09, y, note, va="center", ha="right" if left else "left", fontsize=8, color=INK2)
for n, s in PLAN:
    a.scatter(s, Y[n] + 0.12, s=70, facecolor="white", edgecolor=C[n], linewidth=1.6, linestyle="--", zorder=3)
    a.text(s * 1.09, Y[n] + 0.12, f"start {s}", va="center", fontsize=8, color=INK2)
    a.plot([30, 250], [Y[n] - 0.17] * 2, color=C[n], lw=2, ls="--", alpha=0.8)
    a.scatter(np.geomspace(30, 250, 8), [Y[n] - 0.17] * 8, s=14, facecolor="white", edgecolor=C[n], zorder=3)
    a.text(260, Y[n] - 0.17, "heads 30 to 250", va="center", fontsize=8, color=INK2)
a.set_xscale("log"); a.set_xlim(12, 420); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); a.set_ylim(-0.5, 2.6)
a.set_xticks([30, 60, 100, 120, 200, 250]); a.set_xticklabels(["30", "60", "100", "120", "200", "250"])
a.set_yticks([2, 1, 0]); a.set_yticklabels(["7-gon", "10-gon", "13-gon"])
for t, n in zip(a.get_yticklabels(), (7, 10, 13)): t.set_color(INK)
a.grid(axis="x", color=GRID, lw=0.8); a.set_axisbelow(True)
a.set_xlabel("start s_h (inverse temperature of the cosine attention)")
a.set_title("Outcome per run (120 epochs; scheduled runs: 180)", loc="left", color=INK, fontsize=10.5)
from matplotlib.lines import Line2D
a.legend(handles=[Line2D([], [], marker="o", ls="", color=INK2, ms=8, label="learned the turn (epoch of breakthrough)"),
                  Line2D([], [], marker="x", ls="", color=INK2, ms=7, mew=2, label="stayed at the guessed turn"),
                  Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=8, label="scheduled 21 Nov, 00:05 UTC")],
         loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8, frameon=False)
s = np.geomspace(10, 400, 300)
apass = lambda l, k: ive(l + 0.5, k) / ive(0.5, k)
b.axhspan(0.28, 0.37, color="#f1f0ec", zorder=0)
b.text(11, 0.385, "pass at which the 7-gon learned (s_h 60)", fontsize=8, color=INK2)
for n in (7, 10, 13):
    b.fill_between(s, apass(n, 0.37 * s), apass(n, 0.48 * s), color=C[n], alpha=0.18, lw=0)
    b.plot(s, apass(n, 0.42 * s), color=C[n], lw=2)
    b.text(410, apass(n, 0.42 * 400), f"degree {n}", va="center", fontsize=8.5, color=INK)
MEAS = [(7, 30, 0.073, False), (7, 60, 0.33, True), (7, 60, 0.30, False), (7, 60, 0.34, False), (10, 60, 0.10, False), (13, 60, 0.03, False)]
for n, sv, av, ok in MEAS:
    if ok: b.scatter(sv, av, s=60, color=C[n], edgecolor="white", linewidth=1.5, zorder=3)
    else: b.scatter(sv, av, s=50, marker="x", color=C[n], linewidth=2, zorder=3)
for n, sv in PLAN: b.axvline(sv, color=C[n], lw=1, ls="--"); b.text(sv * 1.03, 0.02, f"{sv}", color=INK2, fontsize=8)
b.set_xscale("log"); b.set_xlim(10, 600); b.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); b.set_ylim(0, 1)
b.set_xticks([10, 30, 60, 100, 200, 400]); b.set_xticklabels(["10", "30", "60", "100", "200", "400"])
b.grid(color=GRID, lw=0.8); b.set_axisbelow(True)
b.set_xlabel("s_h"); b.set_ylabel("a_n: how much of degree n a window passes")
b.set_title("Why: window width in degree n (lines: theory; marks: measured)", loc="left", color=INK, fontsize=10.5)
for ax in (a, b):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../reports/figures/cosdir_s_h_scaling.png")
fig.savefig(out, dpi=200, facecolor="#fcfcfb"); print(out)
