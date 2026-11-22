"""Final median rotation error against the start value of s_h, per pyramid (bottom views; 14x14 map, 368 sphere points,
constant rate, 120 or 180 epochs, final 32-sample evaluation). Dashed lines: the error of a model that knows everything but the
turn (90/n degrees). Stars: the 28x28 feature map (last two ResNet stages dilated), 240 epochs; hollow = no final
evaluation yet (last per-epoch validation median). Writes ../../reports/figures/cosdir_s_h_vs_mre.png."""
import os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker as mt
C = {7: "#2a78d6", 10: "#eb6834", 13: "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# (sides, start s_h, final median error in degrees)
RUNS = [(7, 30, 12.21), (7, 60, 0.40), (7, 60, 12.12), (7, 60, 11.91), (7, 100, 0.46), (10, 60, 9.11), (13, 60, 6.68)]
# 180-epoch runs (2026-11-21): one shared start value, and heads log-spaced from 30 to 250 in every block
LONG = [(10, 120, 0.40), (13, 200, 6.49)]
SPREAD = [(10, 8.99), (13, 6.98)]
# 28x28 map, 240 epochs: (sides, start s_h, error, final?, x offset). Hollow star: no final evaluation (seed 1 stopped at epoch 145;
# seed 2 final 6.66 and the 10-gon final 9.54: both guessing)
G28 = [(13, 200, 0.458, True, 1.10), (13, 200, 0.70, False, 1.16), (13, 200, 6.66, True, 1.13), (10, 120, 9.54, True, 1.10)]
JIT = {(7, 60, 12.12): -4, (7, 60, 11.91): 4, (10, 60, 9.11): -2, (13, 60, 6.68): 2}
plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK})
fig, ax = plt.subplots(figsize=(10, 5.6))
for n in (7, 10, 13):
    ax.hlines(90 / n, 20, 560, color=C[n], lw=1.2, ls="--", alpha=0.8)
    ax.text(23, 90 / n * 1.035, f"guessing level, {n}-gon ({90 / n:.1f}°)", va="bottom", fontsize=8.5, color=INK2)
for n, s, e in RUNS:
    ax.scatter(s + JIT.get((n, s, e), 0), e, s=110, color=C[n], edgecolor="white", linewidth=2, zorder=3)
for n, s, e in LONG:
    ax.scatter(s, e, s=130, marker="D", color=C[n], edgecolor="white", linewidth=2, zorder=3)
ax.text(128, 0.40, "learned\n(240 epochs)", va="center", fontsize=8.5, color=INK2)
for n, s, e, fin, dx in G28:
    ax.scatter(s * dx, e, s=230, marker="*", color=C[n] if fin else "white", edgecolor=C[n], linewidth=1.6 if not fin else 1.2, zorder=4)
ax.text(200 * 1.21, 0.5, "learned\n(28×28)", va="center", fontsize=8.5, color=INK2)
XS = 420   # separate column for the spread runs (no single starting value)
ax.axvline(330, color=GRID, lw=1.5)
for n, e in SPREAD:
    ax.scatter(XS, e, s=130, marker="s", color=C[n], edgecolor="white", linewidth=2, zorder=3)
ax.text(63, 0.40, "learned", va="center", fontsize=9, color=INK2)
ax.text(103, 0.46, "learned", va="center", fontsize=9, color=INK2)
ax.text(60, 14.6, "3 runs at 60", ha="center", fontsize=9, color=INK2)
ax.set_yscale("log"); ax.set_ylim(0.25, 20)
ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}°")); ax.set_yticks([0.3, 1, 3, 10])
ax.yaxis.set_minor_formatter(mt.NullFormatter())
ax.set_xscale("log"); ax.set_xlim(22, 560); ax.set_xticks([30, 60, 100, 120, 200, XS])
ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: "heads\n30–250" if v == XS else f"{v:g}")); ax.xaxis.set_minor_formatter(mt.NullFormatter())
ax.set_xlabel("starting s_h"); ax.set_ylabel("final median rotation error (log scale)")
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C[n], ms=9, label=f"{n}-gon") for n in (7, 10, 13)]
          + [Line2D([], [], marker="o", ls="", color=INK2, ms=8, label="120 epochs"), Line2D([], [], marker="D", ls="", color=INK2, ms=7, label="180 epochs (10-gon: 240)"),
             Line2D([], [], marker="s", ls="", color=INK2, ms=7, label="180 epochs, heads spread"),
             Line2D([], [], marker="*", ls="", color=INK2, ms=12, label="28×28 map, 240 epochs"), Line2D([], [], marker="*", ls="", mfc="white", mec=INK2, ms=12, label="28×28, no final eval (stopped)")],
          loc="center right", bbox_to_anchor=(1.0, 0.45), frameon=False, fontsize=9, ncol=2)
ax.set_title("Bottom-view pyramids: starting s_h vs final error", loc="left", color=INK, fontsize=12)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../reports/figures/cosdir_s_h_vs_mre.png")
fig.savefig(out, dpi=200, facecolor="#fcfcfb"); print(out)
