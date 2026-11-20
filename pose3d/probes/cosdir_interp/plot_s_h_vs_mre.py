"""Final median rotation error against the start value of s_h, per pyramid (bottom views; 14x14 map, 368 sphere points,
constant rate, 120 epochs, final 32-sample evaluation). Dashed lines: the error of a model that knows everything but the
turn (90/n degrees). Writes ../../reports/figures/cosdir_s_h_vs_mre.png."""
import os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker as mt
C = {7: "#2a78d6", 10: "#eb6834", 13: "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# (sides, start s_h, final median error in degrees)
RUNS = [(7, 30, 12.21), (7, 60, 0.40), (7, 60, 12.12), (7, 60, 11.91), (7, 100, 0.46), (10, 60, 9.11), (13, 60, 6.68)]
JIT = {(7, 60, 12.12): -4, (7, 60, 11.91): 4, (10, 60, 9.11): -2, (13, 60, 6.68): 2}
plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK})
fig, ax = plt.subplots(figsize=(8, 5.2))
for n in (7, 10, 13):
    ax.hlines(90 / n, 20, 108, color=C[n], lw=1.2, ls="--", alpha=0.8)
    ax.text(110, 90 / n, f"guessing level, {n}-gon ({90 / n:.1f}°)", va="center", fontsize=9, color=INK2)
for n, s, e in RUNS:
    ax.scatter(s + JIT.get((n, s, e), 0), e, s=110, color=C[n], edgecolor="white", linewidth=2, zorder=3)
ax.text(63, 0.40, "learned", va="center", fontsize=9, color=INK2)
ax.text(103, 0.46, "learned", va="center", fontsize=9, color=INK2)
ax.text(60, 15.5, "3 runs at 60", ha="center", fontsize=9, color=INK2)
ax.set_yscale("log"); ax.set_ylim(0.25, 20)
ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}°")); ax.set_yticks([0.3, 1, 3, 10])
ax.yaxis.set_minor_formatter(mt.NullFormatter())
ax.set_xlim(20, 170); ax.set_xticks([30, 60, 100])
ax.set_xlabel("starting s_h"); ax.set_ylabel("final median rotation error (log scale)")
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C[n], ms=9, label=f"{n}-gon") for n in (7, 10, 13)],
          title="pyramid sides", loc="center left", frameon=False, fontsize=9.5, title_fontsize=9.5)
ax.set_title("Bottom-view pyramids: starting s_h vs final error (120 epochs)", loc="left", color=INK, fontsize=12)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../reports/figures/cosdir_s_h_vs_mre.png")
fig.savefig(out, dpi=200, facecolor="#fcfcfb"); print(out)
