"""Figures of reports/cosdir_mechanism.md from results/*.json."""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from probe import tails, degree_power

here = os.path.dirname(os.path.abspath(__file__)); res = lambda n: json.load(open(os.path.join(here, "results", n)))
out = os.path.join(here, "..", "..", "reports", "figures")
C = {"tet": "#2a78d6", "cube": "#eb6834", "icosa": "#1baf7a", "cone": "#eda100", "cyl": "#e87ba4"}
NAME = {"tet": "tetrahedron", "cube": "cube", "icosa": "icosahedron", "cone": "cone", "cyl": "cylinder"}
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True})
runs = {r["shape"]: r for r in res("spec_all.json") + [r for r in res("spec_tet.json") if r["t"] == 0.5]}
order = ["tet", "cube", "icosa", "cyl"]

# 1. per-degree spectrum of the predicted velocity
fig, axes = plt.subplots(1, 4, figsize=(12, 2.9), sharey=True)
for ax, s in zip(axes, order):
    p = degree_power(runs[s]["spec"]["velocity|mv"], 12)
    ax.bar(range(1, 13), p, width=0.7, color=C[s]); ax.set_title(NAME[s], color=INK, fontsize=11, loc="left")
    ax.set_xticks(range(1, 13, 1)); ax.set_xlabel("degree l of the pose dependence"); ax.grid(axis="x", visible=False)
    l = int(np.argmax(p)) + 1; ax.annotate(f"l = {l}: {p[l - 1]:.0%}", (l, p[l - 1]), xytext=(l + 1.2, p[l - 1] * 0.92), color=INK)
axes[0].set_ylabel("share of the variance over poses")
fig.suptitle("Predicted velocity: share of its pose dependence in each degree (one test image per shape, t = 0.5)", color=INK, x=0.01, ha="left", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(out, "cosdir_velocity_spectrum.png"), dpi=170); plt.close(fig)

# 2. stage by stage: share of the pose token's scalar channels at degree >= 4
stages = ["b0.attn", "b0.mlp", "b1.attn", "b1.mlp", "b2.attn", "b2.mlp", "b3.attn", "b3.mlp"]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for s in order:
    y = [min(tails(runs[s]["spec"][k + "|s"], (4,))[4], 1.0) for k in stages]
    ax.plot(range(len(stages)), y, color=C[s], lw=2, marker="o", ms=5, label=NAME[s])
ax.set_xticks(range(len(stages))); ax.set_xticklabels([k.replace("b", "block ").replace(".attn", "\nattention").replace(".mlp", "\nMLP") for k in stages], fontsize=8)
ax.set_ylabel("share of pose variance in degrees >= 4"); ax.set_xlim(-0.3, len(stages) - 0.7); ax.set_ylim(0, 1.05)
ax.set_title("Pose token, scalar channels: high-degree share after every sublayer", color=INK, loc="left", fontsize=11)
ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(out, "cosdir_stage_spectrum.png"), dpi=170); plt.close(fig)

# 3. causal: all inverse temperatures times a factor, at inference
iv = {r["name"]: r for r in res("interv.json")}; fac = [0.03, 0.1, 0.25, 0.5, 1, 2]; key = lambda f: f"all x{f:g}"
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6))
for s in C:
    y = [iv[key(f)]["err"][s]["median"] for f in fac]; a.plot(fac, y, color=C[s], lw=2, marker="o", ms=5, label=NAME[s])
a.set_xscale("log"); a.set_xticks(fac); a.set_xticklabels([f"x{f:g}" for f in fac]); a.minorticks_off()
a.set_xlabel("factor on every inverse temperature s (trained value about 30)"); a.set_ylabel("median rotation error, degrees")
a.set_title("Error after rescaling the attention temperature", color=INK, loc="left", fontsize=11); a.legend(frameon=False, fontsize=9)
for s in ("tet", "cube", "icosa"):
    y = [iv[key(f)]["spec"][s]["tails"]["4"] for f in fac]; b.plot(fac, y, color=C[s], lw=2, marker="o", ms=5, label=NAME[s])
b.set_xscale("log"); b.set_xticks(fac); b.set_xticklabels([f"x{f:g}" for f in fac]); b.minorticks_off(); b.set_ylim(0, 1)
b.set_xlabel("factor on every inverse temperature s"); b.set_ylabel("share of velocity variance in degrees >= 4")
b.set_title("High-degree share of the predicted velocity", color=INK, loc="left", fontsize=11); b.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(out, "cosdir_temperature.png"), dpi=170); plt.close(fig)
