"""Gated curriculum on the 3D 7-gon: validation median on views with tilt <= 10 deg (the limit never widened) against
training images seen, compared with the exact bottom-view 7-gon run of the same model (s_h 100), which broke through.
Dashed: guessing level 12.9 deg; dotted: the gate (6.4 deg). Input results/chile3d_pyr7_gated.json; writes
../../reports/figures/cosdir_chile3d_gated.png."""
import json, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker as mt
HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "results", "chile3d_pyr7_gated.json")))
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.axhline(90 / 7, color="#52514e", ls="--", lw=1); ax.text(1.2e4, 90 / 7 * 1.07, "guessing level 12.9°", fontsize=8.5, color="#52514e")
ax.axhline(6.4, color="#9a9890", ls=":", lw=1.2); ax.text(1.2e4, 6.4 * 1.06, "gate 6.4°", fontsize=8.5, color="#52514e")
x = [(i + 1) * 10000 for i in range(len(d["mre"]))]
ax.plot(x, d["mre"], color="#e34948", lw=1.8, label=f"3D, gated curriculum, tilt ≤ 10° (final {d['final']:.2f}°)")
b = d["bottom_view_reference"]; xb = [(i + 1) * b["images_per_epoch"] for i in range(len(b["mre"]))]
ax.plot(xb, b["mre"], color="#eda100", lw=1.4, label=f"exact bottom view, same model (final {b['final']:.2f}°)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.3, 90); ax.set_yticks([0.5, 1, 3, 10, 30])
ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}°")); ax.yaxis.set_minor_formatter(mt.NullFormatter())
ax.set_xlabel("training images seen (log scale)"); ax.set_ylabel("validation median error (log scale)"); ax.grid(color="#e4e3df", lw=0.8)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.set_title("Gated curriculum: views within 10° of the bottom are not learned in 750k images", loc="left", fontsize=11)
fig.tight_layout(); out = os.path.join(HERE, "../../reports/figures/cosdir_chile3d_gated.png"); fig.savefig(out, dpi=200, facecolor="white"); print(out)
