"""Figure: what grows on the 7-gon's guessed-turn plateau. Snapshot diagnostics (diag2.py) of the trace, frozen and fast-s_h
runs (results/diag2_snap_*.json) against the final checkpoint of the run that learned (results/diag2_bottom_grid14_s60.json).
Writes ../../reports/figures/cosdir_bottom7_plateau_growth.png."""
import json, glob, os, re, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
runs = {"trace": ("s_h 60 learnable (trace run)", "#762a83", "o-"), "frozen": ("s_h 60 frozen", "#c2a5cf", "s--"),
        "free": ("s_h 30, 30x faster", "#e08214", "^:")}
def stats(d):
    r = [x for x in d["rings"] if x["share"] is not None]; b = max(r, key=lambda x: x["share"])
    return d["velocity_t0.9"]["aligned"], b["share"], b["coherence"], d["window_pass"]["7"]["median"]
ok = stats(json.load(open(f"{R}/diag2_bottom_grid14_s60.json")))
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
out = {}
for run, (lab, c, st) in runs.items():
    fs = sorted(glob.glob(f"{R}/diag2_snap_{run}_e*.json"))
    if not fs: continue
    e = [int(re.search(r"_e(\d+)", f).group(1)) for f in fs]; v = np.array([stats(json.load(open(f))) for f in fs])
    out[run] = {"epochs": e, "aligned_velocity_freq7": v[:, 0].tolist(), "max_ring_share7": v[:, 1].tolist(),
                "coherence": v[:, 2].tolist(), "a7_median": v[:, 3].tolist()}
    for k in range(3): ax[k].plot(e, v[:, k], st, color=c, label=lab, ms=5)
for k, (yl, lg) in enumerate((("turn-following frequency-7 velocity (t = 0.9)", True), ("largest ring share at frequency 7", True),
                               ("phase coherence of that component", False))):
    ax[k].axhline(ok[k], color="#1b7837", lw=1.2); ax[k].text(2, ok[k] * (0.6 if lg else 0.95), "run that learned (final)", color="#1b7837", fontsize=8)
    ax[k].set_xlabel("epoch"); ax[k].set_ylabel(yl)
    if lg: ax[k].set_yscale("log")
ax[0].legend(fontsize=8); ax[2].set_ylim(0, 1.05)
fig.suptitle("7-gon bottom view on the guessed-turn plateau (none of these runs broke through in 120 epochs)", fontsize=11)
fig.tight_layout()
json.dump(out, open(f"{R}/plateau_growth.json", "w"), indent=1)
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../reports/figures/cosdir_bottom7_plateau_growth.png"); fig.savefig(p, dpi=130); print(p)
