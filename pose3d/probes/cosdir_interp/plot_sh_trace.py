"""Figure: s_h trajectories of the 7-gon trace runs (results/s_h_trace_runs.json) next to the error curves of all 14x14
7-gon runs. Writes ../../reports/figures/cosdir_bottom7_s_h_trace.png."""
import json, os, numpy as np, matplotlib, matplotlib.ticker
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
t = json.load(open(f"{R}/s_h_trace_runs.json"))
s60 = json.load(open(f"{R}/bottom7_grid14_mre_by_epoch.json"))["constant_lr_grid14_nside8_s60"]
s100 = json.load(open(f"{R}/bottom7_grid14_s100_mre_by_epoch.json"))["constant_lr_120"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for y, lab, c, ls in ((s60, "s_h 60 learnable, seed 0 (first run)", "#1b7837", "-"), (s100, "s_h 100 learnable", "#5aae61", "-"),
                      (t["trace"]["median_error"], "s_h 60 learnable, seed 0 (trace run)", "#762a83", "-"),
                      (t["frozen"]["median_error"], "s_h 60 frozen", "#c2a5cf", "--"),
                      (t["free"]["median_error"], "s_h 30, log s_h moved 30x faster", "#e08214", ":")):
    ax[0].plot(np.arange(1, len(y) + 1), y, ls, color=c, lw=1.6, label=lab)
ax[0].axhline(90 / 7, color="grey", lw=0.8); ax[0].text(2, 90 / 7 * 1.08, "guessed turn 12.9 deg", color="grey", fontsize=8)
ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("median rotation error (deg)")
ax[0].set_title("7-gon bottom view, 14x14 map, 368 sphere points"); ax[0].legend(fontsize=7.5)
for k, (run, title) in enumerate((("trace", "trace run: start 60, learnable"), ("free", "start 30, log s_h moved 30x faster"))):
    e = np.array(t[run]["epochs"]); s = np.array(t[run]["s"]).reshape(len(e), -1)
    a = ax[k + 1]
    for h in range(s.shape[1]): a.plot(e, s[:, h], color="#999999", lw=0.6, alpha=0.7)
    a.plot(e, np.median(s, 1), color="black", lw=2, label="median of 32 heads")
    err = np.array(t[run]["median_error"]); jump = [i + 1 for i in range(1, len(err)) if err[i - 1] / err[i] > 1.3]
    for j in jump: a.axvline(j, color="#d6604d", lw=1, ls="--")
    a.set_yscale("log"); a.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()); a.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    a.tick_params(axis="y", which="minor", labelsize=7); a.set_xlabel("epoch"); a.set_ylabel("s_h (per head and block)")
    a.set_title(title + "\n(red: epochs where the error drops by >30%)", fontsize=10); a.legend(fontsize=8)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../reports/figures/cosdir_bottom7_s_h_trace.png")
fig.savefig(out, dpi=130); print(out)
