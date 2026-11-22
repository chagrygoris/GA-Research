"""3D picture of the von Mises-Fisher window logic from the arrays of viz3d_compute.py (env PANELS="label|file.npz;...",
OUT). Per model (row):
  left:  the image tokens on the hemisphere (the optical axis points up), coloured by the turn-following feature: a
         learned model shows an n-lobed ring that turns with the object;
  right: the ring-reading head's attention window (the pose token's weights over the image tokens, a soft spotlight of
         width set by s_h) for pose hypotheses turned by 0, 1/4, 1/2, 3/4 of the period 360/n, with an arrow to the
         direction each window picks out; the window is fixed in the object's frame, so it slides along the ring with
         the hypothesis and samples the lobes."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

SLIDE = ["#2a78d6", "#1baf7a", "#eda100", "#e34948"]

def sphere(ax):
    u, v = np.mgrid[0:2 * np.pi:60j, 0:np.pi / 2:20j]
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="#d9d8d3", alpha=0.10, linewidth=0, shade=False)
    for z in (np.cos(np.radians(30)), np.cos(np.radians(60))):     # rings at 30 and 60 deg from the axis
        t = np.linspace(0, 2 * np.pi, 100); r = np.sqrt(1 - z * z)
        ax.plot(r * np.cos(t), r * np.sin(t), z, color="#bdbcb6", lw=0.6)
    t = np.linspace(0, 2 * np.pi, 100); ax.plot(np.cos(t), np.sin(t), 0 * t, color="#bdbcb6", lw=0.8)
    ax.quiver(0, 0, 0, 0, 0, 1.25, color="#52514e", lw=1, arrow_length_ratio=0.08)
    ax.text(0, 0, 1.32, "optical axis", fontsize=7, color="#52514e", ha="center")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 1.3)
    ax.set_box_aspect((2, 2, 1.3)); ax.view_init(elev=52, azim=-60); ax.set_axis_off()

def panel(axL, axR, label, f):
    d = np.load(f); p, dev, slide = d["pts"], d["dev"], d["slide"]; n = int(d["n"]); h, b = int(d["ring_head"]), int(d["ring_block"])
    v = np.abs(dev).max()
    sphere(axL)
    axL.scatter(p[:, 0], p[:, 1], p[:, 2], c=dev, cmap="RdBu_r", vmin=-v, vmax=v, s=10 + 60 * np.abs(dev) / v, depthshade=False, linewidths=0)
    axL.set_title(f"{label}\nturn-following feature on the tokens (frequency {n})", fontsize=9, loc="left")
    sphere(axR)
    axR.scatter(p[:, 0], p[:, 1], p[:, 2], c=dev, cmap="RdBu_r", vmin=-v, vmax=v, s=6 + 30 * np.abs(dev) / v, alpha=0.3, depthshade=False, linewidths=0)
    for k in range(4):
        w = slide[k] / slide[k].max(); top = w > 0.15
        axR.scatter(p[top, 0], p[top, 1], p[top, 2], color=SLIDE[k], s=15 + 110 * w[top], alpha=0.85, edgecolors="white", depthshade=False, linewidths=0.5)
        c = (slide[k][:, None] * p).sum(0); c = c / np.linalg.norm(c)            # direction the window picks out
        axR.quiver(0, 0, 0, *c, color=SLIDE[k], lw=1.6, arrow_length_ratio=0.1)
    axR.set_title(f"head {h} of block {b} (s_h {float(d['s_h'][h]):.0f}): its window as the pose hypothesis turns\n"
                  f"by 0, 1/4, 1/2, 3/4 of {360 / n:.1f} deg (blue, green, yellow, red)", fontsize=9, loc="left")

if __name__ == "__main__":
    rows = [r.split("|") for r in os.environ["PANELS"].split(";")]
    fig = plt.figure(figsize=(12, 5.0 * len(rows)))
    for i, (lab, f) in enumerate(rows):
        panel(fig.add_subplot(len(rows), 2, 2 * i + 1, projection="3d"), fig.add_subplot(len(rows), 2, 2 * i + 2, projection="3d"), lab, f)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, wspace=0.0, hspace=0.06)
    fig.savefig(os.environ["OUT"], dpi=140, facecolor="white"); print(os.environ["OUT"])
