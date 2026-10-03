"""Visualize the flow-matching trajectories of a trained CliffordFlow on SO(3): the path from noise to a pose.

For a few validation images the flow is integrated from several random rotations (the Euler loop of CliffordFlow.predict, but every
intermediate rotation is kept). Each rotation R_t is shown relative to the ground truth, R_gt^T R_t, as an axis-angle vector inside the
ball of radius pi: the true pose is the origin, the distance from the origin is the geodesic error, and a path that ends near the origin
has found the pose. Opposite points of the ball's surface are the same rotation, so a path can jump across it (drawn as a break).

    python -m pose3d.visualize_flow --checkpoint a.pth --label GATr --checkpoint b.pth --label MLP \
        --val_cache pascal_val.pt --out viz/
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation

from pose3d.evaluate import build_model
from pose3d.geometry.flow import exp_map, rotor_multiply
from pose3d.geometry.rotor import random_rotor, rotor_to_matrix


def load(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model, _ = build_model(ckpt, "cpu")
    return model.eval()


def angle_deg(r_a, r_b):
    """Geodesic angle between rotation matrices (..., 3, 3), in degrees."""
    tr = np.einsum("...ji,...ji->...", r_a, r_b)          # trace(A^T B)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


@torch.no_grad()
def trajectories(model, img, k=24, steps=20, seed=0):
    """(steps + 1, k, 3, 3): k noise rotations integrated to the model's pose, every Euler step kept."""
    cond = model.condition(img[None]).repeat(k, 1, 1)
    torch.manual_seed(seed)
    rotor = random_rotor(k)
    path = [rotor_to_matrix(rotor, model.algebra)]
    dt = 1.0 / steps
    for i in range(steps):
        v = model.velocity(rotor, torch.full((k,), i * dt), cond)
        rotor = rotor_multiply(rotor, exp_map(dt * v), model.algebra)
        path.append(rotor_to_matrix(rotor, model.algebra))
    return torch.stack(path).numpy()


@torch.no_grad()
def errors(model, imgs, rots, n_samples=8, batch=16):
    out = []
    for i in range(0, len(imgs), batch):
        pred = model.predict(imgs[i:i + batch], n_samples=n_samples, steps=20).numpy()
        out.append(angle_deg(pred, rots[i:i + batch]))
    return np.concatenate(out)


def relative_vectors(path, gt):
    """Axis-angle vectors of R_gt^T R_t, shape (steps + 1, k, 3); the norm is the angle, at most pi."""
    rel = np.einsum("ji,tkjl->tkil", gt, path)
    return Rotation.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec().reshape(path.shape[:2] + (3,))


def draw_ball(ax):
    u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
    ax.plot_wireframe(np.pi * np.cos(u) * np.sin(v), np.pi * np.sin(u) * np.sin(v), np.pi * np.cos(v), color="0.8", lw=0.3, alpha=0.5)
    ax.scatter([0], [0], [0], marker="*", s=130, c="gold", edgecolors="k", zorder=10, label="true pose")
    ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi); ax.set_zlim(-np.pi, np.pi)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()


def draw_paths(ax, vec, cmap="viridis"):
    """Time-coloured paths; a jump across the ball's surface (antipodal identification) is left as a gap."""
    n_t, k, _ = vec.shape
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, n_t - 1))
    for j in range(k):
        seg = np.stack([vec[:-1, j], vec[1:, j]], axis=1)
        keep = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1) < np.pi
        ax.add_collection3d(Line3DCollection(seg[keep], colors=colors[keep], linewidths=1.2, alpha=0.85))
    ax.scatter(*vec[0].T, s=14, facecolors="none", edgecolors="0.35", linewidths=0.6)                      # noise
    final = np.linalg.norm(vec[-1], axis=1)
    ax.scatter(*vec[-1].T, s=22, c=np.where(np.degrees(final) < 15, "tab:green", "tab:red"), edgecolors="k", linewidths=0.3)


def triad(ax, rot, style, lw, alpha, label=None):
    for c, col in enumerate(("tab:red", "tab:green", "tab:blue")):
        ax.plot([0, rot[0, c]], [0, rot[1, c]], [0, rot[2, c]], color=col, ls=style, lw=lw, alpha=alpha, label=label if c == 0 else None)


def pick_images(model, imgs, rots, n_pool, quantiles):
    idx = np.linspace(0, len(imgs) - 1, n_pool).astype(int)
    err = errors(model, imgs[idx], rots[idx])
    order = np.argsort(err)
    chosen = [int(order[min(int(q * len(order)), len(order) - 1)]) for q in quantiles]
    return [int(idx[c]) for c in chosen], [float(err[c]) for c in chosen]


def figure_paths(label, model, imgs, rots, picks, errs, k, steps, seed, out):
    fig = plt.figure(figsize=(14, 3.7 * len(picks)))
    for r, (i, e) in enumerate(zip(picks, errs)):
        path = trajectories(model, imgs[i], k, steps, seed)
        vec = relative_vectors(path, rots[i])
        dist = np.linalg.norm(vec, axis=-1) * 180 / np.pi
        ax = fig.add_subplot(len(picks), 3, 3 * r + 1); ax.imshow(imgs[i].permute(1, 2, 0).numpy()); ax.axis("off")
        ax.set_title(f"val #{i}: medoid error {e:.1f}°", fontsize=10)
        ax3 = fig.add_subplot(len(picks), 3, 3 * r + 2, projection="3d"); draw_ball(ax3); draw_paths(ax3, vec)
        ax3.set_title("paths in the rotation ball (true pose = ★)", fontsize=10)
        axd = fig.add_subplot(len(picks), 3, 3 * r + 3)
        t = np.linspace(0, 1, dist.shape[0])
        axd.plot(t, dist, color="tab:blue", alpha=0.25, lw=0.9); axd.plot(t, np.median(dist, axis=1), color="k", lw=2, label="median of samples")
        axd.axhline(15, color="tab:green", ls="--", lw=1, label="15°"); axd.set_ylim(0, 180); axd.set_xlabel("flow time t"); axd.set_ylabel("angle to true pose (°)")
        axd.grid(alpha=0.3); axd.legend(fontsize=8, loc="upper right")
        axd.set_title(f"{(dist[-1] < 15).sum()}/{k} samples end within 15°", fontsize=10)
    fig.suptitle(f"{label}: flow from noise to pose on SO(3), {k} noise samples, {steps} Euler steps", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out, dpi=110); plt.close(fig)


def figure_filmstrip(label, model, imgs, rots, i, steps, seed, out, sample=0):
    path = trajectories(model, imgs[i], 8, steps, seed)
    times = [0, steps // 4, steps // 2, 3 * steps // 4, steps]
    fig = plt.figure(figsize=(3 * (len(times) + 1), 3.6))
    ax = fig.add_subplot(1, len(times) + 1, 1); ax.imshow(imgs[i].permute(1, 2, 0).numpy()); ax.axis("off"); ax.set_title(f"val #{i}", fontsize=10)
    gt = rots[i]
    for c, s in enumerate(times):
        a3 = fig.add_subplot(1, len(times) + 1, c + 2, projection="3d")
        triad(a3, gt, "--", 5, 0.25); triad(a3, path[s, sample], "-", 2.2, 1.0)
        a3.set_xlim(-1, 1); a3.set_ylim(-1, 1); a3.set_zlim(-1, 1); a3.set_box_aspect((1, 1, 1)); a3.view_init(22, 40)
        a3.set_xticks([]); a3.set_yticks([]); a3.set_zticks([])
        a3.set_title(f"t = {s / steps:.2f}  ({angle_deg(path[s, sample], gt):.0f}° off)", fontsize=10)
    fig.suptitle(f"{label}: one noise sample, rotating its frame (solid R,G,B axes) into the true pose (faint, wide axes)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(out, dpi=110); plt.close(fig)


def figure_compare(models, imgs, rots, picks, errs, k, steps, seed, out):
    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 3.8), sharey=True)
    t = np.linspace(0, 1, steps + 1)
    for c, (i, ax) in enumerate(zip(picks, np.atleast_1d(axes))):
        for (label, model), col in zip(models.items(), ("tab:orange", "tab:blue", "tab:purple")):
            vec = relative_vectors(trajectories(model, imgs[i], k, steps, seed), rots[i])      # same seed = same noise for every model
            d = np.linalg.norm(vec, axis=-1) * 180 / np.pi
            ax.fill_between(t, np.percentile(d, 25, axis=1), np.percentile(d, 75, axis=1), color=col, alpha=0.2)
            ax.plot(t, np.median(d, axis=1), color=col, lw=2.2, label=f"{label} (median, IQR)")
        ax.axhline(15, color="tab:green", ls="--", lw=1); ax.set_xlabel("flow time t"); ax.set_title(f"val #{i}", fontsize=10); ax.grid(alpha=0.3); ax.set_ylim(0, 180)
        if c == 0: ax.set_ylabel("angle to true pose (°)"); ax.legend(fontsize=8)
    fig.suptitle(f"Same images, same {k} noise rotations: how fast each model's flow converges", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(out, dpi=110); plt.close(fig)


def animation(label, model, imgs, rots, i, k, steps, seed, out):
    vec = relative_vectors(trajectories(model, imgs[i], k, steps, seed), rots[i])
    fig = plt.figure(figsize=(8, 4.3)); ax = fig.add_subplot(1, 2, 1); ax.imshow(imgs[i].permute(1, 2, 0).numpy()); ax.axis("off")
    a3 = fig.add_subplot(1, 2, 2, projection="3d"); draw_ball(a3)
    sc = a3.scatter(*vec[0].T, s=26, c="tab:blue", edgecolors="k", linewidths=0.3); lines = [a3.plot([], [], [], color="tab:blue", lw=0.8, alpha=0.5)[0] for _ in range(k)]
    title = fig.suptitle("", fontsize=11)
    def frame(f):
        sc._offsets3d = tuple(vec[f].T)
        sc.set_color(np.where(np.linalg.norm(vec[f], axis=1) < np.radians(15), "tab:green", "tab:blue"))
        for j, ln in enumerate(lines):
            seg = vec[:f + 1, j]
            ok = np.r_[True, np.linalg.norm(np.diff(seg, axis=0), axis=1) < np.pi]
            seg = np.where(ok[:, None], seg, np.nan); ln.set_data(seg[:, 0], seg[:, 1]); ln.set_3d_properties(seg[:, 2])
        a3.view_init(18, 30 + 90 * f / steps)
        title.set_text(f"{label}: t = {f / steps:.2f}, {int((np.linalg.norm(vec[f], axis=1) < np.radians(15)).sum())}/{k} within 15°")
        return [sc, *lines]
    FuncAnimation(fig, frame, frames=steps + 1, blit=False).save(out, writer=PillowWriter(fps=6)); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", action="append", required=True); p.add_argument("--label", action="append", required=True)
    p.add_argument("--val_cache", required=True, help="pascal_val.pt of the Pascal3D RAM cache"); p.add_argument("--out", default="viz")
    p.add_argument("--n_pool", type=int, default=96, help="validation images scored to choose the examples")
    p.add_argument("--quantiles", type=float, nargs="+", default=[0.1, 0.5, 0.8, 0.95], help="error quantiles of the examples (of the first model)")
    p.add_argument("--k", type=int, default=24); p.add_argument("--steps", type=int, default=20); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); assert len(a.checkpoint) == len(a.label)
    torch.set_num_threads(4); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    blob = torch.load(a.val_cache, map_location="cpu", weights_only=True)
    imgs, rots = blob["imgs"].float() / 255, blob["targets"].float().numpy()
    models = {lab: load(ck) for ck, lab in zip(a.checkpoint, a.label)}
    first = next(iter(models.values()))
    picks, errs = pick_images(first, imgs, rots, a.n_pool, a.quantiles)
    print("examples (val index, medoid error of", next(iter(models)), "):", list(zip(picks, [round(e, 1) for e in errs])), flush=True)
    for lab, m in models.items():
        e_m = errors(m, imgs[picks], rots[picks])
        figure_paths(lab, m, imgs, rots, picks, e_m, a.k, a.steps, a.seed, out / f"paths_{lab}.png")
        figure_filmstrip(lab, m, imgs, rots, picks[1], a.steps, a.seed, out / f"filmstrip_{lab}.png")
        animation(lab, m, imgs, rots, picks[1], a.k, a.steps, a.seed, out / f"flow_{lab}.gif")
        print("done", lab, flush=True)
    if len(models) > 1:
        figure_compare(models, imgs, rots, picks, errs, a.k, a.steps, a.seed, out / "compare_convergence.png")


if __name__ == "__main__":
    main()
