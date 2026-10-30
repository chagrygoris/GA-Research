"""The expected Wigner-D target of a symmetric shape, and what its truncation at degree L can decode.

Setting (as symmetry_spectrum.py). A view of a shape with symmetry group G at pose R0 has the equivalent poses R0 g.
A regressor trained with an MSE on D^l(label) and a uniformly drawn equivalent label (models/wignerd.py, Lee & Cho
2024) converges to the expected Wigner-D target
    c^l = E_g[D^l(R0 g)] = (1 / |G|) sum_g D^l(R0 g) = D^l(R0) P^l,      P^l = (1 / |G|) sum_g D^l(g).
Its inference, like wignerd.predict, is the grid rotation maximising the score
    f_L(R) = sum_{l <= L} <c^l, D^l(R)>          (<A, B> = sum_ij A_ij B_ij, all degrees weighted 1).

Measured, per shape (V random views R0):
  1. size of the target per degree, |c^l|_F / |D^l(R0)|_F = sqrt(n_l / (2l + 1)): 1 for a single pose (G = {1}),
     0 when the degree carries nothing (n_l = rank P^l, symmetry_spectrum.py);
  2. oracle decoding: R_hat_L = argmax over the grid of f_L, error = geodesic distance from R_hat_L to the nearest
     equivalent pose, median over views. If f_L is constant (n_l = 0 for 1 <= l <= L) every rotation ties and the
     decoded pose is a uniform guess: its median error is reported instead (open markers);
  3. margin_L = min_g f_L(R0 g) - max { f_L(R) : R on the grid, farther than --far_deg from every equivalent }.
     margin_L > 0 iff the degrees <= L put every equivalent pose above every far rotation. At the equivalents
     f_L(R0 g) = sum_{l <= L} n_l exactly (tr P^l D^l(g) = tr P^l), so the margin is that minus the far maximum;
     it is 0 when f_L is constant. Median over views.
Grid: --grid super-Fibonacci rotations (Alexa 2022), near-uniform on SO(3); 294,912 is the size of Image2Sphere's
HEALPix level 4 (about 3.7 deg spacing). Report: reports/symmetry_spectrum.md.

usage: python probes/symmetry_spectrum/symmetry_decoding.py [--lmax 10] [--views 50] [--grid 294912] [--out DIR]
"""
import argparse, json, math, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from symmetry_spectrum import groups  # noqa: E402  (also puts pose3d on the path)
from pose3d.models.so3fourier_head import _float64_default  # noqa: E402


def _o3():
    from e3nn import o3
    return o3


def super_fibonacci(n):
    """n near-uniform unit quaternions (Alexa, "Super-Fibonacci Spirals", CVPR 2022), (n, 4) as (w, x, y, z)."""
    phi, psi = math.sqrt(2.0), 1.533751168755204288118041
    s = torch.arange(n, dtype=torch.float64) + 0.5
    r, R = (s / n).sqrt(), (1 - s / n).sqrt()
    a, b = 2 * math.pi * s / phi, 2 * math.pi * s / psi
    return torch.stack([r * a.sin(), r * a.cos(), R * b.sin(), R * b.cos()], -1)


def wigner(R, lmax):
    """[D^l(R)]_{l <= lmax}, each (B, 2l+1, 2l+1), e3nn's convention (as models/so3fourier_head.wigner_blocks)."""
    o3 = _o3()
    with _float64_default():                                                   # e3nn's constants, else 1e-7 errors
        a, b, g = o3.matrix_to_angles(R)
        return [o3.wigner_D(l, a, b, g) for l in range(lmax + 1)]


def nearest_equivalent_deg(q, q_eq):
    """Angle (deg) from each rotation q (n, 4) to the nearest of q_eq (k, 4): 2 acos max |<q, q_eq>|."""
    dot = (q @ q_eq.T).abs().amax(1).clamp(max=1.0)
    return torch.rad2deg(2 * dot.acos())


def run_shape(G, Dgrid, qgrid, args):
    o3 = _o3()
    L, V = args.lmax, args.views
    Gt = torch.from_numpy(G)
    P = [d.mean(0) for d in wigner(Gt, L)]                                     # projectors P^l
    n_l = np.array([int(round(float(p.trace()))) for p in P])
    R0 = o3.rand_matrix(V, dtype=torch.float64)
    C = [d @ p for d, p in zip(wigner(R0, L), P)]                              # c^l = D^l(R0) P^l, (V, 2l+1, 2l+1)
    size = np.array([float((c[0] ** 2).sum().sqrt()) / math.sqrt(2 * l + 1) for l, c in enumerate(C)])
    assert np.allclose(size, np.sqrt(n_l / (2 * np.arange(L + 1) + 1)), atol=1e-8)
    q_eq = o3.matrix_to_quaternion(torch.einsum("vij,gjk->vgik", R0, Gt))     # (V, |G|, 4)

    best = torch.full((L + 1, V), -math.inf, dtype=torch.float64)
    arg = torch.zeros((L + 1, V), dtype=torch.long)
    far_max = torch.full((L + 1, V), -math.inf, dtype=torch.float64)
    for start in range(0, len(qgrid), args.chunk):
        stop = min(start + args.chunk, len(qgrid))
        far = torch.stack([nearest_equivalent_deg(qgrid[start:stop], q_eq[v]) > args.far_deg for v in range(V)])
        score = torch.zeros(V, stop - start, dtype=torch.float64)
        for l in range(L + 1):
            score += C[l].reshape(V, -1) @ Dgrid[l][start:stop].reshape(stop - start, -1).T
            m, i = score.max(1)
            upd = m > best[l]
            best[l] = torch.where(upd, m, best[l])
            arg[l] = torch.where(upd, i + start, arg[l])
            far_max[l] = torch.maximum(far_max[l], score.masked_fill(~far, -math.inf).max(1).values)

    informative = np.cumsum(n_l[1:]) > 0                                       # f_L non-constant, L = 1..lmax
    informative = np.concatenate([[False], informative])
    # a uniform guess: its error has the same law for every view, so one view and 20,000 random rotations do
    uniform = float(nearest_equivalent_deg(o3.matrix_to_quaternion(o3.rand_matrix(20000, dtype=torch.float64)),
                                           q_eq[0]).median())
    err, margin = [], []
    for l in range(L + 1):
        if not informative[l]:
            err.append(uniform)
            margin.append(0.0)
            continue
        e = torch.stack([nearest_equivalent_deg(qgrid[arg[l, v]][None], q_eq[v]) for v in range(V)]).flatten()
        err.append(float(e.median()))
        margin.append(float((n_l[:l + 1].sum() - far_max[l]).median()))
    return {"order": len(G), "n": n_l.tolist(), "size": size.tolist(), "informative": informative.tolist(),
            "median_error_deg": err, "margin": margin, "uniform_guess_deg": uniform}


def plot(res, labels, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
    series = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    plt.rcParams.update({"font.size": 10, "text.color": ink, "axes.labelcolor": ink2, "xtick.color": ink2,
                         "ytick.color": ink2, "axes.edgecolor": grid, "figure.facecolor": surface,
                         "axes.facecolor": surface})
    names = list(res)
    l = np.arange(args.lmax + 1)

    fig, axes = plt.subplots(1, len(names), figsize=(2.6 * len(names), 2.9), sharey=True)
    for ax, k in zip(axes, names):
        ax.bar(l, res[k]["size"], width=0.75, color="#2a78d6")
        ax.set_title(labels[k], fontsize=10)
        ax.set_xticks(l)
        ax.tick_params(labelsize=8, length=0)
        ax.set_xlabel("degree l")
        ax.grid(axis="y", color=grid, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel(r"$\|c^l\|_F\,/\,\sqrt{2l+1}$")
    axes[0].set_ylim(0, 1.05)
    fig.suptitle("Expected Wigner-D target per degree  (1 = a single pose, 0 = the degree carries nothing)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "wigner_target_per_degree.png"), dpi=160)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4))
    for k, c in zip(names, series):
        inf = np.array(res[k]["informative"])
        for ax, key in ((a1, "median_error_deg"), (a2, "margin")):
            y = np.array(res[k][key])
            ax.plot(l, y, color=c, lw=2, label=labels[k], ls=(0, (4, 2)) if k.startswith("pyr") else "-")
            ax.scatter(l[inf], y[inf], s=34, color=c, edgecolor=surface, linewidth=1.5, zorder=3)
            ax.scatter(l[~inf], y[~inf], s=34, facecolor=surface, edgecolor=c, linewidth=1.5, zorder=3)
    a1.axhline(15, color=ink2, lw=1, ls=(0, (4, 3)))
    a1.text(args.lmax, 16, "15°", color=ink2, fontsize=9, ha="right")
    a1.set_ylabel("median error of the argmax (deg)")
    a1.set_title("Oracle decoding of the exact target truncated at L", loc="left", fontsize=10)
    a2.axhline(0, color=ink2, lw=1, ls=(0, (4, 3)))
    a2.set_ylabel(f"margin (> 0: equivalents above all poses > {args.far_deg:g}° away)")
    a2.set_title("Are degrees ≤ L enough to single out the equivalent poses?", loc="left", fontsize=10)
    for ax in (a1, a2):
        ax.set_xticks(l)
        ax.set_xlabel("truncation degree L")
        ax.grid(axis="y", color=grid, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    a1.legend(frameon=False, fontsize=9, title="open marker: target constant, uniform guess\n"
               "pyramid (dashed) and cone coincide", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "truncated_decoding.png"), dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmax", type=int, default=10)
    ap.add_argument("--views", type=int, default=50, help="random views R0 per shape")
    ap.add_argument("--grid", type=int, default=294912, help="super-Fibonacci grid size")
    ap.add_argument("--far_deg", type=float, default=15.0, help="'far' = farther than this from every equivalent")
    ap.add_argument("--pyramid", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reports", "figures",
                                                  "symmetry_spectrum"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    o3 = _o3()

    qgrid = super_fibonacci(args.grid)
    Dgrid = wigner(o3.quaternion_to_matrix(qgrid), args.lmax)
    G = groups(args.pyramid)
    labels = {"tet": "tetrahedron (12)", "cube": "cube (24)", "icosa": "icosahedron (60)", "cone": "cone (360)",
              "cyl": "cylinder (720)", f"pyr{args.pyramid}": f"{args.pyramid}-gon pyramid ({args.pyramid})"}
    res = {}
    for k in labels:
        res[k] = run_shape(G[k], Dgrid, qgrid, args)
        print(f"{labels[k]:24s} error {np.round(res[k]['median_error_deg'], 1).tolist()}")
        print(f"{'':24s} margin {np.round(res[k]['margin'], 3).tolist()}")
    plot(res, labels, args)
    json.dump({"shapes": res, "args": vars(args)}, open(os.path.join(args.out, "truncated_decoding.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
