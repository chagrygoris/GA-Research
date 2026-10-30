"""Which SO(3) irreps the pose density of a symmetric shape needs: the degrees l whose Fourier block vanishes.

Setting. A shape with rotation symmetry group G (a finite set of rotations, or SO(2) / O(2) for an ideal surface of
revolution) seen at pose R0 has the pose density
    p(R) = (1 / |G|) sum_{g in G} delta(R, R0 g),
an equal point mass on every equivalent pose R0 g (SYMSOL's labels). Its Fourier block of degree l is
    c^l = (1 / |G|) sum_g D^l(R0 g) = D^l(R0) P^l,      P^l = (1 / |G|) sum_g D^l(g),
with D^l the (2l+1) x (2l+1) real Wigner matrix (e3nn's convention). P^l is the orthogonal projector onto the
G-invariant vectors of the degree-l irrep, so for every view R0 the block c^l is zero iff P^l = 0, and otherwise has
rank n_l(G) = rank P^l = |c^l|_F^2: the number of degree-l invariants of G. The view only rotates the block.

n_l is computed two ways:
  * Wigner: P^l from the group elements with models/so3fourier_head.wigner_blocks (l <= --lmax_wigner);
  * characters: n_l = (1 / |G|) sum_g chi_l(theta_g), chi_l(theta) = 1 + 2 sum_{m=1..l} cos(m theta), theta_g the
    rotation angle of g (exact, cheap, any l). For SO(2): n_l = 1; for O(2): n_l = 1 for even l, 0 for odd l.
The two agree on every shape (the script asserts it). Groups of the SYMSOL shapes come from datasets/symsol_points.py
(cone and cylinder at 1 degree, 360 / 720 elements, as in SYMSOL's own labels); "pyr<n>" is a regular n-gonal pyramid
(apex on e3: the cyclic group C_n, no end-over-end turn). Report: reports/symmetry_spectrum.md; figures and JSON go to
reports/figures/symmetry_spectrum/.

usage: python probes/symmetry_spectrum/symmetry_spectrum.py [--pyramid 1000] [--lmax 12] [--lmax_long 2200] [--out DIR]
"""
import argparse, json, math, os, sys

import numpy as np
import torch

sys.path.insert(0, os.environ.get("POSE3D_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from pose3d.datasets.symsol_points import _axis_angle, _polyhedra, _revolution_groups  # noqa: E402
from pose3d.models.so3fourier_head import harmonic_basis, wigner_blocks  # noqa: E402


def pyramid_group(n):
    """C_n: the n turns of a regular n-gonal pyramid about its axis e3."""
    return np.stack([_axis_angle([0, 0, 1], 2 * math.pi * k / n) for k in range(n)])


def groups(pyramid):
    _, g = _polyhedra()
    g.update(_revolution_groups())
    g[f"pyr{pyramid}"] = pyramid_group(pyramid)
    return g


def rotation_angles(G):
    return np.arccos(np.clip((np.trace(G, axis1=1, axis2=2) - 1) / 2, -1, 1))


def n_from_characters(G, lmax):
    """n_l = mean over g of chi_l(theta_g), l = 0..lmax, exact up to rounding."""
    m = np.arange(1, lmax + 1)
    cos = np.cos(np.outer(rotation_angles(G), m))                     # (|G|, lmax)
    chi = 1 + 2 * np.concatenate([np.zeros((len(G), 1)), np.cumsum(cos, 1)], 1)
    n = chi.mean(0)
    assert np.abs(n - np.round(n)).max() < 1e-6, "character average is not an integer"
    return np.round(n).astype(int)


def n_from_wigner(G, lmax, n_points=1024):
    """rank P^l and |P^l - (P^l)^2|_max (P^l must be a projector), l = 0..lmax."""
    basis = harmonic_basis(lmax, n_points)                          # Fibonacci points: enough for l <= 12 at 1024
    D = wigner_blocks(torch.from_numpy(G), lmax, basis)
    out, err = [], 0.0
    for Dl in D:
        P = Dl.mean(0)
        err = max(err, float((P - P @ P).abs().max()))
        out.append(int(round(float((P * P).sum()))))                 # |P|_F^2 = trace P = rank for a projector
    return np.array(out), err


def ideal(name, lmax):
    l = np.arange(lmax + 1)
    return np.ones_like(l) if name == "SO(2)" else (l % 2 == 0).astype(int)


def lowest_nonzero(n):
    nz = np.nonzero(n[1:])[0]
    return int(nz[0]) + 1 if len(nz) else None


def plot(table, long, pyramid, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
    plt.rcParams.update({"font.size": 10, "text.color": ink, "axes.labelcolor": ink2, "xtick.color": ink2,
                         "ytick.color": ink2, "axes.edgecolor": grid, "figure.facecolor": surface,
                         "axes.facecolor": surface})

    # 1. heatmap: rows = shapes, columns = degree l, value n_l; zero blocks left empty
    names = list(table)
    n = np.array([table[k]["n"] for k in names])
    L = n.shape[1] - 1
    seq = ListedColormap([surface, "#9ec5f4", "#5598e7", "#256abf", "#104281"])  # blue ramp, 0 = surface
    fig, ax = plt.subplots(figsize=(8.6, 0.48 * len(names) + 1.5))
    ax.imshow(np.minimum(n, 4), cmap=seq, vmin=-0.5, vmax=4.5, aspect="auto")
    for i in range(len(names)):
        for l in range(L + 1):
            ax.text(l, i, str(n[i, l]) if n[i, l] else "·", ha="center", va="center", fontsize=9,
                    color=(surface if n[i, l] >= 2 else ink) if n[i, l] else ink2)
    ax.set_xticks(range(L + 1))
    ax.set_yticks(range(len(names)), [table[k]["label"] for k in names])
    ax.set_xticks(np.arange(-0.5, L + 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names)), minor=True)
    ax.grid(which="minor", color=surface, linewidth=2)
    ax.tick_params(which="both", length=0)
    ax.set_xlabel("degree l of the Wigner-D irrep")
    ax.set_title("Invariants per degree, n_l = rank of the degree-l block of the pose density  (· = block is zero)",
                 loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "symmetry_spectrum.png"), dpi=160)
    plt.close(fig)

    # 2. the pyramid against the cone, far up in l
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    l = np.arange(len(long["cone_ideal"]))
    ax.plot(l, long["cone_ideal"], color=ink2, lw=1.5, ls=(0, (4, 3)), label="ideal cone, SO(2)")
    ax.plot(l, long["cone"], color="#eb6834", lw=2, label="SYMSOL cone (1° labels), C_360")
    ax.plot(l, long[f"pyr{pyramid}"], color="#2a78d6", lw=2, label=f"{pyramid}-gonal pyramid, C_{pyramid}")
    ax.set_xlim(0, l[-1])
    ax.set_ylim(0, max(max(v) for v in long.values()) + 1)
    ax.set_xlabel("degree l")
    ax.set_ylabel("n_l (invariants per degree)")
    ax.grid(axis="y", color=grid, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for x, txt in ((360, "l = 360"), (pyramid, f"l = {pyramid}")):
        ax.axvline(x, color=grid, lw=1)
        ax.text(x + 15, 0.3, txt, color=ink2, fontsize=9)
    ax.text(15, 1.25, "all three: n_l = 1 (only m = 0)", color=ink2, fontsize=9)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title("Where a discrete axial symmetry first differs from the continuous one", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pyramid_vs_cone.png"), dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyramid", type=int, default=1000, help="number of base edges of the extra pyramid")
    ap.add_argument("--lmax", type=int, default=12, help="degrees in the table and the heatmap")
    ap.add_argument("--lmax_wigner", type=int, default=12, help="degrees checked with explicit Wigner matrices")
    ap.add_argument("--lmax_long", type=int, default=2200, help="degrees in the pyramid-vs-cone plot")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reports", "figures",
                                                  "symmetry_spectrum"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    G = groups(args.pyramid)
    labels = {"tet": "tetrahedron (T, 12)", "cube": "cube (O, 24)", "icosa": "icosahedron (I, 60)",
              "cone": "cone (C_360, SYMSOL)", "cyl": "cylinder (D_360, SYMSOL)",
              f"pyr{args.pyramid}": f"{args.pyramid}-gon pyramid (C_{args.pyramid})"}
    table = {}
    for name, g in G.items():
        n = n_from_characters(g, args.lmax)
        nw, err = n_from_wigner(g, min(args.lmax, args.lmax_wigner))
        assert (nw == n[:len(nw)]).all(), f"{name}: Wigner {nw} != characters {n[:len(nw)]}"
        assert err < 1e-6, f"{name}: P^l is not a projector ({err:.1e})"
        table[name] = {"label": labels[name], "order": len(g), "n": n.tolist(), "lowest_nonzero_l": lowest_nonzero(n),
                       "projector_error": err}
    for name, lab in (("SO(2)", "ideal cone (SO(2))"), ("O(2)", "ideal cylinder (O(2))")):
        n = ideal(name, args.lmax)
        table[name] = {"label": lab, "order": None, "n": n.tolist(), "lowest_nonzero_l": lowest_nonzero(n)}

    long = {"cone_ideal": ideal("SO(2)", args.lmax_long).tolist(),
            "cone": n_from_characters(G["cone"], args.lmax_long).tolist(),
            f"pyr{args.pyramid}": n_from_characters(G[f"pyr{args.pyramid}"], args.lmax_long).tolist()}
    first_diff = {k: int(np.nonzero(np.array(v) != 1)[0][0]) for k, v in long.items() if k != "cone_ideal"}

    print(f"{'shape':34s} |G|   lowest l>0   n_l, l = 0..{args.lmax}")
    for k, v in table.items():
        print(f"{v['label']:34s} {str(v['order'] or 'inf'):5s} {str(v['lowest_nonzero_l']):12s} {v['n']}")
    print("first l where n_l differs from the ideal cone:", first_diff)

    order = ["tet", "cube", "icosa", "cone", "cyl", f"pyr{args.pyramid}"]
    plot({k: table[k] for k in order}, long, args.pyramid, args.out)
    json.dump({"table": table, "first_l_differing_from_SO2": first_diff, "args": vars(args)},
              open(os.path.join(args.out, "symmetry_spectrum.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
