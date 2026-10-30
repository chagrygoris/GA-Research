# Symmetry spectrum

For a shape with rotation symmetry group `G`, which degrees `l` of the SO(3) Fourier expansion (Wigner-D irreps) its pose
density can use. The degree-`l` block of the density is `D^l(R0) P^l`, where `P^l` is the average of `D^l(g)` over `G`. That
block is zero for every view iff `P^l = 0`, and otherwise has rank `n_l`, the number of degree-`l` invariants of `G`.

| script | what it does |
|---|---|
| `symmetry_spectrum.py` | `n_l` for the SYMSOL I shapes (groups from `datasets/symsol_points.py`) and an n-gonal pyramid (`--pyramid`, default 1000), from Wigner matrices (`l <= 12`) and from characters (any `l`), asserting they agree; writes the heatmap, the pyramid-vs-cone plot and a JSON to `reports/figures/symmetry_spectrum/` |

```bash
python probes/symmetry_spectrum/symmetry_spectrum.py              # CPU, about 30 s
```

Results and the general case: [`reports/symmetry_spectrum.md`](../../reports/symmetry_spectrum.md).
