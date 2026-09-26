# References

Articles, code and websites that are useful for pose3d. Add anything you found helpful, one line
per entry: a link, then why it matters for this project. Put your name and the month at the end so
people know who to ask.

Format: `- [Title](link) — why it matters (name, YYYY-MM)`

## Benchmarks and baselines

- [Image to Sphere (I2S)](https://arxiv.org/abs/2302.13926) — Klee et al., ICLR 2023. Pascal3D+ 9.6° with synthetic data; the model our `i2s_real` wraps. (seed, 2026-09)
- [Implicit-PDF](https://arxiv.org/abs/2106.05965) — Murphy et al., ICML 2021. Pascal3D+ ~10.2°; basis of `image2pcd_ipdf` and `ipdf_resnet`. (seed, 2026-09)
- [Matrix Fisher for SO(3)](https://arxiv.org/abs/2006.09740) — Mohlin et al., NeurIPS 2020. Pascal3D+ 8.9° / 8.1° with synthetic data: the strongest classical number to beat. (seed, 2026-09)
- [Rotation Laplace](https://arxiv.org/abs/2303.01743) — Yin et al., ICLR 2023 (follow-up: [2305.10465](https://arxiv.org/abs/2305.10465)). Pascal3D+ 9.3°, ModelNet10-SO(3) 12.7°; not implemented here yet. (seed, 2026-09)
- Delving into Discrete Normalizing Flows on SO(3) — Liu et al. Source of the matrix Fisher checkpoint used by `fisher_prior` (RotationNormFlow repo, `github.com/PKU-EPIC/RotationNormFlow`). (seed, 2026-09)

## Flow matching and geometry

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — Lipman et al. The MSE-regression training our flow uses. (seed, 2026-09)
- [Flow Matching on General Geometries](https://arxiv.org/abs/2302.03660) — Chen and Lipman. Geodesic pre-metric on manifolds; what `geometry/flow.py` implements. (seed, 2026-09)
- Clifford Flows — Alesiani and Maruyama, NeurIPS 2024 ML4PhysicalSciences workshop. The CNF ancestor of our model; we swap its training for flow matching and condition on images. (seed, 2026-09)
- [Rectified Flow](https://arxiv.org/abs/2209.03003), Mean Flows for One-step Generative Modeling (Geng et al.), Generative Modeling via Drifting (Deng et al.) — candidates for one-step inference. (seed, 2026-09)

## Geometric algebra

- [Clifford Group Equivariant Neural Networks](https://arxiv.org/abs/2305.11141) — Ruhe et al., NeurIPS 2023. Upstream of the `clifford` layers we use. (seed, 2026-09)
- [GAFL: Geometric Algebra Flow Matching](https://arxiv.org/abs/2411.05238) — NeurIPS 2024. Same geodesic construction on protein frames; its Clifford Frame Attention is a candidate for our conditioning stack. (seed, 2026-09)
- [Calculation of spin group elements revisited](https://arxiv.org/abs/2412.02772) — Shirokov. Theorem 12 is our matrix to rotor conversion (`geometry/rotor.py`). (seed, 2026-09)
- Practical parameterization of rotations using the exponential map — Grassia. Source of the log and exp maps. (seed, 2026-09)

## Backbones and data

- [Depth Anything V2](https://arxiv.org/abs/2406.09414) — the `depth_anything` encoder and the point-cloud models. (seed, 2026-09)
- [Pascal3D+](https://cvgl.stanford.edu/projects/pascal3d.html) — the benchmark. RenderForCNN ([arXiv:1505.05641](https://arxiv.org/abs/1505.05641)) provides the synthetic images behind `use_synth`. (seed, 2026-09)

## Not yet added

Ideas for what belongs here: ModelNet10-SO(3) and SYMSOL benchmark papers, symmetric-object pose
estimation, equivariance testing, ODE solvers on manifolds. The README format may move to Typst
later; keep entries one line each so they convert easily.
