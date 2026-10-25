# Two-tower Clifford Flow (multivector track + STF tensor track) on SYMSOL I, 10 % of the data

Status: code, tests and runner are done; **no run has been launched** (each run is over the 30-minute limit of
`CLAUDE.md` and needs a human's approval; the container that wrote this has no Kaggle credentials either).
Design note written October 2026. Every number marked *estimate* is a calculation from earlier timings, not a
measurement on a GPU.

## 1. The question

Multivectors of Cl(3,0) (and of GATr's Cl(3,0,1)) hold only SO(3) degrees l = 0 and l = 1
(`ga_high_degree_irreps.md`): a GATr-based flow can reach a degree l >= 2 dependence on the rotor only by stacking
products. SYMSOL's tetrahedron, cube and icosahedron have 12, 24, 60 equivalent poses, a function on SO(3) with
sharp peaks, i.e. high degrees. **Does adding explicit degree 2..L content (L = 3 or 6) to the same flow change its
score under an unchanged protocol?** Arm A is the reference model (`clifford_flow`, GATr vector field), arm B is the
same model with `--stf_track`.

## 2. What was built

Symbols used below: **l** is a degree (an irreducible representation of SO(3) of dimension 2l + 1); **S** is a
tensor with l indices; **v**, **u** are vectors in R^3; **delta** is the 3 x 3 identity; **R** is a 3 x 3 rotation
matrix; **Q**, **g** are rotation matrices used as group elements; "(Q . S)" applies Q to every index of S:
(Q . S)_{i1..il} = Q_{i1 j1} ... Q_{il jl} S_{j1..jl}.

* **STF tensor** (`models/stf_track.py`): a rank-l tensor that is symmetric under every index permutation and has
  zero trace over every index pair. It has 2l + 1 independent components and transforms as the degree-l
  irrep. Stored full, as 3^l floats (729 at l = 6), plus `n_components(l)` and an orthonormal coordinate
  system `stf_basis(l)` (found by SVD of many STF_l(v, ..., v); the SVD must show exactly 2l + 1 directions, which
  also checks the recursion).
* **The recursion** `stf_next(S, v)`: for an STF tensor S of rank l - 1 and a vector v, the rank-l STF tensor

  `STF_l(S, v) = Sym(S (x) v) - c_l * Sym(delta (x) (S . v))`, with `c_l = (l - 1) / (2 l - 1)`,

  where `(S . v)_{i1..i(l-2)} = S_{i1..i(l-2) k} v_k`, `delta (x) A` puts the identity on two of the l indices
  and `Sym` averages over index permutations (S is symmetric, so this is l terms for `Sym(S (x) v)` and
  l (l - 1) / 2 terms for the delta part). Why c_l: `Sym(S (x) v)` has trace (2 / l) (S . v) over any index pair
  (S is traceless, so only the terms with v on a traced index survive), while `Sym(delta (x) A)` for a traceless A
  has trace 2 (2 l - 1) / (l (l - 1)) A; the quotient is c_l. `stf2(u, v) = (u v^T + v u^T) / 2 - (u . v) delta / 3`
  is l = 2; l = 3 is CliffordSTF's STF_3 (their 1/5 * 2/3 is c_3 / 3). Tested for l = 2..6 in float64:
  symmetric, traceless over every index pair (< 1e-13), SO(3)-equivariant, parity (-1)^l, and equal to the
  paper's explicit STF_3 formula.
* `contract_down(S, v) = S . v`, rank l -> l - 1 (CliffordSTF's downward coupling). **Deviation from the brief:**
  the result is *not* only symmetric, it is also exactly traceless whenever S is (its trace over two free indices
  is a trace of S times v), so no re-projection is needed; tested for l = 2..6.
* `rotate_to_frame(S, R) = R^T . S`, S in the body frame of R. Tested: rotating S and R jointly from the left,
  S -> g . S, R -> g R, leaves it unchanged, and `R . (R^T . S) = S`.
* `NormGate`: S times sigmoid(W [log(1 + ||S_c||_F), LayerNorm(scalars)] + b) per channel c, an invariant gate
  (norm gating as in CliffordSTF). Linear mixing over the channel axis is equivariant, as all channels of one rank
  transform alike.
* `StfTower(n_vec, lmax, channels, n_scalars)`: input the vector channels V (B, n_vec, 3). *Upward:* `lmax` learned
  linear mixes of V give vector sets g_0, g_1, ... (B, channels, 3), each squashed g / (1 + |g|) so that |g| < 1
  (the l-fold product of vectors scales like |g|^l; without the squash rank 6 would explode or vanish);
  S_2 = stf2(g_0, g_1), S_l = stf_next(S_{l-1}, g_{l-1}); after each rank a channel mix and a `NormGate`.
  *Downward (cross-track coupling):* each S_l is contracted l - 1 times with the vector set that built it, giving
  one vector per channel; a learned linear map takes `channels` vectors to `n_vec` vectors; summed over ranks and
  multiplied by the learned scalar `feedback_scale` (init 0.1). Tested: all outputs equivariant to float64
  rounding for lmax = 2, 3, 6, every parameter gets a gradient.
* **Wiring, `FlowConfig.stf_track` (default off)** in `CliffordFlow._couple` / `velocity`:
  1. *Condition side.* The grade-1 parts of the `n_cond_mv` = 64 condition multivectors (B, 64, 3) go into the
     tower; its grade-0/grade-3 parts are the tower's invariant `scalars`. The feedback vectors are added to the
     grade-1 parts (a residual). The condition is now an `StfCondition(mv, stf)`: the multivectors plus the
     tensors S_2..S_L, each (B, `stf_channels`, 3, .., 3). `StfCondition.repeat_interleave` makes the time-sample
     batching, the multi-sample `predict` and the SYMSOL likelihood work unchanged.
  2. *Vector-field side.* For the current rotor, `rotor_to_matrix` gives R; `stf_invariants(S, R, mode)` turns
     the tensors into bounded invariant scalars; `StfEmbed` (one Linear + SiLU) maps them to `gatr_s_channels`
     scalar channels written onto the pose token (zero on every other token), exactly as GATr gets the query
     flag; the GATr input has that many more scalar channels. The velocity read-out is unchanged. This is where
     the velocity acquires a degree-l dependence on the rotor: the injected numbers are polynomials of degree l in
     the entries of R.
  3. `torch.compile`: the tower and the invariants run eagerly outside the compiled GATr module; the module takes
     the invariants as one more tensor argument. Checked on CPU: the compiled two-tower vector field equals the
     eager one (`test_compiled_two_tower_vector_field_matches_eager`). Not checked on a GPU.
* **Two injection modes, `--stf_inject gram|frame`** (default `gram`). This is a **deviation from the brief**, and
  the reason is the equivariance test the brief asks for. GATr acts on every token by the sandwich g x g^-1, so on the
  rotor token it is *conjugation*, R -> g R g^T, with the condition vectors going to g v (the repo's own
  `_frame_velocity` docstring says the same). The brief's body-frame components of R^T . S are invariant when S
  and R are rotated *from the left* (S -> g . S, R -> g R), but under conjugation they transform like a tensor
  (R^T . S -> g . (R^T . S)), so injecting them as scalars breaks the symmetry the control model has. I verified the
  control model's symmetry numerically (conjugating the condition multivectors and the rotor by one rotor g
  rotates the velocity by g; `test_control_equivariance_reference`). The default `gram` mode builds from
  unit-normalised tensors s_c = S_c / (||S_c||_F + 1e-4) the numbers `<R^T . s_c, s_c'>` over channel pairs
  (c, c') of every rank (`channels^2` per rank) plus `0.1 log ||S_c||_F`: invariant under conjugation, still
  degree l in R (they are matrix elements of the Wigner matrix D^l(R) between two learned vectors in the
  degree-l space). `frame` is the brief's version (components of R^T . s_c in `stf_basis` coordinates,
  (2l + 1) per channel and rank): more absolute information, left-invariant, not conjugation-equivariant (tested
  to *fail* the control's equivariance test, as expected). Note the team's `--gatr_pose_harmonics` is also a
  left-invariant "frame" construction; `frame` is the STF counterpart of it, and the two can be combined.
* **Not supported (clear error):** `--vector_field clifford` (the Clifford-MLP field; injecting there was not cheap),
  `--mlp_heads`, `--fisher_prior`, and `--cond_tokens` / `--pose_tokens` other than the defaults `pooled` / `rotor`
  (the tensors are built from the pooled condition multivectors and injected at the single rotor token).
  `--condition_head clifford|gatr` both work. `evaluate.py` rebuilds two-tower checkpoints.

Flags (all in `FlowConfig`): `stf_track` (False), `stf_lmax` (3, allowed 2..6), `stf_channels` (8),
`stf_inject` (`gram`).

### Verification (CPU, torch 2.9.0, 3.13)

* `tests/test_stf_track.py` (44 tests): the algebra above for l = 2..6, the SVD dimension counts, tower and gate
  equivariance, invariance of `gram` under conjugation and of `frame` under left rotation, and a spectral check
  that `<S, R_z(theta) . S>` has angular frequency l (and nothing above it).
* `tests/test_stf_flow.py` (14 tests): control and two-tower models (lmax 2, 3, 6) both satisfy
  |v(g x g^-1) - g v g^-1| < 1e-4 (float32); `frame` does not; STF scalars change the velocity; training step,
  `predict`, the SYMSOL nearest-equivalent evaluation and the ODE log-likelihood run; the zero-initialised
  read-out gives zero velocity as before; compile equals eager; flag plumbing and errors.
* **Off means unchanged:** with `stf_track` off the model has the same 24,433,669 parameters, state-dict hash,
  `compute_loss` and `predict` for a fixed seed as the tree before any of this work (checked with a git worktree
  of the commit 6af9909; `/scratchpad` script, not kept).
* CPU smoke runs of the whole `python -m pose3d --dataset symsol --sanity_check` path on a synthetic SYMSOL folder
  (20 train / 10 test images; the branch's `tests/test_symsol.py` generator) for the control and for
  `--stf_track --stf_lmax 6`: training epoch, nearest-equivalent evaluation, 2-sample final evaluation, per-shape
  table and SYMSOL log-likelihood all complete.
* Whole test suite after merging the SYMSOL branch: 206 passed, 1 skipped, 1 pre-existing failure deselected
  (`tests/test_macro_metrics.py::test_metrics_record_eval_classes_without_giving_them_to_the_model`: a
  `SimpleNamespace` config without `features`; fails the same way on `main`).

### Size (non-backbone parameters, `--encoder resnet50`, reference sizes, not matched)

| model | non-backbone parameters |
|---|---|
| control | 1,133,689 |
| `--stf_track --stf_lmax 3` (8 channels) | 1,145,002 (+11,313, +1.0 %) |
| `--stf_track --stf_lmax 6` | 1,159,234 (+25,545, +2.3 %) |

## 3. Protocol (both arms identical; only the model flags differ)

The `symsol1_10pct_*_i2s_adamw` protocol of `reports/i2s_symsol1_10pct.md`, on the SYMSOL stack merged from
`claude/optimistic-faraday-6effbq` (the loader keeps every equivalent pose of a test image, the error is to the
**nearest equivalent**, the log-likelihood is the IPDF/Image2Sphere one): SYMSOL I, 4,500 of 45,000 training views
per shape (22,500 images; `--symsol_train_views 4500`), the first 1,000 test views per shape (5,000 images;
`--symsol_test_views 1000`), no augmentation, no EMA, AdamW lr 1e-4 x sqrt(64/32), 5 warmup epochs + cosine,
global batch 64, 40 epochs, one process (`--no-ddp`), `--encoder resnet50` (the encoder of Image2Sphere's SYMSOL
runs in that report), the reference flow recipe otherwise (Clifford MLP condition head, GATr vector field, 64
condition multivectors, 8 time samples per image, dense GATr linears, `torch.compile`, bf16 backbone, 32-sample
medoid final evaluation). Reported: per-epoch single-sample median nearest-equivalent error; at the end the
32-sample medoid median and Acc@15/30 per shape, and the log-likelihood (8 equivalents per image, 50 ODE steps).

Run commands (the runner `notebooks/symsol-stf-runner.ipynb` builds them; `RUN` in cell 0 picks one per kernel):

```bash
COMMON="--model clifford_flow --encoder resnet50 --dataset symsol --symsol_set 1 --symsol_train_views 4500 \
  --symsol_test_views 1000 --no-use_warp --no-use_synth --no-ema --fixed_val_noise --no-ddp --batch_size 64 \
  --n_epochs 40 --optimizer adamw --lr 1e-4 --lr_scaling sqrt --lr_schedule cosine --warmup_epochs 5 --num_workers 4"
python -m pose3d --path_to_datasets $DATA --run_name symsol1_10pct_gatr_control   $COMMON                       # A
python -m pose3d --path_to_datasets $DATA --run_name symsol1_10pct_gatr_stf_l3    $COMMON --stf_track --stf_lmax 3   # B
python -m pose3d --path_to_datasets $DATA --run_name symsol1_10pct_gatr_stf_l6    $COMMON --stf_track --stf_lmax 6   # B, optional
```

Optional extras in the runner: `stf_l3_frame` (`--stf_inject frame`), and `control_nc`, `stf_l3_nc` with
`--symsol_nearest_coupling`. The last two are there because `reports/symsol_p1_stall.md` found that GATr flows trained
from a uniform start rotation switch the polyhedra off (zero-predictor loss 5.29 vs 1.15 with coupling); a plain
control may therefore stall on tet/cube/icosa (log-likelihood ~ -2.29 there) and then a comparison without the
coupling says little. Decide after the first control epochs, or run the `_nc` pair as well.

### What the earlier reports already say (looked up, not re-run)

* No run of the plain reference model (`clifford_flow`, pooled condition, GATr field) under this protocol exists in
  `reports/`. The flow variants there use other heads: best `symsol1_10pct_gatr_spherecloud_harm6_i2s_adamw`
  (16.97 deg median, 19.88 deg mean of per-shape medians, Acc@15 0.48, log-likelihood 3.24),
  `..._spherecloud_small_...` (26.63 / 25.62, 1.24). P1-small stalls at log-likelihood 1.95 and DiT-small reaches
  3.92 (`symsol_p1_stall.md`, 100 epochs). Image2Sphere under the same protocol: 3.07 deg median, 3.21 deg mean of
  per-shape medians, Acc@15 1.0, log-likelihood 3.48 (AdamW), 4.23 / 4.51 / 2.55 with the paper's SGD recipe.
* The encoder of those runs: `--encoder resnet50` (ResNet-50 ImageNet weights; the offline dataset also ships
  ResNet-101). Both arms here use ResNet-50.

## 4. ETA (one process, `RUN` = one variant per kernel)

Inputs: 22,500 train images -> 352 steps per epoch at batch 64 (14,080 steps in 40 epochs); 5,000 scored test views
per epoch; timings of `USAGE.md` on one RTX Pro 6000 with the default speed flags: 80 ms per training step at batch
64, 0.44 s per 1-sample evaluation of 360 images, 3.41 s per 32-sample evaluation of 360 images (ResNet-101 and
EMA included; ResNet-50 without EMA should be somewhat faster, so 80 ms is conservative). Two-tower step time:
x1.3 for `stf_l3`, x1.5 for `stf_l6` (**assumption**: the brief's +30-50 %). Evidence for the assumption: the tower
works on the 64 images of a batch, the invariants on the 512 (image, time-sample) pairs; their FLOPs are small
(a CPU fwd+bwd of condition coupling + vector field at batch 64 x 8 measured +6 % for lmax 3 and +9 % for lmax 6),
but they are many small eager ops outside the compiled graph (an index movedim/sum per term of the recursion and
one matmul per index of every rank, forward and backward), so on a GPU launch overhead, not FLOPs, may decide;
I did not measure that.

| item | control (RTX) | `stf_l3` (RTX) | `stf_l6` (RTX) | T4 control / l3 / l6 |
|---|---|---|---|---|
| step | 80 ms | 104 ms | 120 ms | 400 / 520 / 600 ms (5x) |
| train time per epoch (352 steps) | 28 s | 37 s | 42 s | 141 / 183 / 211 s |
| per-epoch validation (val loss over 5,000 + 1-sample prediction) | 8 s | 11 s | 12 s | 41 / 54 / 62 s |
| 40 epochs | 24 min | 32 min | 36 min | 2.0 / 2.6 / 3.0 h |
| PNG decoding of 27,500 images (22,500 + 5,000), RAM cache | ~3 min | ~3 min | ~3 min | ~5 min |
| `torch.compile` warm-up | ~5 min | ~5 min | ~6 min | ~8 min |
| final 32-sample evaluation of 5,000 views | 47 s | 62 s | 71 s | 4 / 5 / 6 min |
| log-likelihood (8 equivalents x 50 steps x 5,000 images) | 5 min (3-10) | 8 min (5-15) | 10 min (6-18) | 25 / 40 / 50 min |
| imports, model build, startup | ~1 min | ~1 min | ~1 min | ~2 min |
| **total** | **~39 min** | **~50 min** | **~57 min** | **~2.7 / 3.6 / 4.2 h** |

How the pieces were obtained: validation = 79 forward-only batches at ~0.027 s (a third of a step) + 5000/360 x
0.44 s; final evaluation = 5000/360 x 3.41 s; log-likelihood = 5000 x 8 x 50 velocity evaluations with the
divergence taken by autograd (a guess of ~4x the cost of a plain evaluation of the compiled field, with the
STF arms x1.5-2); PNG decoding: measured 1.3 ms per random-noise 224 x 224 PNG on one core of this container
(threads did not help), real renders compress better, the rest is file reading from the mounted dataset; T4 = 5x
per step and per evaluation (the brief's rule of thumb; the log-likelihood uses 5x as well), no bf16 on a T4
(`_native_bf16`), so the T4 numbers are +-40 %. For orientation, Image2Sphere on a T4 took 280-300 s per epoch on
the same data. **Every run is over 30 minutes, so each launch needs a human's approval**; the three RTX runs together
are ~2.4 h, the T4 ones ~10.5 h. Monitoring per `CLAUDE.md`: look 15 minutes after the launch (is the dataset built,
is epoch 1 done: first epoch ~1 min after the ~8 min of decoding and compiling), then hourly.

## 5. Kaggle logistics (nothing uploaded or launched from here)

RTX Pro 6000 kernels have no internet; the notebook follows `i2s-symsol-runner.ipynb`'s offline mode.

1. **Repo snapshot:** `python pose3d/scripts/make_snapshot.py --out <dir>` writes `ga-research-<commit>.tar.gz`
   (tracked files of a clean checkout, 1.4 MB); upload as a Kaggle dataset and attach. The notebook refuses a
   snapshot without the two-tower model.
2. **Dependencies:** attach `syfry5suvzovvakmuj/ga-research-offline-deps` (clifford, e3nn, healpy, image2sphere,
   opt_einsum, ResNet weights) and `syfry5suvzovvakmuj/ga-research-gatr-deps` (the GATr wheel). Their contents were
   not inspected from here: the notebook installs every `*.whl` of both with `--no-index --no-deps`, copies every
   `resnet*-*.pth` into the torch hub cache and imports `gatr` to fail early.
3. **Data:** `syfry5suvzovvakmuj/symsol1-10pct-fulltest` (`symsol1_t4500_v5000/symsol/{train,test}/rotations.npz +
   images/`), found by glob. Only the first 4,500 train and 1,000 test views per shape are decoded.
4. **Launch:** set `RUN` in cell 0 (one variant per kernel), push with the pool's launcher after approval; W&B runs
   offline and `gpu_pool/rtx_monitor.py` replays the `WANDB_SYNC` lines.

Open items: the credentials and the approval to launch (whoever holds the pool tokens); a check that the two
dependency datasets contain the wheels listed above; the decision about the `_nc` variants after seeing the control.
Not done: any GPU measurement (step time of the STF arms, compile with the STF module on a GPU, memory), a
`--vector_field clifford` implementation, an STF4+ parity (O(3)) treatment (like CliffordSTF the tower is SO(3)
only).
