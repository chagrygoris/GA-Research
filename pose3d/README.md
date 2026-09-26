# pose3d

Predict an object's 3D rotation (SO(3)) from one RGB image on Pascal3D+. The main model, **Clifford
Flow**, is conditional flow matching on Spin(3) rotors built from geometric-algebra layers.
Reference result: **10.25°** median rotation error (W&B `k5sblpo8`).

```bash
cd pose3d && poetry install
poetry run python -m pose3d --path_to_datasets /path/to/data --run_name my-run
```

Install, flags, models and code layout: [`USAGE.md`](USAGE.md).
What has been tried so far: [`reports/2026-09-history.typ`](reports/2026-09-history.typ).
Reading material: [`references/`](references/README.md).

## Idea board

This table is the list of experiments for pose3d. Keep it up to date, in the same commit or pull
request as your work.

1. **Propose.** Add a row: next free ID, one line for the idea, your name under *Proposed by*,
   status `proposed`. Anyone can do this.
2. **Claim.** Put your name under *Developer* and set the status to `in progress`. Work on a branch
   named `idea-<ID>-<short-name>` off `main` and write it in *Branch*. One idea, one flag: the
   default must keep reproducing the reference run (see the root README).
3. **Report.** When you have results, copy `reports/_template.typ` to
   `reports/YYYY-MM-<slug>.typ`, write it up, link it under *Report* and set the final status.

| Status | Meaning | What happens on `main` |
|---|---|---|
| `proposed` | nobody works on it yet | nothing |
| `in progress` | has a developer and a branch | nothing |
| `adopted` | measured better than the reference | merged, flag `True` or new default |
| `experimental` | promising, not proven (one run, inside the noise, unfinished) | merged, flag `False` |
| `dropped` | worse, or not worth the complexity | not merged; the report says why |

The first rows were seeded from the summer report and the W\&B audit of 2026-09-26; take any of them.

| ID | Idea | Proposed by | Developer | Status | Branch | Report |
|---|---|---|---|---|---|---|
| 1 | Confirm the augmentation result: rerun the reference with `--use_warp` (and `--raw_cache`). The 9.71° run `lyqxhz1p` has unconfirmed args. | seed | | proposed | | |
| 2 | Train with RenderForCNN synthetic data (`--use_synth`). I2S and matrix Fisher both gain from it (matrix Fisher 8.9° to 8.1°). | seed | | proposed | | |
| 3 | Measure the run-to-run noise: several seeds of the reference, plus the four `_cmv64` reruns of 2026-09-26 (`t6vunfp8`, `pf8v440i`, `50nzm900`, `g41l2cbo`). | seed | | proposed | | |
| 4 | Cut parameters without losing accuracy: confirm `adapter_grid` 9, `adapter_channels` 96 and `vector_field_hidden_dim` (10.49° lead) once the noise is known. | seed | | proposed | | |
| 5 | Add Rotation Laplace as a baseline in this harness (published 9.3°). | seed | | proposed | | |
| 6 | Matrix Fisher as a same-harness baseline (published 8.9°), and a finished `fisher_prior` run. | seed | | proposed | | |
| 7 | Real ModelNet10-SO(3) evaluation. Loaders for ModelNet10 and SYMSOL exist on the `CliffordNet-exps` branch. | seed | | proposed | | |
| 8 | Check the equivariance of the flow rigorously, not by assumption. | seed | | proposed | | |
| 9 | One-step inference: MeanFlow, rectified flow or drifting adapted to rotors. | seed | | proposed | | |
| 10 | Better sampling: higher-order ODE solvers, and drawing the source rotor from a rough estimate of p(R given image). | seed | | proposed | | |
| 11 | Clifford Frame Attention (GAFL) for conditioning, instead of pooling a 16x16 grid down to 64 multivectors. | seed | | proposed | | |
| 12 | CliffordNet backbone with the harmonic projector (`CliffordNet-exps` branch) as encoder. | seed | | proposed | | |
