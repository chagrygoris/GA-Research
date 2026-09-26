# pose3d — 3D pose estimation on SO(3)

Predict an object's 3D rotation from one RGB image (Pascal3D+). The main model, **Clifford Flow**,
is conditional flow matching on Spin(3) rotors built from Clifford (geometric-algebra) layers:

```
image -> ResNet-50 (ImageNet init, fine-tuned) -> ConvAdapter -> 16x16 multivectors
      -> condition head (CGENN-style GA MLP) -> n_cond_mv conditioning multivectors
      -> vector field over (rotor R_t, time t, condition) -> bivector velocity -> MSE in the tangent space
inference: Euler ODE from a random rotor, K samples per image, geodesic medoid
```

Reference result (Pascal3D+, median rotation error, in-repo harness): **10.25 deg** (W&B `k5sblpo8`).
See [`EXPERIMENTS.md`](EXPERIMENTS.md) for everything that was tried, with numbers.

## Setup

Python `>=3.11,<3.15`, [Poetry](https://python-poetry.org/):

```bash
cd pose3d
poetry install
```

`clifford` and `image2sphere` are forks installed from git (see `pyproject.toml`); the
Pascal3D+ dataset is read through `image2sphere.pascal_dataset.Pascal3D`.

## Run

Defaults are the reference recipe, so the minimum is:

```bash
poetry run python -m pose3d --path_to_datasets /path/to/data --run_name my-run
```

Everything below is a flag of the same name. `python -m pose3d --help` lists all of them, grouped
by section.

```bash
# turn an experimental feature on
poetry run python -m pose3d --path_to_datasets ... --use_warp
# turn an adopted feature off
poetry run python -m pose3d --path_to_datasets ... --no-medoid_eval
# other models (their own knobs are documented in config.py)
poetry run python -m pose3d --path_to_datasets ... --model i2s_real --encoder resnet101 --lr 1e-3 --n_epochs 10
```

`--run_name` enables W&B logging (entity/project via `--wandb_entity` / `--wandb_project`) and
saves + uploads a checkpoint at the end. `--sanity_check` trains on one batch. `--path_to_checkpoint`
evaluates a checkpoint before training.

## Configuration

[`pose3d/config.py`](pose3d/config.py) is the single source of truth. The `Features` section holds
one boolean per modification of the reference pipeline:

| Flag | Default | Meaning |
|---|---|---|
| `pretrained_backbone` | **on** | ImageNet-initialised, fine-tuned backbone |
| `ram_memory` | **on** | preload Pascal3D tensors into RAM |
| `time_sample_batching` | **on** | `n_time_samples` (t, r0) pairs per image per step |
| `medoid_eval` | **on** | final evaluation with `eval_samples` draws + geodesic medoid |
| `use_warp` | off | Pascal3D augmentation (flip / jitter) |
| `use_synth` | off | RenderForCNN synthetic training images |
| `raw_cache` | off | RAM-cache file reads so `use_warp` stays random per access |
| `freeze_encoder` | off | freeze the backbone |
| `mlp_heads` | off | swap the GA heads for parameter-matched MLPs (comparison) |
| `fisher_prior` | off | sample the flow's source rotation from a matrix Fisher head |

On = adopted, off = experimental. Numeric options that ablate the architecture (`adapter_grid`,
`adapter_channels`, `vector_field_hidden_dim`) default to the reference values. How a change earns
its `True` is described in [`../docs/experiment-workflow.md`](../docs/experiment-workflow.md).

## Models (`--model`)

| Value | What it is |
|---|---|
| `clifford_flow` | **the main model** |
| `mlp_flow` | non-equivariant flow-matching baseline (MLPs on rotation matrices) |
| `i2s_real` | the published Image2Sphere (`image2sphere.predictor.I2S`) under this harness |
| `i2s`, `ga_i2s` | in-repo I2S-style models with a GA head (average-pooled features / GA pixel encoder) |
| `tralalero`, `mlp` | direct regression of a 3x3 matrix with a GA / linear head |
| `i2s_backbone` (`i2s_resnet`) | ResNet-50 / ConvNeXt + conv adapter + GA head; Fourier, matrix, rotor outputs; several GA head types |
| `i2s_conv_resnet`, `i2s_conv_convnext` | earlier variant with alternative adapters (`conv`, `mlp_block`, `linear`, `geometric`, `inc`) |
| `ipdf_resnet` | Clifford implicit PDF: score rotor-rotated multivector features |
| `image2pcd_ipdf` | Depth-Anything features with an Implicit-PDF scorer |
| `image2pcd` | Depth-Anything features, direct regression |
| `image2pcd_pointnet`, `image2pcd_late_fusion` | Depth-Anything point-cloud regressors |
| `vit_baseline` | ViT / Depth-Anything hidden states with mean / attention / transformer / conv / GA pooling |
| `dummynet` | point-cloud debugging model on a ModelNet10 stub (not a benchmark) |

Encoders (`--encoder`): `resnet`/`resnet50`, `resnet101`, `depth_anything`, `convnext_*`, `ga`,
`ga_canonical`. Losses (`--loss`): `mse`, `mse_ortho`, `geodesic`, `prob`, `rotor`, `mv_rotor`
(CliffordFlow, IPDF and the I2S heads compute their own loss and ignore it).

## Layout

```
pose3d/
  pyproject.toml
  EXPERIMENTS.md            what was tried, what is adopted, what was not
  notebooks/                Kaggle runner + checkpoint re-evaluation (archive/ = old exploration)
  tests/
  pose3d/
    config.py               all options; Features = the on/off switches
    train.py                entry point (python -m pose3d)
    evaluate.py             re-score a checkpoint with multi-sample prediction
    geometry/               rotor <-> matrix, flow-matching maps on Spin(3), quaternion helpers
    data/                   Pascal3D loading, RAM / raw-file caches, ModelNet10 stub
    engine/                 trainer, losses, metrics, checkpoints, W&B
    models/                 clifford_flow.py (main) + baselines and other model families;
                            __init__.py is the registry (build_model)
    pointcloud.py           mesh / depth-map utilities, DummyNet
```

## Kaggle

`notebooks/clifford-runner.ipynb` is the runner template: cell 0 holds `BRANCH`, `RUN_NAME` and the
flags that differ from the defaults; the remaining cells clone the branch, install with Poetry and
run `python -m pose3d`. To reuse a cached RAM dataset between sessions pass `--ram_cache_dir`
(read) or `--ram_cache_save_dir` (write after a normal build).

## Tests

```bash
cd pose3d && poetry run python -m unittest discover -s tests
```
