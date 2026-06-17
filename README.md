# 3D Pose Estimation Experiments

A research framework for experimenting with **3D object pose estimation** (predicting an object's
SO(3) rotation from a single RGB image). The codebase explores several model families, including
standard CNN/ViT baselines and **geometric-algebra (GA) / Clifford-group-equivariant** networks, plus
the **Image2Sphere (I2S)** family of SO(3)-distribution heads.

> ⚠️ This README was generated to document the project layout and how to run experiments. Some
> sections describe model variants the maintainer understands best — feel free to correct or expand
> any descriptions that aren't quite right.

---

## What's in here

The training entry point wires together a dataset, a model, a loss, an optimizer/scheduler, and
Weights & Biases logging, then runs a train/eval loop.

```
3D Pose experiemtns/
├── src/
│   ├── main.py                # Entry point: arg parsing → model build → train/eval
│   ├── config.py              # All CLI flags (argparse) + a dataclass config
│   ├── dataset.py             # Dataloaders (Pascal3D, in-memory cache, sanity/dummy sets)
│   ├── model.py               # All models: baselines, GA heads, I2S, I2S_ResNet
│   ├── image_encoders.py      # Backbone/encoder builders (resnet, ga, ga_canonical)
│   ├── train_utils.py         # Train loop, checkpointing, losses, device/multi-GPU helpers
│   ├── evaluation_metrics.py  # Pose-error metrics
│   ├── wandb_utils.py         # W&B run/code/artifact logging
│   ├── img_to_pcd_2.py        # Image → point cloud model (I2P)
│   └── img_to_pcd_stuff.py    # Older image → point cloud experiments
├── tests/                     # Pytest unit tests
├── I2S_RESNET.md              # Notes specific to the I2S_ResNet model
├── pyproject.toml             # Poetry dependencies
└── *.ipynb                    # Exploratory / runner notebooks
```

Two external research packages are pulled in as git dependencies and are central to the GA models:

- [`clifford`](https://github.com/chagrygoris/clifford-group-equivariant-neural-networks.git) —
  Clifford-group-equivariant layers (`MVLinear`, `MVSiLU`, geometric-product layers, `MVLayerNorm`, …).
- [`image2sphere`](https://github.com/chagrygoris/image2sphere.git) — SO(3) utilities (HEALPix
  grids, Wigner matrices) and the `Pascal3D` dataset wrapper.

---

## Setup

The project uses [Poetry](https://python-poetry.org/) and targets Python `>=3.11,<3.15`.

```bash
cd "3D Pose experiemtns"
poetry install
poetry shell        # or prefix commands with `poetry run`
```

Key dependencies (see `pyproject.toml` for exact pins): `torch 2.9.0`, `torchvision 0.24.0`,
`transformers`, `e3nn`, `healpy`, `open3d`, `wandb`, plus the two git packages above.

---

## Running an experiment

Everything is driven through `src/main.py`. The only required flag is `--path_to_datasets`.

```bash
python -m src.main \
  --path_to_datasets /path/to/data \
  --model tralalero \
  --loss mse \
  --n_epochs 10 \
  --batch_size 32 \
  --lr 1e-3 \
  --run_name my-first-run
```

A quick smoke test that skips checkpoint saving:

```bash
python -m src.main --path_to_datasets /path/to/data --sanity_check
```

To **evaluate** an existing checkpoint instead of training from scratch, pass
`--path_to_checkpoint`; metrics are written to `res.pth`.

---

## Models (`--model`)

| Value          | Description |
|----------------|-------------|
| `tralalero`    | GA pose head over a CNN encoder (geometric-product blocks → rotation/rotor output). |
| `mlp`          | Simple MLP baseline on top of an encoder. |
| `vit_baseline` | ViT (or Depth-Anything-v2) backbone with configurable pooling, incl. a GA pooling/readout path. |
| `i2s`          | Image2Sphere: predicts a distribution over the SO(3) HEALPix grid. |
| `ga_i2s`       | GA-flavored Image2Sphere variant. |
| `i2s_resnet`   | ResNet/ConvNeXt backbone feeding GA blocks; supports rotation-matrix, Fourier, and rotor outputs. See `I2S_RESNET.md`. |
| `image2pcd`    | Image → point cloud model (`I2P`). |
| `dummynet`     | Placeholder/debug model. |

## Losses (`--loss`)

| Value      | Meaning |
|------------|---------|
| `mse`      | Direct rotation-matrix regression (Frobenius). |
| `geodesic` | Geodesic distance on SO(3). |
| `prob`     | Cross-entropy over the SO(3) grid (for the I2S distribution heads). |
| `rotor`    | Rotor-based loss (requires `--algebra_dim 3`, i.e. Cl(3,0)). |
| `mv_rotor` | Multivector-rotor loss (requires `--algebra_dim 3`). |

## Encoders (`--encoder`)

`resnet` (default), `ga`, `ga_canonical`.

---

## Notable flags

A non-exhaustive tour — run `python -m src.main --help` for the full list (defined in `src/config.py`).

**General**
- `--n_epochs`, `--batch_size`, `--lr`, `--run_name`
- `--dataset` (default `pascal`), `--path_to_datasets` *(required)*
- `--algebra_dim` (default `3`; most GA/rotor paths require `3` = Cl(3,0))
- `--path_to_checkpoint` (evaluate instead of train), `--save_checkpoint`, `--sanity_check`

**Weights & Biases**
- `--wandb_project` (default `3D Pose Estimation`), `--wandb_entity` (default `clifforders`), `--wandb_group`

**ViT baseline** (`--model vit_baseline`)
- `--vit_backbone_type {vit,depth_anything_v2}`, `--vit_model_name`, `--vit_layers`, `--freeze_vit`
- `--vit_pooling_type {mean,attention,transformer_attention,convolution,ga}`
- `--vit_ga_readout_type {scalar,mean,linear,grade,rotor}` (`rotor` needs `--algebra_dim 3`)

**I2S family** (`--model i2s | ga_i2s | i2s_resnet`)
- `--lmax`, `--rec_level`, `--n_mv`, `--hidden_dim`, `--temperature`, `--ga_pool_hw`
- `--i2s_resnet_output_mode {auto,rotation_matrix,fourier,rotor,multivector_rotor}`
- `--i2s_resnet_backbone_name {resnet50,convnext_tiny}`, plus several `--i2s_resnet_ga_head_*` knobs

> Several combinations are validated at startup in `instantiate()` (`src/main.py`) — e.g. rotor
> modes and variable `algebra_dim` are only supported for specific model/pooling choices, and will
> raise a clear `ValueError` otherwise.

---

## Datasets

The default pipeline uses **Pascal3D** via `image2sphere.pascal_dataset.Pascal3D`, with an optional
in-memory cache for faster epochs (`src/dataset.py`). There are also sanity-check and dummy
point-cloud datasets used for quick iteration and the image-to-point-cloud experiments
(`ModelNet10`-based). Point `--path_to_datasets` at the directory containing your datasets.

---

## Tests

```bash
poetry run pytest
```

---

## Logging & checkpoints

Runs are tracked with Weights & Biases (project/entity/group configurable via flags). On completion,
unless `--sanity_check` is set and `--save_checkpoint` is enabled, a checkpoint is written and logged
as a W&B artifact. Evaluation metrics from a loaded checkpoint are saved to `res.pth`.
