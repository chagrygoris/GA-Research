# pose3d usage

## Setup

Python `>=3.11,<3.15`, [Poetry](https://python-poetry.org/):

```bash
cd pose3d
poetry install
```

`clifford` and `image2sphere` are forks installed from git (see `pyproject.toml`). The Pascal3D+
dataset is read through `image2sphere.pascal_dataset.Pascal3D`.

## Run

Run from the `pose3d/` folder. The defaults are the reference recipe:

```bash
poetry run python -m pose3d --path_to_datasets /path/to/data --run_name my-run
```

Every option in `pose3d/config.py` is a flag of the same name; `python -m pose3d --help` lists them.

```bash
poetry run python -m pose3d --path_to_datasets ... --use_warp          # feature on
poetry run python -m pose3d --path_to_datasets ... --no-medoid_eval    # feature off
poetry run python -m pose3d --path_to_datasets ... --model i2s_real --encoder resnet101 --lr 1e-3 --n_epochs 10
```

`--run_name` turns on W&B logging and uploads a checkpoint at the end. `--sanity_check` trains on one
batch. `--path_to_checkpoint` evaluates a checkpoint before training.

Re-score a checkpoint from W&B:

```bash
poetry run python -m pose3d.evaluate --artifact <entity/project/name.pth:vN> --path_to_datasets ...
```

## Kaggle

`notebooks/clifford-runner.ipynb` is the runner. Cell 0 holds `BRANCH`, `RUN_NAME` and the flags that
differ from the defaults; the other cells clone the branch, install with Poetry and run
`python -m pose3d`.
