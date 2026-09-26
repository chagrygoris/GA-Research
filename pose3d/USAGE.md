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

## Multi-GPU

`--ddp` trains on every visible GPU (torch DistributedDataParallel). The run relaunches itself under
`torchrun` with one process per GPU, so the command is the same as for one GPU; with one GPU or CPU
the flag changes nothing. `--num_gpus N` limits the count.

```bash
poetry run python -m pose3d --path_to_datasets ... --ddp
```

`--batch_size` is the **global** batch: each GPU gets `batch_size // n_gpus`, so the recipe is the
same on 2 T4 or 4 L4. Raising the global batch on purpose (`--batch_size 128`) is a different recipe;
`--lr_scaling linear|sqrt` rescales the learning rate by `batch_size / lr_reference_batch`. Other
options: `--sync_bn`, `--nccl_p2p` (off by default, Kaggle's T4 x2 can hang with it), `--seed`
(each rank adds its rank). With the RAM cache the tensors are built once before the ranks start.
W&B, the checkpoint and the printed log come from rank 0 only.

## Kaggle

`notebooks/clifford-runner.ipynb` is the runner. Cell 0 holds `BRANCH`, `RUN_NAME` and the flags that
differ from the defaults; the other cells clone the branch, install with Poetry and run
`python -m pose3d`.
