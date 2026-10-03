# pose3d usage

## Setup

Python `>=3.11,<3.15`, [Poetry](https://python-poetry.org/):

```bash
cd pose3d
poetry install
```

`clifford` and `image2sphere` are forks installed from git (see `pyproject.toml`). The Pascal3D+
dataset is read through `image2sphere.pascal_dataset.Pascal3D`.

Poetry is not required where torch is already installed (Kaggle): install only what is missing and
run `python -m pose3d` from `pose3d/`, no install of the package itself.

```bash
pip install e3nn==0.5.9 healpy==1.19.0 \
  git+https://github.com/chagrygoris/image2sphere.git \
  git+https://github.com/chagrygoris/clifford-group-equivariant-neural-networks.git
```

On Kaggle, pass a constraints file pinning the preinstalled packages (`-c constraints.txt`, as in the
runner notebook) so `pip` does not replace torch.

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

## GATr denoiser

`--vector_field gatr` swaps the Clifford MLP vector field of `clifford_flow` for the Geometric
Algebra Transformer ([reference](https://github.com/Qualcomm-AI-research/geometric-algebra-transformer)).
The rotor, the time and the `n_cond_mv` condition multivectors are embedded in Cl(3,0,1) and become
the tokens of one sequence; the velocity is read from the rotor token's rotation bivector. The
condition head stays a Clifford MLP unless `--condition_head gatr`, which runs GATr over the 256
backbone tokens plus `n_cond_mv` learned query tokens (each token also gets a learned scalar
embedding, since GATr treats tokens as an unordered set) and reads the condition multivectors from
the queries. Both share the sizes below. Size them with `--gatr_blocks`, `--gatr_mv_channels`,
`--gatr_s_channels` and `--gatr_heads`. The default is still `--vector_field clifford`.

```bash
pip install --no-deps einops opt_einsum \
  git+https://github.com/Qualcomm-AI-research/geometric-algebra-transformer.git
poetry run python -m pose3d --path_to_datasets ... --vector_field gatr
```

Install GATr with `--no-deps`: its `setup.py` pins `numpy<1.25` and `xformers`, which would replace
the preinstalled torch. `xformers` is only needed for attention masks, which the flow never passes,
so a stub stands in when it is missing.

### `--cond_algebra pga`: the Clifford condition head in Cl(3,0,1)

With `--vector_field gatr` the denoiser already works in the projective algebra Cl(3,0,1) while the
condition head still computes in Cl(3,0) and gets embedded at the boundary. `--cond_algebra pga`
moves the condition head into Cl(3,0,1) as well, so the two halves share one algebra:

```bash
poetry run python -m pose3d --path_to_datasets ... --vector_field gatr --cond_algebra pga
```

What changes: the adapter cuts the 2048 backbone channels into 2048/16 = **128** multivectors
instead of 256, the head's layers are built on `CliffordAlgebra((1, 1, 1, 0))`, and the denoiser
takes its 16-blade tokens directly (`pose3d/geometry/pga.py` holds the blade maps; CGENN orders
blades ShortLex with the degenerate generator last, GATr puts it first and calls it `e0`, so the
reorder carries signs). The rotor, the velocity and the loss stay in Cl(3,0) — only the condition
head and the tokens handed to GATr move. It is therefore only valid together with
`--vector_field gatr --condition_head clifford`, and not with `--mlp_heads` or `--fisher_prior`.

Cost, condition head only, measured at the default `--flow_hidden_dim 88`:

| | blades | adapter `n_mv` | cond-head params | GP MACs/sample |
|---|---|---|---|---|
| `cl3` (default) | 8 | 256 | 1,036,048 | 18.4M |
| `pga` | 16 | 128 | 863,536 | 100.9M |
| `pga --flow_hidden_dim 104` | 16 | 128 | 1,030,480 | 126.1M |

The geometric product contracts over `n_blades**3`, so it costs ~5.5x more while the parameter
count *falls* (the adapter emits half as many multivectors). `--flow_hidden_dim 104` restores the
Cl(3,0) parameter budget to within 0.5%, which is the arm to compare against if the question is
about the algebra rather than the budget.

Two things to watch when reading the result. CGENN's invariants are the grade-wise `q`/`norm`,
and those annihilate the degenerate direction, so `NormalizationLayer` and `MVSiLU` cannot see
the `e0` components they are gating. And GATr's equivariant linear basis in Cl(3,0,1) includes
maps (multiplication by `e0`) that CGENN's `MVLinear` does not have, so this is Cl(3,0,1) without
the part of GATr that exploits it.

## Micro and macro metrics

Every reported number has two versions. Micro is pooled over all validation images, so the big
classes dominate (car, chair). Macro averages over the 12 Pascal3D+ classes, so each counts the
same: `class_mean_median_error` (the mean of the per-class medians, the number the IPDF /
Image2Sphere tables report) every epoch, and at the end also `final_class_mean_acc@15` /
`@30` plus each class's median and accuracies (`final_median_error_class<c>`,
`final_acc@15_class<c>`, ...), printed as a table.

The pre-built RAM cache holds no class labels, so they are read from the annotations of the
mounted Pascal3D+ (no image is decoded) and checked against the cache's ground-truth rotations. If
Pascal3D+ is not mounted, or the check fails, macro metrics are skipped with a message and the
run goes on. Other loaders pass the labels through as before.

## Multi-GPU

Training uses every visible GPU by default (torch DistributedDataParallel, `--ddp`). The run relaunches
itself under `torchrun` with one process per GPU, so the command is the same as for one GPU; with one GPU
or CPU it changes nothing. `--num_gpus N` limits the count and `--no-ddp` turns it off.

```bash
poetry run python -m pose3d --path_to_datasets ... --no-ddp
```

`--batch_size` is the **global** batch: each GPU gets `batch_size // n_gpus`, so the recipe is the
same on 2 T4 or 4 L4. Raising the global batch on purpose (`--batch_size 128`) is a different recipe;
`--lr_scaling linear|sqrt` rescales the learning rate by `batch_size / lr_reference_batch`. Other
options: `--sync_bn`, `--nccl_p2p` (off by default, Kaggle's T4 x2 can hang with it), `--seed`
(each rank adds its rank). With the RAM cache the tensors are built once before the ranks start.
W&B, the checkpoint and the printed log come from rank 0 only.

## Pascal3D tensor cache

`--ram_memory` (on by default) decodes every image once per run (~34 min on Kaggle). The tensors can be
saved and reloaded instead: `--ram_cache_save_dir DIR` writes `pascal_train.pt` / `pascal_val.pt` after a
normal build, `--ram_cache_dir DIR` loads them (Pascal3D is not even constructed). By default
(`--pre_cache`) the run looks for the Kaggle dataset `syfry5suvzovvakmuj/pascal3d-ram-cache` and uses it
when it is mounted; `--no-pre_cache` disables that. The cache holds one un-augmented pass, so it is
skipped with `--use_warp`, `--use_synth`, `--raw_cache` or `--fisher_prior`.

## Kaggle

`notebooks/clifford-runner.ipynb` is the runner. Cell 0 holds `BRANCH`, `RUN_NAME` and the flags that
differ from the defaults; the other cells clone the branch, install what Kaggle lacks (no Poetry) and run
`python -m pose3d`. Attach the datasets `syfry5suvzovvakmuj/pascal3d` and
`syfry5suvzovvakmuj/pascal3d-ram-cache`.
