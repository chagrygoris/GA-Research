"""Fair sharing of a pool of shared accelerators: Kaggle accounts and Slurm clusters (HSE cHARISMa).

Ask where a job can run right now, then send it there:

    from gpu_pool import ComputePool, RunSpec, PoolRouter, SlurmCluster, CHARISMA

    pool = ComputePool(PoolRouter.from_dir("~/pool_tokens"), [SlurmCluster(CHARISMA)])
    pool.probe()
    print(pool.report())

The Kaggle side alone:

    from gpu_pool import PoolRouter, PoolLauncher, git_run_spec

    router = PoolRouter.from_dir("~/pool_tokens")
    router.probe_all()
    print(router.report())

Kaggle goes through the official Kaggle CLI (router.py); Slurm clusters through ssh (slurm.py).
"""

from .launcher import ACCELERATORS, NotebookSpec, PoolLauncher, git_run_spec
from .pool import ComputePool, Offer, RunHandle, RunSpec
from .router import (
    AccountStatus,
    CliError,
    PoolAccount,
    PoolRouter,
    Quota,
    filter_progress,
    shape_label,
)
from .slurm import CHARISMA, ClusterStatus, SlurmCluster, SlurmConfig, SlurmError, SlurmJobSpec

__all__ = [
    "ACCELERATORS",
    "AccountStatus",
    "CHARISMA",
    "CliError",
    "ClusterStatus",
    "ComputePool",
    "Offer",
    "RunHandle",
    "RunSpec",
    "SlurmCluster",
    "SlurmConfig",
    "SlurmError",
    "SlurmJobSpec",
    "NotebookSpec",
    "PoolAccount",
    "PoolLauncher",
    "PoolRouter",
    "Quota",
    "filter_progress",
    "git_run_spec",
    "shape_label",
]
