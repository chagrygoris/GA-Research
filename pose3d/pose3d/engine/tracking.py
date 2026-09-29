"""Weights & Biases helpers. Every function is a no-op when there is no run."""

import json
import os
from pathlib import Path

import wandb


def wandb_create_run(cfg):
    if not cfg.run.run_name:
        print(f"W&B run won't be created, since run name is {cfg.run.run_name}")
        return None
    print("Creating W&B run")
    run = wandb.init(
        project=cfg.run.wandb_project,
        entity=cfg.run.wandb_entity,
        group=cfg.run.wandb_group,
        name=cfg.run.run_name,
        config=cfg.to_dict(),
    )
    print("Created W&B run with name {} at {}".format(run.name, run.project_url))
    return run


def wandb_finish_run(run):
    if run is not None:
        run.finish()


def wandb_log_code(run, code_dir: Path):
    if run is not None:
        run.log_code(code_dir.__str__())


def wandb_log_artifact(run, path_to_artifact: Path, artifact_type="artifact"):
    if run is None:
        return
    artifact = wandb.Artifact(name=path_to_artifact.name, type=artifact_type)
    artifact.add_file(path_to_artifact.__str__())
    run.log_artifact(artifact)


def wandb_load_artifact(run, artifact_full_name):
    if run is None or artifact_full_name is None:
        return None
    artifact = run.use_artifact(artifact_full_name, type='model')
    artifact_dir = artifact.download()
    path = list(Path(artifact_dir).resolve().iterdir())[0]
    return path


#: `wandb.log`/`.summary` reach wandb.ai directly on an online run, but a Kaggle kernel with
#: internet disabled (every RTX / L4 / TPU run so far, see PRE_CACHE_DIRS' neighbour
#: gpu_pool/README.md "Accelerators") sets WANDB_MODE=offline before `wandb.init`, so those
#: calls only write local files under /kaggle/working that nothing reads until the kernel
#: finishes and someone downloads its output. This prints the same numbers as one grep-able
#: line instead, so `gpu_pool/rtx_monitor.py` can tail `kaggle kernels logs` on a no-internet
#: run and push the points to wandb itself. A pure no-op on an online run (WANDB_MODE unset):
#: T4/P100 runs are completely unaffected and keep syncing live exactly as before.
def log_offline_sync(run, cfg, step: int, metrics: dict, final: bool = False) -> None:
    if run is None or os.environ.get("WANDB_MODE") != "offline":
        return
    payload = {
        "run_name": cfg.run.run_name,
        "wandb_project": cfg.run.wandb_project,
        "wandb_entity": cfg.run.wandb_entity,
        "step": step,
        "final": final,
        "metrics": metrics,
    }
    print(("WANDB_SYNC_FINAL " if final else "WANDB_SYNC ") + json.dumps(payload), flush=True)
