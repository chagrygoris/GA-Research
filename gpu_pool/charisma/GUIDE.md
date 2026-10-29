# cHARISMa (HSE) cluster: operating guide for agents

Verified 2026-10-29 by connecting and running jobs. Official docs: https://hpc.hse.ru/support/instructions
(SSH: /instructions/ssh, jobs: /instructions/run, basics: /instructions/base, Python: /instructions/python/anaconda).

In this repository the cluster is the `charisma` backend of the GPU pool: `gpu_pool/slurm.py` talks to it
over SSH and `gpu_pool/pool.py` puts it behind the same interface as the Kaggle accounts (see
`gpu_pool/README.md`, "Slurm clusters"). The rules in section 2 that code can check (time limits, QOS cap,
budget, the reserve window) are enforced by `ClusterStatus.check`; the rest are for you.

## 1. Account and connection

| Item | Value |
|---|---|
| Host / port | `cluster.hpc.hse.ru`, port **2222** (login nodes `login-0x`, Rocky Linux 9.8) |
| SSH login | `sptsinman` (NOT the personal-cabinet login `efilimoshina`) |
| Auth | SSH key only. Key: `~/.ssh/charisma_auth_key.pem`, passphrase-protected, loaded in ssh-agent |
| Slurm account | `proj_1898` ("Эквивариантные нейронные сети на основе геометрических алгебр. Продление (2026)") |
| QOS | `students`: max **4 GPUs** and **48 CPU cores** in use per user, 24 running jobs |
| Location | Moscow (MSK, UTC+3). Compute nodes have **no internet**; the login node does |

Non-interactive command pattern (use everywhere; `IdentitiesOnly` avoids offering other agent keys):

```bash
ssh -p 2222 -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/charisma_auth_key.pem sptsinman@cluster.hpc.hse.ru '<command>'
scp -P 2222 -o IdentitiesOnly=yes -i ~/.ssh/charisma_auth_key.pem <local> sptsinman@cluster.hpc.hse.ru:~/<dir>/
```

If auth fails with "Permission denied (publickey...)": check `ssh-add -l` lists the key; if not, the user must run
`ssh-add --apple-use-keychain ~/.ssh/charisma_auth_key.pem` (agents must never type the passphrase).
A new `.ppk` from https://lk.hpc.hse.ru/manage_access/ is converted with `puttygen key.ppk -O private-openssh -o key.pem` (user-run).

## 2. Rules of the road

1. **Never run compute on the login node.** It is for editing, compiling, package installs, and submitting. Everything else goes through Slurm.
2. **Budget is finite and shared.** ~5000 GPU-h and ~50000 CPU-h until **2026-12-31**. The PI needs a reserve for the reviewer response **2026-11-05 to 2026-11-18**: avoid large runs in that window without asking. Billing is per allocated GPU/CPU per hour, even if idle, so size requests tightly and set short `--time`.
3. **Always set `--time`.** Default is 24 h; a short limit gets higher priority and bounds mistakes. Smoke tests: `--time=00:05:00`.
4. **Test on 1 GPU for minutes before scaling.** Use partition `test` (30 min max, high priority) for debugging.
5. **Stage everything through the login node first**: `pip`/`conda` installs, dataset downloads, `wandb` offline mode. Compute nodes cannot reach the internet.
6. Large data goes to `/scratch/$USER` (Lustre, faster). Home quota is **100 GB / 1M files** (`checkquota`). `$TMPDIR` is a per-node SSD, wiped after the job.
7. Don't leave jobs pending indefinitely on backfill queues without the user knowing; `scancel <jobid>` when no longer needed.

## 3. Partitions and hardware

| Partition | Max time | Nodes | Notes |
|---|---|---|---|
| `rocky` (default) | 30 days | all | Under `students` QOS effectively gives **V100 32 GB** nodes (type_a/b/c, 4 GPUs/node, 44-48 cores, 768 GB-1.5 TB RAM) and CPU-only type_d nodes |
| `test` | 30 min | all | Debug queue, high priority, no preemption. **Use for smoke tests** |
| `gpu-ef-quick` | 3 h | type_e/f/h | Backfill, **preemptible**: A100 80GB (type_e, 8/node), H100 (type_f, 2/node), H200 141GB (type_h). In practice fully occupied; a 1-GPU job waited >19 h on 2026-10-29. Checkpoint frequently if used |
| `cpu-e-quick` | 24 h | type_e | CPU-only backfill on A100 nodes, preemptible |

Pin a node type with `--constraint=type_a` (or `"type_a|type_b|type_c"` for any V100 node). Check live state:

```bash
sinfo -o "%P %a %l %D %T %G"                                   # partitions and GPU types
sinfo -p gpu-ef-quick -N -o "%N %T %G"                         # per-node state on the fast queue
scontrol show node cn-011 | grep -E "Gres|AllocTRES|Features"  # what's allocated on a node
mj          # my jobs;   mj --start  = estimated start times
squeue -p test -o "%.10i %.9P %.8u %.2t %.10M %.6D %R" | head  # queue load
```

## 4. Software environment

`module load python/...` is **broken** on this cluster (Lmod fails on a `cmake`/GCCcore dependency chain, both login and compute nodes). Use environments by absolute path instead:

| Environment | Path | Contents |
|---|---|---|
| Ready-made GPU env | `/opt/software/python/envs/google_colab_gpu_2025` | Python 3.11, **torch 2.6.0+cu124**, Colab-like package set |
| Miniconda base | `/opt/software/python/miniconda/latest` | conda only, no torch; use to create own envs |

```bash
PYENV=/opt/software/python/envs/google_colab_gpu_2025
export PATH="$PYENV/bin:$PATH" LD_LIBRARY_PATH="$PYENV/lib:${LD_LIBRARY_PATH:-}"
```

Own environment (on the login node): `source /opt/software/python/miniconda/latest/etc/profile.d/conda.sh; conda create -n <env> python=3.11; conda activate <env>; pip install torch ...`, then in jobs `source .../conda.sh && conda activate <env>`. Install torch first into an empty env. Verify on a GPU with `python -c 'import torch; print(torch.cuda.is_available())'` inside a `test` job, never on the login node (it has no GPU).

Other modules that do exist: `cuda/12.9`, `intel/oneAPI_2025`, `julia/1.10|1.12`. Singularity is available for containers.

## 5. Submitting a job

The template that ran successfully as job 4467119 is kept as [`gpu_test.sbatch`](gpu_test.sbatch), with one
change: it exits with the command's own status. The original ended in `echo "exit=$?..."`, so the job's
exit code was `echo`'s and a crashed run still showed as COMPLETED in `sacct`.

```bash
#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH --output=<name>_%j.out
#SBATCH --error=<name>_%j.err
#SBATCH --partition=test          # rocky for real runs
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2         # ~4 per GPU for data loading
#SBATCH --gpus=1
#SBATCH --constraint=type_a       # optional, pins GPU type

set -uo pipefail
echo "node=$(hostname) start=$(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv
PYENV=/opt/software/python/envs/google_colab_gpu_2025
export PATH="$PYENV/bin:$PATH" LD_LIBRARY_PATH="$PYENV/lib:${LD_LIBRARY_PATH:-}"
python "$HOME/<project>/<script>.py" "$@"
status=$?
echo "exit=$status end=$(date)"
exit $status
```

Workflow (by hand; `python gpu_pool/pool.py launch ...` does steps 1-2 for a git repository):

```bash
# 1. upload code
scp -P 2222 -o IdentitiesOnly=yes -i ~/.ssh/charisma_auth_key.pem -r ./code sptsinman@cluster.hpc.hse.ru:~/project/
# 2. submit (CLI flags override #SBATCH lines; -A is only needed if the user joins a second project)
ssh ... 'cd ~/project && sbatch --partition=rocky --time=02:00:00 run.sbatch'
# 3. monitor
ssh ... 'mj; tail -n 20 ~/project/<name>_<jobid>.out'
# 4. accounting after completion (Elapsed x GPUs = GPU-hours billed)
ssh ... 'sacct -j <jobid> -o JobID,Partition,State,Elapsed,AllocTRES%50 -P'
# 5. cancel
ssh ... 'scancel <jobid>'
```

Interactive debugging (bounded!): `srun -p test -t 10 --gpus=1 --cpus-per-task=2 --pty bash`.
Multi-GPU on one node: `--gpus=4 --cpus-per-task=16` (QOS cap is 4 GPUs total per user, so no multi-node GPU jobs).
Array sweeps: `#SBATCH --array=0-9%2` runs 10 configs, 2 at a time, `$SLURM_ARRAY_TASK_ID` selects the config.

## 6. Checking results and budget

- Output files appear in the submit directory as `<name>_<jobid>.out/.err`; `sacct` gives the final `State` (COMPLETED / FAILED / TIMEOUT / PREEMPTED / CANCELLED).
- Rough GPU-hour spend: `sacct -S 2026-10-01 -u sptsinman -X -o JobID,Elapsed,AllocTRES%40 -P` and sum `Elapsed x gres/gpu`. Report spend to the user after any run longer than an hour. `python gpu_pool/slurm.py` does this sum for the whole project account since `budget_start`.
- Reference throughput measured here: V100 fp32 4096x4096 matmul ≈ 13.7 TFLOP/s.

## 7. Known pitfalls

- `module purge` / `module load python` fail silently; `python` then resolves to `/usr/bin/python` without torch. Use absolute env paths.
- Running `ssh` via a shell variable in zsh (`SSH="ssh ..."; $SSH host`) fails; write the command out or use [`connect.sh`](connect.sh) in this folder.
- Expected-start estimates on `gpu-ef-quick` can be a day away; check `mj --start` before leaving a job there.
- The SSH login differs from the cabinet login. Don't guess alternative usernames against the server.
- Never put the key passphrase or cabinet password in scripts, chat, or files.
