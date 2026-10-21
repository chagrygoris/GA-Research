# cHARISMa (HSE supercomputer)

Access notes and a GPU smoke test for the HSE cHARISMa cluster.
Official docs: [hardware](https://hpc.hse.ru/hardware/hpc-cluster/) ·
[instructions](https://hpc.hse.ru/support/instructions) ·
[Slurm](https://hpc.hse.ru/instructions/run) ·
[ssh](https://hpc.hse.ru/instructions/ssh)

## Status: not reachable from a Claude Code cloud session

Two independent blockers, neither of which can be fixed from inside the session:

| Route | Blocked by | Detail |
|---|---|---|
| `ssh -p 2222 cluster.hpc.hse.ru` | **our sandbox** | Egress is HTTPS/443 only, through an HTTP proxy that does not relay raw TCP. Port 2222 times out; the proxy's `CONNECT` returns `200` but no SSH banner arrives (`github.com:22` behaves identically, so it is the proxy, not HSE). |
| `https://lk.hpc.hse.ru/` and `/jupyter/` — personal cabinet and JupyterHub, the only HTTPS route in | **HSE** | TCP/443 connects, then HSE's own nginx returns `403 Доступ запрещен — убедитесь, что не используете VPN` with our egress IP echoed back. The portal is restricted to Russian/HSE networks; our egress is a US datacenter range. `hpc.hse.ru` (public docs) serves `200` from the same IP, so the block is specific to the gated portal. |

Widening this environment's network policy would not help: the 403 is HSE's,
and the geo-restriction is deliberate, so it is not something to route around.

**So the smoke test has to be submitted from a machine HSE accepts** — a laptop
on a Russian/HSE network (or HSE VPN). Everything needed for that is in this
folder; it is a two-command job, see below.

## Connecting (from a machine with access)

Authentication is by **SSH key, not password** — the account password is only for
the personal cabinet. Download `auth_key.pem` from
[lk.hpc.hse.ru](https://lk.hpc.hse.ru/), then:

```bash
chmod 600 auth_key.pem
ssh -p 2222 -i auth_key.pem <login>@cluster.hpc.hse.ru     # key has its own passphrase
scp -P 2222 -i auth_key.pem ga_gpu_smoke.* <login>@cluster.hpc.hse.ru:~/
```

JupyterHub (`https://lk.hpc.hse.ru/jupyter/`) is the no-SSH alternative: it logs
in with the personal-cabinet credentials and spawns a JupyterLab on a compute
node, from whose terminal `sbatch` works the same way.

## Running the smoke test

```bash
sbatch ga_gpu_smoke.sbatch        # queue it
mj                                # own jobs  (squeue --start for an ETA)
grep -E 'PASS|FAIL' ga_gpu_smoke.*.log
```

`ga_gpu_smoke.py` spends 60 s multiplying random multivectors on the GPU and
answers three things:

1. **Scheduling** — Slurm ran the job and handed over the GPUs that were asked for
   (the log records partition, nodelist and `CUDA_VISIBLE_DEVICES`).
2. **Correctness** — every visible GPU is checked against a float64 CPU reference
   before it is timed, and the Cayley table against the `Cl` axioms. A GPU that is
   merely *present* is not enough; `nvidia-smi` listing a device proves nothing
   about its arithmetic. Exit status is 0 only if every GPU agreed, so the log
   reads as pass/fail without interpreting the numbers.
3. **Throughput** — products/s and GFLOP/s per GPU, as a baseline to size real
   runs against. Bandwidth-bound, not a peak-FLOPS figure: the product is one
   outer product contracted against a flattened Cayley table, written exactly as
   [`dpd/dpd/ga.py`](../dpd/dpd/ga.py) writes it, so the number is the throughput
   of the kernel the experiments actually use.

The script is standalone — `torch` and nothing else, with the algebra duplicated
from `dpd.ga` rather than imported, so a failure is unambiguously a cluster
problem and never a broken repo install. `pytest cluster/` guards that copy
against drift and checks that both correctness gates actually reject a wrong
table and a miscomputing device; it needs no GPU and no cluster.

Compute nodes have **no internet access**, so `torch` must already be on the node.
Set `GA_SMOKE_PYTHON` to an interpreter with a CUDA `torch` build, or add a
`module load` line in the `.sbatch` — the public docs do not list the module
names, so the first run's `module avail` dump records them.

Knobs: `--seconds`, `--signature p,q,r` (default `3,0,0`; `dpd` uses `2,0,0`),
`--batch`, `--dtype {fp64,fp32,tf32,bf16,fp16}`.

## Slurm on this cluster

Partitions ([docs](https://hpc.hse.ru/instructions/run)):

| Partition | What it is |
|---|---|
| `rocky` | default queue, Rocky Linux 9 |
| `test` | short debug jobs, higher start priority |
| `gpu-ef-quick` | **preemptible** debug/educational queue on the E/F GPU nodes — what the smoke test uses, so it does not spend project GPU hours on a connectivity check |
| `cpu-e-quick` | preemptible CPU-only queue on the E-type GPU nodes |

```bash
sbatch -p gpu-ef-quick -G 1 --cpus-per-gpu=2 --time=00:05:00 job.sbatch
sbatch -N 2 -G 8                      # 8 GPUs over 2 nodes
sbatch --gpus-per-task=1 -n 4         # 1 GPU per task
srun --pty --ntasks=4 --gpus=1 bash   # interactive shell with a GPU
sbatch -C "type_e|type_f" ...         # pin node type (quotes required)
mj / mj --start / squeue --start      # own jobs / estimated start
freenodes                             # current load
scancel <job_id>
```

GPUs are requested with `--gpus` / `-G` (**not** `--gres`). Default wall-clock
limit is 24 h; a shorter `--time` backfills sooner. `-A <project_id>` selects the
project account if the allocation is charged per project.

GPU node types: E = 8× A100 80 GB SXM, F = 2× H100 80 GB, H/H′ = 8×/4× H200
141 GB, A/B/C = 4× V100 32 GB. 188 GPUs over 51 nodes.

Storage: `/home/$USER` and `/scratch/$USER` (both Lustre; prefer scratch for
datasets), plus a node-local `$TMPDIR` on SSD that is wiped when the job ends.
`checkquota` shows the disk quota — there is no documented command for remaining
CPU/GPU hours, so that goes through [support](https://hpc.hse.ru/support).

## Budget

~50 000 CPU-hours and ~5 000 GPU-hours remain, valid until **2026-12-31**.
A reserve must stay untouched for the reviewer response, **5–18 Nov 2026**.
This smoke test costs ~1 GPU-minute on a preemptible queue.
