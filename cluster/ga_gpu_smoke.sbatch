#!/bin/bash
#SBATCH --job-name=ga-gpu-smoke
#SBATCH --partition=gpu-ef-quick
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=00:05:00
#SBATCH --output=ga_gpu_smoke.%j.log
#SBATCH --error=ga_gpu_smoke.%j.log

# One-minute GPU smoke test for cHARISMa: see cluster/README.md.
#
# Submit from this directory:   sbatch ga_gpu_smoke.sbatch
# Watch:                        mj           (own jobs)   /  squeue --start
# Result:                       ga_gpu_smoke.<jobid>.log  -- grep for PASS or FAIL
#
# Partition note: gpu-ef-quick is the preemptible debug queue on the E/F GPU
# nodes, which is the cheapest place to run this and may be preempted -- fine
# for a 60 s test, and the point is to avoid spending the project's GPU hours
# on a connectivity check.  Swap to a regular GPU partition (see `sinfo`) for
# anything real.  If the project is charged per account, add:
#   #SBATCH --account=<project_id>

set -uo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

# --- Python with a CUDA-enabled torch ---------------------------------------
# Compute nodes have no internet access, so torch must already be on the node:
# either from a module or from a conda env built beforehand on the login node.
# The public docs do not list the module names, so point GA_SMOKE_PYTHON at the
# interpreter you want, e.g.
#   export GA_SMOKE_PYTHON=~/miniconda3/envs/ga/bin/python
# or uncomment and fix a module line once `module avail` below shows the names:
#   module load Python/3.11 CUDA/12.1
PYTHON="${GA_SMOKE_PYTHON:-python3}"

# --- Record the allocation ---------------------------------------------------
# The first run doubles as documentation: these dumps answer the questions the
# public docs leave open (module names, partitions, node types, quota), so the
# log is worth keeping. `|| true` throughout: a missing command must not fail
# the job, since the GPU test below is the actual point.
echo "######## allocation ########"
date
echo "submit dir : $SCRIPT_DIR"
echo "nodelist   : ${SLURM_JOB_NODELIST:-?}"
echo "partition  : ${SLURM_JOB_PARTITION:-?}"
scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | sed -n '1,25p' || true

echo; echo "######## nvidia-smi ########"
nvidia-smi 2>&1 || echo "nvidia-smi unavailable -- no GPU on this node?"

echo; echo "######## module avail (records the module names for next time) ########"
module avail 2>&1 | head -120 || true

echo; echo "######## cluster state ########"
sinfo -o '%20P %5a %10l %6D %10t %N' 2>&1 | head -30 || true
freenodes 2>&1 | head -30 || true
checkquota 2>&1 | head -20 || true

echo; echo "######## python ########"
echo "interpreter: $PYTHON"
"$PYTHON" -c 'import torch, sys; print("torch", torch.__version__, "cuda build", torch.version.cuda)' 2>&1 || {
  echo
  echo "FAIL: '$PYTHON' cannot import torch."
  echo "      Set GA_SMOKE_PYTHON to an interpreter that has a CUDA torch build,"
  echo "      or add the right 'module load' above -- see the module avail dump."
  exit 1
}

# --- The actual test ---------------------------------------------------------
echo; echo "######## ga_gpu_smoke ########"
"$PYTHON" "$SCRIPT_DIR/ga_gpu_smoke.py" --seconds 60 --signature 3,0,0
status=$?

echo; echo "######## done, exit $status ########"
exit $status
