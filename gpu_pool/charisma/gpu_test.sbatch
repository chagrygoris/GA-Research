#!/bin/bash
# Smoke test from GUIDE.md section 5 (the template ran as job 4467119). Checks that a V100 and torch work.
#   sbatch gpu_test.sbatch        # ~1 min on the test partition
#SBATCH --job-name=gpu_test
#SBATCH --output=gpu_test_%j.out
#SBATCH --error=gpu_test_%j.err
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
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
status=$?
echo "exit=$status end=$(date)"
exit $status
