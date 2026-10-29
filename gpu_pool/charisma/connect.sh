#!/usr/bin/env bash
# Run a command on the cHARISMa login node (or open a shell with no arguments).
# Uses the key loaded in ssh-agent; never prompts for the passphrase (BatchMode).
#   gpu_pool/charisma/connect.sh 'mj; checkquota'
set -euo pipefail
KEY="${CHARISMA_KEY:-$HOME/.ssh/charisma_auth_key.pem}"
OPTS=(-p 2222 -o IdentitiesOnly=yes -i "$KEY")
if [ $# -eq 0 ]; then
  exec ssh "${OPTS[@]}" sptsinman@cluster.hpc.hse.ru
fi
exec ssh "${OPTS[@]}" -o BatchMode=yes sptsinman@cluster.hpc.hse.ru "$*"
