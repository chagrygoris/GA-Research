# GPU Pool — fair sharing of shared accelerator quota

A team shares several accelerator accounts. This answers **whose account can run this job right
now**, and sends the job there.

`router.py` reports each account's weekly accelerator quota, what it is already running, which
GPU types it can reach, and picks the freest account. `launcher.py` pushes a notebook to the
account the router picks.

The backend is the official `kaggle` CLI (Kaggle/kaggle-cli 2.x) rather than raw HTTP, so the
undocumented REST surface is no longer our problem.

## Requirements

```bash
pip install kaggle          # needs Python 3.11+; the CLI is found on PATH or via $KAGGLE_CLI
```

Commands used: `kaggle quota`, `kaggle kernels list --mine`, `kaggle kernels status|logs|files`,
`kaggle kernels push`, `kaggle config view`.

## Credentials

One directory per account holding a `token` file (a new-style access token from
**Settings -> API -> Create New Token**), passed to the CLI as `KAGGLE_API_TOKEN`:

```
~/pool_tokens/
├── account1_alice/token
├── account2_bob/token
└── ...
```

Legacy `kaggle.json` files (`{"username": ..., "key": ...}`) still work and may be mixed into the
same directory; they are passed as `KAGGLE_USERNAME` / `KAGGLE_KEY`. Tokens carry no username, so
the router resolves it with `kaggle config view`.

Every CLI call runs with its own `KAGGLE_CONFIG_DIR` and with ambient `KAGGLE_*` variables
stripped, so accounts never read each other's state or your own `~/.kaggle`.

Point the tools at the directory with `--tokens-dir`, or export `KAGGLE_TOKENS_DIR`.
Keep it outside the repo (`.gitignore` blocks `kaggle*.json` as a backstop).

## CLI

```bash
export KAGGLE_TOKENS_DIR=~/pool_tokens

python gpu_pool/router.py                           # full table
python gpu_pool/router.py --running                 # only active kernels
python gpu_pool/router.py --best --idle-only \
                              --min-gpu-hours 6         # pick one account to use
python gpu_pool/router.py --best --exclude-premium      # ... but never a premium-capable one
python gpu_pool/router.py --no-kernels              # quota only (~1s, skips status calls)
python gpu_pool/router.py --json                    # machine readable
```

Example:

```
ACCOUNT                  KAGGLE USER      GPU LEFT   TPU LEFT   RUNNING  STATE
------------------------------------------------------------------------------
account5_alice           alice             30.0/30h   20.0/20h  0        FREE
account1_bob             bob               21.1/30h   20.0/20h  1        BUSY
```

## Library

```python
from gpu_pool.router import PoolRouter

router = PoolRouter.from_dir("~/pool_tokens")
router.probe_all()

pick = router.best(min_gpu_hours=6, require_idle=True)   # AccountStatus or None
print(pick.username, pick.gpu.remaining_h)

pick.account.write_kaggle_json()        # install as ~/.kaggle/kaggle.json
env = pick.account.env()                # {"KAGGLE_USERNAME": ..., "KAGGLE_KEY": ...}

router.running()                        # every running/queued kernel in the pool
router.total_remaining_hours("gpu")
```

## What the numbers mean

| Field | Source | Notes |
|---|---|---|
| `gpu` / `tpu` quota | `kaggle quota --format json` | Weekly budget: 30 GPU-h, 20 TPU-h per account |
| `remaining_h` | the CLI's own `remaining` column | Hours left this week |
| `quota_refresh` | `refreshAt` | Weekly reset timestamp (UTC) |
| `active_kernels` | `kernels list --mine` + `kernels status` | Statuses `running`, `queued`, `cancelRequested` |
| `free_slots` | `--max-concurrent` (default 2) | Confirmed by Kaggle refusing a 3rd push: "Maximum batch GPU session count of 2 reached" |

### `--exclude-premium`

One weekly 30-hour budget covers every GPU type, so an hour spent on T4 by a competition-entered
account is an hour that account cannot spend on an RTX Pro 6000 or L4 — and the plain accounts
can never spend it there at all. `--exclude-premium` (on `router.py` and `launcher.py`, and
`exclude_premium=` on `available()` / `best()` / `pick_account()` / `launch()`) drops every
account that can reach a competition-gated accelerator, so ordinary T4 work lands on the plain
ones and the premium quota stays where it is the only option. It forces the competition probe,
since `can_use` otherwise guesses from observed history.

Only the `--recent N` (default 10) most recently run kernels per account are status-checked, and
only those run within `--recent-hours` (default 24), since each check is one CLI call and an older
kernel cannot still be running. Raise either if a long-lived session is missed.

## Caveats

- Kaggle's Terms of Service treat accounts as personal and prohibit sharing credentials. This
  tool coordinates a pool of accounts; using it to run work on someone else's account is between
  you and that ToS.
- Quota figures are per account and reset weekly (`quotaRefreshTime`), not per notebook.

---

# Launching runs: `launcher.py`

`PoolLauncher` turns "run this notebook somewhere with a T4" into a push, choosing the account
via `PoolRouter`. Accelerator choice rides in the push metadata as `machineShape` — the legacy
`kaggle.json` basic-auth path carries it fine, no new CLI needed.

## CLI

```bash
python gpu_pool/launcher.py     --tokens-dir ~/pool_tokens     --title clifford-flow-default     --repo https://github.com/optimist1938/Clifford-Flow-Matching.git     --command "python -m src.main"     --accelerator t4 --min-hours 6 --idle-only --wait 3600
```

`--dry-run` prints the chosen account and the exact push body without touching Kaggle.

## Library

```python
from gpu_pool import PoolRouter, PoolLauncher, git_run_spec

router = PoolRouter.from_dir("~/pool_tokens"); router.probe_all()
launcher = PoolLauncher(router)

spec = git_run_spec(
    title="clifford-flow-default",
    repo="https://github.com/optimist1938/Clifford-Flow-Matching.git",
    command="python -m src.main",      # defaults => --model clifford
    accelerator="t4",
)
handle = launcher.launch(spec, min_hours=6, require_idle=True)
print(handle.url, handle.wait(timeout=3600))
```

`git_run_spec` generates a four-cell notebook: preflight (python version + `nvidia-smi`, plus a
secret check when secrets are declared) -> `git clone --depth 1` -> install -> run. For an existing
notebook use `NotebookSpec(title=..., notebook_path="run.ipynb", accelerator="l4")`.

`launch_many([...])` spreads several specs across distinct accounts.

## Accelerators

| Alias | `machineShape` | Notes |
|---|---|---|
| `cpu` / `none` | *(omitted)* | No accelerator quota consumed |
| `t4` | `NvidiaTeslaT4` | Confirmed in use across the pool |
| `p100` | `NvidiaTeslaP100` | Default image's torch may lack `sm_60` kernels |
| `l4` | `NvidiaL4` | Confirmed |
| `rtx6000` | `NvidiaRtxPro6000` | Confirmed |
| `tpu-v3` / `tpu-v5e` | `Tpu1VmV38` / `TpuV5E8` | **Accepted but silently non-TPU** ([#1197](https://github.com/Kaggle/kaggle-cli/issues/1197)) |

There is no value for the editor's "GPU T4 ×2" ([#1196](https://github.com/Kaggle/kaggle-cli/issues/1196)).

## Secrets are per-account and cannot be provisioned via the API

There is no secrets endpoint. If a notebook calls `UserSecretsClient` (e.g. `wandb_api_key`, a
GitHub PAT for a private repo), each account must have that secret added by hand under
**Add-ons > Secrets**, or the run fails on a routed account. Declare them via
`required_secrets=[...]` / `--secret` so the preflight cell fails in seconds instead of hours.

The Clifford-Flow-Matching default run needs **no secrets**: the repo is public, and `run_name`
defaults to `None`, which makes `wandb_create_run` a no-op.

## Logs

`kaggle kernels logs` returns **nothing while a kernel is running**, then the whole log once it
finishes. The live stream is a separate mode:

```python
handle.log()                      # empty until the run completes
handle.log(follow=True)           # --follow: streams the live session for 30s, then returns
router.kernel_logs(acct, ref, follow=True, follow_seconds=60)
```

A training log is almost entirely progress-bar redraws — one real pose run gave 41,226 lines of
which 129 carried timings, losses and metrics — so pass it through `filter_progress`:

```python
from gpu_pool import filter_progress
for line in filter_progress(handle.log(follow=True), keep_last=10):
    print(line)
```

`handle.status()` gives `queued` -> `running` -> `complete` / `error` / `cancelAcknowledged`.

There is no cancel: neither the CLI nor the API exposes one, so `handle.cancel()` raises and
points at the notebook page instead.

## Tests

```bash
python3 gpu_pool/test_gpu_pool.py     # 31 offline tests, no network or credentials
```
