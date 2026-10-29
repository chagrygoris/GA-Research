"""Run jobs on a Slurm cluster over SSH: the HSE cHARISMa backend of the GPU pool.

A Slurm cluster is the second kind of compute in the pool, next to the Kaggle accounts of
``router.py``. Where a Kaggle account has a weekly quota and two session slots, a Slurm project has
a **fixed GPU-hour budget until an end date** and a **QOS cap** (Slurm's per-user limit on GPUs in
use at once). Everything goes through ``ssh`` to the login node; nothing is installed there.

* ``sacct``  -> the project's GPU-hours spent so far, and this user's queued / running jobs
* ``sinfo``  -> how many GPUs of each node type are idle right now
* ``sbatch`` / ``scancel`` -> submit and cancel (unlike Kaggle, a Slurm job *can* be cancelled)

Compute nodes have no internet, so a run is staged on the login node first: the repository is
cloned there, an optional ``setup`` command (package installs only, never compute) runs there, and
the job script only runs the command. W&B runs in offline mode (``WANDB_MODE=offline``) unless the
spec says otherwise; sync later from the login node with ``wandb sync``.

The cluster's own operating rules live in ``gpu_pool/charisma/GUIDE.md``; ``CHARISMA`` below
encodes the ones the code can enforce.

CLI
---
    python gpu_pool/slurm.py                     # status of cHARISMa: budget, GPUs in use, idle GPUs
    python gpu_pool/slurm.py --json
    python gpu_pool/slurm.py --jobs              # only this user's queued / running jobs

Launching goes through the pool interface (``gpu_pool/pool.py``), which applies the approval rule.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: Slurm job states -> the pool's state vocabulary (the one ``router.ACTIVE_STATES`` uses).
#: ``preempted`` is extra: a backfill job on ``gpu-ef-quick`` can be pushed off its node.
_STATE_MAP = {
    "PENDING": "queued", "CONFIGURING": "queued", "REQUEUED": "queued", "REQUEUE_HOLD": "queued",
    "RESV_DEL_HOLD": "queued", "SUSPENDED": "queued",
    "RUNNING": "running", "COMPLETING": "running", "STAGE_OUT": "running", "SIGNALING": "running",
    "COMPLETED": "complete",
    "CANCELLED": "cancelAcknowledged",
    "PREEMPTED": "preempted",
    "FAILED": "error", "TIMEOUT": "error", "OUT_OF_MEMORY": "error", "NODE_FAIL": "error",
    "BOOT_FAIL": "error", "DEADLINE": "error",
}
ACTIVE = frozenset({"queued", "running"})
TERMINAL = frozenset({"complete", "error", "cancelAcknowledged", "preempted"})


def slurm_state(raw: str) -> str:
    """``"CANCELLED by 123"`` / ``"RUNNING"`` -> ``"cancelAcknowledged"`` / ``"running"``."""
    word = (raw or "").strip().split(" ", 1)[0].rstrip("+").upper()
    return _STATE_MAP.get(word, word.lower() or "unknown")


def _elapsed_hours(text: str) -> float:
    """Slurm duration ``[D-]HH:MM:SS`` / ``MM:SS`` / ``UNLIMITED`` -> hours (0 if unparsable)."""
    text = (text or "").strip()
    days = 0
    if "-" in text:
        head, text = text.split("-", 1)
        days = int(head) if head.isdigit() else 0
    parts = [p for p in text.split(":") if p != ""]
    if not parts or not all(p.replace(".", "", 1).isdigit() for p in parts):
        return 0.0
    nums = [float(p) for p in parts]
    while len(nums) < 3:
        nums.insert(0, 0.0)
    h, m, s = nums[-3:]
    return days * 24 + h + m / 60 + s / 3600


def tres_gpus(tres: str) -> int:
    """GPUs in a TRES string: ``"cpu=2,gres/gpu=1,node=1"`` or ``"gres/gpu:v100=2"`` -> 1 / 2."""
    total = 0
    for match in re.finditer(r"gres/gpu(?::[\w.-]+)?[=:](\d+)", tres or ""):
        total = max(total, int(match.group(1)))   # typed and untyped entries count the same GPUs
    return total


def tres_cpus(tres: str) -> int:
    match = re.search(r"(?:^|,)cpu=(\d+)", tres or "")
    return int(match.group(1)) if match else 0


def _hours_fmt(hours: float) -> str:
    """Hours -> Slurm ``--time`` value, rounded up to the minute: 1.5 -> ``"01:30:00"``."""
    minutes = max(1, int(-(-hours * 60 // 1)))
    days, minutes = divmod(minutes, 24 * 60)
    text = "%02d:%02d:00" % divmod(minutes, 60)
    return "%d-%s" % (days, text) if days else text


def _slug(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")
    return slug[:40] or "run"


class SlurmError(RuntimeError):
    """An ``ssh`` / Slurm command failed, or a job request breaks one of the cluster's rules."""


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Partition:
    """Where one accelerator lives: Slurm partition, node constraint and limits."""

    partition: str
    constraint: Optional[str]
    name: str                 #: readable hardware name
    max_hours: float          #: partition time limit
    gpus_per_node: int
    preemptible: bool = False
    #: Moon Cluster planet this hardware is drawn on (several accelerators may share one).
    planet: str = ""


@dataclass(frozen=True)
class SlurmConfig:
    """A Slurm project reachable over SSH. Holds no secret: the key stays in ssh-agent."""

    name: str
    title: str
    site: str
    host: str
    port: int
    user: str
    key_path: str
    account: str
    #: QOS caps per user: GPUs and CPU cores in use at once.
    gpu_cap: int
    cpu_cap: int
    #: Project budget: GPU-hours from ``budget_start`` until ``budget_end`` (inclusive).
    budget_gpu_h: float
    budget_cpu_h: float
    budget_start: str
    budget_end: str
    accelerators: Dict[str, Partition] = field(default_factory=dict)
    #: Debug partition: short limit, high priority, used for smoke tests.
    test_partition: str = "test"
    test_max_hours: float = 0.5
    #: Date windows (inclusive ISO dates) reserved by the PI. A job larger than
    #: ``reserve_max_gpu_h`` that overlaps one needs ``allow_reserve=True``.
    reserve_windows: Tuple[Tuple[str, str], ...] = ()
    reserve_max_gpu_h: float = 4.0
    python_env: str = ""
    #: Run directories are created under this path, relative to the login home directory.
    remote_root: str = "gpu_pool/runs"
    cpus_per_gpu: int = 4
    docs: str = ""

    def ssh_base(self) -> List[str]:
        """The non-interactive ``ssh`` prefix from the cluster guide (no prompts, one key)."""
        return [
            "ssh", "-p", str(self.port),
            "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
            "-o", "ConnectTimeout=20",
            "-i", os.path.expanduser(self.key_path),
            "%s@%s" % (self.user, self.host),
        ]

    def accelerator(self, label: str) -> Partition:
        try:
            return self.accelerators[label]
        except KeyError:
            raise SlurmError("%s has no accelerator %r; choose from %s"
                             % (self.title, label, ", ".join(sorted(self.accelerators)))) from None

    @classmethod
    def from_file(cls, path: str) -> "SlurmConfig":
        """A JSON file with any of the fields above, applied on top of ``CHARISMA``."""
        with open(os.path.expanduser(path)) as fh:
            blob = json.load(fh)
        accels = blob.pop("accelerators", None)
        if "reserve_windows" in blob:
            blob["reserve_windows"] = tuple(tuple(w) for w in blob["reserve_windows"])
        config = dataclasses.replace(CHARISMA, **blob)
        if accels is not None:
            config = dataclasses.replace(config, accelerators={k: Partition(**v) for k, v in accels.items()})
        return config


#: HSE cHARISMa, project proj_1898, as verified on 2026-10-29 (gpu_pool/charisma/GUIDE.md).
#: Under the ``students`` QOS the default partition ``rocky`` gives V100 32 GB nodes (type_a/b/c);
#: ``gpu-ef-quick`` is a preemptible backfill queue with A100 / H100 / H200 that is usually full.
CHARISMA = SlurmConfig(
    name="charisma",
    title="cHARISMa",
    site="HSE",
    host="cluster.hpc.hse.ru",
    port=2222,
    user="sptsinman",
    key_path="~/.ssh/charisma_auth_key.pem",
    account="proj_1898",
    gpu_cap=4,
    cpu_cap=48,
    budget_gpu_h=5000.0,
    budget_cpu_h=50000.0,
    budget_start="2026-01-01",
    budget_end="2026-12-31",
    accelerators={
        "v100": Partition("rocky", "type_a|type_b|type_c", "V100 32GB", 720.0, 4, planet="v100"),
        "a100": Partition("gpu-ef-quick", "type_e", "A100 80GB", 3.0, 8, True, planet="hpc-ef"),
        "h100": Partition("gpu-ef-quick", "type_f", "H100", 3.0, 2, True, planet="hpc-ef"),
        "h200": Partition("gpu-ef-quick", "type_h", "H200 141GB", 3.0, 8, True, planet="hpc-ef"),
        "hpc-cpu": Partition("rocky", "type_d", "CPU node", 720.0, 0, planet=""),
    },
    reserve_windows=(("2026-11-05", "2026-11-18"),),
    python_env="/opt/software/python/envs/google_colab_gpu_2025",
    docs="gpu_pool/charisma/GUIDE.md",
)

CLUSTERS: Dict[str, SlurmConfig] = {CHARISMA.name: CHARISMA}


def cluster_config(name_or_path: str) -> SlurmConfig:
    """A preset name (``"charisma"``) or a JSON config file."""
    if name_or_path in CLUSTERS:
        return CLUSTERS[name_or_path]
    if os.path.isfile(os.path.expanduser(name_or_path)):
        return SlurmConfig.from_file(name_or_path)
    raise KeyError("unknown cluster %r; presets: %s" % (name_or_path, ", ".join(CLUSTERS)))


# --------------------------------------------------------------------------------------
# what to run
# --------------------------------------------------------------------------------------
@dataclass
class SlurmJobSpec:
    """One batch job. ``hours`` becomes ``--time``, so it bounds both runtime and cost."""

    title: str
    command: str
    hours: float
    accelerator: str = "v100"
    gpus: int = 1
    repo: Optional[str] = None
    branch: Optional[str] = None
    subdir: Optional[str] = None
    #: Run on the **login node** after cloning, e.g. ``pip install --user -e .``. Installs only.
    setup: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    cpus_per_task: Optional[int] = None
    mem: Optional[str] = None
    #: Use the debug partition (30 min max, high priority) instead of the accelerator's own.
    smoke: bool = False
    array: Optional[str] = None
    #: Python environment directory put first on PATH; ``None`` takes the cluster default.
    python_env: Optional[str] = None

    @property
    def slug(self) -> str:
        return _slug(self.title)

    @property
    def gpu_hours(self) -> float:
        """Upper bound on GPU-hours billed: GPUs x time limit (billing counts idle GPUs too)."""
        return self.gpus * self.hours * (self._array_size() or 1)

    def _array_size(self) -> int:
        if not self.array:
            return 0
        body = self.array.split("%", 1)[0]
        count = 0
        for part in body.split(","):
            part = part.split(":", 1)[0]
            if "-" in part:
                lo, hi = part.split("-", 1)
                count += int(hi) - int(lo) + 1 if lo.isdigit() and hi.isdigit() else 1
            elif part:
                count += 1
        return count


def sbatch_script(spec: SlurmJobSpec, config: SlurmConfig, workdir: str) -> str:
    """The job script: the cluster guide's template, run in ``workdir`` (relative to home).

    Unlike the guide's template it exits with the command's own status, so a crash shows up
    as FAILED in ``sacct`` instead of COMPLETED.
    """
    part = config.accelerator(spec.accelerator)
    partition = config.test_partition if spec.smoke else part.partition
    cpus = spec.cpus_per_task or max(2, config.cpus_per_gpu * max(spec.gpus, 1))
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=%s" % spec.slug,
        "#SBATCH --output=slurm-%j.out",
        "#SBATCH --error=slurm-%j.err",
        "#SBATCH --account=%s" % config.account,
        "#SBATCH --partition=%s" % partition,
        "#SBATCH --time=%s" % _hours_fmt(spec.hours),
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=%d" % cpus,
    ]
    if spec.gpus:
        lines.append("#SBATCH --gpus=%d" % spec.gpus)
    if part.constraint:
        lines.append('#SBATCH --constraint="%s"' % part.constraint)
    if spec.mem:
        lines.append("#SBATCH --mem=%s" % spec.mem)
    if spec.array:
        lines.append("#SBATCH --array=%s" % spec.array)
    pyenv = spec.python_env if spec.python_env is not None else config.python_env
    env = {"WANDB_MODE": "offline"}
    env.update(spec.env)
    lines += [
        "",
        "set -uo pipefail",
        'echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date)"',
        "nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo 'no GPU visible'",
    ]
    if pyenv:
        lines += [
            "PYENV=%s" % shlex.quote(pyenv),
            'export PATH="$PYENV/bin:$PATH" LD_LIBRARY_PATH="$PYENV/lib:${LD_LIBRARY_PATH:-}"',
        ]
    lines += ["export %s=%s" % (k, shlex.quote(str(v))) for k, v in env.items()]
    run_dir = "src/%s" % spec.subdir.strip("/") if spec.repo and spec.subdir else ("src" if spec.repo else ".")
    lines += [
        'cd "$HOME/%s/%s"' % (workdir, run_dir),
        spec.command,
        "status=$?",
        'echo "exit=$status end=$(date)"',
        "exit $status",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# probe results
# --------------------------------------------------------------------------------------
@dataclass
class SlurmJob:
    """One job of this user, as ``sacct`` lists it."""

    job_id: str
    name: str
    state: str                 #: pool vocabulary: queued / running / complete / error / ...
    partition: str
    gpus: int
    cpus: int
    elapsed_h: float
    limit_h: float
    raw_state: str = ""

    @property
    def is_active(self) -> bool:
        return self.state in ACTIVE

    def accelerator(self, config: SlurmConfig) -> str:
        """Best guess at the hardware from the partition (``rocky`` -> ``v100``)."""
        for label, part in config.accelerators.items():
            if part.partition == self.partition and (part.gpus_per_node > 0) == (self.gpus > 0):
                return label
        return self.partition

    def as_dict(self) -> Dict[str, Any]:
        return {"job_id": self.job_id, "name": self.name, "state": self.state, "slurm_state": self.raw_state,
                "partition": self.partition, "gpus": self.gpus, "cpus": self.cpus,
                "elapsed_h": round(self.elapsed_h, 2), "limit_h": round(self.limit_h, 2)}


@dataclass
class ClusterStatus:
    """Probe result for one Slurm project."""

    config: SlurmConfig
    ok: bool = False
    error: Optional[str] = None
    spent_gpu_h: float = 0.0
    spent_cpu_h: float = 0.0
    jobs: List[SlurmJob] = field(default_factory=list)
    #: accelerator label -> (idle GPUs, total GPUs) on nodes that are up, from ``sinfo``.
    idle: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    taken_at: Optional[datetime] = None

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def active_jobs(self) -> List[SlurmJob]:
        return [j for j in self.jobs if j.is_active]

    @property
    def gpus_in_use(self) -> int:
        """GPUs held or requested by this user's queued and running jobs (counted by the QOS)."""
        return sum(j.gpus for j in self.active_jobs)

    @property
    def free_gpus(self) -> int:
        """GPUs this user may still request before hitting the QOS cap."""
        return max(0, self.config.gpu_cap - self.gpus_in_use)

    @property
    def remaining_gpu_h(self) -> float:
        return max(0.0, self.config.budget_gpu_h - self.spent_gpu_h)

    @property
    def days_left(self) -> int:
        today = (self.taken_at or datetime.now(timezone.utc)).date()
        return max(0, (date.fromisoformat(self.config.budget_end) - today).days + 1)

    def reserve_window(self, start: Optional[datetime] = None, hours: float = 0.0) -> Optional[Tuple[str, str]]:
        """The reserve window a job starting at ``start`` and lasting ``hours`` overlaps, if any."""
        start = start or datetime.now(timezone.utc)
        end = start + timedelta(hours=hours)
        for lo, hi in self.config.reserve_windows:
            if start.date() <= date.fromisoformat(hi) and end.date() >= date.fromisoformat(lo):
                return lo, hi
        return None

    def check(self, spec: SlurmJobSpec, allow_reserve: bool = False) -> List[str]:
        """Why ``spec`` may not be submitted (empty list = fine). Rules from the cluster guide."""
        problems = []
        part = self.config.accelerator(spec.accelerator)
        limit = self.config.test_max_hours if spec.smoke else part.max_hours
        if spec.hours <= 0:
            problems.append("hours must be > 0: it becomes --time, which Slurm otherwise sets to 24 h")
        elif spec.hours > limit:
            problems.append("%.2f h exceeds the %s partition limit of %g h"
                            % (spec.hours, self.config.test_partition if spec.smoke else part.partition, limit))
        if spec.gpus > self.config.gpu_cap:
            problems.append("%d GPUs exceeds the QOS cap of %d per user" % (spec.gpus, self.config.gpu_cap))
        if part.gpus_per_node and spec.gpus > part.gpus_per_node:
            problems.append("%d GPUs do not fit on one %s node (%d per node; no multi-node GPU jobs under this QOS)"
                            % (spec.gpus, part.name, part.gpus_per_node))
        if self.ok and spec.gpu_hours > self.remaining_gpu_h:
            problems.append("%.1f GPU-h requested but only %.1f left in the project budget"
                            % (spec.gpu_hours, self.remaining_gpu_h))
        window = self.reserve_window(hours=spec.hours)
        if window and spec.gpu_hours > self.config.reserve_max_gpu_h and not allow_reserve:
            problems.append("%.1f GPU-h overlaps the PI's reserve window %s to %s; ask first, then pass "
                            "allow_reserve / --allow-reserve" % (spec.gpu_hours, window[0], window[1]))
        return problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "title": self.config.title, "site": self.config.site, "ok": self.ok,
            "error": self.error, "account": self.config.account,
            "budget_gpu_h": self.config.budget_gpu_h, "spent_gpu_h": round(self.spent_gpu_h, 1),
            "remaining_gpu_h": round(self.remaining_gpu_h, 1), "budget_end": self.config.budget_end,
            "spent_cpu_h": round(self.spent_cpu_h, 1),
            "gpu_cap": self.config.gpu_cap, "gpus_in_use": self.gpus_in_use, "free_gpus": self.free_gpus,
            "idle": {k: list(v) for k, v in self.idle.items()},
            "active_jobs": [j.as_dict() for j in self.active_jobs],
            "reserve_windows": [list(w) for w in self.config.reserve_windows],
        }


# --------------------------------------------------------------------------------------
# the cluster
# --------------------------------------------------------------------------------------
Runner = Callable[[List[str], Optional[str], float], subprocess.CompletedProcess]


def _subprocess_runner(argv: List[str], stdin: Optional[str], timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise SlurmError("ssh not found; install an OpenSSH client") from exc
    except subprocess.TimeoutExpired as exc:
        raise SlurmError("ssh timed out after %ss" % timeout) from exc


_SACCT_FIELDS = "JobID,JobName,State,Partition,ElapsedRaw,TimelimitRaw,AllocTRES,ReqTRES"


class SlurmCluster:
    """Probe a Slurm project over SSH and submit jobs to it.

    >>> cluster = SlurmCluster(CHARISMA)
    >>> cluster.probe().remaining_gpu_h          # doctest: +SKIP
    """

    def __init__(self, config: SlurmConfig = CHARISMA, runner: Optional[Runner] = None,
                 timeout: float = 60.0) -> None:
        self.config = config
        self.runner = runner or _subprocess_runner
        self.timeout = timeout
        self._status: Optional[ClusterStatus] = None

    @property
    def name(self) -> str:
        return self.config.name

    # -- ssh ---------------------------------------------------------------------------
    def run(self, command: str, stdin: Optional[str] = None, check: bool = True,
            timeout: Optional[float] = None) -> subprocess.CompletedProcess:
        """Run ``command`` in a login shell on the login node (in the home directory)."""
        proc = self.runner(self.config.ssh_base() + [command], stdin, timeout or self.timeout)
        if check and proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            hint = ""
            if proc.returncode == 255 and "Permission denied" in (proc.stderr or ""):
                hint = " (is the key loaded? `ssh-add -l`; see %s)" % self.config.docs
            raise SlurmError("%s: `%s` failed (exit %d): %s%s" % (
                self.config.title, command.split(" ", 1)[0], proc.returncode,
                detail[-1] if detail else "no output", hint))
        return proc

    # -- probing -----------------------------------------------------------------------
    def _sacct_rows(self, args: str) -> List[List[str]]:
        out = self.run("sacct -X -n -P -o %s %s" % (_SACCT_FIELDS, args)).stdout or ""
        return [line.split("|") for line in out.splitlines() if line.count("|") >= 7]

    @staticmethod
    def _job(row: List[str]) -> SlurmJob:
        job_id, name, state, partition, elapsed, limit, alloc, req = row[:8]
        tres = alloc if tres_gpus(alloc) or tres_cpus(alloc) else req
        return SlurmJob(
            job_id=job_id, name=name, state=slurm_state(state), raw_state=state, partition=partition,
            gpus=tres_gpus(tres), cpus=tres_cpus(tres),
            elapsed_h=(int(elapsed) / 3600.0) if elapsed.isdigit() else 0.0,
            limit_h=(int(limit) / 60.0) if limit.isdigit() else 0.0,
        )

    def jobs(self) -> List[SlurmJob]:
        """This user's queued and running jobs."""
        rows = self._sacct_rows("-u %s -s PD,R,CG,RQ,S" % shlex.quote(self.config.user))
        return [self._job(r) for r in rows]

    def spent(self) -> Tuple[float, float]:
        """(GPU-hours, CPU-hours) billed to the project since ``budget_start``.

        Billing is per allocated GPU per hour, so this is elapsed time x allocated GPUs over every
        job of the account that ``sacct`` shows (all users where Slurm permits, else this user's).
        """
        rows = self._sacct_rows("-a -A %s -S %s -E now" % (
            shlex.quote(self.config.account), shlex.quote(self.config.budget_start)))
        gpu = cpu = 0.0
        for row in rows:
            job = self._job(row)
            gpu += job.elapsed_h * tres_gpus(row[6])
            cpu += job.elapsed_h * tres_cpus(row[6])
        return gpu, cpu

    def idle_gpus(self) -> Dict[str, Tuple[int, int]]:
        """Accelerator -> (idle, total) GPUs on nodes that are up, from ``sinfo``."""
        out = self.run("sinfo -h -N -O NodeHost:40,StateCompact:16,Gres:60,GresUsed:80,Features:80",
                       check=False).stdout or ""
        nodes: Dict[str, Tuple[str, int, int, set]] = {}
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            host, state, gres, used, features = parts[:5]
            nodes[host] = (state, tres_gpus("gres/" + gres.split("(")[0]) if "gpu" in gres else 0,
                           tres_gpus("gres/" + used.split("(")[0]) if "gpu" in used else 0,
                           set(features.split(",")))
        out_counts: Dict[str, Tuple[int, int]] = {}
        for label, part in self.config.accelerators.items():
            if not part.gpus_per_node or not part.constraint:
                continue
            wanted = set(part.constraint.split("|"))
            idle = total = 0
            for state, have, used, feats in nodes.values():
                if not (feats & wanted) or have == 0:
                    continue
                if re.match(r"(down|drain|fail|maint|unk|boot|futr|resv)", state.rstrip("*~#!%$@^-")):
                    continue
                total += have
                idle += max(0, have - used)
            out_counts[label] = (idle, total)
        return out_counts

    def probe(self, with_idle: bool = True) -> ClusterStatus:
        status = ClusterStatus(config=self.config, taken_at=datetime.now(timezone.utc))
        try:
            status.jobs = self.jobs()
            status.spent_gpu_h, status.spent_cpu_h = self.spent()
            status.ok = True
        except SlurmError as exc:
            status.error = str(exc)
            self._status = status
            return status
        if with_idle:
            try:
                status.idle = self.idle_gpus()
            except SlurmError:
                pass   # idle counts are a nicety; budget and jobs are what routing needs
        self._status = status
        return status

    @property
    def status(self) -> ClusterStatus:
        return self._status or self.probe()

    # -- submitting --------------------------------------------------------------------
    def submit(self, spec: SlurmJobSpec, allow_reserve: bool = False, dry_run: bool = False) -> "SlurmHandle":
        """Stage ``spec`` on the login node and ``sbatch`` it. Raises SlurmError on a rule break."""
        problems = self.status.check(spec, allow_reserve=allow_reserve)
        if problems:
            raise SlurmError("%s refuses %s: %s" % (self.config.title, spec.title, "; ".join(problems)))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        workdir = "%s/%s-%s" % (self.config.remote_root.strip("/"), spec.slug, stamp)
        script = sbatch_script(spec, self.config, workdir)
        stage = ["set -e", "mkdir -p %s" % shlex.quote(workdir), "cd %s" % shlex.quote(workdir)]
        if spec.repo:
            stage.append("git clone -q --depth 1 %s%s src" % (
                ("--branch %s " % shlex.quote(spec.branch)) if spec.branch else "", shlex.quote(spec.repo)))
            stage.append("git -C src log -1 --oneline")
            if spec.setup:
                stage.append("(cd src/%s && %s)" % (shlex.quote((spec.subdir or ".").strip("/") or "."), spec.setup))
        stage += ["cat > run.sbatch", "sbatch --parsable run.sbatch"]
        command = "\n".join(stage)
        if dry_run:
            print("# login-node staging (%s)\n%s\n\n# run.sbatch\n%s" % (self.config.title, command, script))
            return SlurmHandle(self, "dry-run", workdir, spec)
        # Cloning and installing can take minutes on the login node.
        proc = self.run(command, stdin=script, timeout=max(self.timeout, 900.0))
        match = re.search(r"^(\d+)(?:;\S+)?\s*$", (proc.stdout or "").strip().splitlines()[-1] if proc.stdout else "")
        if not match:
            raise SlurmError("sbatch printed no job id: %s" % (proc.stdout or proc.stderr).strip()[-300:])
        return SlurmHandle(self, match.group(1), workdir, spec)

    # -- one job -----------------------------------------------------------------------
    def job_state(self, job_id: str) -> str:
        rows = self._sacct_rows("-j %s" % shlex.quote(job_id))
        if rows:
            return slurm_state(rows[0][2])
        # sacct can lag a few seconds behind a fresh submission
        out = self.run("squeue -h -j %s -o %%T" % shlex.quote(job_id), check=False).stdout or ""
        return slurm_state(out.strip()) if out.strip() else "unknown"

    def job_log(self, job_id: str, lines: int = 40, stream: str = "out") -> str:
        """Last ``lines`` of the job's ``slurm-<id>.out`` (or ``.err``), found under ``remote_root``."""
        pattern = "%s/*/slurm-%s.%s" % (self.config.remote_root.strip("/"), job_id, stream)
        out = self.run("f=$(ls -1 %s 2>/dev/null | head -n 1); [ -n \"$f\" ] && tail -n %d \"$f\" || true"
                       % (pattern, int(lines)), check=False)
        return (out.stdout or "").rstrip()

    def cancel(self, job_id: str) -> None:
        self.run("scancel %s" % shlex.quote(job_id))

    def accounting(self, job_id: str) -> Dict[str, Any]:
        """Final state and GPU-hours billed for one job."""
        rows = self._sacct_rows("-j %s" % shlex.quote(job_id))
        if not rows:
            return {"job_id": job_id, "state": "unknown"}
        job = self._job(rows[0])
        return dict(job.as_dict(), gpu_hours=round(job.elapsed_h * job.gpus, 2))


@dataclass
class SlurmHandle:
    """A submitted job: poll, tail, wait, cancel."""

    cluster: SlurmCluster
    job_id: str
    workdir: str = ""
    spec: Optional[SlurmJobSpec] = None

    @property
    def ref(self) -> str:
        return "%s:%s" % (self.cluster.name, self.job_id)

    def status(self) -> str:
        return self.cluster.job_state(self.job_id)

    def log(self, lines: int = 40) -> str:
        return self.cluster.job_log(self.job_id, lines)

    def cancel(self) -> None:
        self.cluster.cancel(self.job_id)

    def wait(self, timeout: float = 3600.0, poll: float = 60.0, verbose: bool = True) -> str:
        deadline, last = time.time() + timeout, ""
        while time.time() < deadline:
            state = self.status()
            if verbose and state != last:
                print("[%s] %s %s" % (time.strftime("%H:%M:%S"), self.ref, state), flush=True)
                last = state
            if state in TERMINAL:
                return state
            time.sleep(poll)
        return last or "timeout"

    def as_dict(self) -> Dict[str, Any]:
        return {"ref": self.ref, "cluster": self.cluster.name, "job_id": self.job_id, "workdir": self.workdir,
                "accelerator": self.spec.accelerator if self.spec else None,
                "gpus": self.spec.gpus if self.spec else None, "hours": self.spec.hours if self.spec else None}


# --------------------------------------------------------------------------------------
# report / CLI
# --------------------------------------------------------------------------------------
def report(status: ClusterStatus) -> str:
    c = status.config
    head = "%s (%s, %s@%s, account %s)" % (c.title, c.site, c.user, c.host, c.account)
    if not status.ok:
        return "%s\n  UNREACHABLE: %s" % (head, status.error)
    lines = [
        head,
        "  budget   %.1f of %.0f GPU-h left until %s (%d days); %.1f GPU-h spent since %s"
        % (status.remaining_gpu_h, c.budget_gpu_h, c.budget_end, status.days_left, status.spent_gpu_h, c.budget_start),
        "  QOS      %d of %d GPUs in use by %s, %d free" % (status.gpus_in_use, c.gpu_cap, c.user, status.free_gpus),
    ]
    for label, (idle, total) in status.idle.items():
        part = c.accelerators[label]
        lines.append("  %-8s %d of %d GPUs idle on %s (%s, max %g h%s)" % (
            label, idle, total, part.name, part.partition, part.max_hours, ", preemptible" if part.preemptible else ""))
    window = status.reserve_window(hours=24 * 14)
    if window:
        lines.append("  reserve  %s to %s is held for the PI: jobs over %g GPU-h need approval"
                     % (window[0], window[1], c.reserve_max_gpu_h))
    for job in status.active_jobs:
        lines.append("  job %-10s %-10s %-12s %d GPU  %.2f/%.2f h  %s"
                     % (job.job_id, job.state, job.partition, job.gpus, job.elapsed_h, job.limit_h, job.name))
    if not status.active_jobs:
        lines.append("  no jobs queued or running")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Status of a Slurm project in the GPU pool.")
    parser.add_argument("--cluster", default="charisma", help="preset name or JSON config (default: charisma)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--jobs", action="store_true", help="only list this user's queued / running jobs")
    args = parser.parse_args(argv)
    cluster = SlurmCluster(cluster_config(args.cluster))
    status = cluster.probe(with_idle=not args.jobs)
    if args.json:
        print(json.dumps(status.as_dict(), indent=2))
    elif args.jobs:
        for job in status.active_jobs:
            print("%-10s %-10s %-12s %d GPU  %s" % (job.job_id, job.state, job.partition, job.gpus, job.name))
    else:
        print(report(status))
    return 0 if status.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
