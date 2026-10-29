"""Offline tests for the Slurm backend and the unified pool: a fake ``ssh`` answers every command.

    python3 gpu_pool/test_slurm_pool.py      # or: pytest gpu_pool/test_slurm_pool.py
"""

import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpu_pool.pool import APPROVAL_HOURS, ComputePool, RunSpec
from gpu_pool.slurm import (
    CHARISMA,
    ClusterStatus,
    SlurmCluster,
    SlurmError,
    SlurmJobSpec,
    _elapsed_hours,
    _hours_fmt,
    sbatch_script,
    slurm_state,
    tres_gpus,
)

SACCT_JOBS = "4467200|train-v100|RUNNING|rocky|3600|120|billing=8,cpu=8,gres/gpu=2,mem=32G,node=1|cpu=8,gres/gpu=2\n"
SACCT_SPENT = (
    "4467119|gpu_test|COMPLETED|test|60|5|billing=2,cpu=2,gres/gpu=1,node=1|cpu=2,gres/gpu=1\n"
    "4467150|sweep|COMPLETED|rocky|36000|720|cpu=16,gres/gpu=4,node=1|cpu=16,gres/gpu=4\n"
    "4467200|train-v100|RUNNING|rocky|3600|120|cpu=8,gres/gpu=2,node=1|cpu=8,gres/gpu=2\n"
)
SINFO = (
    "cn-001  mix   gpu:4(S:0-1)  gpu:2(IDX:0-1)  type_a,v100\n"
    "cn-002  idle  gpu:4(S:0-1)  gpu:0(IDX:N/A)  type_b,v100\n"
    "cn-003  drain gpu:4(S:0-1)  gpu:0(IDX:N/A)  type_c,v100\n"
    "cn-011  alloc gpu:8(S:0-1)  gpu:8(IDX:0-7)  type_e,a100\n"
)


class FakeSsh:
    """Records every remote command and answers like the login node would."""

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def __call__(self, argv, stdin, timeout):
        cmd = argv[-1]
        self.calls.append((argv, cmd, stdin))
        if self.fail:
            return subprocess.CompletedProcess(argv, 255, "", "Permission denied (publickey).")
        out = ""
        if cmd.startswith("sacct") and " -u " in cmd:
            out = SACCT_JOBS
        elif cmd.startswith("sacct") and " -A " in cmd:
            out = SACCT_SPENT
        elif cmd.startswith("sacct") and " -j " in cmd:
            out = "4470001|smoke|PENDING|test|0|30|cpu=2,gres/gpu=1|cpu=2,gres/gpu=1\n"
        elif cmd.startswith("sinfo"):
            out = SINFO
        elif "sbatch --parsable" in cmd:
            out = "abc1234 init\n4470001\n"
        return subprocess.CompletedProcess(argv, 0, out, "")


def _cluster(fake=None):
    return SlurmCluster(CHARISMA, runner=fake or FakeSsh())


# -- parsing -------------------------------------------------------------------------------
def test_parsers():
    assert slurm_state("CANCELLED by 1234") == "cancelAcknowledged"
    assert slurm_state("RUNNING") == "running" and slurm_state("PENDING") == "queued"
    assert slurm_state("TIMEOUT") == "error" and slurm_state("PREEMPTED") == "preempted"
    assert tres_gpus("billing=2,cpu=2,gres/gpu=1,node=1") == 1
    assert tres_gpus("cpu=4,gres/gpu:v100=2,gres/gpu=2") == 2
    assert tres_gpus("cpu=4,mem=8G") == 0
    assert _elapsed_hours("1-02:30:00") == 26.5 and _elapsed_hours("05:00") == 5 / 60
    assert _hours_fmt(1.5) == "01:30:00" and _hours_fmt(0.01) == "00:01:00" and _hours_fmt(30) == "1-06:00:00"


def test_ssh_command_follows_the_cluster_guide():
    fake = FakeSsh()
    _cluster(fake).run("hostname")
    argv = fake.calls[0][0]
    assert argv[:3] == ["ssh", "-p", "2222"]
    assert "BatchMode=yes" in argv and "IdentitiesOnly=yes" in argv
    assert argv[-2] == "sptsinman@cluster.hpc.hse.ru"
    assert argv[argv.index("-i") + 1].endswith(".ssh/charisma_auth_key.pem")


# -- probe ---------------------------------------------------------------------------------
def test_probe_reads_budget_qos_and_idle_gpus():
    st = _cluster().probe()
    assert st.ok
    assert abs(st.spent_gpu_h - (60 / 3600 + 10 * 4 + 1 * 2)) < 1e-6     # elapsed x allocated GPUs
    assert st.gpus_in_use == 2 and st.free_gpus == 2
    assert st.idle["v100"] == (6, 8)            # cn-003 is drained and does not count
    assert st.idle["a100"] == (0, 8)
    assert st.remaining_gpu_h < 5000


def test_unreachable_cluster_reports_instead_of_raising():
    st = _cluster(FakeSsh(fail=True)).probe()
    assert not st.ok and "ssh-add -l" in st.error


# -- job script and rules -------------------------------------------------------------------
def test_sbatch_script_bounds_time_and_keeps_the_exit_code():
    spec = SlurmJobSpec(title="Clifford flow", command="python -m src.main", hours=2, gpus=2,
                        repo="https://x/y.git", subdir="pose3d")
    script = sbatch_script(spec, CHARISMA, "gpu_pool/runs/clifford-flow-1")
    assert "#SBATCH --time=02:00:00" in script and "#SBATCH --gpus=2" in script
    assert "#SBATCH --partition=rocky" in script and '--constraint="type_a|type_b|type_c"' in script
    assert "#SBATCH --account=proj_1898" in script and "#SBATCH --cpus-per-task=8" in script
    assert "export WANDB_MODE=offline" in script
    assert 'cd "$HOME/gpu_pool/runs/clifford-flow-1/src/pose3d"' in script
    assert script.rstrip().endswith("exit $status")
    smoke = sbatch_script(SlurmJobSpec(title="s", command="true", hours=0.1, smoke=True), CHARISMA, "w")
    assert "#SBATCH --partition=test" in smoke and "#SBATCH --time=00:06:00" in smoke


def test_rules_from_the_guide():
    st = ClusterStatus(CHARISMA, ok=True, spent_gpu_h=100, taken_at=datetime(2026, 10, 29, tzinfo=timezone.utc))
    ok = SlurmJobSpec(title="a", command="x", hours=2)
    assert st.check(ok) == [] or "reserve" in " ".join(st.check(ok))
    assert any("partition limit" in p for p in st.check(SlurmJobSpec(title="a", command="x", hours=1, smoke=True)))
    assert any("QOS cap" in p for p in st.check(SlurmJobSpec(title="a", command="x", hours=1, gpus=8)))
    assert any("partition limit" in p for p in st.check(SlurmJobSpec(title="a", command="x", hours=5, accelerator="h100")))
    poor = ClusterStatus(CHARISMA, ok=True, spent_gpu_h=4990)
    assert any("budget" in p for p in poor.check(SlurmJobSpec(title="a", command="x", hours=3, gpus=4)))
    assert any("hours must be" in p for p in st.check(SlurmJobSpec(title="a", command="x", hours=0)))


def test_reserve_window_needs_explicit_permission():
    st = ClusterStatus(CHARISMA, ok=True)
    nov10 = datetime(2026, 11, 10, tzinfo=timezone.utc)
    assert st.reserve_window(nov10, 1) == ("2026-11-05", "2026-11-18")
    assert st.reserve_window(datetime(2026, 11, 4, 12, tzinfo=timezone.utc), 14) is not None   # runs into it
    assert st.reserve_window(datetime(2026, 11, 20, tzinfo=timezone.utc), 1) is None
    assert st.reserve_window(datetime(2026, 10, 29, tzinfo=timezone.utc), 1) is None


def test_submit_clones_on_the_login_node_and_parses_the_job_id():
    fake = FakeSsh()
    cluster = _cluster(fake)
    handle = cluster.submit(SlurmJobSpec(title="smoke", command="python -c 'import torch'", hours=0.1,
                                         smoke=True, repo="https://github.com/a/b.git", setup="pip install --user -q x"))
    assert handle.job_id == "4470001" and handle.ref == "charisma:4470001"
    _, cmd, stdin = fake.calls[-1]
    assert "git clone -q --depth 1 https://github.com/a/b.git src" in cmd
    assert "pip install --user -q x" in cmd and cmd.rstrip().endswith("sbatch --parsable run.sbatch")
    assert stdin.startswith("#!/bin/bash") and "--partition=test" in stdin
    assert handle.status() == "queued"


def test_submit_refuses_rule_breaks():
    try:
        _cluster().submit(SlurmJobSpec(title="big", command="x", hours=1, gpus=6))
    except SlurmError as exc:
        assert "QOS cap" in str(exc)
    else:
        raise AssertionError("expected a refusal")


# -- unified pool ---------------------------------------------------------------------------
def test_pool_routes_v100_to_the_cluster_and_offers_are_ranked():
    pool = ComputePool(None, [_cluster()])
    pool.probe()
    assert pool.backend_for("v100") == ["charisma"] and pool.backend_for("t4") == []
    best = pool.best("v100", hours=2, gpus=2)
    assert best.backend == "charisma" and best.gpus == 2 and best.free_now
    assert pool.best("v100", hours=2, gpus=3) is None          # 2 of 4 QOS GPUs are already in use
    assert not pool.best("a100", hours=1).free_now             # every A100 is allocated
    assert "cHARISMa" in pool.report()


def test_pool_launch_needs_approval_over_thirty_minutes():
    pool = ComputePool(None, [_cluster()])
    spec = RunSpec(title="long", repo="https://x/y.git", command="python t.py", hours=2, accelerator="v100")
    try:
        pool.launch(spec)
    except PermissionError as exc:
        assert "approval" in str(exc)
    else:
        raise AssertionError("expected an approval refusal")
    short = RunSpec(title="smoke", repo="https://x/y.git", command="python t.py", hours=APPROVAL_HOURS / 2,
                    accelerator="v100", smoke=True)
    handle = pool.launch(short)
    assert handle.ref == "charisma:4470001" and handle.backend == "charisma"


def test_attach_by_reference():
    pool = ComputePool(None, [_cluster()])
    assert pool.attach("charisma:4470001").status() == "queued"
    for bad in ("charisma", "nowhere:1"):
        try:
            pool.attach(bad)
        except (ValueError, KeyError):
            pass
        else:
            raise AssertionError(bad)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok   %s" % name)
        except Exception as exc:
            failed += 1
            print("FAIL %s: %s: %s" % (name, type(exc).__name__, exc))
    print("\n%d/%d passed" % (len(tests) - failed, len(tests)))
    raise SystemExit(1 if failed else 0)
