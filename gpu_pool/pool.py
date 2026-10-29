"""One interface over all of the pool's compute: Kaggle accounts and Slurm clusters (HSE cHARISMa).

The two kinds of compute differ in how they are paid for, and this module puts both behind the
same four questions: *what is free*, *where should this run go*, *launch it*, *how is it doing*.

=================  =====================================  ===========================================
                   Kaggle account (``router.py``)         Slurm cluster (``slurm.py``)
=================  =====================================  ===========================================
budget             30 GPU-h per account, weekly reset     fixed project GPU-h until an end date
concurrency        2 sessions per account                 QOS cap: 4 GPUs per user
hardware           T4 / P100, competition-gated RTX / L4  V100 (rocky), A100 / H100 / H200 (backfill)
internet           on (off for gated GPUs)                login node only, never on compute nodes
cancel             not possible                           ``scancel``
=================  =====================================  ===========================================

An **offer** is one place a run could go right now: (backend, target, accelerator, GPUs, hours of
budget). A **run reference** names a launched run in either backend: ``kaggle:owner/slug`` or
``charisma:4467119``; ``status``, ``logs`` and ``cancel`` take one.

CLAUDE.md requires a stated ETA for every GPU run and human approval for anything over 30
minutes in total, so ``launch`` takes the run time (``hours``) up front, prints the ETA, and refuses
a run over 30 minutes unless ``approved=True`` (``--approved``).

CLI
---
    python gpu_pool/pool.py status                                   # every backend, one table
    python gpu_pool/pool.py best --accelerator v100 --hours 2        # where would this go?
    python gpu_pool/pool.py launch --title t --repo URL --command CMD --accelerator t4 --hours 0.4
    python gpu_pool/pool.py launch ... --accelerator v100 --hours 6 --approved
    python gpu_pool/pool.py job charisma:4467119                     # state of one run
    python gpu_pool/pool.py logs charisma:4467119
    python gpu_pool/pool.py cancel charisma:4467119                  # Slurm only

Kaggle credentials come from ``--tokens-dir`` / ``$KAGGLE_TOKENS_DIR`` (skipped when absent);
clusters from ``--cluster`` (default: ``charisma``; ``--cluster none`` leaves them out).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

try:  # package import
    from .launcher import ACCELERATORS as KAGGLE_ACCELERATORS
    from .launcher import LaunchHandle, PoolLauncher, git_run_spec
    from .router import PoolRouter, accelerator_name, filter_progress, quota_rate, requires_competition
    from .slurm import CHARISMA, SlurmCluster, SlurmError, SlurmHandle, SlurmJobSpec, cluster_config
    from .slurm import report as cluster_report
except ImportError:  # run as a script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from gpu_pool.launcher import ACCELERATORS as KAGGLE_ACCELERATORS
    from gpu_pool.launcher import LaunchHandle, PoolLauncher, git_run_spec
    from gpu_pool.router import PoolRouter, accelerator_name, filter_progress, quota_rate, requires_competition
    from gpu_pool.slurm import CHARISMA, SlurmCluster, SlurmError, SlurmHandle, SlurmJobSpec, cluster_config
    from gpu_pool.slurm import report as cluster_report

#: CLAUDE.md: any operation longer than this (hours) must be approved by a human first.
APPROVAL_HOURS = 0.5


@dataclass
class Offer:
    """One place a run could go right now."""

    backend: str            #: ``kaggle`` or a cluster name (``charisma``)
    target: str             #: Kaggle account label, or the cluster name
    accelerator: str
    gpus: int               #: GPUs a run can get there at once
    hours_left: float       #: budget left there, in wall-clock hours of this accelerator
    free_now: bool          #: a slot is free (Kaggle) / under the QOS cap and GPUs idle (Slurm)
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"backend": self.backend, "target": self.target, "accelerator": self.accelerator,
                "gpus": self.gpus, "hours_left": round(self.hours_left, 1), "free_now": self.free_now,
                "note": self.note}


@dataclass
class RunSpec:
    """A run described once, launchable on either backend.

    ``hours`` is the expected run time: Kaggle routes on it, Slurm turns it into ``--time``.
    ``install`` runs inside the Kaggle notebook, or on the cluster's **login node** before the job.
    """

    title: str
    repo: str
    command: str
    hours: float
    accelerator: str = "t4"
    gpus: int = 1
    branch: Optional[str] = None
    subdir: Optional[str] = None
    install: Optional[str] = None
    runner: str = ""
    env: Dict[str, str] = field(default_factory=dict)
    competitions: List[str] = field(default_factory=list)     #: Kaggle only
    secrets: List[str] = field(default_factory=list)          #: Kaggle only
    smoke: bool = False                                       #: Slurm only: debug partition


@dataclass
class RunHandle:
    """A launched run on either backend, addressed by ``ref``."""

    ref: str
    backend: str
    inner: Any                 #: launcher.LaunchHandle or slurm.SlurmHandle
    eta_hours: float = 0.0

    @property
    def url(self) -> str:
        return getattr(self.inner, "url", "") or self.ref

    def status(self) -> str:
        return self.inner.status()

    def log(self, lines: int = 40) -> str:
        if self.backend == "kaggle":
            return "\n".join(filter_progress(self.inner.log(follow=True), keep_last=lines))
        return self.inner.log(lines)

    def cancel(self) -> None:
        self.inner.cancel()

    def wait(self, timeout: float = 3600.0) -> str:
        return self.inner.wait(timeout=timeout)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.inner.as_dict(), ref=self.ref, backend=self.backend, eta_hours=self.eta_hours)


class ComputePool:
    """Kaggle accounts and Slurm clusters behind one interface.

    >>> pool = ComputePool(PoolRouter.from_dir("~/pool_tokens"), [SlurmCluster(CHARISMA)])
    >>> pool.probe()                                              # doctest: +SKIP
    >>> pool.best("v100", hours=2)                                # doctest: +SKIP
    """

    def __init__(self, kaggle: Optional[PoolRouter] = None, clusters: Sequence[SlurmCluster] = ()) -> None:
        if kaggle is None and not clusters:
            raise ValueError("the pool needs Kaggle credentials or at least one cluster")
        self.kaggle = kaggle
        self.clusters = {c.name: c for c in clusters}
        self._probed = False

    # -- probing -----------------------------------------------------------------------
    def probe(self, kaggle_competitions: bool = True) -> None:
        if self.kaggle is not None:
            self.kaggle.probe_all(with_competitions=kaggle_competitions)
        for cluster in self.clusters.values():
            cluster.probe()
        self._probed = True

    def _ensure(self) -> None:
        if not self._probed:
            self.probe()

    def backend_for(self, accelerator: str) -> List[str]:
        """Backends that have ``accelerator`` at all: ``"t4"`` -> kaggle, ``"v100"`` -> charisma."""
        out = []
        if self.kaggle is not None and accelerator in KAGGLE_ACCELERATORS:
            out.append("kaggle")
        out += [n for n, c in self.clusters.items() if accelerator in c.config.accelerators]
        return out

    def accelerators(self) -> List[str]:
        names = list(KAGGLE_ACCELERATORS) if self.kaggle is not None else []
        for cluster in self.clusters.values():
            names += [a for a in cluster.config.accelerators if a not in names]
        return names

    # -- offers ------------------------------------------------------------------------
    def offers(self, accelerator: Optional[str] = None) -> List[Offer]:
        """Every place a run could go, best first (free now, then most hours left)."""
        self._ensure()
        out: List[Offer] = []
        if self.kaggle is not None:
            for st in self.kaggle.statuses:
                if not st.ok:
                    continue
                labels = ["t4", "p100"] + st.premium_accelerators
                for label in labels:
                    if accelerator and label != accelerator:
                        continue
                    via = st.competition_for(label) if requires_competition(label) else None
                    out.append(Offer("kaggle", st.name, label, 4 if label == "l4" else 1,
                                     st.gpu.remaining_h / quota_rate(label),
                                     st.free_slots > 0 if st.kernels_checked else not st.busy,
                                     ("via " + via) if via else "weekly quota"))
        for name, cluster in self.clusters.items():
            st = cluster.status
            if not st.ok:
                continue
            for label, part in cluster.config.accelerators.items():
                if accelerator and label != accelerator:
                    continue
                if not part.gpus_per_node:
                    continue
                idle = st.idle.get(label)
                gpus = min(st.free_gpus, part.gpus_per_node)
                free = gpus > 0 and (idle is None or idle[0] > 0)
                note = "%s, max %g h%s" % (part.partition, part.max_hours, ", preemptible" if part.preemptible else "")
                if idle is not None:
                    note += ", %d/%d idle" % idle
                out.append(Offer(name, name, label, gpus, st.remaining_gpu_h, free, note))
        out.sort(key=lambda o: (not o.free_now, -o.hours_left))
        return out

    def best(self, accelerator: str, hours: float = 1.0, gpus: int = 1, require_free: bool = False,
             backend: Optional[str] = None) -> Optional[Offer]:
        """The best offer that has ``gpus`` GPUs and ``hours`` (x GPUs for Slurm) of budget."""
        for offer in self.offers(accelerator):
            if backend and offer.backend != backend:
                continue
            need = hours * (gpus if offer.backend != "kaggle" else 1)
            if offer.hours_left < need or offer.gpus < gpus:
                continue
            if require_free and not offer.free_now:
                continue
            return offer
        return None

    # -- launching ---------------------------------------------------------------------
    def launch(self, spec: RunSpec, approved: bool = False, backend: Optional[str] = None,
               target: Optional[str] = None, require_free: bool = False, allow_reserve: bool = False,
               dry_run: bool = False) -> RunHandle:
        """Route ``spec`` to the best offer and launch it.

        Refuses a run over ``APPROVAL_HOURS`` unless ``approved``: the pool cannot stop a Kaggle run
        once started, and a Slurm run bills every allocated GPU-hour.
        """
        if spec.hours <= 0:
            raise ValueError("state the expected run time: hours must be > 0")
        if spec.hours > APPROVAL_HOURS and not approved and not dry_run:
            raise PermissionError(
                "ETA %.2f h x %d GPU exceeds %d minutes: CLAUDE.md requires a human's approval first; "
                "pass approved=True / --approved once you have it" % (spec.hours, spec.gpus, APPROVAL_HOURS * 60))
        if backend is None and target and target in self.clusters:
            backend = target
        offer = self.best(spec.accelerator, spec.hours, spec.gpus, require_free, backend)
        if offer is None:
            have = self.backend_for(spec.accelerator)
            raise RuntimeError("no offer for %d x %s for %.2f h%s (backends with it: %s)" % (
                spec.gpus, spec.accelerator, spec.hours, " free now" if require_free else "",
                ", ".join(have) or "none"))
        print("ETA %.2f h on %s/%s (%s x%d)%s" % (spec.hours, offer.backend, offer.target,
                                               spec.accelerator, spec.gpus, " [dry run]" if dry_run else ""))
        if offer.backend == "kaggle":
            return self._launch_kaggle(spec, target or offer.target, dry_run)
        return self._launch_slurm(spec, self.clusters[offer.backend], allow_reserve, dry_run)

    def _launch_kaggle(self, spec: RunSpec, account: str, dry_run: bool) -> RunHandle:
        if spec.gpus != 1:
            raise ValueError("a Kaggle run gets one machine; use gpus=1 (4xL4 is one 'l4' machine)")
        kwargs: Dict[str, Any] = dict(env=spec.env or None, required_secrets=list(spec.secrets),
                                      competitions=list(spec.competitions))
        if spec.install is not None:
            kwargs["install"] = spec.install
        nb = git_run_spec(title=spec.title, repo=spec.repo, command=spec.command, accelerator=spec.accelerator,
                          branch=spec.branch, subdir=spec.subdir,
                          runner=spec.runner if spec.runner or spec.install is not None else "poetry run",
                          **kwargs)
        handle = PoolLauncher(self.kaggle).launch(nb, min_hours=spec.hours, account=account, dry_run=dry_run)
        return RunHandle("kaggle:" + handle.ref, "kaggle", handle, spec.hours)

    def _launch_slurm(self, spec: RunSpec, cluster: SlurmCluster, allow_reserve: bool, dry_run: bool) -> RunHandle:
        command = " ".join(x for x in ((spec.runner or ""), spec.command) if x)
        job = SlurmJobSpec(title=spec.title, command=command, hours=spec.hours, accelerator=spec.accelerator,
                           gpus=spec.gpus, repo=spec.repo, branch=spec.branch, subdir=spec.subdir,
                           setup=spec.install, env=dict(spec.env), smoke=spec.smoke)
        handle = cluster.submit(job, allow_reserve=allow_reserve, dry_run=dry_run)
        return RunHandle(handle.ref, cluster.name, handle, spec.hours)

    # -- existing runs -----------------------------------------------------------------
    def attach(self, ref: str) -> RunHandle:
        """A handle for an already launched run, from its reference."""
        backend, _, ident = ref.partition(":")
        if not ident:
            raise ValueError("a run reference looks like kaggle:owner/slug or charisma:12345, got %r" % ref)
        if backend in self.clusters:
            return RunHandle(ref, backend, SlurmHandle(self.clusters[backend], ident))
        if backend == "kaggle":
            if self.kaggle is None:
                raise RuntimeError("no Kaggle credentials loaded (--tokens-dir / $KAGGLE_TOKENS_DIR)")
            owner = ident.split("/", 1)[0]
            account = next((a for a in self.kaggle.accounts if owner in (a.username, a.label)), None)
            if account is None:
                self.kaggle.probe_all(check_kernels=False)
                account = next((a for a in self.kaggle.accounts if owner == a.username), None)
            if account is None:
                raise KeyError("no pool account owns %s" % ident)
            return RunHandle(ref, "kaggle", LaunchHandle(PoolLauncher(self.kaggle), account, ident))
        raise KeyError("unknown backend %r (have: %s)" % (backend, ", ".join(["kaggle"] + list(self.clusters))))

    # -- reporting ---------------------------------------------------------------------
    def report(self) -> str:
        self._ensure()
        parts = []
        if self.kaggle is not None:
            parts.append("== Kaggle (%d accounts) ==\n%s" % (len(self.kaggle.accounts), self.kaggle.report()))
        for cluster in self.clusters.values():
            parts.append("== %s ==\n%s" % (cluster.config.title, cluster_report(cluster.status)))
        free = [o for o in self.offers() if o.free_now]
        by_accel: Dict[str, int] = {}
        for o in free:
            by_accel[o.accelerator] = by_accel.get(o.accelerator, 0) + 1
        parts.append("free now: %s" % (", ".join("%s at %d place%s" % (accelerator_name(a) if a in KAGGLE_ACCELERATORS
                                                                         else a.upper(), n, "" if n == 1 else "s")
                                                 for a, n in by_accel.items()) or "nothing"))
        return "\n\n".join(parts)

    def as_dict(self) -> Dict[str, Any]:
        self._ensure()
        return {
            "kaggle": self.kaggle.as_dict() if self.kaggle is not None else None,
            "clusters": {n: c.status.as_dict() for n, c in self.clusters.items()},
            "offers": [o.as_dict() for o in self.offers()],
        }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_pool(tokens_dir: Optional[str], clusters: Sequence[str]) -> ComputePool:
    router = None
    if tokens_dir and os.path.isdir(os.path.expanduser(tokens_dir)):
        router = PoolRouter.from_dir(tokens_dir)
    names = [c for c in clusters if c and c != "none"]
    if router is None and not names:
        raise SystemExit("nothing to use: give --tokens-dir / $KAGGLE_TOKENS_DIR or a --cluster")
    return ComputePool(router, [SlurmCluster(cluster_config(c)) for c in names])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="The GPU pool: Kaggle accounts and Slurm clusters, one interface.")
    parser.add_argument("--tokens-dir", default=os.environ.get("KAGGLE_TOKENS_DIR"),
                        help="Kaggle credentials directory (default: $KAGGLE_TOKENS_DIR; skipped if absent)")
    parser.add_argument("--cluster", action="append", default=None,
                        help="Slurm cluster preset or JSON config (repeatable; default: charisma; 'none' = no clusters)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("status", help="every backend in one report")
    st.add_argument("--json", action="store_true")
    be = sub.add_parser("best", help="where would a run go")
    be.add_argument("--accelerator", required=True)
    be.add_argument("--hours", type=float, default=1.0)
    be.add_argument("--gpus", type=int, default=1)
    be.add_argument("--free-now", action="store_true")
    la = sub.add_parser("launch", help="route and launch a clone -> install -> run job")
    la.add_argument("--title", required=True)
    la.add_argument("--repo", required=True)
    la.add_argument("--command", required=True)
    la.add_argument("--hours", type=float, required=True, help="expected run time (ETA); Slurm --time")
    la.add_argument("--accelerator", default="t4")
    la.add_argument("--gpus", type=int, default=1)
    la.add_argument("--branch")
    la.add_argument("--subdir")
    la.add_argument("--install", help="Kaggle: in the notebook; Slurm: on the login node before submitting")
    la.add_argument("--runner", default="")
    la.add_argument("--env", action="append", default=[], help="KEY=VALUE (repeatable)")
    la.add_argument("--competition", action="append", default=[])
    la.add_argument("--secret", action="append", default=[])
    la.add_argument("--backend", help="pin a backend: kaggle or a cluster name")
    la.add_argument("--target", help="pin a Kaggle account")
    la.add_argument("--free-now", action="store_true")
    la.add_argument("--smoke", action="store_true", help="Slurm: debug partition, 30 min max")
    la.add_argument("--approved", action="store_true", help="a human approved this run (needed over 30 min)")
    la.add_argument("--allow-reserve", action="store_true", help="Slurm: the PI approved using the reserve window")
    la.add_argument("--dry-run", action="store_true")
    la.add_argument("--wait", type=float, default=0.0)
    for name in ("job", "logs", "cancel"):
        p = sub.add_parser(name)
        p.add_argument("ref", help="kaggle:owner/slug or charisma:JOBID")
        if name == "logs":
            p.add_argument("--lines", type=int, default=40)
    args = parser.parse_args(argv)

    pool = build_pool(args.tokens_dir, args.cluster if args.cluster is not None else [CHARISMA.name])
    if args.cmd in ("job", "logs", "cancel"):
        handle = pool.attach(args.ref)
        if args.cmd == "job":
            print("%s %s" % (args.ref, handle.status()))
        elif args.cmd == "logs":
            print(handle.log(args.lines))
        else:
            try:
                handle.cancel()
            except NotImplementedError as exc:
                print(exc)
                return 1
            print("cancelled %s" % args.ref)
        return 0

    pool.probe()
    if args.cmd == "status":
        print(json.dumps(pool.as_dict(), indent=2) if args.json else pool.report())
        return 0
    if args.cmd == "best":
        offer = pool.best(args.accelerator, args.hours, args.gpus, args.free_now)
        if offer is None:
            print("no offer for %d x %s for %.1f h" % (args.gpus, args.accelerator, args.hours))
            return 1
        print("%s/%s: %s x%d, %.1f h left, %s (%s)" % (offer.backend, offer.target, offer.accelerator, offer.gpus,
                                                     offer.hours_left, "free now" if offer.free_now else "busy", offer.note))
        return 0

    spec = RunSpec(title=args.title, repo=args.repo, command=args.command, hours=args.hours,
                   accelerator=args.accelerator, gpus=args.gpus, branch=args.branch, subdir=args.subdir,
                   install=args.install, runner=args.runner,
                   env=dict((e.split("=", 1) + [""])[:2] for e in args.env),
                   competitions=args.competition, secrets=args.secret, smoke=args.smoke)
    try:
        handle = pool.launch(spec, approved=args.approved, backend=args.backend, target=args.target,
                             require_free=args.free_now, allow_reserve=args.allow_reserve, dry_run=args.dry_run)
    except (PermissionError, SlurmError) as exc:
        print("refused: %s" % exc)
        return 2
    if args.dry_run:
        return 0
    print("launched %s -> %s" % (spec.title, handle.ref))
    print(handle.url)
    if args.wait:
        print("final status:", handle.wait(timeout=args.wait))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
