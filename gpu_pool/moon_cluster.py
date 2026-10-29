"""Moon Cluster: draw the accelerator pool as a night-sky dial page.

The page (``moon_cluster_template.html``) shows every account on an outer orbit with a quota gauge,
the competitions that grant premium GPUs on an inner orbit, and the hardware around a moon that
carries the pool's spare weekly Kaggle GPU-hours. Slurm clusters (``slurm.py``, e.g. HSE cHARISMa)
ride the outer orbit too, drawn as hexagons with a project-budget gauge and linked straight to their
hardware, since no competition gates them. This module fills it in.

Two steps, so an agent can change anything in between:

1. **snapshot** -- a plain JSON dict describing the pool (accounts, quota, routes, active runs,
   competition deadlines, free-form notes). ``probe_snapshot`` builds one from a credentials
   directory through :class:`~gpu_pool.router.PoolRouter`. It never contains credentials.
2. **render** -- ``render(snapshot, previous)`` lays the dial out and returns the HTML page. The
   snapshot is embedded in the page, so the published page itself is the ``previous`` input of the
   next render, and the "changes" cards (new accounts, new routes, removed accounts, ...) are
   computed by comparing the two.

CLI
---
    python gpu_pool/moon_cluster.py --tokens-dir DIR --out moon_cluster.html
    python gpu_pool/moon_cluster.py --tokens-dir DIR --previous last_page.html --out new.html
    python gpu_pool/moon_cluster.py --tokens-dir DIR --dump-snapshot snap.json     # probe only
    python gpu_pool/moon_cluster.py --from-snapshot snap.json --out new.html      # render only

    --note "KICKER|Title|Text"      add a card (repeatable)
    --removed account7=deleted      leave an account out and show it as removed (repeatable)
    --title / --subtitle            page name and the italic word after it
    --cluster charisma              also probe this Slurm cluster over SSH (repeatable)
    --cluster-unprobed charisma     show the cluster from its config only, as *awaiting probe*
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # package import
    from .router import (
        ACCELERATOR_NAMES,
        AccountStatus,
        PoolAccount,
        PoolRouter,
        quota_rate,
    )
    from .slurm import CLUSTERS, ClusterStatus, SlurmCluster, SlurmConfig, cluster_config
except ImportError:  # run as a script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from gpu_pool.router import (
        ACCELERATOR_NAMES,
        AccountStatus,
        PoolAccount,
        PoolRouter,
        quota_rate,
    )
    from gpu_pool.slurm import CLUSTERS, ClusterStatus, SlurmCluster, SlurmConfig, cluster_config

SNAPSHOT_VERSION = 1
TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moon_cluster_template.html")
DEFAULT_TITLE = "Moon Cluster"

#: How each accelerator is drawn. ``rank`` orders them on the page and picks a gauge's colour when an
#: account reaches several; ``color`` is a CSS token defined in the template.
HARDWARE_STYLE: Dict[str, Dict[str, Any]] = {
    "rtx6000": {"short": "RTX 6000", "color": "--rtx", "rank": 0, "spec": "1 × Blackwell · 96 GB",
                "extra": "Internet off."},
    "l4": {"short": "4xL4", "color": "--l4", "rank": 1, "spec": "4 × L4 · 96 GB total", "extra": ""},
    "tpu-v5e": {"short": "TPU", "color": "--tpu", "rank": 2, "spec": "8 TPU v5e cores",
                "extra": "The push API currently provisions a non-TPU image (kaggle-cli #1197)."},
    "tpu-v3": {"short": "TPU v3", "color": "--tpu", "rank": 3, "spec": "8 TPU v3 cores",
               "extra": "The push API currently provisions a non-TPU image (kaggle-cli #1197)."},
    # Slurm cluster hardware; keys are the ``planet`` of ``slurm.Partition``.
    "v100": {"short": "V100", "color": "--v100", "rank": 5, "spec": "V100 32 GB · 4 per node", "name": "V100 32GB",
             "extra": "Slurm partition rocky: jobs up to 30 days, billed per allocated GPU-hour."},
    "hpc-ef": {"short": "A100+", "color": "--ef", "rank": 6, "spec": "A100 80GB · H100 · H200",
               "name": "A100 · H100 · H200",
               "extra": "Backfill queue gpu-ef-quick: preemptible, 3 h per job, usually full, so checkpoint often."},
}
#: Colour tokens for accelerators missing from ``HARDWARE_STYLE``, handed out in order.
SPARE_COLORS = ("--x1", "--x2", "--x3")
OPEN_HARDWARE = {"name": "T4 / P100", "short": "T4 · P100", "color": "--open", "rank": 99, "rate": 1.0,
                 "spec": "16 GB", "note": "No competition needed. Every account in the cluster can use it."}

#: Readable names for known competitions: slug -> (short label on the dial, full name).
COMPETITION_NAMES = {
    "arc-prize-2026-arc-agi-3": ("ARC-AGI-3", "ARC Prize 2026 · ARC-AGI-3"),
    "arc-prize-2026-arc-agi-2": ("ARC-AGI-2", "ARC Prize 2026 · ARC-AGI-2"),
    "gemma-4-developer-agent": ("Gemma 4", "Gemma 4 Developer Agent"),
    "gemma-4-developer-agent-paper": ("Gemma 4 Paper", "Gemma 4 Developer Agent · Paper Track"),
    "nvidia-nemotron-model-reasoning-challenge": ("Nemotron", "NVIDIA Nemotron Model Reasoning Challenge"),
}

_LABEL = re.compile(r"(?i)^account(\d+)[_\-]?(.*?)\s*(?:\[([^\]]+)\])?$")

# ---------------------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------------------
def parse_label(label: str) -> Tuple[Optional[int], str, Optional[str]]:
    """``"account7_CityOfTver[suspended]"`` -> ``(7, "CityOfTver", "suspended")``."""
    match = _LABEL.match(label.strip())
    if not match:
        return None, label, None
    return int(match.group(1)), match.group(2) or label, (match.group(3) or None)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if value else None


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "").split("+")[0]).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def account_entry(status: AccountStatus) -> Dict[str, Any]:
    """One probed account as a snapshot row (no credentials)."""
    n, name, _ = parse_label(status.name)
    entry: Dict[str, Any] = {
        "label": status.name, "n": n, "name": name, "user": status.username,
        "status": "live" if status.ok else "unreachable",
        "h": round(status.gpu.remaining_h, 1), "allowed": round(status.gpu.allowed_h or 30.0, 1),
        "runs": [[k.accelerator or "?", k.ref.split("/", 1)[-1]] for k in status.active_kernels],
        "routes": {},
    }
    if not status.ok:
        entry["error"] = status.error
    for accel, comps in status.routes().items():
        for comp in comps:
            seen = any(k.accelerator == accel and comp in k.competitions for k in status.kernels)
            entry["routes"][comp] = {"hw": accel, "how": "seen" if seen else "rules"}
    return entry


def cluster_entry(config: SlurmConfig, status: Optional[ClusterStatus] = None) -> Dict[str, Any]:
    """One Slurm cluster as a snapshot row. Without ``status`` it is *pending* (not probed yet)."""
    hardware: Dict[str, Dict[str, Any]] = {}
    for label, part in config.accelerators.items():
        if not part.planet:
            continue
        row = hardware.setdefault(part.planet, {"how": "preemptible" if part.preemptible else "queue",
                                                "accels": [], "idle": None})
        row["accels"].append(label)
        if status is not None and label in status.idle:
            idle, total = status.idle[label]
            old = row["idle"] or [0, 0]
            row["idle"] = [old[0] + idle, old[1] + total]
    entry: Dict[str, Any] = {
        "name": config.name, "title": config.title, "site": config.site, "account": config.account,
        "status": "pending" if status is None else ("live" if status.ok else "unreachable"),
        "budget": config.budget_gpu_h, "until": config.budget_end, "gpu_cap": config.gpu_cap,
        "reserve": [list(w) for w in config.reserve_windows], "hardware": hardware,
        "spent": None, "left": None, "in_use": None, "jobs": [],
    }
    if status is not None and status.ok:
        planet = {label: part.planet for label, part in config.accelerators.items()}
        entry.update(spent=round(status.spent_gpu_h, 1), left=round(status.remaining_gpu_h, 1),
                     in_use=status.gpus_in_use,
                     jobs=[[planet.get(j.accelerator(config)) or "", j.name, j.state, j.gpus] for j in status.active_jobs])
    elif status is not None:
        entry["error"] = status.error
    return entry


def snapshot_from_statuses(
    statuses: Iterable[AccountStatus],
    pending: Sequence[str] = (),
    removed: Optional[Dict[str, str]] = None,
    taken_at: Optional[datetime] = None,
    max_concurrent: int = 2,
    clusters: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    """Build a snapshot from probe results (``probe_all(..., with_competitions=True)``)."""
    statuses = list(statuses)
    accounts = [account_entry(s) for s in statuses]
    for label in pending:
        n, name, _ = parse_label(label)
        accounts.append({"label": label, "n": n, "name": name, "status": "pending", "runs": [], "routes": {}})
    for label, reason in (removed or {}).items():
        n, name, tag = parse_label(label)
        accounts.append({"label": label, "n": n, "name": name, "status": "removed",
                         "reason": reason or tag or "removed", "runs": [], "routes": {}})
    competitions: Dict[str, Dict[str, Any]] = {}
    for status in statuses:
        for slug, deadline in status.competition_deadlines.items():
            if any(slug in a["routes"] for a in accounts):
                competitions[slug] = {"deadline": _iso(deadline)}
    refresh = next((s.quota_refresh for s in statuses if s.quota_refresh), None)
    return {
        "version": SNAPSHOT_VERSION,
        "title": DEFAULT_TITLE,
        "subtitle": None,
        "taken_at": _iso(taken_at or datetime.now(timezone.utc)),
        "quota_reset": _iso(_parse_iso(refresh)) if refresh else None,
        "max_concurrent": max_concurrent,
        "accounts": accounts,
        "clusters": list(clusters),
        "competitions": competitions,
        "notes": [],
    }


def scan_tokens_dir(directory: str, removed: Optional[Dict[str, str]] = None
                    ) -> Tuple[List[PoolAccount], List[str], Dict[str, str]]:
    """Split a credentials directory into probe-able accounts, pending folders and removed ones.

    A folder without a ``token`` is *pending* (credentials not supplied yet). A folder whose name
    ends in a bracketed tag (``account7_CityOfTver[suspended]``) or that is named in ``removed`` is
    *removed* and never probed.
    """
    directory = os.path.expanduser(directory)
    removed = dict(removed or {})
    accounts, pending = [], []
    for entry in sorted(os.listdir(directory)):
        path = os.path.join(directory, entry)
        if entry.startswith(".") or not (os.path.isdir(path) or entry.lower().endswith(".json")):
            continue
        label = os.path.splitext(entry)[0] if not os.path.isdir(path) else entry
        n, _, tag = parse_label(label)
        hit = next((k for k in removed if k == label or (n is not None and parse_label(k)[0] == n)), None)
        if hit is not None:
            removed[label] = removed.pop(hit) or tag or "removed"
            continue
        if tag:
            removed[label] = tag
            continue
        try:
            if os.path.isdir(path):
                if os.path.isfile(os.path.join(path, "token")):
                    accounts.append(PoolAccount.from_token_dir(path))
                else:
                    pending.append(label)
            else:
                accounts.append(PoolAccount.from_json(path))
        except (OSError, ValueError):
            pending.append(label)
    return accounts, pending, removed


def probe_clusters(names: Sequence[str] = (), unprobed: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """Snapshot rows for Slurm clusters: ``names`` probed over SSH, ``unprobed`` from config only."""
    rows = [cluster_entry(c.config, c.probe()) for c in (SlurmCluster(cluster_config(n)) for n in names)]
    return rows + [cluster_entry(cluster_config(n)) for n in unprobed]


def probe_snapshot(directory: Optional[str], removed: Optional[Dict[str, str]] = None,
                   clusters: Sequence[Dict[str, Any]] = (), **router_kwargs: Any) -> Dict[str, Any]:
    """Probe every account in ``directory`` and return a snapshot (about a minute for 10 accounts)."""
    if not directory:
        return snapshot_from_statuses([], clusters=clusters)
    accounts, pending, removed = scan_tokens_dir(directory, removed)
    if not accounts:
        raise FileNotFoundError("no usable credentials in %s" % directory)
    router = PoolRouter(accounts, **router_kwargs)
    statuses = router.probe_all(check_kernels=True, recent=10, recent_within_hours=0,
                                with_accelerators=True, with_competitions=True)
    return snapshot_from_statuses(statuses, pending, removed, max_concurrent=router.max_concurrent,
                                  clusters=clusters)


# ---------------------------------------------------------------------------------------
# previous page -> snapshot
# ---------------------------------------------------------------------------------------
_DATA_TAG = re.compile(r'<script type="application/json" id="moon-data">(.*?)</script>', re.S)


def load_snapshot(path: str) -> Dict[str, Any]:
    """A snapshot from a ``.json`` file, or the one embedded in a rendered Moon Cluster page."""
    with open(os.path.expanduser(path)) as fh:
        text = fh.read()
    if path.lower().endswith((".html", ".htm")) or text.lstrip().startswith("<"):
        match = _DATA_TAG.search(text)
        if not match:
            raise ValueError("%s is not a Moon Cluster page (no embedded snapshot)" % path)
        return json.loads(match.group(1))["snapshot"]
    return json.loads(text)


# ---------------------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------------------
def _circ_mean(angles: Sequence[float]) -> float:
    x = sum(math.cos(math.radians(a)) for a in angles)
    y = sum(math.sin(math.radians(a)) for a in angles)
    return math.degrees(math.atan2(y, x))


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def spread(angles: Dict[str, float], min_gap: float, rounds: int = 200) -> Dict[str, float]:
    """Push angles apart on the circle until neighbours are at least ``min_gap`` degrees apart."""
    out = dict(angles)
    if len(out) < 2:
        return out
    min_gap = min(min_gap, 360.0 / len(out))
    for _ in range(rounds):
        moved = False
        keys = sorted(out, key=lambda k: out[k] % 360)
        for i, k in enumerate(keys):
            nxt = keys[(i + 1) % len(keys)]
            gap = (out[nxt] - out[k]) % 360
            if gap < min_gap - 1e-6:
                push = (min_gap - gap) / 2
                out[k] = _wrap(out[k] - push)
                out[nxt] = _wrap(out[nxt] + push)
                moved = True
        if not moved:
            break
    return out


def _hardware(label: str, used: Dict[str, str]) -> Dict[str, Any]:
    style = HARDWARE_STYLE.get(label)
    if style is None:
        color = SPARE_COLORS[len(used) % len(SPARE_COLORS)]
        style = {"short": label.upper(), "color": color, "rank": 10 + len(used), "spec": "", "extra": ""}
    used[label] = style["color"]
    return {"name": style.get("name") or ACCELERATOR_NAMES.get(label, label), "short": style["short"], "color": style["color"],
            "rank": style["rank"], "spec": style["spec"], "rate": quota_rate(label), "extra": style["extra"]}


def _fmt_day(when: datetime) -> str:
    return "%d %s" % (when.day, when.strftime("%b"))


def page_data(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """What the template's script draws: hardware, competitions and accounts with dial angles."""
    taken = _parse_iso(snapshot.get("taken_at")) or datetime.now(timezone.utc)
    rows = [a for a in snapshot["accounts"] if a.get("status") != "removed"]
    live = [a for a in rows if a.get("status") == "live"]

    # competitions that some live account reaches, keyed by slug
    comp_hw: Dict[str, str] = {}
    for a in live:
        for slug, route in a["routes"].items():
            comp_hw.setdefault(slug, route["hw"])
    used: Dict[str, str] = {}
    hardware = {}
    clusters = [dict(c) for c in snapshot.get("clusters", []) if c.get("status") != "removed"]
    cluster_hw = sorted({p for c in clusters for p in c.get("hardware", {})} - set(comp_hw.values()),
                        key=lambda l: HARDWARE_STYLE.get(l, {}).get("rank", 50))
    for label in sorted(set(comp_hw.values()), key=lambda l: HARDWARE_STYLE.get(l, {}).get("rank", 50)) + cluster_hw:
        hardware[label] = _hardware(label, used)

    # account order: by the hardware/competitions they reach, so shared routes sit together
    comp_order = sorted(comp_hw, key=lambda s: (hardware[comp_hw[s]]["rank"], s))
    def acct_key(a: Dict[str, Any]) -> Tuple:
        idx = sorted(comp_order.index(s) for s in a["routes"] if s in comp_order)
        state = {"live": 0, "unreachable": 1, "pending": 2}.get(a.get("status"), 3)
        return (state, not idx, (sum(idx) / len(idx)) if idx else 0, -len(idx), a.get("n") or 0)
    rows.sort(key=acct_key)
    # clusters ride the same ring, right after the live accounts
    n_live = sum(1 for a in rows if a.get("status") == "live")
    ring = rows[:n_live] + clusters + rows[n_live:]
    step = 360.0 / max(1, len(ring))
    for i, a in enumerate(ring):
        a["_angle"] = -90 + i * step

    comp_angles = spread({s: _circ_mean([a["_angle"] for a in live if s in a["routes"]]) for s in comp_hw},
                         min_gap=36)
    hw_seed = {h: _circ_mean([comp_angles[s] for s in comp_hw if comp_hw[s] == h]) for h in hardware if h not in cluster_hw}
    for h in cluster_hw:
        hw_seed[h] = _circ_mean([c["_angle"] for c in clusters if h in c.get("hardware", {})])
    hw_angles = spread(hw_seed, min_gap=80 if len(hw_seed) <= 4 else 360.0 / len(hw_seed))

    competitions = []
    for slug in comp_order:
        deadline = _parse_iso((snapshot.get("competitions", {}).get(slug) or {}).get("deadline"))
        short, full = COMPETITION_NAMES.get(slug, (slug.replace("-", " ").title()[:18], slug.replace("-", " ").title()))
        days = max(0, (deadline.date() - taken.date()).days) if deadline else None
        angle = comp_angles[slug]
        competitions.append({
            "id": slug, "name": short, "full": full, "slug": slug, "hw": comp_hw[slug],
            "ends": _fmt_day(deadline) if deadline else "no deadline", "days": days if days is not None else 999,
            "angle": round(angle, 1), "side": "above" if math.sin(math.radians(angle)) < 0 else "below",
        })
    # hardware notes: billing rate, what grants it and when that ends
    for label, hw in hardware.items():
        grants = [c for c in competitions if c["hw"] == label]
        parts = [hw.pop("extra")] if hw.get("extra") else []
        if len(grants) == 1:
            parts.append("Every route runs through %s, which closes %s." % (grants[0]["name"], grants[0]["ends"]))
        elif grants:
            parts.append("Granted by %s." % ", ".join("%s (until %s)" % (c["name"], c["ends"]) for c in grants))
        if hw["rate"] > 1:
            parts.append("Billed at %g× quota, so spare hours buy %s as much running time."
                         % (hw["rate"], "half" if hw["rate"] == 2 else "1/%g" % hw["rate"]))
        hw["note"] = " ".join(p for p in parts if p)
        hw.pop("extra", None)
    hardware_all = dict(hardware)
    hardware_all["open"] = dict(OPEN_HARDWARE)

    accounts = []
    for a in rows:
        accounts.append({
            "angle": round(a["_angle"], 2),
            "n": a.get("n") if a.get("n") is not None else a["label"], "name": a.get("name") or a["label"],
            "user": a.get("user") or "", "h": float(a.get("h") or 0.0), "allowed": float(a.get("allowed") or 30.0),
            "runs": [r for r in a.get("runs", []) if r[0] in hardware_all or r[0] in ("t4", "p100", "cpu")],
            "routes": {s: r["how"] for s, r in a.get("routes", {}).items() if s in comp_hw},
            "pending": a.get("status") in ("pending", "unreachable"),
            "unreachable": a.get("status") == "unreachable",
        })
    for acc in accounts:  # T4/P100/CPU runs are drawn with the open-hardware colour
        acc["runs"] = [["open" if r[0] in ("t4", "p100", "cpu") else r[0], r[1]] for r in acc["runs"]]

    cluster_rows = []
    for c in clusters:
        days = None
        until = _parse_iso(c.get("until"))
        if until:
            days = max(0, (until.date() - taken.date()).days + 1)
        reserve = None
        for lo, hi in c.get("reserve", []):
            lo_d, hi_d = _parse_iso(lo), _parse_iso(hi)
            if lo_d and hi_d and hi_d.date() >= taken.date():
                reserve = {"from": _fmt_day(lo_d), "to": _fmt_day(hi_d), "days": (lo_d.date() - taken.date()).days}
                break
        cluster_rows.append({
            "id": c["name"], "name": c.get("title") or c["name"], "site": c.get("site") or "",
            "account": c.get("account") or "", "angle": round(c["_angle"], 2),
            "budget": float(c.get("budget") or 0), "left": c.get("left"), "spent": c.get("spent"),
            "until": _fmt_day(until) if until else "", "days": days if days is not None else 999,
            "cap": c.get("gpu_cap") or 0, "in_use": c.get("in_use"),
            "hw": {p: {"how": r.get("how"), "idle": r.get("idle"), "accels": r.get("accels", [])}
                   for p, r in c.get("hardware", {}).items() if p in hardware},
            "jobs": [j for j in c.get("jobs", []) if j and j[0] in hardware],
            "pending": c.get("status") != "live", "unreachable": c.get("status") == "unreachable",
            "error": c.get("error"), "reserve": reserve,
        })

    # the "T4 · P100" lettering goes where no competition sits
    candidates = [c * 15.0 for c in range(24)]
    taken_angles = list(comp_angles.values()) + [c["_angle"] for c in clusters]
    bus = max(candidates, key=lambda c: min([abs(_wrap(c - t)) for t in taken_angles] or [180]))

    pending = [a for a in snapshot["accounts"] if a.get("status") == "pending"]
    removed = [a for a in snapshot["accounts"] if a.get("status") == "removed"]
    unreachable = [a for a in snapshot["accounts"] if a.get("status") == "unreachable"]
    notes = []
    if pending:
        notes.append("%d more awaiting token%s" % (len(pending), "" if len(pending) == 1 else "s"))
    if unreachable:
        notes.append("%d unreachable" % len(unreachable))
    notes += ["account %s %s" % (a.get("n") or a["label"], a.get("reason") or "removed") for a in removed]
    return {
        "hardware": hardware_all,
        "hardware_angles": {k: round(v, 1) for k, v in hw_angles.items()},
        "competitions": competitions,
        "accounts": accounts,
        "clusters": cluster_rows,
        "meta": {"max_concurrent": snapshot.get("max_concurrent", 2), "bus_center": bus,
                 "live_note": " · ".join(notes) or "every folder has a working token"},
    }


# ---------------------------------------------------------------------------------------
# changes since the previous snapshot
# ---------------------------------------------------------------------------------------
def _key(a: Dict[str, Any]) -> str:
    return "n%s" % a["n"] if a.get("n") is not None else a["label"]


def _reach(snapshot: Dict[str, Any], hw: str) -> int:
    return sum(1 for a in snapshot["accounts"] if a.get("status") == "live"
               and any(r["hw"] == hw for r in a["routes"].values()))


def changes(snapshot: Dict[str, Any], previous: Optional[Dict[str, Any]] = None, limit: int = 8) -> List[Dict[str, str]]:
    """Cards for the strip under the dial: notes first, then what changed, then standing issues."""
    hw_name = lambda h: ACCELERATOR_NAMES.get(h, h)
    comp_name = lambda s: COMPETITION_NAMES.get(s, (s,))[0]
    cards = [dict(n) for n in snapshot.get("notes", [])]
    now = {_key(a): a for a in snapshot["accounts"]}
    before = {_key(a): a for a in (previous or {}).get("accounts", [])}

    if previous is not None:
        for k, a in now.items():
            if a.get("status") != "live":
                continue
            old = before.get(k)
            reach = sorted({r["hw"] for r in a["routes"].values()}, key=lambda h: HARDWARE_STYLE.get(h, {}).get("rank", 50))
            if old is None or old.get("status") != "live":
                via = "; ".join("%s through %s" % (hw_name(h), ", ".join(comp_name(s) for s, r in a["routes"].items() if r["hw"] == h))
                                for h in reach) or "T4 / P100 only"
                low = " Only %.1f h of quota left this week." % a["h"] if a["h"] < 5 else ""
                cards.append({"kicker": "new account", "tone": HARDWARE_STYLE.get(reach[0], {}).get("color", "--open") if reach else "--open",
                              "title": "%s · %s" % (a.get("n"), a.get("name")), "text": "Reaches %s.%s" % (via, low)})
                continue
            for slug in sorted(set(a["routes"]) - set(old.get("routes", {}))):
                h = a["routes"][slug]["hw"]
                cards.append({"kicker": "new route", "tone": HARDWARE_STYLE.get(h, {}).get("color", "--x1"),
                              "title": "Account %s joined %s" % (a.get("n"), comp_name(slug)),
                              "text": "%s now reaches %s. %s is open to %d accounts, up from %d."
                                      % (a.get("name"), hw_name(h), hw_name(h), _reach(snapshot, h), _reach(previous, h))})
            for slug in sorted(set(old.get("routes", {})) - set(a["routes"])):
                h = old["routes"][slug]["hw"]
                cards.append({"kicker": "route closed", "tone": "--muted",
                              "title": "Account %s lost %s" % (a.get("n"), comp_name(slug)),
                              "text": "%s no longer reaches %s through %s." % (a.get("name"), hw_name(h), comp_name(slug))})
        for k, old in before.items():
            if old.get("status") == "live" and (k not in now or now[k].get("status") == "removed"):
                reason = (now.get(k) or {}).get("reason") or "removed"
                cards.append({"kicker": "removed", "tone": "--hot", "title": "Account %s %s" % (old.get("n"), reason),
                              "text": "%s is gone, along with its %.1f h of quota." % (old.get("name"), old.get("h") or 0)})
    else:
        for a in snapshot["accounts"]:
            if a.get("status") == "removed":
                cards.append({"kicker": "removed", "tone": "--hot", "title": "Account %s %s" % (a.get("n"), a.get("reason") or "removed"),
                              "text": "%s is left out of the cluster." % a.get("name")})

    pending = [a for a in snapshot["accounts"] if a.get("status") == "pending"]
    if pending:
        nums = [str(a.get("n")) for a in pending]
        cards.append({"kicker": "awaiting tokens", "tone": "--muted",
                      "title": "Account%s %s" % ("s" if len(nums) > 1 else "", " and ".join([", ".join(nums[:-1]), nums[-1]] if len(nums) > 1 else nums)),
                      "text": "%s arrived without a token. Add a token file to bring %s into the cluster."
                              % (" and ".join(a.get("name") for a in pending), "them" if len(nums) > 1 else "it")})
    for a in snapshot["accounts"]:
        if a.get("status") == "unreachable":
            cards.append({"kicker": "unreachable", "tone": "--hot", "title": "Account %s · %s" % (a.get("n"), a.get("name")),
                          "text": "The probe failed: %s" % (a.get("error") or "unknown error")})
    taken = _parse_iso(snapshot.get("taken_at")) or datetime.now(timezone.utc)
    cards += cluster_changes(snapshot, previous, taken)
    # premium hardware about to disappear
    by_hw: Dict[str, List[datetime]] = {}
    for a in snapshot["accounts"]:
        for slug, r in a.get("routes", {}).items():
            d = _parse_iso((snapshot.get("competitions", {}).get(slug) or {}).get("deadline"))
            if d:
                by_hw.setdefault(r["hw"], []).append(d)
    for h, ds in by_hw.items():
        last = max(ds)
        days = (last - taken).total_seconds() / 86400
        if 0 <= days <= 7:
            cards.append({"kicker": "ending soon", "tone": "--hot", "title": "%s ends %s" % (hw_name(h), _fmt_day(last)),
                          "text": "Every route to %s closes within %d days. %d accounts lose it." % (hw_name(h), math.ceil(days), _reach(snapshot, h))})
    return cards[:limit]


def cluster_changes(snapshot: Dict[str, Any], previous: Optional[Dict[str, Any]], taken: datetime) -> List[Dict[str, str]]:
    """Cards about Slurm clusters: joined, unreachable, awaiting probe, reserve window, low budget."""
    cards = []
    before = {c["name"] for c in (previous or {}).get("clusters", [])}
    for c in snapshot.get("clusters", []):
        title = "%s · %s" % (c.get("title") or c["name"], c.get("site") or "")
        hw = " and ".join(HARDWARE_STYLE.get(p, {}).get("name", p) + (" (preemptible)" if r.get("how") == "preemptible" else "")
                          for p, r in c.get("hardware", {}).items())
        until = _parse_iso(c.get("until"))
        if previous is not None and c["name"] not in before:
            cards.append({"kicker": "new cluster", "tone": "--v100", "title": "%s joins" % title,
                          "text": "Slurm project %s: %.0f GPU-h until %s, up to %d GPUs at once, on %s."
                                  % (c.get("account"), c.get("budget") or 0, _fmt_day(until) if until else "?",
                                     c.get("gpu_cap") or 0, hw)})
        if c.get("status") == "pending":
            cards.append({"kicker": "awaiting probe", "tone": "--muted", "title": title,
                          "text": "Shown from its config. Probe it over SSH (--cluster %s) from a machine "
                                  "that holds the key to fill in the budget and the queue." % c["name"]})
        elif c.get("status") == "unreachable":
            cards.append({"kicker": "unreachable", "tone": "--hot", "title": title,
                          "text": "The SSH probe failed: %s" % (c.get("error") or "unknown error")})
        elif c.get("left") is not None and c.get("budget") and c["left"] < 0.1 * c["budget"]:
            cards.append({"kicker": "budget low", "tone": "--hot", "title": "%s: %.0f GPU-h left" % (title, c["left"]),
                          "text": "Under a tenth of the project budget remains until %s." % (_fmt_day(until) if until else "?")})
        for lo, hi in c.get("reserve", []):
            lo_d, hi_d = _parse_iso(lo), _parse_iso(hi)
            if lo_d and hi_d and (lo_d - taken).total_seconds() <= 14 * 86400 and hi_d.date() >= taken.date():
                cards.append({"kicker": "reserve window", "tone": "--l4",
                              "title": "%s held %s to %s" % (c.get("title") or c["name"], _fmt_day(lo_d), _fmt_day(hi_d)),
                              "text": "Budget reserved for the PI's reviewer response: ask before any large run in it."})
    return cards


# ---------------------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------------------
def render(snapshot: Dict[str, Any], previous: Optional[Dict[str, Any]] = None,
           template_path: str = TEMPLATE_PATH) -> str:
    """The Moon Cluster page for ``snapshot``; ``previous`` feeds the changes cards."""
    with open(template_path) as fh:
        page = fh.read()
    data = page_data(json.loads(json.dumps(snapshot)))
    data["snapshot"] = snapshot
    esc = html.escape
    title = snapshot.get("title") or DEFAULT_TITLE
    subtitle = snapshot.get("subtitle")
    taken = _parse_iso(snapshot.get("taken_at"))
    reset = _parse_iso(snapshot.get("quota_reset"))
    stamp = []
    if taken:
        stamp.append("<span>live probe <b>%d %s</b></span>" % (taken.day, esc(taken.strftime("%b %Y · %H:%M UTC"))))
    if reset:
        stamp.append("<span>quota resets <b>%d %s</b></span>" % (reset.day, esc(reset.strftime("%b · %H:%M UTC"))))
    cards = changes(snapshot, previous)
    strip = ""
    if cards:
        strip = '  <section class="changes" aria-label="Changes since the last snapshot">\n' + "".join(
            '    <div class="change"><span class="kicker" style="color:var(%s)">%s</span><h3>%s</h3><p>%s</p></div>\n'
            % (esc(c.get("tone") or "--muted"), esc(c.get("kicker", "note")), esc(c.get("title", "")), esc(c.get("text", "")))
            for c in cards) + "  </section>"
    # "<" is escaped so no text in the snapshot can open or close a tag inside the data block
    blob = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    for key, value in (
        ("{{TITLE}}", esc(title)),
        ("{{SUBTITLE}}", "<em>%s</em>" % esc(subtitle) if subtitle else ""),
        ("{{STAMP}}", "".join(stamp)),
        ("{{CHANGES}}", strip),
        ("{{FOOT}}", ""),
        ("{{DATA}}", blob),
    ):
        page = page.replace(key, value)
    return page


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the Moon Cluster page for the accelerator pool.")
    parser.add_argument("--tokens-dir", default=os.environ.get("KAGGLE_TOKENS_DIR"),
                        help="credentials directory to probe (default: $KAGGLE_TOKENS_DIR)")
    parser.add_argument("--from-snapshot", help="render this snapshot JSON instead of probing")
    parser.add_argument("--previous", help="previous page (.html) or snapshot (.json) to diff against")
    parser.add_argument("--dump-snapshot", help="write the snapshot JSON here")
    parser.add_argument("--out", help="write the page here (default: moon_cluster.html unless only dumping)")
    parser.add_argument("--title", help="page name (default: Moon Cluster)")
    parser.add_argument("--subtitle", help="italic word after the name")
    parser.add_argument("--note", action="append", default=[], help='extra card "KICKER|Title|Text" (repeatable)')
    parser.add_argument("--cluster", action="append", default=[],
                        help="also probe this Slurm cluster over SSH: preset (%s) or JSON config (repeatable)"
                        % ", ".join(CLUSTERS))
    parser.add_argument("--cluster-unprobed", action="append", default=[],
                        help="show this cluster from its config only, as awaiting probe (repeatable)")
    parser.add_argument("--removed", action="append", default=[],
                        help="leave an account out and show it as removed: account7 or account7=deleted (repeatable)")
    args = parser.parse_args(argv)
    if not args.from_snapshot and not args.tokens_dir and not (args.cluster or args.cluster_unprobed):
        parser.error("give --tokens-dir (or set KAGGLE_TOKENS_DIR), --cluster or --from-snapshot")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    removed = dict((r.split("=", 1) + [""])[:2] for r in args.removed)
    clusters = probe_clusters(args.cluster, args.cluster_unprobed)
    if args.from_snapshot:
        snapshot = load_snapshot(args.from_snapshot)
        for a in snapshot["accounts"]:
            n = a.get("n")
            hit = next((k for k in removed if k == a["label"] or k == "account%s" % n), None)
            if hit is not None:
                a.update(status="removed", reason=removed[hit] or "removed", routes={}, runs=[])
        # a freshly probed cluster replaces the snapshot's row of the same name
        fresh = {c["name"] for c in clusters}
        snapshot["clusters"] = [c for c in snapshot.get("clusters", []) if c["name"] not in fresh] + clusters
    else:
        snapshot = probe_snapshot(args.tokens_dir, removed, clusters=clusters)
    if args.title:
        snapshot["title"] = args.title
    if args.subtitle is not None:
        snapshot["subtitle"] = args.subtitle or None
    for note in args.note:
        parts = (note.split("|") + ["", ""])[:3]
        snapshot.setdefault("notes", []).append({"kicker": parts[0] or "note", "title": parts[1], "text": parts[2], "tone": "--l4"})

    if args.dump_snapshot:
        with open(args.dump_snapshot, "w") as fh:
            json.dump(snapshot, fh, indent=1)
        print("snapshot -> %s" % args.dump_snapshot)
    out = args.out or (None if args.dump_snapshot else "moon_cluster.html")
    if out:
        previous = load_snapshot(args.previous) if args.previous else None
        with open(out, "w") as fh:
            fh.write(render(snapshot, previous))
        live = [a for a in snapshot["accounts"] if a.get("status") == "live"]
        print("page -> %s (%d live accounts, %.1f GPU-hours spare, %d cluster(s))"
              % (out, len(live), sum(a["h"] for a in live), len(snapshot.get("clusters", []))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
