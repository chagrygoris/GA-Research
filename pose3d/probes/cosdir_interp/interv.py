"""Causal tests: (1) rescale the attention inverse temperatures s at inference, (2) untrained field, cosdir vs dot."""
import json, math, time, copy, torch, numpy as np
from lib import *
from probe import run, tails

def errors(m, x, eq, cls, n_samples=2, seed=0):
    torch.manual_seed(seed); out = {s: [] for s in SHAPES}
    with torch.no_grad():
        for i in range(0, len(x), 20):
            xb = x[i:i + 20].repeat_interleave(n_samples, 0)
            R = m.predict(xb, steps=20).view(-1, n_samples, 3, 3)
            for j in range(len(R)):
                e = eq[i + j]; tr = torch.einsum("sab,nab->sn", R[j], e)
                ang = torch.rad2deg(torch.acos(((tr - 1) / 2).clamp(-1, 1))).min(-1).values
                out[SHAPES[cls[i + j]]] += ang.tolist()
    return {s: dict(median=float(np.median(v)), acc15=float(np.mean(np.array(v) < 15))) for s, v in out.items()}

if __name__ == "__main__":
    m = load_model(); x, eq, cls = load_data(12); att = m.vector_field.angular_attention
    base = [a.log_scale.detach().clone() for a in att]
    with torch.no_grad(): cond = m.condition(x)
    first = {s: int((cls == c).nonzero()[0]) for c, s in enumerate(SHAPES)}
    conds = [("all x1", {b: 1.0 for b in range(4)})] + [(f"all x{f}", {b: f for b in range(4)}) for f in (2, 0.5, 0.25, 0.1, 0.03)] \
          + [(f"block {b} x0.1", {b: 0.1}) for b in range(4)]
    res = []
    for name, fac in conds:
        with torch.no_grad():
            for b, a in enumerate(att): a.log_scale.copy_(base[b] + math.log(fac.get(b, 1.0)))
        t0 = time.time(); r = dict(name=name, err=errors(m, x, eq, cls), spec={})
        for s in ("tet", "cube", "icosa"):
            sp = run(m.vector_field, cond[first[s]:first[s] + 1], 0.5, n_circ=32, n_pts=128, keep=lambda k: k == "velocity")
            r["spec"][s] = dict(tails=tails(sp["velocity|mv"], (2, 3, 4, 6, 8, 12)), var=sp["velocity|mv"]["total"] - sp["velocity|mv"]["p0"])
        res.append(r); json.dump(res, open("interv.json", "w"))
        print(name, {s: round(v["median"], 1) for s, v in r["err"].items()}, {s: round(v["tails"][4], 2) for s, v in r["spec"].items()}, round(time.time() - t0), flush=True)
    with torch.no_grad():
        for b, a in enumerate(att): a.log_scale.copy_(base[b])
    # (2) untrained vector field on the trained image tokens: the prior of the architecture
    init = []
    for seed in (0, 1):
        from pose3d.config import parse_args
        import sys
        from pose3d.models import build_model
        from clifford.algebra.cliffordalgebra import CliffordAlgebra
        old = sys.argv; sys.argv = ["x"] + ARGS; cfg = parse_args(); sys.argv = old; cfg.device = "cpu"
        torch.manual_seed(100 + seed); vf = build_model(cfg, CliffordAlgebra((1, 1, 1))).vector_field.eval()
        for mode, s in (("cosdir", 30.0), ("cosdir", 1.0), ("dot", 1.0)):
            for a in vf.angular_attention:
                a.fused = False; a.mode = mode
                with torch.no_grad(): a.log_scale.fill_(math.log(s))
            sp = run(vf, cond[first["cube"]:first["cube"] + 1], 0.5, n_circ=32, n_pts=128,
                     keep=lambda k: k in ("velocity", "b0.attn.delta", "b3.mlp"))
            row = dict(seed=seed, mode=mode, s=s, tails={k: tails(v, (2, 3, 4, 6, 8, 12)) for k, v in sp.items()})
            init.append(row); print("init", seed, mode, s, {k: round(v[4], 3) for k, v in row["tails"].items()}, flush=True)
            json.dump(init, open("init.json", "w"))
