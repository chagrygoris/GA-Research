"""Train only the GATr vector field of a bottom-view model on frozen condition tokens (CPU).
env: COSDIR_SNAPSHOT, COSDIR_CKPT, COSDIR_DATA (any directory), BDATA (.../symsol_chile_bottom), NGON; args: name, passes, lr, init (trained|scratch), coupling (indep|nearest|exact), k time samples, [polar] [lift]"""
import os, sys, math, json, time, numpy as np, torch
from lib import *
from PIL import Image
n = int(os.environ["NGON"]); D = os.environ["BDATA"]
name, epochs, lr, init, coupling, k = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6])
polar = "polar" in sys.argv[7:]; lift = "lift" in sys.argv[7:]; torch.manual_seed(0); torch.set_num_threads(int(os.environ.get("THREADS", 2)))
m = load_model(fused=True)
from pose3d.geometry.flow import geodesic_interpolate, relative_log
from pose3d.geometry.rotor import matrix_to_rotor, random_rotor, rotor_to_matrix

def cache(split, N):
    f = f"{D}/cond_{split}_pyr{n}.pt"
    if os.path.exists(f): return torch.load(f)
    rots = np.load(f"{D}/{split}/rotations.npz")[f"pyr{n}"][:N]; out = []
    with torch.no_grad():
        for i in range(0, N, 50):
            x = torch.from_numpy(np.stack([np.asarray(Image.open(f"{D}/{split}/images/pyr{n}_{j:05d}.png").convert("RGB")) for j in range(i, min(i + 50, N))])).permute(0, 3, 1, 2).float() / 255
            out.append(m.condition(x))
    r = (torch.cat(out), torch.from_numpy(rots[:, 0]).float()); torch.save(r, f); return r
ctr, Rtr = cache("train", 4500); cte, Rte = cache("test", 1000)
def cache_maps(split, N):
    f = f"{D}/fmap_{split}_pyr{n}.pt"
    if os.path.exists(f): return torch.load(f)
    out = []
    with torch.no_grad():
        for i in range(0, N, 50):
            x = torch.from_numpy(np.stack([np.asarray(Image.open(f"{D}/{split}/images/pyr{n}_{j:05d}.png").convert("RGB")) for j in range(i, min(i + 50, N))])).permute(0, 3, 1, 2).float() / 255
            out.append(m.adapter.backbone(x).half())
    r = torch.cat(out); torch.save(r, f); return r
if lift:   # the 1x1 projection and the lift to the sphere are trained too, on cached ResNet maps
    ftr, fte = cache_maps("train", 4500), cache_maps("test", 1000); head = m.adapter.conv_adapter
    with torch.no_grad(): print("tokens from cached maps match:", (head(fte[:50].float()) - cte[:50]).abs().max().item(), flush=True)
    tokens = lambda idx, train=True: head((ftr if train else fte)[idx].float())
else:
    tokens = lambda idx, train=True: (ctr if train else cte)[idx]
if init == "scratch":
    import sys as _s
    from pose3d.config import parse_args; from pose3d.models import build_model; from clifford.algebra.cliffordalgebra import CliffordAlgebra
    old = _s.argv; _s.argv = ["x"] + ARGS; cfg = parse_args(); _s.argv = old; cfg.device = "cpu"; torch.manual_seed(0)
    fresh = build_model(cfg, CliffordAlgebra((1, 1, 1))); m.vector_field.load_state_dict(fresh.vector_field.state_dict())
    for a in m.vector_field.angular_attention: a.fused = True
if polar:   # ablation: the pose's first axis as an ordinary vector instead of an axial bivector
    def pin(rotor, t, cond):
        frame = rotor_to_matrix(rotor, m.algebra).transpose(-1, -2); pose = frame.new_zeros(frame.shape[0], 3, 8)
        a = frame[:, 0]; pose[:, 0, 4], pose[:, 0, 5], pose[:, 0, 6] = a[:, 2], -a[:, 1], a[:, 0]      # the trained axial copy, kept
        pose[:, 0, 1:4], pose[:, 1, 1:4], pose[:, 2, 1:4] = frame[:, 0], frame[:, 1], frame[:, 2]   # plus R e1 as an ordinary vector
        return m._spatial_to_body(rotor, m.vector_field(pose, t, cond))
    m._pin_velocity = pin
ang = 2 * math.pi / n; Rz = lambda a: torch.stack([torch.stack([a.cos(), -a.sin(), 0 * a], -1), torch.stack([a.sin(), a.cos(), 0 * a], -1), torch.stack([0 * a, 0 * a, 1 + 0 * a], -1)], -2)
def equivalents(R): return torch.einsum("bij,kjl->bkil", R, Rz(torch.arange(n) * ang))                     # (B, n, 3, 3)
def angle(A, B): return torch.rad2deg(torch.acos(((torch.einsum("...ij,...ij->...", A, B) - 1) / 2).clamp(-1, 1)))

def vstar(rt, t, eq):
    """Exact marginal velocity of independent-coupling flow matching with uniform noise: the posterior mean of
    log(R_t^-1 r1_k) / (1 - t) over the equivalents r1_k (eq (B, K, 3, 3)); weights from the density of R_t given r1_k."""
    B, K = eq.shape[:2]; s = (1 - t).clamp_min(1e-3)[:, None]
    al = torch.deg2rad(angle(rotor_to_matrix(rt, m.algebra)[:, None], eq)).clamp_min(1e-4)
    hav = lambda a: 2 * torch.sin(a.double() / 2) ** 2
    w = (hav((al / s).clamp(max=math.pi)) / (s.double() * hav(al)) * (al <= s * math.pi + 1e-3)).float()
    w = w + 1e-12 * (al == al.min(-1, keepdim=True).values); w = w / w.sum(-1, keepdim=True)
    u = relative_log(rt[:, None].expand(B, K, 4).reshape(-1, 4), matrix_to_rotor(eq.reshape(-1, 3, 3)), m.algebra).view(B, K, 3) / s[..., None]
    return (w[..., None] * torch.nan_to_num(u)).sum(1)

RING = Rz(torch.arange(360) * (2 * math.pi / 360))
@torch.no_grad()
def gap(N=1000, reps=4):
    """On test tokens: mean squared distance of the model, and of the best image-blind field, to the exact velocity."""
    torch.manual_seed(2); a = b = c = 0.0
    for _ in range(reps):
        idx = torch.arange(N); eq = equivalents(Rte[idx]); r0 = random_rotor(N); t = torch.rand(N)
        r1 = matrix_to_rotor(eq[idx, torch.randint(n, (N,))]); rt = geodesic_interpolate(r0, r1, t, m.algebra)
        vs = vstar(rt, t, eq); blind = vstar(rt, t, RING[None].expand(N, -1, -1, -1))
        pred = torch.cat([m.velocity(rt[i:i + 250], t[i:i + 250], tokens(idx[i:i + 250], False)) for i in range(0, N, 250)])
        late = t > 0.75
        a += (pred - vs).pow(2).sum(-1).mean().item() / reps; b += (blind - vs).pow(2).sum(-1).mean().item() / reps
        c += (pred - blind).pow(2).sum(-1).mean().item() / reps
    return dict(model_to_exact=a, blind_to_exact=b, model_to_blind=c)

@torch.no_grad()
def evaluate(N=200, steps=20):
    from pose3d.models.clifford_flow import exp_map, rotor_multiply
    g = torch.Generator().manual_seed(1); torch.manual_seed(1); rotor = random_rotor(N); c = tokens(torch.arange(N), False)
    for i in range(steps):
        v = m.velocity(rotor, torch.full((N,), i / steps), c); rotor = rotor_multiply(rotor, exp_map(v / steps), m.algebra)
    R = rotor_to_matrix(rotor, m.algebra); e = angle(R[:, None], equivalents(Rte[:N])).min(-1).values
    return e.median().item()

params = list(m.vector_field.parameters()) + (list(m.adapter.conv_adapter.parameters()) if lift else []); opt = torch.optim.AdamW(params, lr=lr); log = []
print(name, "start median error", round(evaluate(), 2), {a: round(b, 3) for a, b in gap().items()}, flush=True)
for ep in range(epochs):
    perm = torch.randperm(4500); tot, cnt, t0 = 0.0, 0, time.time(); m.vector_field.train()
    for i in range(0, 4500 - 63, 64):
        idx = perm[i:i + 64].repeat_interleave(k); cond = tokens(idx); eq = equivalents(Rtr[idx]); B = len(idx)
        r0 = random_rotor(B); t = torch.rand(B)
        if coupling == "nearest":
            R0 = rotor_to_matrix(r0, m.algebra); R1 = eq[torch.arange(B), angle(R0[:, None], eq).argmin(-1)]
        else:
            R1 = eq[torch.arange(B), torch.randint(n, (B,))]
        r1 = matrix_to_rotor(R1); rt = geodesic_interpolate(r0, r1, t, m.algebra); target = relative_log(r0, r1, m.algebra)
        if coupling == "exact": target = vstar(rt, t, eq)
        loss = (m.velocity(rt, t, cond) - target).pow(2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); cnt += 1
    m.vector_field.eval(); e = evaluate(); g = gap(); log.append(dict(epoch=ep + 1, loss=tot / cnt, median=e, **g, s=[a.log_scale.exp().mean().item() for a in m.vector_field.angular_attention]))
    print(name, ep + 1, f"loss {tot / cnt:.3f} median {e:.1f} to-exact {g['model_to_exact']:.3f} ({time.time() - t0:.0f}s)", flush=True); json.dump(log, open(f"ft_{name}.json", "w"))
torch.save(m.vector_field.state_dict(), f"ft_{name}.pt")
