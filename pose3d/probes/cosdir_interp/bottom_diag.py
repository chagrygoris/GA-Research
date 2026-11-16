"""Where is the turn of a bottom-view n-gon lost? Probes on the trained cosdir GATr (env: COSDIR_SNAPSHOT, COSDIR_CKPT, COSDIR_DATA (any directory), BDATA (.../symsol_chile_bottom), NGON)."""
import os, sys, glob, json, math, numpy as np, torch
from lib import *
from PIL import Image
from probe import pose_tokens
n = int(os.environ["NGON"]); D = os.environ["BDATA"]; N = int(os.environ.get("NIMG", 1000))
rots = np.load(D + "/test/rotations.npz")[f"pyr{n}"][:N]
theta = np.arctan2(rots[:, 0, 1, 0], rots[:, 0, 0, 0])                     # the turn of each test image
x = torch.from_numpy(np.stack([np.asarray(Image.open(D + f"/test/images/pyr{n}_{i:05d}.png").convert("RGB")) for i in range(N)])).permute(0, 3, 1, 2).float() / 255
Y = np.stack([np.cos(n * theta), np.sin(n * theta)], 1)                     # what the model must know: the turn modulo 360/n

def feats(m):
    grab = {}
    h = m.adapter.conv_adapter.register_forward_pre_hook(lambda mod, a: grab.setdefault("f", []).append(a[0]))
    with torch.no_grad(): cond = torch.cat([m.condition(x[i:i + 50]) for i in range(0, N, 50)])
    h.remove(); return torch.cat(grab["f"]), cond

def probe(F, name, k=200):
    """Ridge regression of (cos n theta, sin n theta) on the top-k principal components; R^2 on held-out images."""
    F = F.reshape(len(F), -1).double().numpy(); tr, te = slice(0, int(.8 * N)), slice(int(.8 * N), N)
    mu = F[tr].mean(0); U, s, Vt = np.linalg.svd(F[tr] - mu, full_matrices=False); k = min(k, len(s))
    A, B = (F[tr] - mu) @ Vt[:k].T, (F[te] - mu) @ Vt[:k].T; A, B = A / (s[:k] / math.sqrt(len(A))), B / (s[:k] / math.sqrt(len(A)))
    best = -9
    for lam in (1e-3, 1e-1, 1, 10, 100):
        W = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(k), A.T @ Y[tr]); r2 = 1 - ((B @ W - Y[te]) ** 2).sum() / ((Y[te] - Y[tr].mean(0)) ** 2).sum()
        best = max(best, r2)
    # the same with a random feature expansion (nonlinear read-out)
    rng = np.random.default_rng(0); Wr = rng.normal(size=(k, 2000)) / math.sqrt(k) * 2; b = rng.uniform(0, 2 * math.pi, 2000)
    A2, B2 = np.cos(A[:, :k] @ Wr + b), np.cos(B[:, :k] @ Wr + b); bestn = -9
    for lam in (1e-4, 1e-2, 1):
        W = np.linalg.solve(A2.T @ A2 + lam * len(A2) * np.eye(2000), A2.T @ Y[tr]); r2 = 1 - ((B2 @ W - Y[te]) ** 2).sum() / ((Y[te] - Y[tr].mean(0)) ** 2).sum()
        bestn = max(bestn, r2)
    print(f"  {name:46s} linear R2 = {best:5.2f}   nonlinear R2 = {bestn:5.2f}", flush=True); return dict(name=name, linear=best, nonlinear=bestn)

res = dict(n=n, probes=[])
m = load_model(); fmap, cond = feats(m); pts = cond[0, :88, 1:4].numpy(); sig = cond[:, :88, 8:72]            # (N, 88, 64)
print("turn decodable from (held-out R2 of cos/sin of n*theta):")
res["probes"] += [probe(fmap, "trained ResNet map 7x7x2048"), probe(fmap.mean((2, 3)), "trained ResNet map, mean over cells"),
                  probe(sig, "cloud scalars 88x64 (what the GATr sees)"), probe(sig.mean(1), "cloud scalars, mean over tokens")]
# per ring of the hemisphere grid: the azimuthal frequency-n part of the signal and whether its phase follows the turn
z = np.round(pts[:, 2], 4); psi = np.arctan2(pts[:, 1], pts[:, 0]); res["rings"] = []
print("rings: polar angle, points, share of ring variance at azimuthal frequency n, phase coherence with the turn")
for v in sorted(set(z), reverse=True):
    idx = np.where(z == v)[0]; f = sig[:, idx].double().numpy()                    # (N, pts, 64)
    if len(idx) <= n: print(f"  {np.degrees(np.arccos(v)):5.1f} deg, {len(idx):2d} points: cannot hold frequency {n} (needs more than {n} points)"); continue
    a = (f * np.exp(-1j * n * psi[idx])[None, :, None]).mean(1)                    # (N, 64) coefficient at frequency n
    var = f.var(1).mean(); share = 2 * (np.abs(a) ** 2).mean() / max(var, 1e-12)
    coh = np.abs((a * np.exp(1j * n * theta)[:, None]).mean(0)).sum() / np.abs(a).mean(0).sum()
    alias = "  (Nyquist: one quadrature only)" if len(idx) == 2 * n else ""
    print(f"  {np.degrees(np.arccos(v)):5.1f} deg, {len(idx):2d} points: share {share:.3f}, coherence {coh:.2f}{alias}")
    res["rings"].append(dict(polar=float(np.degrees(np.arccos(v))), points=int(len(idx)), share=float(share), coherence=float(coh)))
# the model's velocity about the axis as a function of the pose's turn phi, for a fixed image
K = 64; phi = torch.arange(K) * (2 * math.pi / K); Rz = torch.zeros(K, 3, 3); Rz[:, 0, 0] = phi.cos(); Rz[:, 1, 1] = phi.cos(); Rz[:, 0, 1] = -phi.sin(); Rz[:, 1, 0] = phi.sin(); Rz[:, 2, 2] = 1
out = {}
for t in (0.5, 0.9):
    amp, al, tot = [], [], []
    with torch.no_grad():
        for i in range(0, 60):
            w = m.vector_field(pose_tokens(Rz), torch.full((K,), t), cond[i:i + 1].expand(K, -1, -1))   # spatial bivector (e12, e13, e23)
            wz = w[:, 0].double().numpy(); c = np.fft.rfft(wz) / K
            # correct field ~ -sin(n (phi - theta)): coefficient at frequency n proportional to i * exp(-i n theta)
            amp.append(2 * abs(c[n])); al.append((c[n] * np.exp(1j * n * theta[i]))); tot.append(wz.std())
    out[t] = dict(amp_n=float(np.mean(amp)), std=float(np.mean(tot)), aligned=float(abs(np.mean(al)) * 2))
    print(f"velocity about the axis vs pose turn, t={t}: std over turn {np.mean(tot):.4f}, amplitude at frequency n {np.mean(amp):.4f}, part aligned with the true turn {abs(np.mean(al)) * 2:.4f}")
res["velocity"] = out; json.dump(res, open(f"bottom{n}_diag.json", "w"))
