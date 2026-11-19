"""Why does the 14x14 cosine-attention GATr learn the 7-gon's turn but not the 10-/13-gon's? Per checkpoint (env: SNAPDIR,
CKPT, BDATA, NGON, EXTRA_ARGS='--dircloud_nside 8 --dircloud_grid 14', NIMG):
 1. is the turn readable from the sphere tokens (linear, held-out R2 of cos/sin of n*theta)?
 2. is it laid out on the sphere as an n-lobed pattern (per ring: share of the ring's variance at azimuthal frequency n,
    and phase coherence of that component with the turn)?
 3. does the predicted velocity contain the frequency-n component about the axis, and how large is it?
 4. how sharp are the pose token's attention windows (concentration kappa per head, fitted as in attn.py), and how much of
    a degree-n signal can such a window pass: a_n = i_n(kappa) / i_0(kappa)?"""
import os, sys, glob, json, math, numpy as np, torch
from lib import *
from PIL import Image
from probe import pose_tokens, rand_rot
from scipy.special import spherical_in
n = int(os.environ["NGON"]); D = os.environ["BDATA"]; N = int(os.environ.get("NIMG", 600)); torch.set_num_threads(4)
rots = np.load(D + "/test/rotations.npz")[f"pyr{n}"][:N]
theta = np.arctan2(rots[:, 0, 1, 0], rots[:, 0, 0, 0])
Y = np.stack([np.cos(n * theta), np.sin(n * theta)], 1)
m = load_model(fused=False); res = dict(n=n, ckpt=os.path.basename(os.environ["CKPT"]))
with torch.no_grad():
    cond = torch.cat([m.condition(torch.from_numpy(np.stack([np.asarray(Image.open(D + f"/test/images/pyr{n}_{j:05d}.png").convert("RGB"))
                      for j in range(i, min(i + 25, N))])).permute(0, 3, 1, 2).float() / 255) for i in range(0, N, 25)])
P = cond.shape[1] - 1; pts = cond[0, :P, 1:4].numpy(); sig = cond[:, :P, 8:72].double().numpy()
def ridge(F):
    F = F.reshape(len(F), -1); tr, te = slice(0, int(.8 * N)), slice(int(.8 * N), N)
    mu = F[tr].mean(0); U, s, Vt = np.linalg.svd(F[tr] - mu, full_matrices=False); k = min(200, len(s))
    A, B = (F[tr] - mu) @ Vt[:k].T, (F[te] - mu) @ Vt[:k].T; sc = s[:k] / math.sqrt(A.shape[0]); A, B = A / sc, B / sc
    return max(1 - ((B @ np.linalg.solve(A.T @ A + lam * len(A) * np.eye(k), A.T @ Y[tr]) - Y[te]) ** 2).sum() / ((Y[te] - Y[tr].mean(0)) ** 2).sum()
               for lam in (1e-2, 1e-1, 1, 10))
res["R2_tokens"] = float(ridge(sig)); res["R2_token_mean"] = float(ridge(sig.mean(1)))
z = np.round(pts[:, 2], 4); psi = np.arctan2(pts[:, 1], pts[:, 0]); rings = []
for v in sorted(set(z), reverse=True):
    idx = np.where(z == v)[0]
    if len(idx) <= 2 * n: rings.append(dict(polar=float(np.degrees(np.arccos(v))), points=int(len(idx)), share=None, coherence=None)); continue
    f = sig[:, idx]; a = (f * np.exp(-1j * n * psi[idx])[None, :, None]).mean(1)
    share = 2 * (np.abs(a) ** 2).mean() / max(f.var(1).mean(), 1e-12)
    coh = np.abs((a * np.exp(1j * n * theta)[:, None]).mean(0)).sum() / np.abs(a).mean(0).sum()
    rings.append(dict(polar=float(np.degrees(np.arccos(v))), points=int(len(idx)), share=float(share), coherence=float(coh)))
res["rings"] = rings
K = 128; phi = torch.arange(K) * (2 * math.pi / K); Rz = torch.zeros(K, 3, 3); Rz[:, 0, 0] = phi.cos(); Rz[:, 1, 1] = phi.cos(); Rz[:, 0, 1] = -phi.sin(); Rz[:, 1, 0] = phi.sin(); Rz[:, 2, 2] = 1
amp, al, sd = [], [], []
with torch.no_grad():
    for i in range(40):
        w = m.vector_field(pose_tokens(Rz), torch.full((K,), 0.9), cond[i:i + 1].expand(K, -1, -1))[:, 0].double().numpy()
        c = np.fft.rfft(w) / K; amp.append(2 * abs(c[n])); al.append(c[n] * np.exp(1j * n * theta[i])); sd.append(w.std())
res["velocity_t0.9"] = dict(std_over_turn=float(np.mean(sd)), amp_freq_n=float(np.mean(amp)), aligned=float(2 * abs(np.mean(al))))
# attention windows of the pose token (query 0) toward the sphere tokens
from gatr.primitives.attention import _INNER_PRODUCT_WO_TRI_IDX
blades = _INNER_PRODUCT_WO_TRI_IDX[1:]; rec = []
def hook(mod, args):
    q, k = args[0], args[1]; d = lambda mv: mod.unit(mv[..., blades].flatten(-2))
    s = mod.log_scale.clamp(max=math.log(mod.max_scale)).exp()[:, None, None]
    rec.append((s * d(q) @ d(k).transpose(-1, -2))[:, :, 0, 1:P + 1].detach())
hs = [a.register_forward_pre_hook(hook) for a in m.vector_field.angular_attention]
R = rand_rot(256, torch.Generator().manual_seed(0))
with torch.no_grad(): m.vector_field(pose_tokens(R), torch.full((256,), 0.9), cond[:1].expand(256, -1, -1))
for h in hs: h.remove()
A = torch.einsum("jc,ncd->njd", torch.from_numpy(pts).float(), R).reshape(-1, 3); kap = []
for b, ang in enumerate(rec):
    row = []
    for h in range(ang.shape[1]):
        y = ang[:, h].reshape(-1, 1).double(); yc = y - y.mean(); Ad = A.double()
        sol = torch.linalg.lstsq(Ad, yc).solution; r2 = 1 - (Ad @ sol - yc).square().sum() / yc.square().sum()
        row.append((float(sol.norm()), float(r2)))
    kap.append(row)
res["kappa"] = [[k for k, _ in r] for r in kap]; res["kappa_R2"] = [[r2 for _, r2 in r] for r in kap]
ks = np.array(res["kappa"]); pass_n = lambda k, l: spherical_in(l, k) / spherical_in(0, k)
res["window_pass"] = {str(l): dict(median=float(np.median(pass_n(ks, l))), max=float(np.max(pass_n(ks, l)))) for l in (4, 5, 7, 10, 13)}
json.dump(res, open(f"diag2_{os.environ.get('TAG', n)}.json", "w"), indent=1)
print(f"n={n} {res['ckpt']}\n  turn readable from tokens: R2 {res['R2_tokens']:.2f} (mean over tokens {res['R2_token_mean']:.2f})")
print("  rings (polar deg, points, share at freq n, coherence):", "; ".join(f"{r['polar']:.0f}:{r['points']}" + ("" if r['share'] is None else f" {r['share']:.3f}/{r['coherence']:.2f}") for r in rings))
print("  velocity at t=0.9:", {k: round(v, 4) for k, v in res["velocity_t0.9"].items()})
print("  kappa per block (median over heads):", [round(float(np.median(r)), 1) for r in res["kappa"]], " max:", round(float(ks.max()), 1), " fit R2 median:", round(float(np.median(res["kappa_R2"])), 2))
print("  window pass a_l (median / max over heads):", {l: (round(v['median'], 3), round(v['max'], 3)) for l, v in res["window_pass"].items()})
