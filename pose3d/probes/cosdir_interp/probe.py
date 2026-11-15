"""Degree spectrum over the pose R of every stage of the GATr vector field (pose token = token 0).

Circles R(theta) = exp(theta n) R0 with Haar-random R0 and axis n. A function of degree l on SO(3) spreads its
power P_l evenly over the 2l+1 frequencies -l..l along such a circle (averaged over circles), so with
E_m = mean power at frequencies +m and -m:   tail(L) = sum_{l>=L} P_l = (2L+1)/2 E_L + sum_{m>L} E_m.
"""
import math, json, sys, time
import numpy as np, torch
from lib import *
from gatr.utils.tensors import construct_reference_multivector

def rand_rot(n, g):
    q = torch.randn(n, 4, generator=g); q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), 2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1).view(n, 3, 3)

def circles(n_circ, n_pts, g):
    R0, G = rand_rot(n_circ, g), rand_rot(n_circ, g)
    th = torch.arange(n_pts) * (2 * math.pi / n_pts)
    Rz = torch.zeros(n_pts, 3, 3); Rz[:, 0, 0] = th.cos(); Rz[:, 1, 1] = th.cos(); Rz[:, 0, 1] = -th.sin(); Rz[:, 1, 0] = th.sin(); Rz[:, 2, 2] = 1
    R = torch.einsum("cij,pjk,clk,clm->cpim", G, Rz, G, R0)      # G Rz G^T R0
    return R.reshape(-1, 3, 3)

def pose_tokens(R):
    frame = R.transpose(-1, -2)                                  # rows R e1, R e2, R e3
    pose = frame.new_zeros(len(R), 3, 8); a = frame[:, 0]
    pose[:, 0, 4], pose[:, 0, 5], pose[:, 0, 6] = a[:, 2], -a[:, 1], a[:, 0]
    pose[:, 1, 1:4], pose[:, 2, 1:4] = frame[:, 1], frame[:, 2]
    return pose

def embed(vf, pose, t, cond):
    b, n = cond.shape[:2]
    mv = pose.new_zeros(b, 1 + n, 3, 16); mv[:, 0, :, vf._pga_index] = pose
    mv[:, 1:, 0, vf._pga_index] = cond[..., :8]
    s = pose.new_zeros(b, 1 + n, vf.cond_scalars + 2)
    s[:, 1:, :vf.cond_scalars] = cond[..., 8:]; s[..., -2] = t.unsqueeze(1); s[:, 0, -1] = 1.0
    return mv, s

def stream(vf, pose, t, cond, hook=None):
    """Returns {stage: (mv_token0 (B,C,16), s_token0 (B,S))} for the residual stream and each sublayer's update,
    plus 'cloud*' entries with the cloud tokens' stream (for how much of the pose reaches them)."""
    net = vf.net; mv, s = embed(vf, pose, t, cond)
    ref = construct_reference_multivector("data", mv); add_mv, add_s = net._construct_reinserted_channels(mv, s)
    out = {}
    h_mv, h_s = net.linear_in(mv, scalars=s); out["in"] = (h_mv[:, 0], h_s[:, 0])
    for i, blk in enumerate(net.blocks):
        a_mv, a_s = blk._attention_block(multivectors=h_mv, scalars=h_s, additional_qk_features_mv=add_mv,
                                         additional_qk_features_s=add_s, attention_mask=None)
        out[f"b{i}.attn.delta"] = (a_mv[:, 0], a_s[:, 0]); h_mv, h_s = h_mv + a_mv, h_s + a_s
        out[f"b{i}.attn"] = (h_mv[:, 0], h_s[:, 0])
        out[f"b{i}.cloud"] = (h_mv[:, 1::11].flatten(1, 2), h_s[:, 1::11].flatten(1, 2))
        m_mv, m_s = blk._mlp_block(h_mv, scalars=h_s, reference_mv=ref)
        out[f"b{i}.mlp.delta"] = (m_mv[:, 0], m_s[:, 0]); h_mv, h_s = h_mv + m_mv, h_s + m_s
        out[f"b{i}.mlp"] = (h_mv[:, 0], h_s[:, 0])
    o_mv, _ = net.linear_out(h_mv, scalars=h_s)
    w = o_mv[:, 0, 0][:, vf._rot_index]
    out["velocity"] = (w, w[:, :0])
    return out

def spectrum(x, n_circ, n_pts):
    """x (n_circ*n_pts, D) -> dict(total, p0, E (n_pts//2,) two-sided power per frequency, summed over D)."""
    x = x.double().view(n_circ, n_pts, -1)
    c = torch.fft.rfft(x, dim=1) / n_pts                         # (n_circ, n_pts//2+1, D)
    pw = c.abs().square().mean(0).sum(-1)                        # per frequency, summed over components
    E = 2 * pw[1:n_pts // 2]                                     # +m and -m, m = 1 .. n_pts/2 - 1
    total = x.square().mean((0, 1)).sum(); p0 = x.mean((0, 1)).square().sum()
    return dict(total=total.item(), p0=p0.item(), E=E.tolist())

def tails(sp, Ls):
    E = np.array(sp["E"]); var = sp["total"] - sp["p0"]
    out = {}
    for L in Ls:
        out[L] = float(((2 * L + 1) / 2 * E[L - 1] + E[L:].sum()) / max(var, 1e-30))
    return out

def degree_power(sp, lmax):
    E = np.array(sp["E"]); var = sp["total"] - sp["p0"]
    return [float((2 * l + 1) / 2 * (E[l - 1] - E[l]) / max(var, 1e-30)) for l in range(1, lmax + 1)]

@torch.no_grad()
def run(vf, cond1, t, n_circ=128, n_pts=128, seed=0, batch=512, keep=None):
    g = torch.Generator().manual_seed(seed)
    R = circles(n_circ, n_pts, g); pose = pose_tokens(R)
    acc = {}
    for i in range(0, len(R), batch):
        p = pose[i:i + batch]; o = stream(vf, p, torch.full((len(p),), t), cond1.expand(len(p), -1, -1))
        for k, (a, b) in o.items():
            if keep and not keep(k): continue
            acc.setdefault(k + "|mv", []).append(a.flatten(1).float().clone()); acc.setdefault(k + "|s", []).append(b.flatten(1).float().clone())
    return {k: spectrum(torch.cat(v), n_circ, n_pts) for k, v in acc.items() if v[0].shape[1]}

if __name__ == "__main__":
    out_path = sys.argv[1]; per = int(sys.argv[2]); ts = [float(a) for a in sys.argv[3:]]
    m = load_model(); x, eq, cls = load_data(per)
    with torch.no_grad(): cond = m.condition(x)
    res = []
    import os
    for i in range(len(x)):
        if SHAPES[cls[i]] in os.environ.get('SKIP', '').split(','): continue
        for t in ts:
            t0 = time.time(); r = run(m.vector_field, cond[i:i + 1], t)
            res.append(dict(shape=SHAPES[cls[i]], image=i, t=t, spec=r)); print(SHAPES[cls[i]], i, t, round(time.time() - t0), flush=True)
            json.dump(res, open(out_path, "w"))
