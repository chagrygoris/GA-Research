"""Anatomy of the cosdir attention for the pose token (query 0): per block and head."""
import math, json, torch, numpy as np
from lib import *
from probe import rand_rot, pose_tokens, stream
from gatr.primitives.attention import _INNER_PRODUCT_WO_TRI_IDX, _TRIVECTOR_IDX, _build_dist_basis, _build_dist_vec

def parts(att, q_mv, k_mv, q_s, k_s):
    """The three logit terms of mode cosdir: angular s*cos, scalar dot / root, distance / root. (B, H, T, T) each."""
    bq, bk = _build_dist_basis(q_mv.device, q_mv.dtype)
    q_dist = _build_dist_vec(q_mv[..., _TRIVECTOR_IDX], bq, att.normalizer)
    q_dist = (q_dist * att.log_weights.exp()[..., None]).flatten(-2)
    k_dist = _build_dist_vec(k_mv[..., _TRIVECTOR_IDX], bk, att.normalizer).flatten(-2)
    blades = _INNER_PRODUCT_WO_TRI_IDX[1:]
    q_geo = q_mv[..., _INNER_PRODUCT_WO_TRI_IDX].flatten(-2)
    root = math.sqrt(q_geo.shape[-1] + q_dist.shape[-1] + q_s.shape[-1])
    pair = lambda a, b: a @ b.transpose(-1, -2)
    scale = att.log_scale.clamp(max=math.log(att.max_scale)).exp()[:, None, None]
    d = lambda mv: att.unit(mv[..., blades].flatten(-2))
    ang = scale * pair(d(q_mv), d(k_mv))
    sc = (pair(q_mv[..., 0], k_mv[..., 0]) + pair(q_s, k_s)) / root
    dist = pair(q_dist, k_dist) / root
    return ang, sc, dist, d(q_mv), d(k_mv), q_mv, k_mv

@torch.no_grad()
def collect(m, cond1, t, N=1024, seed=0):
    vf = m.vector_field; g = torch.Generator().manual_seed(seed)
    R = rand_rot(N, g); rec = []
    hooks = [a.register_forward_pre_hook(lambda mod, args, rec=rec: rec.append(parts(mod, args[0], args[1], args[3], args[4])))
             for a in vf.angular_attention]
    stream(vf, pose_tokens(R), torch.full((N,), t), cond1.expand(N, -1, -1))
    for h in hooks: h.remove()
    return R, rec

if __name__ == "__main__":
    m = load_model(); x, eq, cls = load_data(2)
    with torch.no_grad(): cond = m.condition(x)
    pts = cond[0, :, 1:4]                                   # the 89 key directions (88 hemisphere points + e3)
    out = []
    for i in range(len(x)):
        R, rec = collect(m, cond[i:i + 1], 0.5)
        for b, (ang, sc, dist, dq, dk, q_mv, k_mv) in enumerate(rec):
            logits = ang + sc + dist; w = logits.softmax(-1)
            wp = w[:, :, 0]                                 # pose query: (N, H, T)
            H = -(wp * wp.clamp_min(1e-30).log()).sum(-1)
            row = dict(image=i, shape=SHAPES[cls[i]], block=b,
                       s=m.vector_field.angular_attention[b].log_scale.exp().tolist(),
                       # spread of each logit term over the 89 cloud keys, for the pose query (mean over poses)
                       ang_std=ang[:, :, 0, 1:].std(-1).mean(0).tolist(), ang_range=(ang[:, :, 0, 1:].max(-1).values - ang[:, :, 0, 1:].min(-1).values).mean(0).tolist(),
                       sc_std=sc[:, :, 0, 1:].std(-1).mean(0).tolist(), dist_std=dist[:, :, 0, 1:].std(-1).mean(0).tolist(),
                       n_eff=H.exp().mean(0).tolist(), self_mass=wp[:, :, 0].mean(0).tolist(), top1=wp[:, :, 1:].max(-1).values.mean(0).tolist(),
                       cloud_to_pose=w[:, :, 1:, 0].mean((0, 2)).tolist(),
                       # how much of the cloud queries' angular logit spread comes from the pose key
                       cloud_n_eff=(-(w[:, :, 1:] * w[:, :, 1:].clamp_min(1e-30).log()).sum(-1)).exp().mean((0, 2)).tolist())
            # fit ang(pose -> cloud j) = x_j^T R m_h : a body-fixed probe direction per head (exact at block 0 if the
            # key direction is the token's own vector)
            A = torch.einsum("jc,ncd->njd", pts, R).reshape(-1, 3)                 # (N*89, 3)
            fits = []
            for h in range(ang.shape[1]):
                y = ang[:, h, 0, 1:].reshape(-1, 1); yc = y - y.mean()
                sol = torch.linalg.lstsq(A, yc).solution
                r2 = 1 - (A @ sol - yc).square().sum() / yc.square().sum()
                fits.append(dict(kappa=sol.norm().item(), dir=(sol[:, 0] / sol.norm()).tolist(), r2=r2.item()))
            row["probe"] = fits; out.append(row)
    json.dump(out, open("attn.json", "w"))
    import collections
    for b in range(4):
        rows = [r for r in out if r["block"] == b]; f = lambda k: np.round(np.mean([r[k] for r in rows], 0), 2)
        print(f"block {b}: s", f("s")); print("  angular logit std over keys", f("ang_std"), " range", f("ang_range"))
        print("  scalar-term std", f("sc_std"), " distance-term std", f("dist_std"))
        print("  n_eff (pose query)", f("n_eff"), " top-1 weight", f("top1"), " pose->pose", f("self_mass"))
        print("  cloud->pose mass", f("cloud_to_pose"), " cloud n_eff", f("cloud_n_eff"))
        print("  probe fit: kappa", np.round(np.mean([[p["kappa"] for p in r["probe"]] for r in rows], 0), 1),
              " R2", np.round(np.mean([[p["r2"] for p in r["probe"]] for r in rows], 0), 2))
        if b == 0: print("  probe body directions (image 0):", [np.round(p["dir"], 2).tolist() for p in rows[0]["probe"]])
