"""Does the image encoder know the turn of the 7-gonal pyramid about its own axis? Linear read-out test.

For each image with pose R (body -> camera; rotations.npz index 0), split R into a swing and a twist: the axis
a = R e3, S = the shortest rotation taking e3 to a, and Rz(theta) = S^T R, the turn theta about the pyramid's own
axis. The target y = (cos 7 theta, sin 7 theta) is the same for all 7 equivalent poses (theta shifts by 2 pi / 7). A
ridge regression from features to y, 5-fold cross-validated, gives the held-out R^2 (1 = the turn is linearly
readable, 0 = nothing). Features, all computed by the trained model on CPU:
  tokens  the 368 x 64 scalar features of the image tokens that the GATr receives (flattened);
  mean    their mean over the tokens (64);
  map     the ResNet feature map after global average pooling (2048).
Control target: the axis direction a (3 numbers), which a model at the guessing level has learned.
R^2 is also reported per bin of the tilt (angle between the axis and the optical axis), from the held-out predictions.

env: COSDIR_SNAPSHOT (pose3d/ of the run's patched snapshot), COSDIR_CKPT (checkpoint with a "model" state dict),
     IMGDIR (directory with <split>/images/pyr7_<i>.png and <split>/rotations.npz), SPLITS ("test:1000,train:2000"),
     EXTRA_ARGS ("--dircloud_nside 8 --dircloud_grid 14"), N (sides, 7), OUT (.json)."""
import os, sys, json, math
os.environ.setdefault("COSDIR_DATA", "/nonexistent")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from PIL import Image
import lib

N = int(os.environ.get("N", 7)); D = os.environ["IMGDIR"]
torch.set_num_threads(int(os.environ.get("THREADS", 8)))

def twist(R):
    """theta of R = S Rz(theta), S the shortest rotation e3 -> R e3; also the axis a and its tilt (deg)."""
    a = R[:, :, 2]
    th, tilt = [], []
    for Ri, ai in zip(R, a):
        z = np.array([0., 0., 1.]); v = np.cross(z, ai); s = np.linalg.norm(v); c = ai @ z
        if s < 1e-8:
            S = np.eye(3) if c > 0 else np.diag([1., -1., -1.])
        else:
            k = v / s; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            S = np.eye(3) + s * K + (1 - c) * K @ K
        T = S.T @ Ri
        th.append(math.atan2(T[1, 0], T[0, 0])); tilt.append(math.degrees(math.acos(np.clip(c, -1, 1))))
    return np.array(th), a, np.array(tilt)

def ridge_cv(X, Y, lams=(1e-2, 1e-1, 1, 10, 100, 1e3, 1e4), k=5, seed=0):
    """Held-out predictions of a ridge regression (standardised X), lambda chosen by inner CV on each training fold."""
    n = len(X); idx = np.random.default_rng(seed).permutation(n); folds = np.array_split(idx, k)
    pred = np.zeros_like(Y)
    def fit(Xa, Ya, lam):
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-6; Z = (Xa - mu) / sd; ym = Ya.mean(0)
        if Z.shape[1] > Z.shape[0]:   # dual form
            A = Z @ Z.T + lam * np.eye(len(Z)); W = Z.T @ np.linalg.solve(A, Ya - ym)
        else:
            W = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (Ya - ym))
        return lambda Xb: ((Xb - mu) / sd) @ W + ym
    for f in range(k):
        te = folds[f]; tr = np.concatenate([folds[g] for g in range(k) if g != f])
        inner = np.array_split(tr, 3); best = None
        for lam in lams:
            err = 0
            for g in range(3):
                vi = inner[g]; ti = np.concatenate([inner[h] for h in range(3) if h != g])
                err += ((fit(X[ti], Y[ti], lam)(X[vi]) - Y[vi]) ** 2).sum()
            if best is None or err < best[0]: best = (err, lam)
        pred[te] = fit(X[tr], Y[tr], best[1])(X[te])
    return pred

def r2(Y, P):
    return float(1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())

if __name__ == "__main__":
    m = lib.load_model(fused=False)
    bb = dict(m.named_modules())[[k for k, _ in m.named_modules() if k.endswith("backbone")][0]]
    pooled = []
    bb.register_forward_hook(lambda mod, i, o: pooled.append((o[0] if isinstance(o, (tuple, list)) else o).mean((-1, -2)).detach()))
    imgs, rots = [], []
    for spec in os.environ.get("SPLITS", "test:1000").split(","):
        split, n = spec.split(":"); r = np.load(f"{D}/{split}/rotations.npz")[f"pyr{N}"][:int(n)]
        rots.append(r[:, 0]); imgs += [f"{D}/{split}/images/pyr{N}_{j:05d}.png" for j in range(len(r))]
    R = np.concatenate(rots).astype(np.float64)
    toks = []
    with torch.no_grad():
        for i in range(0, len(imgs), 25):
            x = torch.from_numpy(np.stack([np.asarray(Image.open(f).convert("RGB")) for f in imgs[i:i + 25]])).permute(0, 3, 1, 2).float() / 255
            c = m.condition(x); P = c.shape[1] - 1
            toks.append(c[:, :P, 8:72].reshape(len(x), -1).double().numpy())
            if i % 500 == 0: print(i, flush=True)
    Xt = np.concatenate(toks); Xm = Xt.reshape(len(Xt), P, 64).mean(1); Xb = torch.cat(pooled).double().numpy()
    th, a, tilt = twist(R)
    Y = np.stack([np.cos(N * th), np.sin(N * th)], 1)
    bins = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 180)]
    out = dict(n_images=len(R), n_tokens=P, results={})
    for name, X in (("tokens", Xt), ("mean", Xm), ("map", Xb)):
        Pt = ridge_cv(X, Y); Pa = ridge_cv(X, a)
        per = {f"{lo}-{hi}": (r2(Y[(tilt >= lo) & (tilt < hi)], Pt[(tilt >= lo) & (tilt < hi)]), int(((tilt >= lo) & (tilt < hi)).sum()))
               for lo, hi in bins}
        out["results"][name] = dict(turn_r2=r2(Y, Pt), axis_r2=r2(a, Pa), turn_r2_by_tilt=per)
        print(name, json.dumps(out["results"][name]), flush=True)
    json.dump(out, open(os.environ["OUT"], "w"), indent=1)
