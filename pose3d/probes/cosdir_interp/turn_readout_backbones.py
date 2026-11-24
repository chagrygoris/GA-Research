"""Is the turn of the 3D SYMSOL-CHILE 7-gon (about its own axis) in the image, and in which backbone's features?

Same target and read-out as turn_readout.py: theta from the swing-twist split of the pose, y = (cos 7 theta,
sin 7 theta), ridge regression, 5-fold cross-validated, held-out R^2 (1 = linearly readable, 0 = nothing); also R^2
for the axis direction a (3 numbers) as a control, and R^2 of the turn per tilt bin. Plus a non-linear read-out: k
nearest neighbours (k = 10, cosine distance on standardised features), so that "not linear" and "not there" can be told
apart. Feature sets (env FEATS, comma-separated):
  pixels      the image at 56 x 56, grey (3,136 numbers);
  imagenet50  torchvision ResNet-50 with ImageNet weights (the start of every SYMSOL run): the last stage, average-pooled
              globally (2048) and to 4 x 4 (32,768);
  pascal101   the ResNet-101 of the Pascal3D+ cosine-GATr run (reports/pascal_cosdir.md; EMA weights, last stage dilated
              as trained): the same two poolings.
env: IMGDIR (<split>/images/pyr7_<i>.png, <split>/rotations.npz), SPLITS ("test:1000,train:879"), FEATS, R50 (resnet50
     ImageNet .pth), PASCAL_CKPT (the run's resume checkpoint), OUT (.json)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torchvision
from PIL import Image
from turn_readout import twist, ridge_cv, r2

torch.set_num_threads(int(os.environ.get("THREADS", 4)))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def knn_pred(X, Y, k=10, folds=5, seed=0):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-6); Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    idx = np.random.default_rng(seed).permutation(len(X)); parts = np.array_split(idx, folds); P = np.zeros_like(Y)
    for f in range(folds):
        te = parts[f]; tr = np.concatenate([parts[g] for g in range(folds) if g != f])
        s = Z[te] @ Z[tr].T; nn = np.argsort(-s, 1)[:, :k]; P[te] = Y[tr][nn].mean(1)
    return P

def backbone(name):
    if name == "imagenet50":
        m = torchvision.models.resnet50(); m.load_state_dict(torch.load(os.environ["R50"], map_location="cpu"))
    else:
        m = torchvision.models.resnet101(replace_stride_with_dilation=[False, False, True])
        sd = torch.load(os.environ["PASCAL_CKPT"], map_location="cpu", weights_only=False)["ema"]
        pre = "module.adapter.backbone.encoder.layers."
        seq = torch.nn.Sequential(*list(m.children())[:8])
        res = seq.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}, strict=True); print(name, res)
        return seq.eval()
    return torch.nn.Sequential(*list(m.children())[:8]).eval()

def features(name, files):
    if name == "pixels":
        return {"pixels": np.stack([np.asarray(Image.open(f).convert("L").resize((56, 56), Image.BILINEAR), dtype=np.float64).ravel()
                                    for f in files])}
    net = backbone(name); g, s4 = [], []
    with torch.no_grad():
        for i in range(0, len(files), 16):
            x = torch.from_numpy(np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files[i:i + 16]])).permute(0, 3, 1, 2).float() / 255
            o = net((x - MEAN) / STD)
            g.append(o.mean((-1, -2)).double().numpy()); s4.append(torch.nn.functional.adaptive_avg_pool2d(o, 4).flatten(1).double().numpy())
            if i % 400 == 0: print(name, i, flush=True)
    return {f"{name}_global": np.concatenate(g), f"{name}_4x4": np.concatenate(s4)}

if __name__ == "__main__":
    D = os.environ["IMGDIR"]; files, rots = [], []
    for spec in os.environ.get("SPLITS", "test:1000").split(","):
        split, n = spec.split(":"); r = np.load(f"{D}/{split}/rotations.npz")["pyr7"][:int(n)]
        rots.append(r[:, 0]); files += [f"{D}/{split}/images/pyr7_{j:05d}.png" for j in range(len(r))]
    R = np.concatenate(rots).astype(np.float64); th, a, tilt = twist(R)
    Y = np.stack([np.cos(7 * th), np.sin(7 * th)], 1)
    bins = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 180)]
    out = dict(imgdir=D, n_images=len(R), results={})
    for name in os.environ["FEATS"].split(","):
        for fname, X in features(name, files).items():
            Pl = ridge_cv(X, Y); Pk = knn_pred(X, Y)
            per = {f"{lo}-{hi}": round(r2(Y[(tilt >= lo) & (tilt < hi)], Pl[(tilt >= lo) & (tilt < hi)]), 3) for lo, hi in bins}
            out["results"][fname] = dict(turn_r2_linear=r2(Y, Pl), turn_r2_knn=r2(Y, Pk), axis_r2_linear=r2(a, ridge_cv(X, a)),
                                         turn_r2_linear_by_tilt=per, dims=X.shape[1])
            print(fname, json.dumps(out["results"][fname]), flush=True)
            json.dump(out, open(os.environ["OUT"], "w"), indent=1)
