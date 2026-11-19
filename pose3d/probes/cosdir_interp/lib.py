"""Shared loader for the cosdir GATr interpretability probes."""
import os, sys, math, glob
# COSDIR_SNAPSHOT: the pose3d/ directory of the run's code snapshot (the Kaggle notebook output: it holds
# pose3d/models/cos_attention.py and the checkpoint). COSDIR_DATA: a directory with SYMSOL test images
# (<shape>_<index>.png) and rotations.npz anywhere below it.
SNAP = os.environ["COSDIR_SNAPSHOT"]
DATA = os.environ["COSDIR_DATA"]
sys.path.insert(0, SNAP)
os.environ["POSE3D_SYMSOL_COSMV"] = "1"; os.environ["POSE3D_SYMSOL_ATTENTION"] = "cosdir"
import numpy as np, torch
from PIL import Image
from pose3d.models.gatr_denoiser import _ensure_xformers_stub; _ensure_xformers_stub()
CKPT = os.environ.get("COSDIR_CKPT", SNAP + "/symsol1_10pct_gatr_spherecloud_cosdir_h8_coslr_adamw_l4x4.pth")
ARGS = ['--path_to_datasets', '/nonexistent', '--vector_field', 'gatr', '--cond_tokens', 'spherecloud', '--pose_tokens', 'frame_pin', '--condition_head', 'none', '--dircloud_scalars', '64', '--gatr_mv_channels', '8', '--gatr_s_channels', '32', '--no-gatr_zero_readout', '--encoder', 'resnet50', '--dataset', 'symsol', '--symsol_set', '1', '--no-use_warp', '--no-use_synth', '--no-ema', '--batch_size', '64', '--gatr_heads', '8', '--seed', '0']
SHAPES = ("tet", "cube", "icosa", "cone", "cyl")

def load_model(fused=False):
    from pose3d.config import parse_args
    from pose3d.models import build_model
    from clifford.algebra.cliffordalgebra import CliffordAlgebra
    make_algebra = lambda: CliffordAlgebra((1, 1, 1))
    old = sys.argv; sys.argv = ["x"] + ARGS + os.environ.get("EXTRA_ARGS", "").split()   # e.g. "--dircloud_nside 8 --dircloud_grid 14"
    try: cfg = parse_args()
    finally: sys.argv = old
    cfg.device = "cpu"
    model = build_model(cfg, make_algebra())
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    print("load:", model.load_state_dict(sd, strict=False))
    model.eval()
    for a in model.vector_field.angular_attention: a.fused = fused
    return model

def load_data(n=24):
    root = glob.glob(DATA + "/**/rotations.npz", recursive=True)[0]
    rots = np.load(root)
    imgs, R, cls, eq = [], [], [], []
    for c, name in enumerate(SHAPES):
        for i in range(n):
            f = glob.glob(DATA + f"/**/{name}_{i:05d}.png", recursive=True)
            if not f: continue
            imgs.append(np.asarray(Image.open(f[0]).convert("RGB")))
            eq.append(torch.from_numpy(rots[name][i].astype(np.float32))); cls.append(c)
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255
    return x, eq, torch.tensor(cls)
