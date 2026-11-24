"""Render the SYMSOL-CHILE 7-gonal pyramid at given poses, standard and with black edges (the "make the sides visible"
idea). Uses datasets/symsol_chile.py of the SYMSOL-CHILE branch (env GEN = the directory that holds that file):
same camera, light and shading; with EDGE_W > 0 every pixel whose hit point lies within EDGE_W (object units; the unit
ball is about 134 px wide, so 0.03 is about 2 px) of a second face plane is drawn black. All 14 edges (7 side edges, 7
base edges) are treated alike, so the 7-fold symmetry is kept.
env: GEN, ROTS (an .npz with key pyr7, (N, ...,3,3), index 0 used), OUT (directory), EDGE_W (0 = standard), N."""
import os, sys, math
sys.path.insert(0, os.environ["GEN"])
import numpy as np, torch
from PIL import Image
import symsol_chile as sc

def render_edges(faces, rots, rays, w):
    f = torch.as_tensor(faces, dtype=torch.float32); t = torch.tensor([0.0, 0.0, sc.CAM_DIST])
    a = torch.einsum("bij,fj->bfi", rots.float(), f[:, :3]); b = f[:, 3] + a @ t
    on_ball = (rays @ t) ** 2 - (sc.CAM_DIST ** 2 - 1) >= 0; v = rays[on_ball]
    av = torch.einsum("bfi,pi->bpf", a, v); tf = b[:, None, :] / av
    t_in, face = torch.where(av < 0, tf, torch.full_like(tf, -math.inf)).max(-1)
    t_out = torch.where(av > 0, tf, torch.full_like(tf, math.inf)).min(-1).values
    hit = (t_in <= t_out) & (t_in > 0)
    normal = torch.gather(a, 1, face[..., None].expand(-1, -1, 3)); to_eye = -v.expand_as(normal)
    lam = (normal * to_eye).sum(-1).clamp(min=0)
    grey = sc.AMBIENT + sc.ALBEDO * lam + sc.SPECULAR * lam ** sc.SHININESS
    if w > 0:
        x = t_in[..., None] * v                                                   # hit points, camera frame (B, P, 3)
        d = torch.einsum("bfi,bpi->bpf", a, x) - b[:, None, :]                    # signed distances to the planes (<= 0 inside)
        d2 = d.topk(2, dim=-1).values[..., 1]                                     # the second-closest plane
        grey = torch.where(d2 > -w, torch.zeros_like(grey), grey)
    out = torch.zeros(len(rots), rays.shape[0]); out[:, on_ball] = torch.where(hit, grey, torch.zeros_like(grey))
    m = sc.SIZE * sc.SUPERSAMPLE
    out = torch.nn.functional.avg_pool2d(out.view(-1, 1, m, m), sc.SUPERSAMPLE).view(-1, sc.SIZE, sc.SIZE)
    return (out.clamp(0, 1) * 255).round().to(torch.uint8)

if __name__ == "__main__":
    R = np.load(os.environ["ROTS"])["pyr7"]; R = R[:, 0] if R.ndim == 4 else R
    N = int(os.environ.get("N", len(R))); R = torch.from_numpy(R[:N].astype(np.float64))
    out = os.environ["OUT"]; os.makedirs(out + "/images", exist_ok=True); w = float(os.environ.get("EDGE_W", 0))
    faces = sc.pyramid_faces(7); rays = sc._rays("cpu"); torch.set_num_threads(4)
    for i in range(0, N, 8):
        imgs = render_edges(faces, R[i:i + 8], rays, w).numpy()
        for j, im in enumerate(imgs):
            Image.fromarray(np.repeat(im[..., None], 3, -1)).save(f"{out}/images/pyr7_{i + j:05d}.png")
    np.savez(out + "/rotations.npz", pyr7=R.numpy()[:, None].astype(np.float32)); print("rendered", N, "to", out)
