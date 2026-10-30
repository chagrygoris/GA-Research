"""SYMSOL-CHILE: SYMSOL-style renders of regular n-gonal pyramids, up to the chiliagon (n = 1000).

Why (reports/symmetry_spectrum.md): an n-gonal pyramid looks the same after a turn of 360/n degrees about its
axis, so its symmetry group is the cyclic group C_n and its pose density needs the Wigner-D degree l = n to be
told apart from a cone's. A model that only reaches degrees below n gets the axis right and guesses the turn, an
error spread evenly over [0, 180/n] degrees: up to 25.7 deg for pyr7, 18 deg for pyr10, 0.18 deg for pyr1000.

Shapes (SHAPES): "pyr<n>", a pyramid with a regular n-gon base. Proportions as the SYMSOL cone of
datasets/symsol_points.py: apex (0, 0, 1), base vertices at radius 1 in the plane z = -1 (the k-th at angle 2 pi k / n),
all scaled by 1 / sqrt 2 into the unit ball. Its rotations are the n turns about e3 (pyramid_group); equivalent
poses of a view at pose R are R g for those g, as in SYMSOL.

Rendering, in this repository's camera frame (datasets/symsol.py: e1 image columns, e2 rows down, e3 the optical
axis; SYMSOL's own poses are already in it): the shape is placed at x_cam = R x + (0, 0, CAM_DIST) and seen by a
pinhole camera at the origin. Every pixel is a ray o + t v; since a pyramid is convex (the points with
<a_f, x> <= b_f for each face f), the ray enters it at t_in = max of (b_f - <a_f, o>) / <a_f, v> over the faces with
<a_f, v> < 0 and leaves at t_out = min over the faces with <a_f, v> > 0; it hits iff t_in <= t_out, and the face
attaining t_in gives the normal. Shading as SYMSOL's renders (untextured grey solid on black): a point light at the
camera, ambient + Lambert + Blinn-Phong specular. 2 x 2 supersampling, then 224 x 224 RGB, uint8.

Output, the layout of the official SYMSOL zip, so datasets/symsol.Symsol reads it (--symsol_set 5):
    <root>/symsol_chile/{train,test}/rotations.npz      {shape: (n_views, n, 3, 3)}, index 0 = rendered pose
    <root>/symsol_chile/{train,test}/images/<shape>_<index:05d>.png
Poses are uniform on SO(3), seeded per split and shape, so a run regenerates the same data.

usage: python -m pose3d.datasets.symsol_chile --root DIR [--train 45000] [--test 5000] [--shapes pyr7,pyr10,pyr1000]
"""

import argparse
import math
import os
import time

import numpy as np
import torch
from PIL import Image

SHAPES = ("pyr7", "pyr10", "pyr1000")
SIZE = 224
SUPERSAMPLE = 2
CAM_DIST = 4.0
HALF_FOV = math.radians(23.0)      # the unit ball spans about 0.6 of the image width, as SYMSOL's shapes do
ALBEDO, AMBIENT, SPECULAR, SHININESS = 0.85, 0.08, 0.35, 24.0


def sides(shape: str) -> int:
    if not shape.startswith("pyr") or not shape[3:].isdigit() or int(shape[3:]) < 3:
        raise ValueError(f"SYMSOL-CHILE shapes are 'pyr<n>' with n >= 3, got {shape!r}")
    return int(shape[3:])


def pyramid_group(n: int) -> np.ndarray:
    """C_n: the n turns about e3 by 2 pi k / n, (n, 3, 3) float64."""
    a = 2 * np.pi * np.arange(n) / n
    G = np.zeros((n, 3, 3))
    G[:, 0, 0], G[:, 0, 1], G[:, 1, 0], G[:, 1, 1], G[:, 2, 2] = np.cos(a), -np.sin(a), np.sin(a), np.cos(a), 1
    return G


def pyramid_faces(n: int) -> np.ndarray:
    """Half-spaces <a_f, x> <= b_f of the pyramid, rows (a_f, b_f) with unit a_f, (n + 1, 4): n sides, then the base."""
    s = 1 / math.sqrt(2)
    apex = np.array([0.0, 0.0, 1.0]) * s
    ang = 2 * np.pi * np.arange(n) / n
    base = np.stack([np.cos(ang), np.sin(ang), -np.ones(n)], -1) * s
    a = np.cross(base - apex, np.roll(base, -1, 0) - apex)          # counter-clockwise base: outward normals
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = a @ apex
    return np.concatenate([np.concatenate([a, b[:, None]], 1), [[0.0, 0.0, -1.0, s]]])


def _rays(device):
    m = SIZE * SUPERSAMPLE
    c = (torch.arange(m, device=device, dtype=torch.float32) + 0.5) / m * 2 - 1      # pixel centres in [-1, 1]
    y, x = torch.meshgrid(c, c, indexing="ij")                                       # rows down = +e2, columns = +e1
    v = torch.stack([x * math.tan(HALF_FOV), y * math.tan(HALF_FOV), torch.ones_like(x)], -1).reshape(-1, 3)
    return v / v.norm(dim=-1, keepdim=True)


def render(faces: np.ndarray, rots: torch.Tensor, rays: torch.Tensor = None) -> torch.Tensor:
    """Images (B, 224, 224) uint8 grey of the convex solid `faces` (object frame) at the poses rots (B, 3, 3)."""
    device = rots.device
    rays = _rays(device) if rays is None else rays
    f = torch.as_tensor(faces, dtype=torch.float32, device=device)
    t = torch.tensor([0.0, 0.0, CAM_DIST], device=device)
    # faces in the camera frame: <R a, x - t> <= b, i.e. normal R a, offset b + <R a, t>; the ray origin is 0
    a = torch.einsum("bij,fj->bfi", rots.float(), f[:, :3])                         # (B, F, 3)
    b = f[:, 3] + a @ t                                                               # (B, F)
    on_ball = (rays @ t) ** 2 - (CAM_DIST ** 2 - 1) >= 0                             # only rays that meet the unit ball
    v = rays[on_ball]                                                                 # (P, 3)
    av = torch.einsum("bfi,pi->bpf", a, v)                                            # (B, P, F)
    tf = b[:, None, :] / av
    t_in, face = torch.where(av < 0, tf, torch.full_like(tf, -math.inf)).max(-1)
    t_out = torch.where(av > 0, tf, torch.full_like(tf, math.inf)).min(-1).values
    hit = (t_in <= t_out) & (t_in > 0)
    normal = torch.gather(a, 1, face[..., None].expand(-1, -1, 3))                   # (B, P, 3), outward
    to_eye = -v.expand_as(normal)                                                     # light and eye at the origin
    lam = (normal * to_eye).sum(-1).clamp(min=0)                                      # half-vector = to_eye
    grey = AMBIENT + ALBEDO * lam + SPECULAR * lam ** SHININESS
    out = torch.zeros(len(rots), rays.shape[0], device=device)
    out[:, on_ball] = torch.where(hit, grey, torch.zeros_like(grey))
    m = SIZE * SUPERSAMPLE
    out = torch.nn.functional.avg_pool2d(out.view(-1, 1, m, m), SUPERSAMPLE).view(-1, SIZE, SIZE)
    return (out.clamp(0, 1) * 255).round().to(torch.uint8)


def random_rotations(n: int, generator: torch.Generator) -> torch.Tensor:
    """n rotations uniform on SO(3) (unit quaternions from a Gaussian), (n, 3, 3) float64."""
    q = torch.randn(n, 4, generator=generator, dtype=torch.float64)
    w, x, y, z = (q / q.norm(dim=1, keepdim=True)).unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).view(n, 3, 3)


def generate(root: str, split: str, views: int, shapes=SHAPES, batch: int = 16, device: str = "cpu", log=print):
    """Render `views` poses of every shape into <root>/symsol_chile/<split>/, writing rotations.npz last."""
    path = os.path.join(root, "symsol_chile", split)
    os.makedirs(os.path.join(path, "images"), exist_ok=True)
    rays = _rays(device)
    rotations = {}
    for k, shape in enumerate(shapes):
        n = sides(shape)
        gen = torch.Generator().manual_seed(1000 * (split == "test") + k)
        R = random_rotations(views, gen)
        rotations[shape] = np.einsum("vij,gjk->vgik", R.numpy(), pyramid_group(n)).astype(np.float32)
        faces, t0 = pyramid_faces(n), time.time()
        step = max(1, min(batch, int(3e7 // (len(rays) * 0.3 * len(faces)))))   # rays x faces per batch: memory
        for i in range(0, views, step):
            imgs = render(faces, R[i:i + step].to(device), rays).cpu().numpy()
            for j, img in enumerate(imgs):
                Image.fromarray(np.repeat(img[..., None], 3, -1)).save(
                    os.path.join(path, "images", f"{shape}_{i + j:05d}.png"))
        log(f"{split} {shape}: {views} views in {time.time() - t0:.0f} s")
    np.savez(os.path.join(path, "rotations.npz"), **rotations)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="cfg.run.path_to_datasets: writes <root>/symsol_chile/")
    ap.add_argument("--train", type=int, default=45000, help="views per shape (SYMSOL: 45,000)")
    ap.add_argument("--test", type=int, default=5000, help="views per shape (SYMSOL: 5,000)")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    shapes = tuple(args.shapes.split(","))
    for split, views in (("test", args.test), ("train", args.train)):
        if views:
            generate(args.root, split, views, shapes, args.batch, args.device)


if __name__ == "__main__":
    main()
