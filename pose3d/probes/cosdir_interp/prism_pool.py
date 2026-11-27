"""Render an n-gonal prism ("n-gonal cylinder") with the SYMSOL-CHILE renderer (same camera, light and shading).

The prism: a regular n-gon cross-section with circumradius 1/sqrt 2, vertices at angles 2 pi k / n about e3 (vertex 0
on e1), half-height 1/sqrt 2, so it fits in the unit ball like the SYMSOL-CHILE pyramids. Faces as half-spaces
<a_f, x> <= b_f: n sides (outward normals at angles 2 pi (k + 1/2) / n, offset r cos(pi / n)), then the top and the
bottom. Its rotation group is the dihedral group D_n (2n elements): the turns Rz(2 pi k / n) and the same composed
with the half-turn Rx(pi) about e1, which swaps the two ends.
Poses uniform on SO(3) (as SYMSOL-CHILE's --view random). Output in the SYMSOL layout:
<root>/<dir>/<split>/images/<name>_<i>.png and rotations.npz {name: (views, 2n, 3, 3)}, index 0 = the rendered pose.
usage: python prism_pool.py GEN_DIR ROOT DIR SPLIT VIEWS SEED [N] [DEVICE]   (GEN_DIR holds symsol_chile.py)"""
import sys, os, math
import numpy as np, torch
from PIL import Image

def prism_faces(n):
    s = 1 / math.sqrt(2); r, h = s, s
    ang = 2 * np.pi * (np.arange(n) + 0.5) / n
    sides = np.stack([np.cos(ang), np.sin(ang), np.zeros(n), np.full(n, r * math.cos(math.pi / n))], 1)
    return np.concatenate([sides, [[0, 0, 1, h], [0, 0, -1, h]]])

def dihedral_group(n):
    a = 2 * np.pi * np.arange(n) / n
    Z = np.zeros((n, 3, 3)); Z[:, 0, 0], Z[:, 0, 1], Z[:, 1, 0], Z[:, 1, 1], Z[:, 2, 2] = np.cos(a), -np.sin(a), np.sin(a), np.cos(a), 1
    X = np.diag([1.0, -1.0, -1.0])                        # Rx(pi)
    return np.concatenate([Z, Z @ X])

if __name__ == "__main__":
    sys.path.insert(0, sys.argv[1]); import symsol_chile as sc
    gen, root, d, split, views, seed = sys.argv[1:7]; views, seed = int(views), int(seed)
    n = int(sys.argv[7]) if len(sys.argv) > 7 else 7; dev = sys.argv[8] if len(sys.argv) > 8 else ("cuda" if torch.cuda.is_available() else "cpu")
    name = f"prism{n}"; path = os.path.join(root, d, split); os.makedirs(path + "/images", exist_ok=True)
    R = sc.random_rotations(views, torch.Generator().manual_seed(seed))
    faces, rays = prism_faces(n), sc._rays(dev)
    for i in range(0, views, 16):
        imgs = sc.render(faces, R[i:i + 16].to(dev), rays).cpu().numpy()
        for j, im in enumerate(imgs):
            Image.fromarray(np.repeat(im[..., None], 3, -1)).save(f"{path}/images/{name}_{i + j:05d}.png")
    G = dihedral_group(n)
    np.savez(path + "/rotations.npz", **{name: np.einsum("vij,gjk->vgik", R.numpy(), G).astype(np.float32)})
    print(f"prism pool: {name} {split} {views} views, D{n} ({len(G)} equivalents)", flush=True)
