"""Runner: easy-to-hard curriculum on the 3D SYMSOL-CHILE 7-gon, from bottom views to all tilts (requested 2026-11-25).

Same model and recipe as make_chile3d_runner.py (ImageNet ResNet-50, 14x14 map, 368 sphere points, start s_h 100,
constant lr), 4xL4. Changed:
  * training pool: N poses R = Rz(phi) Ry(tilt) Rz(theta) with the tilt (angle between the pyramid's axis R e3 and the
    camera's +z; tilt 0 = the bottom view) uniform on [0, 180] deg, phi and theta uniform, so near-bottom views are
    plentiful (a pool uniform on SO(3) has 0.8 % of its views within 10 deg). Rendered in the kernel with the
    SYMSOL-CHILE renderer (curriculum_pool.py below); the 1,000 test views stay uniform on SO(3) (the generator's seeds);
  * curriculum (POSE3D_CURRICULUM="T0,FRAC,EPOCHS"): in epoch e only pool views with tilt <= T(e) are drawn, T(e) rising
    linearly from T0 to 180 deg at epoch FRAC x EPOCHS; from then on views are drawn with weight sin(tilt), so the
    training distribution equals the uniform-SO(3) test distribution. The draw replaces the sampler's index inside
    Symsol.__getitem__; the epoch reaches the (persistent) DataLoader workers through a shared-memory tensor set by the
    training loop.

usage: python make_chile3d_curriculum.py N STEPS T0 FRAC OUT.ipynb   -> prints the run name
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_chile3d_runner import build

POOL_SRC = r'''
import sys, os, math, numpy as np, torch
from PIL import Image
sys.path.insert(0, '/kaggle/working/GA-Research/pose3d')
import pose3d.datasets.symsol_chile as sc
root, N = sys.argv[1], int(sys.argv[2])
path = os.path.join(root, 'symsol_chile', 'train'); os.makedirs(path + '/images', exist_ok=True)
g = torch.Generator().manual_seed(4242)
tilt = math.pi * torch.rand(N, generator=g, dtype=torch.float64)
phi = 2 * math.pi * torch.rand(N, generator=g, dtype=torch.float64)
th = 2 * math.pi * torch.rand(N, generator=g, dtype=torch.float64)
def rz(a):
    c, s, o, z = torch.cos(a), torch.sin(a), torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([c, -s, z, s, c, z, z, z, o], -1).view(-1, 3, 3)
def ry(a):
    c, s, o, z = torch.cos(a), torch.sin(a), torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([c, z, s, z, o, z, -s, z, c], -1).view(-1, 3, 3)
R = rz(phi) @ ry(tilt) @ rz(th)
faces, rays = sc.pyramid_faces(7), sc._rays('cuda')
for i in range(0, N, 32):
    imgs = sc.render(faces, R[i:i + 32].cuda(), rays).cpu().numpy()
    for j, im in enumerate(imgs):
        Image.fromarray(np.repeat(im[..., None], 3, -1)).save(f'{path}/images/pyr7_{i + j:05d}.png')
np.savez(path + '/rotations.npz', pyr7=np.einsum('vij,gjk->vgik', R.numpy(), sc.pyramid_group(7)).astype(np.float32))
print('curriculum pool:', N, 'views, tilt uniform on [0, 180] deg', flush=True)
'''

CURRIC_PATCH = r'''

def apply_curriculum(root):
    """POSE3D_CURRICULUM="T0,FRAC,EPOCHS": tilt-limited draws in Symsol (train) + the epoch set by the training loop."""
    f = os.path.join(root, "pose3d", "datasets/symsol.py")
    edit(f, "        self.max_equivalents = max([MAX_EQUIVALENTS] + [len(r) for r in self.rots])\n",
         "        self.max_equivalents = max([MAX_EQUIVALENTS] + [len(r) for r in self.rots])\n"
         "        import os as _os\n"
         "        _c = _os.environ.get('POSE3D_CURRICULUM') if train else None\n"
         "        self.curriculum = None\n"
         "        if _c:\n"
         "            _t0, _frac, _ne = (float(x) for x in _c.split(','))\n"
         "            _r0 = np.concatenate([rotations[n][:num_views][:, 0] for n in self.shapes])\n"
         "            _tilt = np.degrees(np.arccos(np.clip(_r0[:, 2, 2], -1, 1)))\n"
         "            self.curriculum = (_t0, _frac * _ne, _tilt)\n"
         "            self.curriculum_epoch = torch.zeros(1, dtype=torch.long).share_memory_()\n"
         "            self._cur_cache = {}\n"
         "            print(f'curriculum: tilt <= {_t0:g} deg at epoch 0, 180 deg at epoch {_frac * _ne:.0f}, then sin(tilt) weights; '\n"
         "                  f'pool tilts {_tilt.min():.1f}..{_tilt.max():.1f} deg', flush=True)\n")
    edit(f, "    def __getitem__(self, i):\n        rots = self.rots[i]\n",
         "    def _curriculum_index(self, i):\n"
         "        t0, e_full, tilt = self.curriculum\n"
         "        e = int(self.curriculum_epoch[0])\n"
         "        if e not in self._cur_cache:\n"
         "            T = t0 + (180.0 - t0) * min(1.0, e / max(e_full, 1.0))\n"
         "            idx = np.nonzero(tilt <= T + 1e-9)[0]\n"
         "            w = np.sin(np.radians(tilt[idx])) + 1e-12 if e >= e_full else np.ones(len(idx))\n"
         "            self._cur_cache = {e: (idx, np.cumsum(w / w.sum()))}\n"
         "        idx, cw = self._cur_cache[e]\n"
         "        u = np.random.default_rng(e * 1000003 + i).random()\n"
         "        return int(idx[min(int(np.searchsorted(cw, u)), len(idx) - 1)])\n\n"
         "    def __getitem__(self, i):\n"
         "        if self.train and getattr(self, 'curriculum', None) is not None:\n"
         "            i = self._curriculum_index(i)\n"
         "        rots = self.rots[i]\n")
    f = os.path.join(root, "pose3d", "engine/trainer.py")
    edit(f, "        if hasattr(sampler, \"set_epoch\"):\n            sampler.set_epoch(i)\n",
         "        if hasattr(sampler, \"set_epoch\"):\n            sampler.set_epoch(i)\n"
         "        _ds = getattr(train_loader, 'dataset', None)\n"
         "        if getattr(_ds, 'curriculum', None) is not None:\n"
         "            _ds.curriculum_epoch[0] = i\n"
         "            if is_main():\n"
         "                _t0, _ef, _ = _ds.curriculum\n"
         "                print(f'curriculum epoch {i + 1}: tilt <= {_t0 + (180 - _t0) * min(1.0, i / max(_ef, 1.0)):.0f} deg', flush=True)\n")

apply_curriculum('/kaggle/working/GA-Research/pose3d')
print('patched: POSE3D_CURRICULUM', flush=True)
'''

def build_curriculum(n, steps=20000, t0=15, frac=0.5):
    ep = math.ceil(steps / (n / 64)); nb, name = build(0, ep, 690); cells = nb["cells"]
    def sub(old, new):
        hits = [c for c in cells if old in "".join(c["source"])]; assert len(hits) == 1, (old, len(hits))
        t = "".join(hits[0]["source"]); assert t.count(old) == 1, old; hits[0]["source"] = t.replace(old, new)
    sub("'--symsol_train_views', '4500'", f"'--symsol_train_views', '{n}'")
    sub("'--num_gpus', '2'", "'--num_gpus', '4'")
    # test views from the generator (uniform SO(3), its seeds); training pool from curriculum_pool.py
    sub("'--train', '4500'", "'--train', '0'")
    sub("'--view', 'random', '--shapes', 'pyr7'], check=True)\n",
        "'--view', 'random', '--shapes', 'pyr7'], check=True)\n"
        f"    open('/kaggle/working/curriculum_pool.py', 'w').write(POOL_SRC)\n"
        f"    subprocess.run([sys.executable, '-u', '/kaggle/working/curriculum_pool.py', DATA, '{n}'], check=True)\n")
    sub("os.environ['POSE3D_SYMSOL_COS_SCALE'] = ", f"os.environ['POSE3D_CURRICULUM'] = '{t0},{frac},{ep}'\nos.environ['POSE3D_SYMSOL_COS_SCALE'] = ")
    t = "".join(cells[2]["source"]); k = "apply_chile3d('/kaggle/working/GA-Research/pose3d')"
    assert t.count(k) == 1; cells[2]["source"] = t.replace(k, k + CURRIC_PATCH)
    new = name.replace("chile3d_pyr7", f"chile3d_pyr7_curric_t{t0}_n{n}")
    sub(f"RUN_NAME = '{name}'", f"RUN_NAME = '{new}'\nPOOL_SRC = {POOL_SRC!r}")
    cells[0]["source"] = (f"# CURRICULUM (2026-11-25, human-requested): 3D 7-gon, training pool of {n} views with tilt uniform on [0, 180] deg; "
                          f"tilt <= {t0} deg at first, rising linearly to 180 deg by epoch {frac * ep:.0f} of {ep}, then uniform-SO(3) "
                          f"weights; ~{steps} steps; test views uniform on SO(3). Recipe otherwise unchanged (ImageNet ResNet-50). "
                          "4xL4.\n") + "".join(cells[0]["source"])
    return nb, new

if __name__ == "__main__":
    a = sys.argv[1:]
    nb, name = build_curriculum(int(a[0]), int(a[1]), int(a[2]), float(a[3]))
    json.dump(nb, open(a[4], "w")); print(name)
