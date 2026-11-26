"""Runner: GATED easy-to-hard curriculum on the 3D SYMSOL-CHILE 7-gon (requested 2026-11-26).

The widening curriculum (make_chile3d_curriculum.py) moved on before the easy stage was learned: the tilt limit stayed
under 30 deg for about 80k images, while the bottom view needs about 300k to break through its plateau. Here the tilt
limit T only widens when the model has learned the current range:
  * training pool: N poses R = Rz(phi) Ry(tilt) Rz(theta), tilt (angle between the pyramid's axis R e3 and the camera's
    +z; 0 = the bottom view) uniform on [0, 180] deg; validation pool: M poses of the same law (another seed), rendered
    in the kernel by the SYMSOL-CHILE renderer; both replace the generator's views;
  * only views with tilt <= T are drawn, for training and for the per-epoch validation (validation draws are fixed
    within a stage, so epochs of one stage are compared on the same views);
  * gate: after each epoch, if the validation median on the current range is below GATE deg (half the guessing level
    90/7 = 12.9 deg, i.e. the turn is resolved there), T grows by STEP deg; rank 0 decides and broadcasts;
  * at T = 180 deg views are drawn with weight sin(tilt), so the last stage is the uniform-SO(3) distribution, and the
    final evaluation (if T reaches 180) is on uniform-SO(3) draws from the validation pool.
env POSE3D_GATED="T0,STEP,GATE" (T0 can be raised for a resumed run). Model and recipe as make_chile3d_runner.py
(ImageNet ResNet-50, 14x14 map, s_h 100, constant lr), 2xT4.

usage: python make_chile3d_gated.py N M EPOCHS T0 STEP GATE OUT.ipynb   -> prints the run name
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_chile3d_runner import build

POOL_SRC = r'''
import sys, os, math, numpy as np, torch
from PIL import Image
sys.path.insert(0, '/kaggle/working/GA-Research/pose3d')
import pose3d.datasets.symsol_chile as sc
root, split, N, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
path = os.path.join(root, 'symsol_chile', split); os.makedirs(path + '/images', exist_ok=True)
g = torch.Generator().manual_seed(seed)
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
print('gated pool:', split, N, 'views, tilt uniform on [0, 180] deg', flush=True)
'''

GATED_PATCH = r'''

def apply_gated(root):
    """POSE3D_GATED="T0,STEP,GATE": tilt-limited draws in Symsol (train and test) + the gate in the training loop."""
    f = os.path.join(root, "pose3d", "datasets/symsol.py")
    edit(f, "        self.max_equivalents = max([MAX_EQUIVALENTS] + [len(r) for r in self.rots])\n",
         "        self.max_equivalents = max([MAX_EQUIVALENTS] + [len(r) for r in self.rots])\n"
         "        import os as _os\n"
         "        _c = _os.environ.get('POSE3D_GATED')\n"
         "        self.curriculum = None\n"
         "        if _c:\n"
         "            _t0, _step, _gate = (float(x) for x in _c.split(','))\n"
         "            _r0 = np.concatenate([rotations[n][:num_views][:, 0] for n in self.shapes])\n"
         "            _tilt = np.degrees(np.arccos(np.clip(_r0[:, 2, 2], -1, 1)))\n"
         "            self.curriculum = (_step, _gate, _tilt)\n"
         "            self.curriculum_T = torch.full((1,), _t0, dtype=torch.float64).share_memory_()\n"
         "            self._cur_cache = {}\n"
         "            print(f'gated curriculum ({\"train\" if train else \"test\"}): T0 {_t0:g} deg, step {_step:g}, gate {_gate:g} deg; '\n"
         "                  f'{len(_tilt)} views, tilts {_tilt.min():.1f}..{_tilt.max():.1f}', flush=True)\n")
    edit(f, "    def __getitem__(self, i):\n        rots = self.rots[i]\n",
         "    def _curriculum_index(self, i):\n"
         "        _step, _gate, tilt = self.curriculum\n"
         "        T = float(self.curriculum_T[0])\n"
         "        if T not in self._cur_cache:\n"
         "            idx = np.nonzero(tilt <= T + 1e-9)[0]\n"
         "            w = np.sin(np.radians(tilt[idx])) + 1e-12 if T >= 180.0 else np.ones(len(idx))\n"
         "            self._cur_cache = {T: (idx, np.cumsum(w / w.sum()))}\n"
         "        idx, cw = self._cur_cache[T]\n"
         "        if self.train:\n"
         "            self._draws = getattr(self, '_draws', 0) + 1\n"
         "            u = np.random.default_rng((int(T * 10) * 1000003 + i) * 7919 + self._draws).random()\n"
         "        else:   # fixed within a stage: the epochs of one stage are scored on the same views\n"
         "            u = np.random.default_rng(int(T * 10) * 1000003 + i).random()\n"
         "        return int(idx[min(int(np.searchsorted(cw, u)), len(idx) - 1)])\n\n"
         "    def __getitem__(self, i):\n"
         "        if getattr(self, 'curriculum', None) is not None:\n"
         "            i = self._curriculum_index(i)\n"
         "        rots = self.rots[i]\n")
    f = os.path.join(root, "pose3d", "engine/trainer.py")
    edit(f, "        if run is not None:\n            run.log(metrics)\n        scheduler.step()\n",
         "        _tr, _va = getattr(train_loader, 'dataset', None), getattr(val_loader, 'dataset', None)\n"
         "        if getattr(_tr, 'curriculum', None) is not None:\n"
         "            _step, _gate, _ = _tr.curriculum\n"
         "            _T = float(_tr.curriculum_T[0])\n"
         "            metrics['curriculum_tilt_limit'] = _T\n"
         "            _new = min(180.0, _T + _step) if (mre is not None and mre < _gate and _T < 180.0) else _T\n"
         "            import torch.distributed as _dist\n"
         "            if _dist.is_available() and _dist.is_initialized():\n"
         "                _t = torch.tensor([_new], dtype=torch.float64, device=cfg.device); _dist.broadcast(_t, 0); _new = float(_t.item())\n"
         "            if is_main():\n"
         "                print(f'GATE epoch {i + 1}: tilt limit {_T:g} deg, median {mre} -> ' + (f'widen to {_new:g} deg' if _new != _T else 'stay'), flush=True)\n"
         "            for _d in (_tr, _va):\n"
         "                if getattr(_d, 'curriculum', None) is not None:\n"
         "                    _d.curriculum_T[0] = _new\n"
         "        if run is not None:\n            run.log(metrics)\n        scheduler.step()\n")

apply_gated('/kaggle/working/GA-Research/pose3d')
print('patched: POSE3D_GATED', flush=True)
'''

def build_gated(n=10000, m=1000, epochs=80, t0=10, step=10, gate=6.4):
    nb, name = build(0, epochs, 690); cells = nb["cells"]
    def sub(old, new):
        hits = [c for c in cells if old in "".join(c["source"])]; assert len(hits) == 1, (old, len(hits))
        t = "".join(hits[0]["source"]); assert t.count(old) == 1, old; hits[0]["source"] = t.replace(old, new)
    sub("'--symsol_train_views', '4500'", f"'--symsol_train_views', '{n}'")
    sub("'--symsol_test_views', '1000'", f"'--symsol_test_views', '{m}'")
    # both pools from gated_pool.py instead of the generator (tilt uniform on [0, 180] deg)
    sub("if not os.path.exists(f'{DATA}/symsol_chile/train/rotations.npz'):\n",
        "if not os.path.exists(f'{DATA}/symsol_chile/train/rotations.npz'):\n"
        "    open('/kaggle/working/gated_pool.py', 'w').write(POOL_SRC)\n"
        f"    subprocess.run([sys.executable, '-u', '/kaggle/working/gated_pool.py', DATA, 'test', '{m}', '5151'], check=True)\n"
        f"    subprocess.run([sys.executable, '-u', '/kaggle/working/gated_pool.py', DATA, 'train', '{n}', '4242'], check=True)\n"
        "if False:\n")
    sub("os.environ['POSE3D_SYMSOL_COS_SCALE'] = ", f"os.environ['POSE3D_GATED'] = '{t0},{step},{gate}'\nos.environ['POSE3D_SYMSOL_COS_SCALE'] = ")
    t = "".join(cells[2]["source"]); k = "apply_chile3d('/kaggle/working/GA-Research/pose3d')"
    assert t.count(k) == 1; cells[2]["source"] = t.replace(k, k + GATED_PATCH)
    new = name.replace("chile3d_pyr7", f"chile3d_pyr7_gated_t{t0}s{step}_n{n}")
    sub(f"RUN_NAME = '{name}'", f"RUN_NAME = '{new}'\nPOOL_SRC = {POOL_SRC!r}")
    cells[0]["source"] = (f"# GATED CURRICULUM (2026-11-26, human-requested): 3D 7-gon; train pool {n}, validation pool {m} views with tilt "
                          f"uniform on [0, 180] deg; tilt limit starts at {t0} deg and widens by {step} deg whenever the validation "
                          f"median on the current range is below {gate} deg; at 180 deg uniform-SO(3) weights. {epochs} epochs. "
                          "Recipe otherwise unchanged (ImageNet ResNet-50). 2xT4.\n") + "".join(cells[0]["source"])
    return nb, new

if __name__ == "__main__":
    a = sys.argv[1:]
    nb, name = build_gated(int(a[0]), int(a[1]), int(a[2]), int(a[3]), int(a[4]), float(a[5]))
    json.dump(nb, open(a[6], "w")); print(name)
