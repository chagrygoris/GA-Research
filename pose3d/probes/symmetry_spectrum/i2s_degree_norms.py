"""Learned vs target density of Image2Sphere L = 10 on the 7-gon: size of each degree's Wigner-D coefficient block.

For a density p on SO(3) the degree-l coefficients are c^l = E_{R~p}[D^l(R)] (a (2l+1) x (2l+1) block). Plotted:
|c^l|_F / |D^l(R)|_F, i.e. 1 for a single exact pose and 0 for no content at degree l. Target: equal mass on the 7
correct poses R0 g. Learned: softmax of the model's logits on its training grid (HEALPix level 3). Mean (and spread)
over test views. Paths as in i2s_degree_probe.py. Run from pose3d/.
"""
import json, sys
import numpy as np, torch
sys.path.insert(0, '.')
from pose3d.config import parse_args
from pose3d.models import build_model
from clifford.algebra.cliffordalgebra import CliffordAlgebra
from e3nn import o3
from image2sphere import so3_utils
from PIL import Image
from pose3d.datasets.symsol_chile import pyramid_group

cfg = parse_args(['--path_to_datasets=/x', '--dataset', 'symsol', '--symsol_set', '5', '--model', 'i2s_real', '--lmax', '10',
                  '--encoder', 'resnet50', '--i2s_eval_rec_level', '5', '--no-pretrained_backbone', '--no-ddp'])
cfg.device = torch.device('cpu')
model = build_model(cfg, CliffordAlgebra((1.0, 1.0, 1.0)))
model.load_state_dict(torch.load('/home/user/optimist1938/i2s_l10_out/GA-Research/pose3d/chile_10pct_i2s_l10_adamw.pth',
                                 map_location='cpu', weights_only=False)['model'])
model.eval()
L, V = 10, 200
root = '/home/user/optimist1938/chile7/symsol_chile/test'
rots = np.load(f'{root}/rotations.npz')['pyr7']
imgs = torch.stack([torch.from_numpy(np.array(Image.open(f'{root}/images/pyr7_{i:05d}.png').convert('RGB'))).permute(2, 0, 1).float() / 255
                    for i in range(V)])
with torch.no_grad():
    psi = torch.cat([model.forward(imgs[i:i + 25]).squeeze(1).float() for i in range(0, V, 25)])
offs = np.cumsum([0] + [(2 * l + 1) ** 2 for l in range(L + 1)])
a, b, g = so3_utils.so3_healpix_grid(rec_level=3)
Wg = so3_utils.flat_wigner(L, a, b, g)
c_learned = (torch.softmax(psi @ Wg.T, dim=1) @ Wg).double()                 # (V, M)
G = torch.from_numpy(pyramid_group(7))
c_target, c_single = [], []
for v in range(V):
    E = torch.einsum('ij,gjk->gik', torch.from_numpy(rots[v, 0]).double(), G)
    W = so3_utils.flat_wigner(L, *[x.float() for x in o3.matrix_to_angles(E)]).double()
    c_target.append(W.mean(0)); c_single.append(W[0])
c_target, c_single = torch.stack(c_target), torch.stack(c_single)
def norms(C):
    return torch.stack([C[:, offs[l]:offs[l + 1]].norm(dim=1) for l in range(L + 1)], 1)
ref = norms(c_single)                                                       # a single exact pose, per degree
out = {k: (norms(C) / ref).numpy() for k, C in (('target', c_target), ('learned', c_learned))}
res = {k: {'mean': v.mean(0).tolist(), 'p10': np.percentile(v, 10, 0).tolist(), 'p90': np.percentile(v, 90, 0).tolist()}
       for k, v in out.items()}
json.dump(res, open('/home/user/optimist1938/i2s_degree_norms.json', 'w'))
for k in res: print(k, np.round(res[k]['mean'], 3).tolist())
print('target formula sqrt(n_l/(2l+1)):', np.round([np.sqrt((1 if l < 7 else 3) / (2 * l + 1)) for l in range(L + 1)], 3).tolist())
