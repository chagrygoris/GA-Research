"""Learned (softmax on the training grid) vs target (7 correct poses) density of Image2Sphere L = 10 on the 7-gon, as expected
Wigner-D coefficients, analysed along the ring of turns. Paths as in i2s_degree_probe.py. Run from pose3d/."""
import json, math, sys
import numpy as np, torch
sys.path.insert(0, '.')
from pose3d.config import parse_args
from pose3d.models import build_model
from clifford.algebra.cliffordalgebra import CliffordAlgebra
from e3nn import o3
from image2sphere import so3_utils
from PIL import Image
from pose3d.datasets.symsol_chile import pyramid_group
args = ['--path_to_datasets=/x', '--dataset', 'symsol', '--symsol_set', '5', '--model', 'i2s_real', '--lmax', '10',
        '--encoder', 'resnet50', '--i2s_eval_rec_level', '5', '--no-pretrained_backbone', '--no-ddp']
cfg = parse_args(args); cfg.device = torch.device('cpu')
model = build_model(cfg, CliffordAlgebra((1.0, 1.0, 1.0)))
ck = torch.load('/home/user/optimist1938/i2s_l10_out/GA-Research/pose3d/chile_10pct_i2s_l10_adamw.pth', map_location='cpu', weights_only=False)
model.load_state_dict(ck['model']); model.eval()
L = 10; V = 200
root = '/home/user/optimist1938/chile7/symsol_chile/test'
rots = np.load(f'{root}/rotations.npz')['pyr7']
imgs = torch.stack([torch.from_numpy(np.array(Image.open(f'{root}/images/pyr7_{i:05d}.png').convert('RGB'))).permute(2, 0, 1).float() / 255 for i in range(V)])
with torch.no_grad():
    psi = torch.cat([model.forward(imgs[i:i + 25]).squeeze(1).float() for i in range(0, V, 25)])
offs = np.cumsum([0] + [(2 * l + 1) ** 2 for l in range(L + 1)])
# the training grid (HEALPix level 3): the model's density is softmax of its logits there
a, b, g = so3_utils.so3_healpix_grid(rec_level=3)
Wg = so3_utils.flat_wigner(L, a, b, g)                                   # (Q, M)
print('grid', Wg.shape, flush=True)
p = torch.softmax(psi @ Wg.T, dim=1)                                    # (V, Q) learned density on the grid
c_model = (p @ Wg).double()                                             # (V, M) E_model[D(R)]
G = torch.from_numpy(pyramid_group(7))
c_true = []
for v in range(V):
    E = torch.from_numpy(rots[v, 0]).double() @ G.transpose(1, 2).transpose(1, 2)   # R0 g, g in C7
    E = torch.einsum('ij,gjk->gik', torch.from_numpy(rots[v, 0]).double(), G)
    ae, be, ge = o3.matrix_to_angles(E)
    c_true.append(so3_utils.flat_wigner(L, ae.float(), be.float(), ge.float()).double().mean(0))
c_true = torch.stack(c_true)
K = 360; phi = torch.arange(K, dtype=torch.float64) * 2 * math.pi / K
Rz = torch.zeros(K, 3, 3, dtype=torch.float64)
Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = phi.cos(), -phi.sin(), phi.sin(), phi.cos(), 1
res = {}
for name, C in (('target', c_true), ('learned', c_model)):
    m0 = np.zeros((V, L + 1)); m7 = np.zeros((V, L + 1)); oth = np.zeros((V, L + 1)); prof = []
    for v in range(V):
        R = torch.from_numpy(rots[v, 0]).double() @ Rz
        ar, br, gr = o3.matrix_to_angles(R)
        W = so3_utils.flat_wigner(L, ar.float(), br.float(), gr.float()).double()
        f_l = torch.stack([W[:, offs[l]:offs[l + 1]] @ C[v, offs[l]:offs[l + 1]] for l in range(L + 1)], 1)   # (K, L+1)
        F = torch.fft.rfft(f_l, dim=0)[:12] / K
        m0[v] = F[0].real.numpy()
        m7[v] = (2 * (f_l * torch.cos(7 * phi)[:, None]).mean(0)).numpy()                                    # aligned 7-fold part
        others = [m for m in range(1, 12) if m != 7]
        oth[v] = (2 * F[others].abs()).mean(0).numpy()
        if v < 3: prof.append(f_l.sum(1).numpy().tolist())
    res[name] = {'m0': m0.mean(0).tolist(), 'm7': m7.mean(0).tolist(), 'm7_sd': m7.std(0).tolist(), 'other': oth.mean(0).tolist(), 'profiles': prof}
    print(name, 'm0', np.round(m0.mean(0), 3).tolist()); print(name, 'm7', np.round(m7.mean(0), 3).tolist()); print(name, 'other', np.round(oth.mean(0), 3).tolist())
json.dump(res, open('/home/user/optimist1938/i2s_density_probe.json', 'w'))
