"""Per-degree contribution of a trained Image2Sphere (L = 10) logit along the ring of turns of the 7-gon (reports/symmetry_spectrum.md).

Paths are those of the 2026-10-31 run: the checkpoint of Kaggle kernel samson12345423/symsol-chile-i2s-l10-rtx and 200 rendered pyr7
test views (python -m pose3d.datasets.symsol_chile --root ... --test 200 --train 0 --shapes pyr7). Run from pose3d/."""
import json, math, sys
import numpy as np, torch
sys.path.insert(0, '.')
from pose3d.config import parse_args
from pose3d.models import build_model
from pose3d.datasets.symsol import Symsol
from clifford.algebra.cliffordalgebra import CliffordAlgebra
from e3nn import o3
from image2sphere import so3_utils

args = ['--path_to_datasets=/home/user/optimist1938/chile7', '--dataset', 'symsol', '--symsol_set', '5', '--model', 'i2s_real',
        '--lmax', '10', '--encoder', 'resnet50', '--i2s_eval_rec_level', '5', '--no-pretrained_backbone', '--no-ddp']
cfg = parse_args(args); cfg.device = torch.device('cpu')
model = build_model(cfg, CliffordAlgebra((1.0, 1.0, 1.0)))
ck = torch.load('/home/user/optimist1938/i2s_l10_out/GA-Research/pose3d/chile_10pct_i2s_l10_adamw.pth', map_location='cpu', weights_only=False)
r = model.load_state_dict(ck['model'], strict=False)
print('missing', r.missing_keys[:5], 'unexpected', r.unexpected_keys[:5], flush=True)
model.eval()

L = 10
test = Symsol('/home/user/optimist1938/chile7', train=False, set_number=5, num_views=None, workers=4) if False else None
import os
from PIL import Image
root = '/home/user/optimist1938/chile7/symsol_chile/test'
rots = np.load(f'{root}/rotations.npz')['pyr7']                      # (V, 7, 3, 3), index 0 = rendered pose
V = 200
imgs = torch.stack([torch.from_numpy(np.asarray(Image.open(f'{root}/images/pyr7_{i:05d}.png').convert('RGB'))).permute(2, 0, 1).float() / 255
                    for i in range(V)])
with torch.no_grad():
    psi = torch.cat([model.forward(imgs[i:i + 25]).squeeze(1).float() for i in range(0, V, 25)])   # (V, M)
sizes = [(2 * l + 1) ** 2 for l in range(L + 1)]; offs = np.cumsum([0] + sizes)
assert psi.shape[1] == offs[-1], psi.shape

K = 360                                                                 # turns phi about the pyramid axis, 1 deg steps
phi = torch.arange(K, dtype=torch.float64) * 2 * math.pi / K
Rz = torch.zeros(K, 3, 3, dtype=torch.float64)
Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = phi.cos(), -phi.sin(), phi.sin(), phi.cos(), 1
res = {'per_degree_turn': [], 'per_degree_std': [], 'profiles': []}
turn = np.zeros((V, L + 1)); amp = np.zeros((V, L + 1)); prof = []
for v in range(V):
    R = torch.from_numpy(rots[v, 0]).double() @ Rz                      # the ring R0 Rz(phi): equivalents at phi = 2 pi k / 7
    a, b, g = o3.matrix_to_angles(R)
    W = so3_utils.flat_wigner(L, a.float(), b.float(), g.float())        # (K, M), the model's own convention
    f_l = torch.stack([W[:, offs[l]:offs[l + 1]] @ psi[v, offs[l]:offs[l + 1]] for l in range(L + 1)], 1).double()   # (K, L+1)
    # 7-fold component aligned with the true equivalents: 2 * mean f_l(phi) cos(7 phi); > 0 means peaks at the right turns
    turn[v] = (2 * (f_l * torch.cos(7 * phi)[:, None]).mean(0)).numpy()
    amp[v] = f_l.std(0).numpy()                                           # total variation of degree l along the ring
    if v < 4: prof.append(f_l.numpy().tolist())
# true turn errors of the model's argmax along this ring, for reference
out = {'turn_mean': turn.mean(0).tolist(), 'turn_sem': (turn.std(0) / math.sqrt(V)).tolist(),
       'ring_std_mean': amp.mean(0).tolist(), 'profiles': prof, 'V': V}
json.dump(out, open('/home/user/optimist1938/i2s_degree_probe.json', 'w'))
print('turn coefficient per degree (mean):', np.round(turn.mean(0), 4).tolist())
print('ring variation per degree (std):', np.round(amp.mean(0), 4).tolist())

# ---- checks: is the ring the right set of poses, which frequencies vary, where is the ring's argmax ----
torch.manual_seed(0)
ring_mean, rand_mean, spec, phase = [], [], np.zeros((L + 1, 15)), []
U = o3.rand_matrix(2000, dtype=torch.float64)
au, bu, gu = o3.matrix_to_angles(U); WU = so3_utils.flat_wigner(L, au.float(), bu.float(), gu.float())
for v in range(V):
    R = torch.from_numpy(rots[v, 0]).double() @ Rz
    a, b, g = o3.matrix_to_angles(R)
    W = so3_utils.flat_wigner(L, a.float(), b.float(), g.float())
    f = (W @ psi[v]).double()
    ring_mean.append(float(f.mean())); rand_mean.append(float((WU @ psi[v]).double().mean()))
    f_l = torch.stack([W[:, offs[l]:offs[l + 1]] @ psi[v, offs[l]:offs[l + 1]] for l in range(L + 1)], 1).double()
    F = torch.fft.rfft(f_l, dim=0).abs()[:15].T / K * 2                    # (L+1, 15): amplitude of frequency m along the ring
    spec += F.numpy()
    k = int(f.argmax()); phase.append((math.degrees(float(phi[k])) % (360 / 7)))
spec /= V
out.update({'ring_logit_mean': float(np.mean(ring_mean)), 'random_logit_mean': float(np.mean(rand_mean)),
            'spectrum': spec.tolist(), 'argmax_phase_deg': phase})
json.dump(out, open('/home/user/optimist1938/i2s_degree_probe.json', 'w'))
print('mean logit on ring', round(np.mean(ring_mean), 3), 'vs random rotations', round(np.mean(rand_mean), 3))
print('ring spectrum, rows l=0..10, cols m=0..9:'); print(np.round(spec[:, :10], 3))
ph = np.array(phase); print('argmax phase within the 51.4 deg period: mean', round(ph.mean(), 1), 'std', round(ph.std(), 1), '(uniform: 25.7 / 14.8)')
