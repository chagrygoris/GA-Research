"""How much does knowing the turn of a bottom-view n-gon lower the flow-matching loss? For the exact (Bayes) velocity field
v* (posterior mean over the n equivalent poses, independent coupling, uniform noise) and the best image-blind field
(the same with all 360 turns equivalent): mean squared distance between them, over the flow's training distribution."""
import sys, math, json, torch
sys.path.insert(0, "/tmp/claude-0/-home-user-GA-Research/47fc9bb1-1306-56b6-a408-21242df5b300/scratchpad/ns8/snap/pose3d")
from clifford.algebra.cliffordalgebra import CliffordAlgebra
from pose3d.geometry.flow import geodesic_interpolate, relative_log
from pose3d.geometry.rotor import matrix_to_rotor, random_rotor, rotor_to_matrix
alg = CliffordAlgebra((1, 1, 1)); torch.manual_seed(0)
Rz = lambda a: torch.stack([torch.stack([a.cos(), -a.sin(), 0 * a], -1), torch.stack([a.sin(), a.cos(), 0 * a], -1), torch.stack([0 * a, 0 * a, 1 + 0 * a], -1)], -2)
angle = lambda A, B: torch.acos(((torch.einsum("...ij,...ij->...", A, B) - 1) / 2).clamp(-1, 1))
def vstar(rt, t, eq):
    B, K = eq.shape[:2]; s = (1 - t).clamp_min(1e-3)[:, None]
    al = angle(rotor_to_matrix(rt, alg)[:, None], eq).clamp_min(1e-4)
    hav = lambda a: 2 * torch.sin(a.double() / 2) ** 2
    w = (hav((al / s).clamp(max=math.pi)) / (s.double() * hav(al)) * (al <= s * math.pi + 1e-3)).float()
    w = w + 1e-12 * (al == al.min(-1, keepdim=True).values); w = w / w.sum(-1, keepdim=True)
    u = relative_log(rt[:, None].expand(B, K, 4).reshape(-1, 4), matrix_to_rotor(eq.reshape(-1, 3, 3)), alg).view(B, K, 3) / s[..., None]
    return (w[..., None] * torch.nan_to_num(u)).sum(1)
ring = Rz(torch.arange(360) * (2 * math.pi / 360)); out = {}
for n in (4, 5, 7, 10, 13):
    tot = 0.0; reps = 6; N = 1500
    for _ in range(reps):
        th = torch.rand(N) * 2 * math.pi; eq = Rz(th[:, None] + torch.arange(n) * (2 * math.pi / n))      # (N, n, 3, 3)
        r0 = random_rotor(N); t = torch.rand(N); r1 = matrix_to_rotor(eq[torch.arange(N), torch.randint(n, (N,))])
        rt = geodesic_interpolate(r0, r1, t, alg)
        tot += (vstar(rt, t, eq) - vstar(rt, t, ring[None].expand(N, -1, -1, -1))).pow(2).sum(-1).mean().item() / reps
    out[n] = dict(gap=tot, guessed_turn_deg=90 / n); print(n, round(tot, 3), f"(guessed-turn error {90 / n:.1f} deg)", flush=True)
json.dump(out, open("gap_vs_n.json", "w"), indent=1)
