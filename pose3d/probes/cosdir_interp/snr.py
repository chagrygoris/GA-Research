"""Gradient signal-to-noise of flow matching at the plateau checkpoint: per-batch gradients of the vector field under the
sampled target (independent or nearest coupling) against the gradient under the exact velocity on the same samples."""
import os, sys, math, json, torch, numpy as np
sys.argv = ["snr", "snr", "0", "1e-4", "trained", "indep", "1"]
src = open("ft.py").read().split("params = list(m.vector_field.parameters())")[0]
exec(src)
params = [p for p in m.vector_field.parameters() if p.requires_grad]
def grad(loss):
    g = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True); return torch.cat([(x if x is not None else torch.zeros_like(p)).flatten() for x, p in zip(g, params)])
torch.manual_seed(0); m.vector_field.train(); out = {}
for coupling in ("indep", "nearest"):
    S_, N_ = [], []
    for rep in range(24):
        idx = torch.randint(4500, (64,)); cond = ctr[idx]; eq = equivalents(Rtr[idx]); B = 64
        r0 = random_rotor(B); t = torch.rand(B)
        if coupling == "nearest":
            R0 = rotor_to_matrix(r0, m.algebra); pick = angle(R0[:, None], eq).argmin(-1); R1 = eq[torch.arange(B), pick]
        else: R1 = eq[torch.arange(B), torch.randint(n, (B,))]
        r1 = matrix_to_rotor(R1); rt = geodesic_interpolate(r0, r1, t, m.algebra); u = relative_log(r0, r1, m.algebra)
        pred = m.velocity(rt, t, cond)
        # the exact velocity of this coupling: independent -> posterior mean over equivalents; nearest -> the sample's own target
        # is already deterministic given (R_t, t) up to ties, so its noise is ~0 and the reference is u itself
        ref = vstar(rt, t, eq) if coupling == "indep" else u
        g_s = grad((pred - u).pow(2).sum(-1).mean()); g_e = grad((pred - ref).pow(2).sum(-1).mean())
        S_.append(g_e); N_.append(g_s - g_e)
    G = torch.stack(S_); Nn = torch.stack(N_); sig = G.mean(0)
    out[coupling] = dict(signal_norm=sig.norm().item(), noise_norm_b64=Nn.norm(dim=1).mean().item(),
                         exact_batch_scatter=(G - sig).norm(dim=1).mean().item())
    print(coupling, out[coupling], "SNR(batch 64) = %.4f, SNR(batch 512) = %.4f" % ((sig.norm() / Nn.norm(dim=1).mean()).item(), (sig.norm() / Nn.norm(dim=1).mean() * math.sqrt(8)).item()), flush=True)
json.dump(out, open("snr.json", "w"))
