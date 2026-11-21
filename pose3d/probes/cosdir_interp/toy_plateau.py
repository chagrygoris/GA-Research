"""Toy model of the bottom-view plateau: why learning the turn of an n-gon waits for a long time and then happens fast.

The model keeps only the geometry of the real one (reports/cosdir_bottom_view.md):
  * the image tokens sit on one ring of P points at azimuths psi_j = 2 pi j / P (P = 32, the outer rings of the
    368-point grid);
  * the image of a pyramid turned by theta gives every point the signal g_theta(psi) = cos(n (psi - theta)) (only the
    n-fold pattern carries the turn) plus a turn-independent constant;
  * a learnable circular convolution W (C channels) turns the signal into token features h = W * g ("encoder");
  * H heads look at the ring through fixed von Mises-Fisher windows: weights softmax_j(kappa cos(psi_j - phi_h)) around
    random centres phi_h ("cosine attention" with inverse temperature kappa);
  * a learnable linear readout R maps the H x C head outputs to the target (cos n theta, sin n theta) ("velocity").
W and R start small (scale eps), as the turn-dependent part of the real model does on the plateau. The loss of the
best turn-blind answer (output 0) is 0.5. Training: full batch over 256 turns, plain gradient descent or Adam.

Prediction (deep linear networks, Saxe, McClelland & Ganguli 2014): the turn component of the output is a product
u = a_n * (feature amplitude at frequency n) * (readout), where a_n = |sum_j w(psi_j) e^{i n psi_j}| is how much of
degree n a window passes. Under gradient descent each factor grows at a rate set by the others, so u grows like
exp(2 eta s t) from u_0 ~ eps^2 with s proportional to a_n, and the plateau lasts about ln(1/u_0) / (2 eta s): inversely
proportional to a_n and logarithmic in the starting size. Then the escape is sudden.

usage: python toy_plateau.py [--out results/toy_plateau.json]
"""
import argparse, json, math, os
import numpy as np, torch

P, C, H = 32, 4, 8

def run(n, kappa, eps, seed, opt="gd", lr=None, steps=20000, ntheta=256, batch=64, noise=2.0):
    g = torch.Generator().manual_seed(seed)
    psi = torch.arange(P) * 2 * math.pi / P
    def data(theta):
        sig = torch.cos(n * (psi[None] - theta[:, None]))                   # (T, P)
        sig = torch.stack([sig, torch.ones_like(sig)], 1)                   # turn signal + constant channel
        return sig, torch.stack([torch.cos(n * theta), torch.sin(n * theta)], 1)
    theta_ev = torch.arange(ntheta) * 2 * math.pi / ntheta; sig_ev, y_ev = data(theta_ev)
    phi = torch.rand(H, generator=g) * 2 * math.pi
    w = torch.softmax(kappa * torch.cos(psi[None] - phi[:, None]), 1)       # (H, P) fixed windows
    a_n = float((w * torch.exp(1j * n * psi)[None]).sum(1).abs().median())
    W = (eps * torch.randn(C, 2, P, generator=g)).requires_grad_()          # circular conv kernels
    R = (eps * torch.randn(2, H * C, generator=g)).requires_grad_()
    idx = (torch.arange(P)[:, None] + torch.arange(P)[None]) % P             # circular shifts
    o = torch.optim.SGD([W, R], lr=lr or 0.5) if opt == "gd" else torch.optim.Adam([W, R], lr=lr or 1e-3)
    loss_t, amp_t = [], []
    def forward(sig):
        h = torch.einsum("cim,tijm->tcj", W, sig[:, :, idx])                 # (T, C, P): circular convolution
        return torch.einsum("hj,tcj->thc", w, h).reshape(len(sig), -1) @ R.T
    for t in range(steps):
        # minibatch of random turns; the target carries noise of std `noise` per component, as the flow-matching
        # target does (its turn-dependent mean is a small part of its variance)
        th = torch.rand(batch, generator=g) * 2 * math.pi; sig, y = data(th)
        y = y + noise * torch.randn(y.shape, generator=g)
        loss = ((forward(sig) - y) ** 2).sum(1).mean() / 2
        o.zero_grad(); loss.backward(); o.step()
        if t % 5 == 0:
            with torch.no_grad():                                            # clean loss and the output's projection on the target
                out = forward(sig_ev); le = float(((out - y_ev) ** 2).sum(1).mean() / 2)
            loss_t.append(le); amp_t.append(float((out * y_ev).sum(1).mean()))
            if le < 0.02:
                break
    ls = np.array(loss_t)
    escape = int(np.argmax(ls < 0.25)) * 5 if (ls < 0.25).any() else None   # first step below half the blind loss
    return dict(n=n, kappa=kappa, eps=eps, seed=seed, opt=opt, a_n=a_n, escape_step=escape, loss=loss_t[::4], amp=amp_t[::4], log_every=20)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/toy_plateau.json"))
    a = ap.parse_args(); torch.set_num_threads(4); res = []
    LR = {"gd": 0.05, "adam": 1e-3}
    for opt in ("gd", "adam"):
        for kappa in (8, 10, 12, 15, 18, 27, 40):                            # window width for the 10-gon
            for seed in range(3):
                r = run(10, kappa, 1e-3, seed, opt, lr=LR[opt]); res.append(r)
                print(opt, "kappa", kappa, "a_10 %.3f" % r["a_n"], "seed", seed, "escape", r["escape_step"], flush=True)
        for eps in (1e-1, 1e-2, 1e-4, 1e-5):                                 # starting size, kappa 12
            for seed in range(3):
                r = run(10, 12, eps, seed, opt, lr=LR[opt]); res.append(r)
                print(opt, "eps", eps, "seed", seed, "escape", r["escape_step"], flush=True)
        json.dump(res, open(a.out, "w"))
    json.dump(res, open(a.out, "w"))

if __name__ == "__main__":
    main()
