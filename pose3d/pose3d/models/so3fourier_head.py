"""SO(3)-Fourier velocity field (image2sphere / Howell style) for CliffordFlow.

Design and proofs: pose3d/reports/equivariance_alt_head.md. The GATr heads carry Cl(3,0) multivectors,
i.e. angular frequencies l <= 1 per token; here the image becomes C_f signals on SO(3) with Fourier
coefficients up to l = lmax, and the velocity at a pose hypothesis R is read from those signals expressed
around R.

Condition path (once per image), SphereConditionHead:
    fmap (B, C, 7, 7) -> 1x1 conv -> C_s scalar maps
      -> D4-steerable orthographic lift onto a HEALPix hemisphere (P points, optical axis e3):
         Gaussian weights in |cell - coverage * point| with per-radius gains, so a 90 degree turn or a
         left-right flip of the map permutes cells and points alike
      -> ReLU on the grid -> S2 Fourier transform (l <= lmax) -> per-(channel, l) gains (isotropic kernel)
      -> S2 -> SO(3) convolution with Fourier-domain filters psi (image2sphere's S2 conv)
      -> norm gate per (channel, l) block (exactly equivariant; a grid ReLU is only approximately so)
    = c (B, C_f, sum_l (2l+1)^2), the Wigner coefficients of C_f signals f(g) = sum_l <c_l, D^l(g)>.
    Camera roll G acts as c_l -> D^l(G) c_l (left action).

Velocity read-out (every flow step), SO3FourierVelocity:
    c~_l = D^l(R)^T c_l     (the signals around the hypothesis: invariant under (R, c) -> (G R, D(G) c))
    v = MLP(flatten(c~), t) the body-frame velocity bivector (e12, e13, e23), last layer zero-initialised
    Z2 Reynolds average over the mirror F = diag(-1, 1, 1): v = 1/2 [m(c~) + S m(D(F) c~)], S = (-1, -1, +1).
Body velocity is invariant under R -> G R, so the Euler step r <- r exp(dt v) commutes with camera roll;
the flow of a mirrored image is the mirrored flow. Exact for 90 degree turns and the flip of the map
(tests/test_so3fourier.py); the ResNet in front is not equivariant.

canonicalize=False gives the non-equivariant twin for the fair test: same condition path, the read-out sees
[c, R] instead of D(R)^T c (and no mirror average), with the parameter count kept within ~0.3 %.

Conventions: e1 = image columns (right), e2 = rows (down), e3 = optical axis;
torch.rot90(map, 1, (2, 3)) <-> R -> R_z(-90 deg) R; left-right flip <-> R -> F R F.
"""

import math

import numpy as np
import torch
import torch.nn as nn


def _o3():
    from e3nn import o3
    return o3


class _float64_default:
    """e3nn builds some constants in torch's default dtype; compute the fixed geometry in float64."""

    def __enter__(self):
        self.prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    def __exit__(self, *exc):
        torch.set_default_dtype(self.prev)


def hemisphere_grid(nside: int = 4) -> torch.Tensor:
    """HEALPix pixel centres with z > 0, (P, 3). The grid is D4-symmetric about e3."""
    import healpy as hp
    x, y, z = hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))
    pts = np.stack([x, y, z], -1)[z > 1e-9]
    return torch.tensor(pts, dtype=torch.float64)


def wigner_blocks(R: torch.Tensor, lmax: int):
    """[D^l(R)]_{l <= lmax} for a batch of rotation matrices (B, 3, 3), each (B, 2l+1, 2l+1)."""
    o3 = _o3()
    abc = o3.matrix_to_angles(R)
    return [o3.wigner_D(l, *abc) for l in range(lmax + 1)]


def n_coeffs(lmax: int) -> int:
    return sum((2 * l + 1) ** 2 for l in range(lmax + 1))


class SphereConditionHead(nn.Module):
    """fmap (B, c_in, grid, grid) -> SO(3) Fourier coefficients (B, c_f, sum_l (2l+1)^2)."""

    def __init__(self, c_in: int = 2048, c_s: int = 32, c_f: int = 8, lmax: int = 6, grid: int = 7,
                 nside: int = 4, coverage: float = 0.9, sigma: float = 0.2,
                 taper_beta: float = math.radians(75), max_beta: float = math.radians(90)):
        super().__init__()
        if grid % 2 == 0 or grid < 3:
            raise ValueError("grid must be odd (>= 3) so the centre cell sits on the optical axis")
        o3 = _o3()
        self.lmax, self.c_s, self.c_f, self.grid = lmax, c_s, c_f, grid
        self.proj = nn.Conv2d(c_in, c_s, 1)
        pts = hemisphere_grid(nside)                                       # (P, 3)
        max_r = pts[:, :2].norm(dim=1).max()
        self.register_buffer("q", pts[:, :2] / max_r, persistent=False)   # point -> image position in [-1, 1]
        off = (torch.arange(grid, dtype=torch.float64) - grid // 2) / (grid // 2)
        rows, cols = torch.meshgrid(off, off, indexing="ij")
        p = torch.stack([cols.flatten(), rows.flatten()], -1)             # (grid^2, 2): (u, v) = (col, row)
        self.register_buffer("p", p, persistent=False)
        rad2 = (p ** 2).sum(-1)
        uniq = rad2.unique()
        self.register_buffer("ridx", torch.searchsorted(uniq, rad2), persistent=False)
        self.radial_gain = nn.Parameter(torch.ones(len(uniq)))             # D4-invariant per-cell gain
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma)))
        self.coverage = nn.Parameter(torch.tensor(coverage))
        beta = torch.acos(pts[:, 2].clamp(-1, 1))
        taper = ((beta - max_beta) / (taper_beta - max_beta)).clamp(max=1.0)
        self.register_buffer("taper", taper, persistent=False)             # fades out towards the rim
        with _float64_default():
            Y = o3.spherical_harmonics(list(range(lmax + 1)), pts, normalize=True, normalization="component")
        # Geometry buffers stay float64 (the D4 symmetry of the grid is exact only to their precision) and are
        # cast to the feature dtype at use.
        self.register_buffer("Y", Y.to(torch.float64), persistent=False)  # (P, (L+1)^2), real, natural parity
        self.l_gain = nn.Parameter(torch.ones(c_s, lmax + 1))              # isotropic learned S2 kernel
        self.psi = nn.ParameterList([
            nn.Parameter(torch.randn(c_s, c_f, 2 * l + 1) / math.sqrt(c_s * (2 * l + 1)))
            for l in range(lmax + 1)])                                     # S2 -> SO(3) filters, Fourier domain
        self.gate_a = nn.Parameter(torch.ones(c_f, lmax + 1))
        self.gate_b = nn.Parameter(torch.zeros(c_f, lmax + 1))

    @property
    def width(self) -> int:
        return n_coeffs(self.lmax)

    def lift_kernel(self, dtype):
        """(grid^2, P) weights from cell to hemisphere point."""
        p, q = self.p.to(dtype), self.q.to(dtype)
        d2 = ((p[:, None, :] - self.coverage.to(dtype) * q[None, :, :]) ** 2).sum(-1)
        k = torch.exp(-d2 / (2 * torch.exp(2 * self.log_sigma.to(dtype))))
        k = k / k.sum(0, keepdim=True)
        return k * self.radial_gain.to(dtype)[self.ridx][:, None] * self.taper.to(dtype)[None, :]

    def s2_coeffs(self, fmap):
        b, _, h, w = fmap.shape
        if (h, w) != (self.grid, self.grid):
            raise ValueError(f"SphereConditionHead expects a {self.grid}x{self.grid} map, got {h}x{w}")
        feats = self.proj(fmap).flatten(2)
        sig = torch.relu(feats @ self.lift_kernel(feats.dtype))            # (B, C_s, P)
        psi = sig @ self.Y.to(feats.dtype) / math.sqrt(self.Y.shape[0])    # equal-area grid: a plain mean
        gains = torch.cat([self.l_gain[:, l:l + 1].expand(-1, 2 * l + 1) for l in range(self.lmax + 1)], 1)
        return psi * gains[None]

    def forward(self, fmap):
        psi = self.s2_coeffs(fmap)
        out, start = [], 0
        for l in range(self.lmax + 1):
            n = 2 * l + 1
            c = torch.einsum("bxm,xyn->bymn", psi[:, :, start:start + n], self.psi[l])   # (B, C_f, m, n)
            norm = c.flatten(2).norm(dim=-1) / n                                        # left/right invariant
            gate = torch.sigmoid(self.gate_a[:, l] * norm + self.gate_b[:, l])
            out.append((c * gate[..., None, None]).flatten(2))
            start += n
        return torch.cat(out, -1)


class SO3FourierVelocity(nn.Module):
    """(R (B, 3, 3), t (B,), c (B, c_f, N)) -> body velocity bivector (B, 3), (e12, e13, e23)."""

    def __init__(self, c_f: int = 8, lmax: int = 6, hidden: int = 44, t_dim: int = 64,
                 flip_symmetrize: bool = True, canonicalize: bool = True):
        super().__init__()
        self.c_f, self.lmax, self.t_dim = c_f, lmax, t_dim
        self.canonicalize = bool(canonicalize)
        self.flip_symmetrize = bool(flip_symmetrize) and self.canonicalize
        n_in = c_f * n_coeffs(lmax) + (0 if self.canonicalize else 9)
        self.first = nn.Linear(n_in, hidden)        # a bank of `hidden` SO(3) correlations evaluated at R
        self.film = nn.Linear(t_dim, 2 * hidden)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.out = nn.Linear(hidden, 3)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        if self.flip_symmetrize:
            o3 = _o3()
            with _float64_default():
                abc = o3.matrix_to_angles(torch.diag(torch.tensor([1.0, -1.0, -1.0])))   # R_x(pi)
                DF = [((-1) ** l) * o3.wigner_D(l, *abc) for l in range(lmax + 1)]   # D^l(F), F = diag(-1, 1, 1)
            for l, D in enumerate(DF):
                self.register_buffer(f"DF{l}", D.to(torch.float64), persistent=False)
        self.register_buffer("S", torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64), persistent=False)

    def _t_embed(self, t):
        half = self.t_dim // 2
        freqs = torch.exp(-math.log(1e4) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
        a = 1000.0 * t[:, None] * freqs[None]
        return torch.cat([a.cos(), a.sin()], -1)

    def _blocks(self, c):
        out, start = [], 0
        for l in range(self.lmax + 1):
            n = 2 * l + 1
            out.append(c[:, :, start:start + n * n].view(c.shape[0], self.c_f, n, n))
            start += n * n
        return out

    def _mlp(self, x, t):
        g = self.first(x)
        gamma, beta = self.film(self._t_embed(t.to(x.dtype))).chunk(2, -1)
        return self.out(self.mlp(g * (1 + gamma) + beta))

    def forward(self, R, t, c):
        if not self.canonicalize:
            return self._mlp(torch.cat([c.flatten(1), R.flatten(1)], 1), t)
        ct = [torch.einsum("bmk,bymn->bykn", D.to(c.dtype), cl)            # D^l(R)^T c_l
              for D, cl in zip(wigner_blocks(R, self.lmax), self._blocks(c))]
        v = self._mlp(torch.cat([x.flatten(1) for x in ct], 1), t)
        if not self.flip_symmetrize:
            return v
        mirrored = torch.cat([torch.einsum("mk,bykn->bymn", getattr(self, f"DF{l}").to(x.dtype), x).flatten(1)
                              for l, x in enumerate(ct)], 1)
        return 0.5 * (v + self.S.to(v.dtype) * self._mlp(mirrored, t))
