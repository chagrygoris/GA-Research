"""Versor (Truong & Hirst 2026, arXiv 2602.10195) as the Clifford Flow denoiser.

`VersorVectorField` is a drop-in replacement for `GATrVectorField` (pose_tokens='rotor'): same
(B, 2 + n_cond_mv, 8) Cl(3,0) input, same (B, 1, 8) output with the velocity in the bivector part.

Versor works in the conformal algebra Cl(4,1): 5 basis vectors e1, e2, e3, e+, e- with squares
+1, +1, +1, +1, -1, so a multivector has 2^5 = 32 components. A basis blade is indexed by the
bitmask of its basis vectors (e1 = 1, e2 = 2, e3 = 4, e+ = 8, e- = 16; e13 = 1 | 4 = 5), which makes
the geometric product of two blades the blade of their XOR. Cl(3,0) is the subalgebra spanned by
the blades with bits 0-2 only, so the embedding is a plain copy of components and the read-out of
e12, e13, e23 is a copy of components 3, 5, 6.

The layers are ported from the reference code (https://github.com/VersorAI/Versor, Apache-2.0:
Model/layers.py, Model/model.py, library/gacore/kernel.py), in plain torch without its Triton / MLX
kernels, so they run anywhere torch does; the functions are those of its CPU path:

- `VersorLinear`: out_n = Normalize(sum_k x_k w_nk), a geometric product of every input channel
  x_k with a learned multivector w_nk (right multiplication), then manifold normalization.
- manifold normalization: A / sqrt(|sum_i s_i A_i^2| + eps), s_i the product of the squares of the
  basis vectors in blade i, A_i the components. The reference's fallback to the identity for a
  denominator below 1e-4 is kept but never fires with eps = 1e-6 (the denominator is >= 1e-3).
- `VersorAttention` (Geometric Product Attention in the paper): score_ij = (<q_i k_j>_0 +
  gamma * sqrt(|q_i|^2 |k_j|^2 - <q_i k_j>_0^2)) / sqrt(d_head), plus the code's 0.1 x
  softmax(score) @ score "triangle" term, softmax over j, values averaged per channel.
- `VersorBlock`: pre-LayerNorm attention and Tanh MLP residuals with LayerScale, each followed
  by manifold normalization.

Unlike GATr, these layers are not rotation equivariant (right multiplication by a fixed w,
LayerNorm and Tanh over components). In the pooled reference recipe the condition multivectors do
not rotate with the image anyway (config.FlowConfig.cond_tokens), so the comparison is between
two denoisers, not between an equivariant and a non-equivariant flow.

Scalars: Versor has no scalar track. What GATr receives as scalar channels (the time t on every
token, and which token is the pose) goes into a second input multivector channel: a learned role
multivector (pose / time / condition) plus t on its scalar component. The read-out `out` is the
same geometric product without the normalization (a normalized output cannot be zero), zeroed by
CliffordFlow so the initial velocity is 0, as GATr's.
"""

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

SIGNATURE = (1, 1, 1, 1, -1)   # e1, e2, e3, e+, e-
GA_DIM = 32
# Cl(3,0) blades in this repo's order (1, e1, e2, e3, e12, e13, e23, e123) -> Cl(4,1) bitmasks.
_CL41_INDEX = (0, 1, 2, 4, 3, 5, 6, 7)
_ROTATION_BIVECTOR = (3, 5, 6)   # e12, e13, e23


def _popcount(x: int) -> int:
    return bin(x).count("1")


@functools.lru_cache(maxsize=None)
def _tables():
    """sign[i, k] = sign of e_i e_{i ^ k} (kernel.get_sign_matrix), the metric sign s_i of
    `manifold_normalization`, and the scalar sign <e_i e_i>_0 of the attention score."""
    sign = torch.zeros(GA_DIM, GA_DIM)
    for i in range(GA_DIM):
        for k in range(GA_DIM):
            j = i ^ k
            swaps = sum(_popcount(i & ~((1 << (b + 1)) - 1)) for b in range(5) if (j >> b) & 1)
            metric = 1.0
            for b in range(5):
                if ((i & j) >> b) & 1:
                    metric *= SIGNATURE[b]
            sign[i, k] = (-1.0 if swaps % 2 else 1.0) * metric
    norm_sign = torch.tensor([float(torch.tensor([SIGNATURE[b] for b in range(5) if (i >> b) & 1]).prod())
                              for i in range(GA_DIM)])
    return sign, norm_sign, sign[:, 0].clone()


def geometric_product(a, b):
    """(..., 32) x (..., 32) -> (..., 32): c_k = sum_i a_i b_{i ^ k} sign[i, k]."""
    sign = _tables()[0].to(a.device, a.dtype)
    idx = torch.arange(GA_DIM, device=a.device)
    xor = idx.unsqueeze(1) ^ idx.unsqueeze(0)
    return (a.unsqueeze(-1) * b[..., xor] * sign).sum(-2)


def manifold_normalization(x, eps: float = 1e-6):
    norm_sign = _tables()[1].to(x.device, x.dtype)
    denom = torch.sqrt((x * x * norm_sign).sum(-1, keepdim=True).abs() + eps)
    normalized = x / torch.where(denom < 1e-6, torch.ones_like(denom), denom)
    identity = torch.zeros_like(x)
    identity[..., 0] = 1.0
    return torch.where(denom < 1e-4, identity, normalized)


class VersorLinear(nn.Module):
    """out_n = sum_k x_k w_nk (geometric product), normalized unless normalize=False.

    Runs as one dense matmul: the weight multivectors are expanded into the (32 out, 32 in) matrix
    of right multiplication on every call, so they stay trainable (geometric_linear_cpu)."""

    def __init__(self, in_features: int, out_features: int, normalize: bool = True):
        super().__init__()
        self.in_features, self.out_features, self.normalize = in_features, out_features, normalize
        self.weight = nn.Parameter(torch.randn(out_features, in_features, GA_DIM)
                                   * (2.0 / (in_features * GA_DIM)) ** 0.5)
        idx = torch.arange(GA_DIM)
        j, i = torch.meshgrid(idx, idx, indexing="ij")
        self.register_buffer("_xor", i ^ j, persistent=False)                    # [j, i] -> i ^ j
        self.register_buffer("_sign", _tables()[0].T.contiguous(), persistent=False)   # [j, i] -> sign[i, j]

    def forward(self, x):                                   # (..., in, 32) -> (..., out, 32)
        n, k = self.out_features, self.in_features
        w = (self.weight[:, :, self._xor] * self._sign.to(self.weight.dtype))   # (n, k, j, i)
        w = w.permute(0, 2, 1, 3).reshape(n * GA_DIM, k * GA_DIM)
        out = F.linear(x.reshape(*x.shape[:-2], k * GA_DIM), w).reshape(*x.shape[:-2], n, GA_DIM)
        return manifold_normalization(out) if self.normalize else out


class VersorAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        if channels % num_heads:
            raise ValueError(f"versor channels ({channels}) must be divisible by the heads ({num_heads})")
        self.num_heads, self.head_dim = num_heads, channels // num_heads
        self.q_proj, self.k_proj = VersorLinear(channels, channels), VersorLinear(channels, channels)
        self.v_proj, self.o_proj = VersorLinear(channels, channels), VersorLinear(channels, channels)
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.register_buffer("_scalar_sign", _tables()[2], persistent=False)

    def forward(self, x):                                   # (B, S, C, 32)
        b, s, c, _ = x.shape
        h, d = self.num_heads, self.head_dim

        def heads(t):                                       # (B * h, S, d, 32)
            return t.view(b, s, h, d, GA_DIM).transpose(1, 2).reshape(b * h, s, d, GA_DIM)

        q, k, v = heads(self.q_proj(x)), heads(self.k_proj(x)), heads(self.v_proj(x))
        scalar = torch.bmm((q * self._scalar_sign.to(q.dtype)).reshape(b * h, s, -1),
                           k.reshape(b * h, s, -1).transpose(1, 2))           # <q_i k_j>_0
        q2 = (q ** 2).sum((-1, -2)).unsqueeze(2)
        k2 = (k ** 2).sum((-1, -2)).unsqueeze(1)
        bivector = torch.sqrt(torch.relu(q2 * k2 - scalar ** 2) + 1e-6)
        score = (scalar + self.gamma * bivector) / d ** 0.5
        if s > 1:
            score = score + 0.1 * torch.bmm(torch.softmax(score, dim=-1), score)
        out = torch.einsum("bij,bjdl->bidl", torch.softmax(score, dim=-1), v)
        out = out.view(b, h, s, d, GA_DIM).transpose(1, 2).reshape(b, s, c, GA_DIM)
        return self.o_proj(out)


class VersorBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, expansion: int = 4, init_values: float = 1e-5):
        super().__init__()
        self.attn = VersorAttention(channels, num_heads)
        self.ln1, self.ln2 = nn.LayerNorm([channels, GA_DIM]), nn.LayerNorm([channels, GA_DIM])
        self.gamma1 = nn.Parameter(init_values * torch.ones(channels, GA_DIM))   # LayerScale
        self.gamma2 = nn.Parameter(init_values * torch.ones(channels, GA_DIM))
        self.mlp = nn.Sequential(VersorLinear(channels, expansion * channels), nn.Tanh(),
                                 VersorLinear(expansion * channels, channels))

    def forward(self, x):
        x = manifold_normalization(x + self.gamma1 * self.attn(self.ln1(x)))
        return manifold_normalization(x + self.gamma2 * self.mlp(self.ln2(x)))


class VersorVectorField(nn.Module):
    """(B, 2 + n_cond_mv, 8) Cl(3,0) multivectors -> (B, 1, 8), like GATrVectorField.

    Token 0 is the noisy rotor R_t, token 1 the time embedding (a scalar multivector), tokens 2..
    the condition multivectors. Each token gets two input channels in Cl(4,1): its multivector, and
    its role multivector (learned, one each for pose / time / condition) plus t on the scalar part.
    A VersorLinear lifts them to `mv_channels`, `num_blocks` VersorBlocks follow, and the
    unnormalized read-out `out` maps token 0 to one multivector whose e12, e13, e23 are the velocity.
    """

    def __init__(self, n_cond_mv: int, mv_channels: int = 8, num_blocks: int = 4, num_heads: int = 4,
                 expansion: int = 4):
        super().__init__()
        self.n_cond_mv = n_cond_mv
        self.role = nn.Parameter(0.1 * torch.randn(3, GA_DIM))
        self.lift = VersorLinear(2, mv_channels)
        self.blocks = nn.ModuleList(VersorBlock(mv_channels, num_heads, expansion) for _ in range(num_blocks))
        self.out = VersorLinear(mv_channels, 1, normalize=False)   # zeroed by CliffordFlow
        self.register_buffer("_index", torch.tensor(_CL41_INDEX), persistent=False)
        self.register_buffer("_rot_index", torch.tensor(_ROTATION_BIVECTOR), persistent=False)
        self.register_buffer("_role_of_token", torch.tensor([0, 1] + [2] * n_cond_mv), persistent=False)

    def forward(self, x):
        b, n, _ = x.shape
        mv = x.new_zeros(b, n, 2, GA_DIM)
        mv[:, :, 0, self._index] = x
        mv[:, :, 1] = self.role.to(x.dtype)[self._role_of_token[:n]]
        mv[:, :, 1, 0] = mv[:, :, 1, 0] + x[:, 1, 0].unsqueeze(1)   # t: scalar part of the time token
        h = self.lift(mv)
        for block in self.blocks:
            h = block(h)
        query = self.out(h[:, :1])[:, 0, 0]                          # (B, 32)
        result = x.new_zeros(b, 1, x.shape[-1])
        result[:, 0, 4:7] = query[:, self._rot_index]
        return result
