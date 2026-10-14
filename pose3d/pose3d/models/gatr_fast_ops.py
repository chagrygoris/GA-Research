"""Compile-friendly replacements for GATr 1.4's primitives: the same functions, written so that
torch.compile fuses them.

The ideas are those of lgatr 2.0 (https://github.com/heidelberg-hepml/lgatr, MIT), carried over to
the projective algebra G(3,0,1):

- GATr contracts its geometric product and join with dense 16x16x16 einsum tables. Each table
  holds 192 / 81 non-zero entries, all +-1, with at most one k per (i, j). Here they are
  straight-line sums over exactly those entries, read from GATr's own tables, so the blade order
  and the signs match by construction.
- Python-list indices (`x[..., [0]]`), which copy an index tensor host -> device on every call,
  become slices.
- Attention no longer sits behind `@torch.compiler.disable`, which split every block into
  separate graphs.

Unrolled sums launch hundreds of tiny kernels in eager mode, so `install` is meant for
--gatr_compile only (`gatr_denoiser._compile` calls it). Timings: reports/flash_clifford/frameworks.md.
"""

import importlib
import math

import torch
import torch.nn.functional as F

_GP_TERMS = None
_JOIN_TERMS = None


def _terms(table):
    """For each output blade i, the (j, k, negative) of every non-zero table[i, j, k] (all +-1)."""
    table = table.to("cpu", torch.float64)
    assert set(table[table != 0].tolist()) <= {-1.0, 1.0}
    return [[(j, k, bool(table[i, j, k] < 0)) for j, k in (table[i] != 0).nonzero().tolist()]
            for i in range(table.shape[0])]


def _bilinear(terms, x, y):
    x, y = torch.broadcast_tensors(x, y)
    xs, ys = x.unbind(-1), y.unbind(-1)
    out = []
    for blade in terms:
        acc = None
        for j, k, negative in blade:
            term = xs[j] * ys[k]
            if acc is None:
                acc = -term if negative else term
            else:
                acc = acc - term if negative else acc + term
        out.append(torch.zeros_like(xs[0]) if acc is None else acc)
    return torch.stack(out, dim=-1)


def geometric_product(x, y):
    return _bilinear(_GP_TERMS, x, y)


def equivariant_join(x, y, reference):
    return reference[..., 14:15] * _bilinear(_JOIN_TERMS, x, y)


def inner_product(x, y):
    """GATr's inner product: the blades without e0 (0, e1-e3, e12-e23, e123), shape (..., 1)."""
    return (x[..., 0:1] * y[..., 0:1] + (x[..., 2:5] * y[..., 2:5]).sum(-1, keepdim=True)
            + (x[..., 8:11] * y[..., 8:11]).sum(-1, keepdim=True) + x[..., 14:15] * y[..., 14:15])


def _without_trivectors(x):
    """The inner-product blades except e123: GATr's _INNER_PRODUCT_WO_TRI_IDX = [0, 2, 3, 4, 8, 9, 10]."""
    return torch.cat([x[..., 0:1], x[..., 2:5], x[..., 8:11]], dim=-1)


def _dist_vec(tri, basis, normalizer):
    tri_normed = tri * normalizer(tri[..., 3:4])
    return torch.einsum("xyz,...x,...y->...z", basis, tri_normed, tri_normed)


def geometric_attention(q_mv, k_mv, v_mv, q_s, k_s, v_s, normalizer, weights=None, attn_mask=None):
    """gatr.primitives.attention.geometric_attention (1.4.2), step by step, with slices and SDPA
    in the graph. xformers masks fall back to GATr's own function."""
    attention = importlib.import_module("gatr.primitives.attention")
    from gatr.utils.tensors import to_nd

    if attn_mask is not None and not isinstance(attn_mask, torch.Tensor):
        return attention.geometric_attention(q_mv, k_mv, v_mv, q_s, k_s, v_s, normalizer, weights, attn_mask)
    bh_shape = q_mv.shape[:-3]
    q_mv, k_mv, v_mv = to_nd(q_mv, 5), to_nd(k_mv, 5), to_nd(v_mv, 5)
    q_s, k_s, v_s = to_nd(q_s, 4), to_nd(k_s, 4), to_nd(v_s, 4)
    if isinstance(attn_mask, torch.Tensor) and attn_mask.dim() > 2:
        attn_mask = attn_mask.view(-1, *attn_mask.shape[-3:])
    n_mv_v, n_s_v = v_mv.shape[-2], v_s.shape[-1]
    n_mv_qk, n_s_qk = q_mv.shape[-2], q_s.shape[-1]

    basis_q, basis_k = attention._build_dist_basis(q_mv.device, q_mv.dtype)
    q_dist = _dist_vec(q_mv[..., 11:15], basis_q, normalizer)
    k_dist = _dist_vec(k_mv[..., 11:15], basis_k, normalizer)
    if weights is not None:
        q_dist = q_dist * weights[..., None].to(q_dist.dtype)

    n_qk = n_mv_qk * (7 + 5) + n_s_qk
    n_v = n_mv_v * 16 + n_s_v
    n = 8 * -(-max(n_qk, n_v) // 8)  # ceil to a multiple of 8
    pad = q_s.new_zeros(*q_s.shape[:3], n - n_qk)
    q = torch.cat([_without_trivectors(q_mv).flatten(-2), q_dist.flatten(-2), q_s, pad], -1)
    k = torch.cat([_without_trivectors(k_mv).flatten(-2), k_dist.flatten(-2), k_s,
                   k_s.new_zeros(*k_s.shape[:3], n - n_qk)], -1)
    v = torch.cat([v_mv.flatten(-2), v_s, v_s.new_zeros(*v_s.shape[:3], n - n_v)], -1)
    k = k * math.sqrt(n / n_qk)  # correct for the zero padding

    lead = torch.broadcast_shapes(q.shape[:-2], k.shape[:-2], v.shape[:-2])  # multi-query heads
    q, k, v = (t.expand(*lead, *t.shape[-2:]) for t in (q, k, v))
    v_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    out_mv = v_out[..., : n_mv_v * 16].unflatten(-1, (n_mv_v, 16))
    out_s = v_out[..., n_mv_v * 16: n_mv_v * 16 + n_s_v]
    return out_mv.view(*bh_shape, *out_mv.shape[-3:]), out_s.view(*bh_shape, *out_s.shape[-2:])


def _gated_forward(self, multivectors, scalars):
    """ScalarGatedNonlinearity.forward with the scalar gate as a slice."""
    return (self.gated_nonlinearity(multivectors, gates=multivectors[..., 0:1]),
            self.scalar_nonlinearity(scalars))


def install():
    """Point GATr's layers at these functions (process-wide). Returns a function that undoes it."""
    global _GP_TERMS, _JOIN_TERMS
    from gatr.primitives.bilinear import _load_bilinear_basis
    from gatr.primitives.dual import _compute_efficient_join

    _GP_TERMS = _terms(_load_bilinear_basis("gp"))
    _JOIN_TERMS = _terms(_compute_efficient_join())
    # importlib: `import gatr.primitives.dual as m` would bind the function `dual` instead
    bilinears = importlib.import_module("gatr.layers.mlp.geometric_bilinears")
    nonlinearities = importlib.import_module("gatr.layers.mlp.nonlinearities")
    normalization = importlib.import_module("gatr.primitives.normalization")
    attention_layer = importlib.import_module("gatr.layers.attention.attention")
    targets = [(bilinears, "geometric_product", geometric_product),
               (bilinears, "equivariant_join", equivariant_join),
               (normalization, "inner_product", inner_product),
               (attention_layer, "geometric_attention", geometric_attention),
               (nonlinearities.ScalarGatedNonlinearity, "forward", _gated_forward)]
    saved = [(obj, name, getattr(obj, name)) for obj, name, _ in targets]
    for obj, name, new in targets:
        setattr(obj, name, new)

    def undo():
        for obj, name, old in saved:
            setattr(obj, name, old)
    return undo
