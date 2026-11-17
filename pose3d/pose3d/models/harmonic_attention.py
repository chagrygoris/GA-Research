"""Harmonics-aware GATr attention: GATr's attention score plus an inner product of spherical harmonics.

GATr (Brehmer et al. 2023, arXiv:2305.18415, Appendix B) computes, per head, the attention logit between tokens i and j
as a sum over the multivector channels c of three terms, divided by sqrt(12 n_mv + n_s):

    alpha_c <q_ic, k_jc>  +  beta_c phi(q_ic) . psi(k_jc)  +  scalar part q^s_i . k^s_j

where <., .> is the PGA inner product (the 8 blades without e0), phi / psi are the distance-aware features of the
trivector part (their product is minus the squared distance between the points the trivectors represent), n_mv / n_s
are the numbers of multivector / scalar channels per head and alpha, beta are positive learnable weights.

Harmonics-aware attention (--gatr_harmonic_attn_lmax L) adds, in the same dot-product form, a fourth term:

    sum_c sum_{l=l0..L} w_hcl < Y_l(u(q_ic)), Y_l(u(k_jc)) >

  * l0 (--gatr_harmonic_attn_lmin): 2 by default; 1 adds the degree-1 term <Y_1(u), Y_1(u')> = cos of the angle
    between the two directions (for unit u, u'). GATr's own inner product already contains the vector parts' dot
    product q_v . k_v, but unnormalised, so it mixes length and angle; the degree-1 term is the angle alone;

  * v(x) = (x_e1, x_e2, x_e3): the vector part of a query / key multivector (a PGA plane's normal direction; it rotates
    with the scene and does not change under translations);
  * u(x) = v / sqrt(|v|^2 + eps): its direction, shrunk smoothly to zero for short vectors (eps = 1e-3);
  * Y_l: the 2l + 1 real spherical harmonics of degree l as homogeneous polynomials, normalised so that |Y_l(u)| = 1
    for a unit u (e3nn, normalization="norm"). For unit u, u': <Y_l(u), Y_l(u')> = P_l(u . u') (addition theorem),
    the Legendre polynomial of the cosine of the angle between the two directions. Degree l responds to angular
    patterns with l-fold structure, which the plain inner product (degree 1: u . u') cannot resolve;
  * w_hcl = exp(log_w_hcl) > 0: a learnable weight per head h, channel c and degree l, initialised to 1.

The term shares GATr's normaliser 1 / sqrt(12 n_mv + n_s): GATr's own terms keep exactly their scale, so the default
(L = 0, no fourth term) is unchanged. Every factor is an O(3) invariant of the pair (rotating or reflecting both
directions leaves P_l(u . u') unchanged) and translation invariant, so GATr stays E(3) equivariant.

Implementation: a copy of gatr.primitives.attention.geometric_attention (gatr 1.4.2) with the harmonic features
concatenated to the query and key features, swapped into every block of a built GATr by enable_harmonic_attention.
"""
import math

import torch
from einops import rearrange
from torch import nn

_VECTOR_IDX = [2, 3, 4]   # e1, e2, e3 within the 16 PGA blades


def harmonic_degrees(lmax: int, lmin: int = 2):
    """The degrees of the fourth term: lmin..lmax (lmin 1 or 2)."""
    if int(lmin) not in (1, 2) or not int(lmin) <= int(lmax) <= 10:
        raise ValueError(f"harmonic attention needs lmin in (1, 2) and lmin <= lmax <= 10, got {lmin}, {lmax}")
    return list(range(int(lmin), int(lmax) + 1))


def harmonic_features(mv: torch.Tensor, lmax: int, eps: float = 1e-3, lmin: int = 2) -> torch.Tensor:
    """mv (..., 16) PGA multivectors -> (..., sum_l (2l+1)) features Y_l(u(v(mv))) for l = lmin..lmax."""
    from e3nn import o3
    v = mv[..., _VECTOR_IDX]
    u = v / torch.sqrt(v.square().sum(-1, keepdim=True) + eps)
    return o3.spherical_harmonics(harmonic_degrees(lmax, lmin), u, normalize=False, normalization="norm")


def harmonic_geometric_attention(q_mv, k_mv, v_mv, q_s, k_s, v_s, normalizer, weights, harmonic_weights, lmax,
                                 attn_mask=None, lmin=2):
    """gatr.primitives.attention.geometric_attention plus the harmonic term (module docstring).

    q_mv, k_mv (..., heads, items, C, 16), v_mv (..., heads, items, C_v, 16); q_s, k_s, v_s (..., heads, items, S);
    weights (heads, 1, C): GATr's distance-term weights; harmonic_weights (heads, 1, C, n_l), n_l = lmax - lmin + 1.
    Returns (..., heads, items, C_v, 16) and (..., heads, items, S_v), as GATr's function.
    """
    from gatr.primitives.attention import (_INNER_PRODUCT_WO_TRI_IDX, _TRIVECTOR_IDX, _build_dist_basis,
                                           _build_dist_vec, _sdpa_graph_breaking)
    from gatr.utils.tensors import to_nd
    bh_shape = q_mv.shape[:-3]
    q_mv, k_mv, v_mv = to_nd(q_mv, 5), to_nd(k_mv, 5), to_nd(v_mv, 5)
    q_s, k_s, v_s = to_nd(q_s, 4), to_nd(k_s, 4), to_nd(v_s, 4)
    if isinstance(attn_mask, torch.Tensor) and len(attn_mask.shape) > 2:
        attn_mask = attn_mask.view(-1, *attn_mask.shape[-3:])

    num_mv_channels_v, num_s_channels_v = v_mv.shape[-2], v_s.shape[-1]
    num_mv_channels_qk, num_s_channels_qk = q_mv.shape[-2], q_s.shape[-1]

    # GATr's distance-aware features of the trivector parts (unchanged).
    q_tri, k_tri = q_mv[..., _TRIVECTOR_IDX], k_mv[..., _TRIVECTOR_IDX]
    basis_q, basis_k = _build_dist_basis(q_tri.device, q_tri.dtype)
    q_dist = _build_dist_vec(q_tri, basis_q, normalizer)
    k_dist = _build_dist_vec(k_tri, basis_k, normalizer)
    if weights is not None:
        q_dist = q_dist * weights[..., None].to(q_dist.dtype)

    # The harmonic features: (B, H, N, C, sum_l (2l+1)); the query side carries the per-degree weights.
    q_harm = harmonic_features(q_mv, lmax, lmin=lmin).to(q_mv.dtype)
    k_harm = harmonic_features(k_mv, lmax, lmin=lmin).to(k_mv.dtype)
    sizes = torch.tensor([2 * l + 1 for l in harmonic_degrees(lmax, lmin)], device=q_mv.device)
    w = harmonic_weights.repeat_interleave(sizes, dim=-1)                 # (H, 1, C, sum_l (2l+1))
    q_harm = q_harm * w.to(q_harm.dtype)
    n_harm = q_harm.shape[-1]

    device, dtype = q_mv.device, q_mv.dtype
    num_channels_gatr = num_mv_channels_qk * (7 + 5) + num_s_channels_qk      # GATr's own query / key width
    num_channels_qk = num_channels_gatr + num_mv_channels_qk * n_harm
    num_channels_v = num_mv_channels_v * 16 + num_s_channels_v
    num_channels = max(num_channels_qk, num_channels_v)
    num_channels = 8 * -(-num_channels // 8)  # Ceil to multiple of 8

    def cat(mv, dist, harm, s):
        return torch.cat([rearrange(mv[..., _INNER_PRODUCT_WO_TRI_IDX], "... c x -> ... (c x)"),
                          rearrange(dist, "... c d -> ... (c d)"),
                          rearrange(harm, "... c d -> ... (c d)"),
                          s,
                          torch.zeros(*s.shape[:3], num_channels - num_channels_qk, device=device, dtype=dtype)], -1)

    q, k = cat(q_mv, q_dist, q_harm, q_s), cat(k_mv, k_dist, k_harm, k_s)
    v = torch.cat([rearrange(v_mv, "... c x -> ... (c x)"), v_s,
                   torch.zeros(*v_s.shape[:3], num_channels - num_channels_v, device=device, dtype=dtype)], -1)
    # SDPA divides by sqrt(num_channels); this keeps GATr's own normaliser 1 / sqrt(12 n_mv + n_s) for every term.
    k = k * math.sqrt(num_channels / num_channels_gatr)
    q, k, v_out = _sdpa_graph_breaking(q, k, v, attn_mask=attn_mask)

    v_out_mv = rearrange(v_out[..., : num_mv_channels_v * 16], "... (c x) -> ...  c x", x=16)
    v_out_s = v_out[..., num_mv_channels_v * 16: num_mv_channels_v * 16 + num_s_channels_v]
    return v_out_mv.view(*bh_shape, *v_out_mv.shape[-3:]), v_out_s.view(*bh_shape, *v_out_s.shape[-2:])


class HarmonicGeometricAttention(nn.Module):
    """Drop-in for gatr.layers.attention.attention.GeometricAttention with the harmonic term (module docstring)."""

    def __init__(self, base: nn.Module, lmax: int, lmin: int = 2):
        super().__init__()
        self.normalizer = base.normalizer
        self.log_weights = base.log_weights                              # GATr's own distance weights, kept
        heads, _, channels = base.log_weights.shape
        self.lmax, self.lmin = int(lmax), int(lmin)
        self.harmonic_log_weights = nn.Parameter(torch.zeros(heads, 1, channels, len(harmonic_degrees(lmax, lmin))))

    def forward(self, q_mv, k_mv, v_mv, q_s, k_s, v_s, attention_mask=None):
        return harmonic_geometric_attention(q_mv, k_mv, v_mv, q_s, k_s, v_s, normalizer=self.normalizer,
                                            weights=self.log_weights.exp(),
                                            harmonic_weights=self.harmonic_log_weights.exp(),
                                            lmax=self.lmax, attn_mask=attention_mask, lmin=self.lmin)


def enable_harmonic_attention(net: nn.Module, lmax: int, lmin: int = 2) -> nn.Module:
    """Swap the attention of every block of a built gatr.GATr for HarmonicGeometricAttention (degrees lmin..lmax)."""
    harmonic_degrees(lmax, lmin)
    for block in net.blocks:
        block.attention.attention = HarmonicGeometricAttention(block.attention.attention, lmax, lmin)
    return net
