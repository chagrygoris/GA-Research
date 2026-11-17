"""ATTENTION CHANGE (labelled, not a drop-in activation): the similarity inside GATr's ONE self-attention.

PROVENANCE: written by optimist1938 and shipped inside his Kaggle kernel arynabiaroza/symsol-gatr-spherecloud-cosdir-h8
(2026-11-14, the champion SYMSOL I run, pose3d/USAGE.md). This repository copy keeps his code verbatim up to and
including the FUSED path (fused=True: modes dot, cos, cosmv, cosdir, ktab), which is the path the champion run used.
His unfused reference path (fused=False) and the kcos / kadd modes, which have no fused form, were not recovered and
raise NotImplementedError here; replace_attention at the end is a short re-implementation (swap every GATr block's
attention for ScaledAttention). Enabled through FlowConfig.gatr_attention (cosdir / cosmv); default "dot" = stock GATr.

GATr:    logits(i, j) = [ <q_i, k_j> + distance features ] / sqrt(width)                      then softmax over j
here:    logits(i, j) = s_h * cos(q_i, k_j) + distance features / sqrt(width)               then softmax over j

q, k are the same queries and keys GATr builds (invariant blades of the multivector channels and the scalar
channels of one head); cos divides their inner product by both norms. s_h is one inverse temperature per head
and block, stored as a logarithm, learnable, capped at `max_scale`. This is scaled cosine attention as in
Swin Transformer V2 (cos / tau, tau learnable per head, tau >= 0.01, timm starts 1 / tau at 10) and
query-key normalisation (Henry et al. 2020: l2-normalised q and k times a learned scalar).

Why it is the pair kernel in one attention: softmax_j(s cos(theta_ij)) is a von Mises-Fisher kernel on the angle
between two tokens. It contains every Legendre degree (roughly up to sqrt(s) .. s) and has no cutoff; s -> inf
is "attend to the nearest token". No second token exchange, no extra projections: the only new parameters
are the temperatures (heads x blocks).

modes
  dot    GATr's own logits times s_h. s = 1, not learned, reproduces GATr exactly. Control: is it the
         temperature alone?
  cos    cosine of the whole query and key (multivector blades and scalars together).
  cosmv  cosine of the multivector parts only; the scalar channels keep GATr's dot product.
  kcos   logits = phi_h(cos_dir) + GATr's scalar and distance terms, then softmax. phi_h is a learned function of
         one variable per head (16 GELU units, slopes x `scale`, kinks spread over [-1, 1], as in pair-mlp-g30);
         cos_dir is the cosine of the vector and bivector blades only. phi = s * t would be `cosmv`: the
         softmax of a cosine is monotone ("nearest token"), a learned phi can peak at any angle.
  kadd   weights = softmax(GATr's logits) + phi_h(cos_dir) / tokens: GATr's attention unchanged plus a signed
         kernel term that shares its queries, keys, values and output projection. phi starts at zero.
  ktab   `kcos` with the same parameters and the same phi_h, but phi_h is evaluated on `knots` points of [-1, 1]
         once per forward pass (heads x knots x 16 GELUs) and each token pair reads it by linear interpolation:
         two table reads per pair instead of 16 GELUs. The result enters the fused call as an additive bias.

speed (`fused=True`)
  s * cos(q, k) is the dot product of unit vectors, so `dot`, `cos` and `cosmv` are GATr's own fused
  scaled-dot-product call on rescaled queries and keys: q <- s_h * sqrt(width) * q / |q|, k <- k / |k|. The cost
  over GATr is two normalisations per token (linear in tokens); nothing is computed per token pair in Python.
  `scale=(low, high)` starts the heads at log-spaced temperatures instead of one value: heads with different
  temperatures, mixed with signed weights by the output projection, give a learned kernel (a signed mixture of
  von Mises-Fisher kernels) at no cost per pair.
"""
import math

import torch
from torch import nn


class ScaledAttention(nn.Module):
    equivariant = True
    token_exchange = False     # no exchange is added; the existing attention is changed
    attention_change = True

    def __init__(self, old, mode="cos", scale=10.0, learn=True, max_scale=100.0, eps=1e-6, fused=False, knots=256):
        super().__init__()
        self.mode, self.max_scale, self.eps, self.fused = mode, max_scale, eps, fused or mode == "ktab"
        self.normalizer, self.log_weights = old.normalizer, old.log_weights
        heads = old.log_weights.shape[0]
        if isinstance(scale, (tuple, list)):    # one temperature per head, log-spaced from low to high
            start = torch.linspace(math.log(scale[0]), math.log(scale[1]), heads)
        else:
            start = torch.as_tensor(scale, dtype=torch.float32).expand(heads).clone().log()
        if learn:
            self.log_scale = nn.Parameter(start)
        else:
            self.register_buffer("log_scale", start)
        if mode in ("kcos", "kadd", "ktab"):    # the learned 1-D function; these draw random numbers
            hidden = 16
            self.log_scale.data.zero_()
            self.log_scale.requires_grad_(False)
            self.slope = nn.Parameter(torch.empty(heads, hidden).uniform_(-1, 1) * scale)
            self.shift = nn.Parameter(-self.slope.data * torch.empty(heads, hidden).uniform_(-1, 1))
            self.weight = nn.Parameter(torch.zeros(heads, hidden) if mode == "kadd"
                                       else torch.randn(heads, hidden) / math.sqrt(hidden))
            self.register_buffer("knots", torch.linspace(-1, 1, knots), persistent=False)
        self.description = f"attention similarity {mode}, inverse temperature starts at {scale}, learnable {learn}"
        self.last_entropy = None

    def unit(self, x):
        squared = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        return x * torch.rsqrt(squared.square().sum(-1, keepdim=True) + self.eps).to(x.dtype)

    def table(self, cosine):
        """phi_h(cosine) read from its values at the knots. cosine (B, heads, T, T)."""
        heads, count = self.slope.shape[0], len(self.knots)
        inner = self.slope[:, None, :] * self.knots[None, :, None] + self.shift[:, None, :]     # (heads, knots, 16)
        values = (torch.nn.functional.gelu(inner) * self.weight[:, None, :]).sum(-1)            # (heads, knots)
        position = (cosine + 1) * ((count - 1) / 2)
        with torch.no_grad():
            index = position.floor().clamp_(0, count - 2)
        part = (position - index).transpose(0, 1).reshape(heads, -1)
        index = index.long().transpose(0, 1).reshape(heads, -1)     # gather per head: its backward is a scatter-add
        low, high = values.gather(1, index), values.gather(1, index + 1)
        return (low + part * (high - low)).view(heads, -1, *cosine.shape[2:]).transpose(0, 1)

    def attend(self, q, k, v_mv, v_s, valid, bias=None):
        """torch's fused scaled-dot-product attention on prepared queries and keys (it divides by sqrt(width))."""
        heads, channels = q.shape[1], v_mv.shape[-2]
        v = torch.cat((v_mv.flatten(-2), v_s), -1).expand(-1, heads, -1, -1)
        mask = valid
        if bias is not None:
            mask = bias if valid is None else bias.masked_fill(~valid, float("-inf"))
        # Equal, 8-aligned Q/K/V widths allow SDPA to select a fused CUDA backend.
        # Explicit scaling preserves the logits when padding changes the width.
        width = q.shape[-1]
        value_width = v.shape[-1]
        padded_width = 8 * math.ceil(max(width, value_width) / 8)
        q = torch.nn.functional.pad(q, (0, padded_width - width))
        k = torch.nn.functional.pad(k, (0, padded_width - width)).expand(-1, heads, -1, -1)
        v = torch.nn.functional.pad(v, (0, padded_width - value_width))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                              scale=1 / math.sqrt(width))
        out = out[..., :value_width]
        return out[..., :channels * 16].unflatten(-1, (channels, 16)), out[..., channels * 16:]

    def forward(self, q_mv, k_mv, v_mv, q_s, k_s, v_s, attention_mask=None):
        from gatr.primitives.attention import (_INNER_PRODUCT_WO_TRI_IDX, _TRIVECTOR_IDX, _build_dist_basis,
                                               _build_dist_vec)

        # q_mv (B, heads, T, C, 16), k_mv and v_mv (B, 1, T, C, 16), q_s (B, heads, T, S), k_s and v_s (B, 1, T, S)
        basis_q, basis_k = _build_dist_basis(q_mv.device, q_mv.dtype)
        q_dist = _build_dist_vec(q_mv[..., _TRIVECTOR_IDX], basis_q, self.normalizer)
        q_dist = (q_dist * self.log_weights.exp()[..., None].to(q_dist.dtype)).flatten(-2)
        k_dist = _build_dist_vec(k_mv[..., _TRIVECTOR_IDX], basis_k, self.normalizer).flatten(-2)
        q_geo = q_mv[..., _INNER_PRODUCT_WO_TRI_IDX].flatten(-2)
        k_geo = k_mv[..., _INNER_PRODUCT_WO_TRI_IDX].flatten(-2)
        blades = _INNER_PRODUCT_WO_TRI_IDX[1:]                                         # vector and bivector
        root = math.sqrt(q_geo.shape[-1] + q_dist.shape[-1] + q_s.shape[-1])
        pair = lambda a, b: a @ b.transpose(-1, -2)                                    # (B, heads, T, T)
        scale = self.log_scale.clamp(max=math.log(self.max_scale)).exp().to(q_mv.dtype)[:, None, None]
        valid = None if attention_mask is None else attention_mask.view(-1, *attention_mask.shape[-3:])
        if self.fused:
            join = lambda *parts: torch.cat(parts, -1)
            if self.mode == "dot":
                q, k = scale * join(q_geo, q_dist, q_s), join(k_geo, k_dist, k_s)
            elif self.mode == "cos":
                q = join(scale * root * self.unit(join(q_geo, q_s)), q_dist)
                k = join(self.unit(join(k_geo, k_s)), k_dist)
            elif self.mode == "cosmv":
                q, k = join(scale * root * self.unit(q_geo), q_dist, q_s), join(self.unit(k_geo), k_dist, k_s)
            elif self.mode == "cosdir":
                # Cosine of the vector and bivector blades only: a true angle between directions, so the angular
                # gain is exactly s_h. The scalar blade joins the scalar channels in GATr's dot product.
                direction = lambda mv: self.unit(mv[..., blades].flatten(-2))
                q = join(scale * root * direction(q_mv), q_mv[..., 0], q_dist, q_s)
                k = join(direction(k_mv), k_mv[..., 0], k_dist, k_s)
            elif self.mode == "ktab":
                direction = lambda mv: self.unit(mv[..., blades].flatten(-2))
                bias = self.table(pair(direction(q_mv), direction(k_mv)))
                q, k = join(q_mv[..., 0], q_dist, q_s), join(k_mv[..., 0], k_dist, k_s)
                return self.attend(q * (math.sqrt(q.shape[-1]) / root), k, v_mv, v_s, valid, bias)
            else:
                raise ValueError(f"no fused form for {self.mode}")
            return self.attend(q, k, v_mv, v_s, valid)
        raise NotImplementedError(
            "this copy keeps only the fused path (fused=True, used by the champion run); the unfused reference path "
            "and the kcos / kadd modes were not recovered (module docstring)")


def replace_attention(model, **options):
    """Swap the attention of every block of a built gatr.GATr for ScaledAttention(old, **options); return the new modules.

    Re-implemented (his original body was not recovered): the same effect for a gatr.GATr, whose blocks hold their
    GeometricAttention at block.attention.attention.
    """
    modules = []
    for block in model.blocks:
        new = ScaledAttention(block.attention.attention, **options)
        block.attention.attention = new
        modules.append(new)
    return modules
