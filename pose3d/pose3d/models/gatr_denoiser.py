"""GATr (Geometric Algebra Transformer) as the Clifford Flow denoiser and condition head.

`GATrVectorField` is a drop-in replacement for the CGENN-style vector field of `CliffordFlow`:
same (B, 2 + n_cond_mv, 8) Cl(3,0) input, same (B, 8) output. The multivectors are embedded into
the projective algebra Cl(3,0,1) that GATr works in, the rotor / time / condition multivectors
become tokens of one sequence, and the velocity is read from the rotor token's rotation bivector.

`GATrConditionHead` replaces the condition head the same way: (B, n_in, 8) backbone multivectors
in, (B, n_out, 8) condition multivectors out.

https://github.com/Qualcomm-AI-research/geometric-algebra-transformer

The reference package hard-imports `xformers` (only used for optional attention masks, which
this model never passes). A stub is installed when it is missing, so torch is never replaced.
"""

import sys
import types

import torch
import torch.nn as nn

# Cl(3,0) blade -> PGA Cl(3,0,1) blade, both in grade-major order:
#   1, e1, e2, e3, e12, e13, e23, e123  ->  1, e1..e3 (2-4), e12..e23 (8-10), e123 (14)
_PGA_INDEX = (0, 2, 3, 4, 8, 9, 10, 14)
_ROTATION_BIVECTOR = (8, 9, 10)  # e12, e13, e23 within the 16 PGA blades
_VECTOR = (2, 3, 4)              # e1, e2, e3 within the 16 PGA blades


def _ensure_xformers_stub():
    try:
        import xformers.ops  # noqa: F401
        return
    except ImportError:
        pass

    class AttentionBias:  # never instantiated: no attention mask is ever passed
        pass

    def memory_efficient_attention(*args, **kwargs):
        raise RuntimeError("xformers is not installed; GATr attention masks are unsupported")

    ops = types.ModuleType("xformers.ops")
    ops.AttentionBias = AttentionBias
    ops.memory_efficient_attention = memory_efficient_attention
    pkg = types.ModuleType("xformers")
    pkg.ops = ops
    sys.modules["xformers"], sys.modules["xformers.ops"] = pkg, ops


def _import_gatr():
    _ensure_xformers_stub()
    try:
        from gatr import GATr
    except ImportError as e:
        raise ImportError(
            "--vector_field gatr needs the GATr package: pip install --no-deps einops opt_einsum "
            "git+https://github.com/Qualcomm-AI-research/geometric-algebra-transformer.git"
        ) from e
    return GATr


class GATrVectorField(nn.Module):
    """(B, 2 + n_cond_mv, 8) Cl(3,0) multivectors -> (B, 1, 8), like TralaleroTralala.

    Token 0 carries the noisy rotor R_t, tokens 1.. the conditioning multivectors (token 1 is
    the time embedding, which the flow builds as a scalar multivector). Scalar channels tell the
    tokens apart: the time t, broadcast to every token, and a query flag that is 1 on token 0.
    With n_pose_tokens=3 the pose is the frame R e1, R e2, R e3 on tokens 0-2 (time on token 3),
    each with its own flag so the network knows which axis is which. With pose_channels=3 the frame
    is instead three multivector channels of token 0 (input (B, n, 3, 8)), so per-token geometric
    products can combine R e_i with image directions attention brings in, from the first block.
    """

    def __init__(self, n_cond_mv: int, mv_channels: int = 8, s_channels: int = 32,
                 num_blocks: int = 4, num_heads: int = 4, n_pose_tokens: int = 1,
                 pose_channels: int = 1, extra_scalars: int = 0):
        super().__init__()
        self.extra_scalars = extra_scalars   # per-token scalars passed to forward (CliffordFlow's pose harmonics)
        # 1: the rotor token; 3: the frame R e1, R e2, R e3 (CliffordFlow pose_tokens='frame').
        # The time token follows the pose tokens; the query flag and the read-out stay on token 0.
        self.n_pose_tokens = n_pose_tokens
        self.pose_channels = pose_channels
        GATr = _import_gatr()
        self.net = GATr(
            in_mv_channels=pose_channels, out_mv_channels=1, hidden_mv_channels=mv_channels,
            in_s_channels=1 + n_pose_tokens + extra_scalars, out_s_channels=None, hidden_s_channels=s_channels,
            attention={"num_heads": num_heads}, mlp={}, num_blocks=num_blocks,
        )
        self.register_buffer("_pga_index", torch.tensor(_PGA_INDEX), persistent=False)
        self.register_buffer("_rot_index", torch.tensor(_ROTATION_BIVECTOR), persistent=False)
        self.register_buffer("_vec_index", torch.tensor(_VECTOR), persistent=False)
        self.out = self.net.linear_out  # zeroed by CliffordFlow so the initial velocity is 0

    def forward(self, x, extra=None):
        if x.dim() == 3:
            x = x.unsqueeze(2)                     # (B, n, channels, 8)
        b, n, c, _ = x.shape
        pga = x.new_zeros(b, n, c, 16)
        pga[..., self._pga_index] = x
        scalars = x.new_zeros(b, n, 1 + self.n_pose_tokens)
        scalars[..., 0] = x[:, self.n_pose_tokens, 0, 0].unsqueeze(1)  # t: scalar part of the time token
        for i in range(self.n_pose_tokens):
            scalars[:, i, 1 + i] = 1.0
        if self.extra_scalars:
            scalars = torch.cat([scalars, extra.to(scalars.dtype)], dim=-1)
        out_mv, _ = self.net(pga, scalars=scalars)  # (B, n, 1, 16)
        query = out_mv[:, 0, 0]  # (B, 16)
        result = x.new_zeros(b, 1, x.shape[-1])
        if self.n_pose_tokens == 1 and self.pose_channels == 1:
            result[:, 0, 4:7] = query[:, self._rot_index]  # e12, e13, e23 of Cl(3,0)
        else:
            # Frame tokens carry no bivector part, so at the zero-initialised read-out a bivector
            # read-out sees almost no signal and training stalls at a saddle. Read the vector part
            # instead (token 0 is R e1 itself) and take its dual e123 w, still a bivector under
            # rotations: e12 = w3, e13 = -w2, e23 = w1.
            w = query[:, self._vec_index]
            result[:, 0, 4:7] = torch.stack([w[:, 2], -w[:, 1], w[:, 0]], dim=-1)
        return result


class _NormThenLinear(nn.Module):
    """EquiLayerNorm, then GATr's EquiLinear read-out, with the read-out's calling convention."""

    def __init__(self, norm, linear):
        super().__init__()
        self.norm, self.linear = norm, linear

    def forward(self, multivectors, scalars=None):
        multivectors, scalars = self.norm(multivectors, scalars=scalars)
        return self.linear(multivectors, scalars=scalars)


class GATrCloudField(nn.Module):
    """Pose + time + a direction cloud -> the spatial angular velocity as a bivector (P1).

    Token 0 is the pose as three multivector channels with the right parity under reflections:
    l = (R e1) e123, a bivector (axial), then R e2 and R e3 (vectors, polar). Two polar axes and one
    axial axis transform under a reflection F exactly as the axes of F R F do, so with GATr's
    Pin(3,0,1) equivariance the flow of a left-right flipped image is the flipped flow. The
    velocity is read from the bivector part (e12, e13, e23) of token 0's output: angular velocity is
    axial, so a bivector is its honest type (the frame_ch vector-dual read-out passes rotation tests
    but fails the flip test). l carries bivector signal at the input, so the zero-initialised
    read-out still gets a gradient from step 0.

    Tokens 1.. are the condition cloud, packed (B, n, 8 + cond_scalars): a Cl(3,0) multivector in
    channel 0 and per-token scalar channels. Every token also gets the flow time t and a pose flag.
    """

    def __init__(self, cond_scalars: int, mv_channels: int = 8, s_channels: int = 32,
                 num_blocks: int = 4, num_heads: int = 4, final_norm: bool = False, cond_mv: int = 8,
                 pose_lmax: int = 0):
        super().__init__()
        GATr = _import_gatr()
        self.cond_scalars = cond_scalars
        # pose_lmax = L > 0: every condition token also gets the (L+1)^2 real spherical harmonics Y_lm(u), l <= L, of
        # its vector part expressed in the current pose's frame, u = R^T x / |R^T x| (49 numbers for L = 6). They
        # are invariant under turning input and pose together, so the field stays exactly equivariant, and they give
        # it every angular frequency up to L in the pose from the first step (reports/frequency_hypothesis_ru.md).
        self.pose_lmax = int(pose_lmax)
        n_harm = (self.pose_lmax + 1) ** 2 if self.pose_lmax else 0
        self.n_harm = n_harm
        cond_scalars = cond_scalars + n_harm
        self.cond_mv = cond_mv   # 8: Cl(3,0) condition multivectors (embedded); 16: already PGA (PointGATrEncoder)
        self.net = GATr(
            in_mv_channels=3, out_mv_channels=1, hidden_mv_channels=mv_channels,
            in_s_channels=cond_scalars + 2, out_s_channels=None, hidden_s_channels=s_channels,
            attention={"num_heads": num_heads}, mlp={}, num_blocks=num_blocks,
        )
        self.register_buffer("_pga_index", torch.tensor(_PGA_INDEX), persistent=False)
        self.register_buffer("_rot_index", torch.tensor(_ROTATION_BIVECTOR), persistent=False)
        if final_norm:
            # GATr goes from its last block straight to linear_out, so a token's output can be switched off by
            # shrinking its hidden state, which also shrinks its gradients. On SYMSOL P1 did exactly that to
            # the polyhedra and never recovered (reports/symsol_p1_stall.md); DiT's read-out is normalised
            # (norm_final). An equivariant LayerNorm before the read-out removes that route.
            from gatr.layers.layer_norm import EquiLayerNorm
            self.net.linear_out = _NormThenLinear(EquiLayerNorm(), self.net.linear_out)
            self.out = self.net.linear_out.linear
        else:
            self.out = self.net.linear_out  # zeroed by CliffordFlow so the initial velocity is 0

    def forward(self, pose, t, cond, R=None):
        """pose (B, 3, 8) Cl(3,0), t (B,), cond (B, n, 8 + cond_scalars) -> (B, 3) e12, e13, e23.
        R (B, 3, 3), the pose as a matrix, is needed with pose_lmax."""
        b, n = cond.shape[:2]
        if self.pose_lmax:
            from e3nn import o3
            vec = cond[..., 2:5] if self.cond_mv == 16 else cond[..., 1:4]      # the token's vector part
            y = torch.einsum("bji,bnj->bni", R.to(vec.dtype), vec)               # R^T x: the pose's own frame
            u = y / y.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            harm = o3.spherical_harmonics(list(range(self.pose_lmax + 1)), u, normalize=True,
                                          normalization="component")
            cond = torch.cat([cond, harm], dim=-1)
        mv = pose.new_zeros(b, 1 + n, 3, 16)
        mv[:, 0, :, self._pga_index] = pose
        if self.cond_mv == 16:
            mv[:, 1:, 0] = cond[..., :16]
        else:
            mv[:, 1:, 0, self._pga_index] = cond[..., :8]
        scalars = pose.new_zeros(b, 1 + n, self.cond_scalars + self.n_harm + 2)
        scalars[:, 1:, :self.cond_scalars + self.n_harm] = cond[..., self.cond_mv:]
        scalars[..., -2] = t.to(pose.dtype).unsqueeze(1)
        scalars[:, 0, -1] = 1.0
        out_mv, _ = self.net(mv, scalars=scalars)   # (B, 1 + n, 1, 16)
        return out_mv[:, 0, 0][:, self._rot_index]


class PointGATrEncoder(nn.Module):
    """A point cloud (B, N, 3) -> n_out condition tokens, GATr's own way (the 8.95 recipe's condition head on
    real geometry): every point is a token with two multivector channels, the point itself as a PGA trivector
    (gatr.interface.embed_point, how the GATr paper embeds positions) and x_i as a vector (the plane through the
    centre normal to x_i, so inner products see where the point sits relative to the centre), plus |x_i| as a
    scalar; n_out query tokens carry no geometry (zero multivectors) and a learned scalar embedding each. GATr runs
    over points + queries; the queries' outputs are the condition, packed (B, n_out, 16 + out_scalars). Queries
    and scalars are rotation invariant, so the encoder is exactly equivariant to turning the cloud.
    """

    def __init__(self, n_out: int = 32, mv_channels: int = 8, s_channels: int = 32, num_blocks: int = 2,
                 num_heads: int = 4, query_dim: int = 8, out_scalars: int = 8):
        super().__init__()
        GATr = _import_gatr()
        self.n_out, self.out_scalars = n_out, out_scalars
        self.net = GATr(
            in_mv_channels=2, out_mv_channels=1, hidden_mv_channels=mv_channels,
            in_s_channels=1 + query_dim, out_s_channels=out_scalars, hidden_s_channels=s_channels,
            attention={"num_heads": num_heads}, mlp={}, num_blocks=num_blocks,
        )
        self.query = nn.Parameter(torch.randn(n_out, query_dim))
        self.register_buffer("_vec_index", torch.tensor(_VECTOR), persistent=False)

    @property
    def width(self) -> int:
        return 16 + self.out_scalars

    def forward(self, points):                      # (B, N, 3)
        from gatr.interface import embed_point
        b, n, _ = points.shape
        mv = points.new_zeros(b, n + self.n_out, 2, 16)
        mv[:, :n, 0] = embed_point(points)
        mv[:, :n, 1, self._vec_index] = points
        scalars = points.new_zeros(b, n + self.n_out, self.query.shape[1] + 1)
        scalars[:, :n, 0] = points.norm(dim=-1)
        scalars[:, n:, 1:] = self.query.to(points.dtype)
        out_mv, out_s = self.net(mv, scalars=scalars)
        return torch.cat([out_mv[:, n:, 0], out_s[:, n:]], dim=-1)


class GATrConditionHead(nn.Module):
    """(B, n_in, 8) Cl(3,0) multivectors -> (B, n_out, 8), like TralaleroTralala.

    The n_in inputs are tokens, followed by n_out learned query multivectors; the output is read
    from the query positions, which is how GATr (no cross-attention) maps n_in tokens to n_out.
    The inputs are chunks of the pooled backbone vector, so which chunk is which matters; GATr
    treats tokens as an unordered set, hence every token also gets a learned scalar embedding.
    """

    def __init__(self, n_in: int, n_out: int, mv_channels: int = 8, s_channels: int = 32,
                 num_blocks: int = 4, num_heads: int = 4, token_dim: int = 16):
        super().__init__()
        GATr = _import_gatr()
        self.net = GATr(
            in_mv_channels=1, out_mv_channels=1, hidden_mv_channels=mv_channels,
            in_s_channels=token_dim, out_s_channels=None, hidden_s_channels=s_channels,
            attention={"num_heads": num_heads}, mlp={}, num_blocks=num_blocks,
        )
        self.n_in, self.n_out = n_in, n_out
        self.query = nn.Parameter(0.1 * torch.randn(n_out, 8))
        self.token_embedding = nn.Parameter(torch.randn(n_in + n_out, token_dim))
        self.register_buffer("_pga_index", torch.tensor(_PGA_INDEX), persistent=False)

    def forward(self, x):
        b = x.shape[0]
        tokens = torch.cat([x, self.query.to(x.dtype).expand(b, -1, -1)], dim=1)
        pga = x.new_zeros(b, self.n_in + self.n_out, 16)
        pga[..., self._pga_index] = tokens
        scalars = self.token_embedding.to(x.dtype).expand(b, -1, -1)
        out_mv, _ = self.net(pga.unsqueeze(2), scalars=scalars)  # (B, n_in + n_out, 1, 16)
        return out_mv[:, self.n_in:, 0][..., self._pga_index]
