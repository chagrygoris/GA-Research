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

Two speed options compute the same function with the same parameters, so checkpoints load either
way: `dense_linear` runs every EquiLinear as one dense matmul (`use_dense_linears`), and `compile`
runs the forward through torch.compile. Timings: pose3d/reports/flash_clifford.
"""

import functools
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

# Cl(3,0) blade -> PGA Cl(3,0,1) blade, both in grade-major order:
#   1, e1, e2, e3, e12, e13, e23, e123  ->  1, e1..e3 (2-4), e12..e23 (8-10), e123 (14)
_PGA_INDEX = (0, 2, 3, 4, 8, 9, 10, 14)
_ROTATION_BIVECTOR = (8, 9, 10)  # e12, e13, e23 within the 16 PGA blades


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


@functools.lru_cache(maxsize=None)
def _dense_equilinear_class():
    from gatr.layers.linear import EquiLinear
    from gatr.primitives.linear import _compute_pin_equi_linear_basis

    class DenseEquiLinear(EquiLinear):
        """EquiLinear as ONE matmul over [multivector components, scalars]: same parameters, same map.

        GATr contracts the 9 Pin-equivariant basis maps with an einsum whose backward is slow (half
        of the vector field's training step). Here the coefficients are folded into a dense
        (16 out_mv + out_s, 16 in_mv + in_s) matrix on every call, so they stay trainable; the block
        layout is the one of gatr.utils.compile_linear.compile_equi_linear (frozen weights only).
        """

        def forward(self, multivectors, scalars=None):
            om, im, _ = self.weight.shape
            basis = _compute_pin_equi_linear_basis(self.weight.device, self.weight.dtype)
            mv_weight = torch.einsum("yxa,aij->yixj", self.weight, basis).reshape(16 * om, 16 * im)
            with_s = scalars is not None and self.s2mvs is not None
            n_s_out = self.mvs2s.out_features if self.mvs2s is not None else 0
            n_s_in = scalars.shape[-1] if with_s else 0

            weight = mv_weight.new_zeros(16 * om + n_s_out, 16 * im + n_s_in)
            bias = mv_weight.new_zeros(16 * om + n_s_out)
            weight[:16 * om, :16 * im] = mv_weight
            if self.bias is not None:  # exists only without input scalars; on the scalar components
                bias[:16 * om:16] = self.bias.flatten()
            if with_s:
                weight[:16 * om:16, 16 * im:] = self.s2mvs.weight
                if self.s2mvs.bias is not None:
                    bias[:16 * om:16] = self.s2mvs.bias
            if n_s_out:
                weight[16 * om:, :16 * im:16] = self.mvs2s.weight
                if self.mvs2s.bias is not None:
                    bias[16 * om:] = self.mvs2s.bias
                if self.s2s is not None and with_s:
                    weight[16 * om:, 16 * im:] = self.s2s.weight

            lead = multivectors.shape[:-2]
            inputs = multivectors.reshape(*lead, 16 * im)
            if with_s:
                inputs = torch.cat([inputs, scalars.expand(*lead, n_s_in)], dim=-1)
            out = F.linear(inputs, weight, bias)
            return out[..., :16 * om].reshape(*lead, om, 16), (out[..., 16 * om:] if n_s_out else None)

    return DenseEquiLinear


def use_dense_linears(module):
    """Swap every GATr EquiLinear in `module` (itself included) for DenseEquiLinear, in place.

    Only the class changes, so the parameters, state_dict keys and checkpoints stay the same."""
    from gatr.layers.linear import EquiLinear

    dense = _dense_equilinear_class()
    for m in module.modules():
        if type(m) is EquiLinear:
            m.__class__ = dense
    return module


def _compile(module):
    """module.compile(): unlike torch.compile(module) it keeps the state_dict keys unprefixed.

    GATr's cached einsum paths do not trace, so they are switched off (process-wide). Every GATr
    block, the EMA copy, train / eval mode and each batch shape get their own compiled frames,
    beyond torch's default limit of 8, after which dynamo silently runs the rest eager."""
    from gatr.utils.einsum import enable_cached_einsum

    enable_cached_einsum(False)
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
    module.compile()


class _Compilable(nn.Module):
    """The `compile` option, kept when the module is copied.

    nn.Module drops Module.compile()'s compiled call from its pickled / deep-copied state, so the
    EMA weights (a deepcopy, torch.optim.swa_utils.AveragedModel) would otherwise run eager."""

    def _set_compiled(self, compile: bool):
        self.compiled = bool(compile)
        if compile:
            _compile(self)

    def __setstate__(self, state):
        super().__setstate__(state)
        if self.__dict__.get("compiled"):
            _compile(self)


class GATrVectorField(_Compilable):
    """(B, 2 + n_cond_mv, 8) Cl(3,0) multivectors -> (B, 1, 8), like TralaleroTralala.

    Token 0 carries the noisy rotor R_t, tokens 1.. the conditioning multivectors (token 1 is
    the time embedding, which the flow builds as a scalar multivector). Scalar channels tell the
    tokens apart: the time t, broadcast to every token, and a query flag that is 1 on token 0.
    """

    def __init__(self, n_cond_mv: int, mv_channels: int = 8, s_channels: int = 32,
                 num_blocks: int = 4, num_heads: int = 4, dense_linear: bool = False,
                 compile: bool = False):
        super().__init__()
        GATr = _import_gatr()
        self.net = GATr(
            in_mv_channels=1, out_mv_channels=1, hidden_mv_channels=mv_channels,
            in_s_channels=2, out_s_channels=None, hidden_s_channels=s_channels,
            attention={"num_heads": num_heads}, mlp={}, num_blocks=num_blocks,
        )
        self.register_buffer("_pga_index", torch.tensor(_PGA_INDEX), persistent=False)
        self.register_buffer("_rot_index", torch.tensor(_ROTATION_BIVECTOR), persistent=False)
        self.out = self.net.linear_out  # zeroed by CliffordFlow so the initial velocity is 0
        if dense_linear:
            use_dense_linears(self.net)
        self._set_compiled(compile)

    def forward(self, x):
        b, n, _ = x.shape
        pga = x.new_zeros(b, n, 16)
        pga[..., self._pga_index] = x
        scalars = x.new_zeros(b, n, 2)
        scalars[..., 0] = x[:, 1, 0].unsqueeze(1)  # t: the scalar part of token 1
        scalars[:, 0, 1] = 1.0
        out_mv, _ = self.net(pga.unsqueeze(2), scalars=scalars)  # (B, n, 1, 16)
        query = out_mv[:, 0, 0]  # (B, 16)
        result = x.new_zeros(b, 1, x.shape[-1])
        result[:, 0, 4:7] = query[:, self._rot_index]  # e12, e13, e23 of Cl(3,0)
        return result


class GATrConditionHead(_Compilable):
    """(B, n_in, 8) Cl(3,0) multivectors -> (B, n_out, 8), like TralaleroTralala.

    The n_in inputs are tokens, followed by n_out learned query multivectors; the output is read
    from the query positions, which is how GATr (no cross-attention) maps n_in tokens to n_out.
    The inputs are chunks of the pooled backbone vector, so which chunk is which matters; GATr
    treats tokens as an unordered set, hence every token also gets a learned scalar embedding.
    """

    def __init__(self, n_in: int, n_out: int, mv_channels: int = 8, s_channels: int = 32,
                 num_blocks: int = 4, num_heads: int = 4, token_dim: int = 16,
                 dense_linear: bool = False, compile: bool = False):
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
        if dense_linear:
            use_dense_linears(self.net)
        self._set_compiled(compile)

    def forward(self, x):
        b = x.shape[0]
        tokens = torch.cat([x, self.query.to(x.dtype).expand(b, -1, -1)], dim=1)
        pga = x.new_zeros(b, self.n_in + self.n_out, 16)
        pga[..., self._pga_index] = tokens
        scalars = self.token_embedding.to(x.dtype).expand(b, -1, -1)
        out_mv, _ = self.net(pga.unsqueeze(2), scalars=scalars)  # (B, n_in + n_out, 1, 16)
        return out_mv[:, self.n_in:, 0][..., self._pga_index]
