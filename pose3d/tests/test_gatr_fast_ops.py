"""Compile-friendly GATr primitives (pose3d.models.gatr_fast_ops) compute GATr 1.4's functions."""

import copy

import pytest
import torch

from pose3d.models.gatr_denoiser import _ensure_xformers_stub

_ensure_xformers_stub()
pytest.importorskip("gatr", reason="GATr package not installed")

D = torch.float64


@pytest.fixture
def fast_ops():
    from pose3d.models import gatr_fast_ops

    undo = gatr_fast_ops.install()
    yield gatr_fast_ops
    undo()


def _value_and_grads(fn, *inputs):
    inputs = [x.clone().requires_grad_() for x in inputs]
    out = fn(*inputs)
    weight = torch.randn(out.shape, generator=torch.Generator().manual_seed(1), dtype=out.dtype)
    grads = torch.autograd.grad((out * weight).sum(), inputs)
    return [out.detach(), *grads]


def _assert_same(fn, ref, *inputs):
    for a, b in zip(_value_and_grads(ref, *inputs), _value_and_grads(fn, *inputs)):
        torch.testing.assert_close(b, a, atol=1e-12, rtol=1e-12)


def test_unrolled_geometric_product_matches_gatr(fast_ops):
    from gatr.primitives.bilinear import geometric_product

    torch.manual_seed(0)
    x, y = torch.randn(5, 3, 16, dtype=D), torch.randn(5, 3, 16, dtype=D)
    _assert_same(fast_ops.geometric_product, geometric_product, x, y)
    _assert_same(fast_ops.geometric_product, geometric_product, x, y[:1])   # broadcasting


def test_unrolled_join_matches_gatr(fast_ops):
    from gatr.primitives.dual import equivariant_join

    torch.manual_seed(0)
    x, y, ref = torch.randn(5, 3, 16, dtype=D), torch.randn(5, 3, 16, dtype=D), torch.randn(5, 1, 16, dtype=D)
    _assert_same(fast_ops.equivariant_join, equivariant_join, x, y, ref)


def test_inner_product_on_slices_matches_gatr(fast_ops):
    from gatr.primitives.invariants import inner_product

    torch.manual_seed(0)
    _assert_same(fast_ops.inner_product, inner_product, torch.randn(5, 3, 16, dtype=D), torch.randn(5, 3, 16, dtype=D))


def _use_gatr_primitives(monkeypatch):
    """Point GATr's layers back at the functions where GATr defines them (gatr_fast_ops only
    rebinds the names the layers call), whatever an earlier --gatr_compile test installed."""
    import importlib

    from gatr.primitives.attention import geometric_attention
    from gatr.primitives.bilinear import geometric_product
    from gatr.primitives.dual import equivariant_join
    from gatr.primitives.invariants import inner_product

    def gated_forward(self, multivectors, scalars):  # GATr 1.4.2's ScalarGatedNonlinearity.forward
        gates = multivectors[..., [0]]
        return self.gated_nonlinearity(multivectors, gates=gates), self.scalar_nonlinearity(scalars)

    layers = {name: importlib.import_module(f"gatr.{name}") for name in (
        "layers.mlp.geometric_bilinears", "primitives.normalization", "layers.attention.attention",
        "layers.mlp.nonlinearities")}
    monkeypatch.setattr(layers["layers.mlp.geometric_bilinears"], "geometric_product", geometric_product)
    monkeypatch.setattr(layers["layers.mlp.geometric_bilinears"], "equivariant_join", equivariant_join)
    monkeypatch.setattr(layers["primitives.normalization"], "inner_product", inner_product)
    monkeypatch.setattr(layers["layers.attention.attention"], "geometric_attention", geometric_attention)
    monkeypatch.setattr(layers["layers.mlp.nonlinearities"].ScalarGatedNonlinearity, "forward", gated_forward)


@pytest.mark.parametrize("dense_linear", [False, True])
def test_vector_field_with_fast_ops_matches_gatr(dense_linear, monkeypatch):
    """The whole network (attention, gating and norms included) against GATr's own primitives."""
    from pose3d.models import gatr_fast_ops
    from pose3d.models.gatr_denoiser import GATrVectorField

    _use_gatr_primitives(monkeypatch)

    torch.manual_seed(0)
    field = GATrVectorField(8, mv_channels=4, s_channels=8, num_blocks=2, num_heads=2,
                            dense_linear=dense_linear).double()
    for p in field.parameters():
        torch.nn.init.normal_(p, std=0.3)
    x = torch.randn(3, 10, 8, dtype=D)

    def run():
        xx = x.clone().requires_grad_()
        field.zero_grad()
        out = field(xx)
        (out * torch.randn(out.shape, generator=torch.Generator().manual_seed(2), dtype=D)).sum().backward()
        return [out.detach(), xx.grad] + [torch.zeros_like(p) if p.grad is None else p.grad.clone()
                                          for p in field.parameters()]

    ref = run()
    undo = gatr_fast_ops.install()
    try:
        got = run()
    finally:
        undo()
    for a, b in zip(ref, got):
        torch.testing.assert_close(b, a, atol=1e-12, rtol=1e-12)


def test_compile_runs_on_the_fast_ops():
    import gatr.layers.mlp.geometric_bilinears as bilinears
    from pose3d.models import gatr_fast_ops
    from pose3d.models.gatr_denoiser import GATrVectorField

    GATrVectorField(8, mv_channels=4, s_channels=8, num_blocks=2, num_heads=2, compile=True)
    assert bilinears.geometric_product is gatr_fast_ops.geometric_product
    assert bilinears.equivariant_join is gatr_fast_ops.equivariant_join
