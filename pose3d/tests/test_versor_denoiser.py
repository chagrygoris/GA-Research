"""Versor denoiser: the Cl(4,1) algebra, the layers, and CliffordFlow integration (no pretrained weights)."""

import itertools

import pytest
import torch

from pose3d.models.versor_denoiser import (_CL41_INDEX, GA_DIM, SIGNATURE, VersorLinear, VersorVectorField,
                                           geometric_product, manifold_normalization)


def _blade(i):
    out = torch.zeros(GA_DIM)
    out[i] = 1.0
    return out


def test_basis_vectors_square_to_the_signature_and_anticommute():
    vectors = [1 << b for b in range(5)]
    for b, i in enumerate(vectors):
        torch.testing.assert_close(geometric_product(_blade(i), _blade(i)), SIGNATURE[b] * _blade(0))
    for i, j in itertools.combinations(vectors, 2):
        ij, ji = geometric_product(_blade(i), _blade(j)), geometric_product(_blade(j), _blade(i))
        torch.testing.assert_close(ij, -ji)
        torch.testing.assert_close(ij, _blade(i | j))   # e_i e_j = e_ij for i < j


def test_geometric_product_is_associative():
    a, b, c = torch.randn(3, 5, GA_DIM).unbind(0)
    torch.testing.assert_close(geometric_product(geometric_product(a, b), c),
                               geometric_product(a, geometric_product(b, c)), atol=1e-4, rtol=1e-4)


def test_cl30_embedding_rotates_vectors_like_a_quaternion():
    # rotor exp(-theta/2 e12) turns e1 towards e2 by theta
    theta = 0.7
    r = torch.zeros(8)
    r[0], r[4] = torch.cos(torch.tensor(theta / 2)), -torch.sin(torch.tensor(theta / 2))
    r_rev = r.clone()
    r_rev[4:7] = -r_rev[4:7]

    def emb(x):
        out = torch.zeros(GA_DIM)
        out[list(_CL41_INDEX)] = x
        return out

    e1 = torch.zeros(8)
    e1[1] = 1.0
    turned = geometric_product(geometric_product(emb(r), emb(e1)), emb(r_rev))
    expected = torch.zeros(GA_DIM)
    expected[1], expected[2] = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
    torch.testing.assert_close(turned, expected, atol=1e-6, rtol=1e-6)


def test_linear_is_right_multiplication_by_the_weights():
    lin = VersorLinear(3, 2, normalize=False)
    x = torch.randn(4, 3, GA_DIM)
    expected = sum(geometric_product(x[:, k].unsqueeze(1), lin.weight[:, k]) for k in range(3))
    torch.testing.assert_close(lin(x), expected, atol=1e-5, rtol=1e-5)


def test_manifold_normalization_unit_norm_and_zero_stays_zero():
    x = torch.randn(6, GA_DIM)
    x[0] = 0.0
    y = manifold_normalization(x)
    assert y[0].abs().max() == 0   # as in the reference: eps keeps the denominator >= 1e-3
    from pose3d.models.versor_denoiser import _tables
    norm = (y[1:] ** 2 * _tables()[1]).sum(-1).abs()
    torch.testing.assert_close(norm, torch.ones(5), atol=1e-4, rtol=1e-4)


def test_vector_field_shape_zero_readout_gradients():
    torch.manual_seed(0)
    field = VersorVectorField(n_cond_mv=6, mv_channels=8, num_blocks=2, num_heads=2)
    torch.nn.init.zeros_(field.out.weight)
    x = torch.randn(3, 8, 8)
    out = field(x)
    assert out.shape == (3, 1, 8)
    assert out.abs().max() == 0
    assert out[:, :, [0, 1, 2, 3, 7]].abs().max() == 0   # only e12, e13, e23 are ever written
    # the zeroed read-out still gets a gradient, and through it everything upstream does after one step
    out[:, 0, 4:7].sum().backward()
    assert field.out.weight.grad.abs().sum() > 0
    with torch.no_grad():
        field.out.weight.normal_()
    field.zero_grad()
    field(x)[:, 0, 4:7].pow(2).sum().backward()
    for name, p in field.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert field.role.grad.abs().sum() > 0 and field.lift.weight.grad.abs().sum() > 0


def test_vector_field_depends_on_time_and_rotor():
    torch.manual_seed(0)
    field = VersorVectorField(n_cond_mv=4, mv_channels=8, num_blocks=1, num_heads=2)
    for block in field.blocks:   # LayerScale starts at 1e-5; open it so the blocks matter
        block.gamma1.data.fill_(1.0)
        block.gamma2.data.fill_(1.0)
    x = torch.randn(2, 6, 8)
    x_t, x_r = x.clone(), x.clone()
    x_t[:, 1, 0] += 0.5
    x_r[:, 0] = torch.randn(2, 8)
    assert not torch.allclose(field(x), field(x_t))
    assert not torch.allclose(field(x), field(x_r))


def _flow(vector_field):
    pytest.importorskip("clifford", reason="clifford (CGENN fork) not installed")
    from clifford.algebra.cliffordalgebra import CliffordAlgebra

    from pose3d.models.clifford_flow import CliffordFlow
    return CliffordFlow(CliffordAlgebra((1, 1, 1)), hidden_dim=[16], n_cond_mv=8, pretrained_backbone=False,
                        encoder_type="resnet50", conv_adapter=False, n_time_samples=2, vector_field=vector_field,
                        versor=dict(num_blocks=2, mv_channels=4, num_heads=2))


def test_flow_trains_and_samples():
    torch.manual_seed(0)
    model = _flow("versor")
    assert type(model.vector_field).__name__ == "VersorVectorField"
    img = torch.randn(2, 3, 64, 64)
    rot = torch.linalg.qr(torch.randn(2, 3, 3))[0]
    rot = rot * torch.linalg.det(rot).sign().view(-1, 1, 1)
    loss = model.compute_loss(img, rot)
    loss.backward()
    assert model.vector_field.out.weight.grad.abs().sum() > 0
    assert model.predict(img, n_samples=2, steps=3).shape == (2, 3, 3)
