"""Spherical GATr (models/gatr_denoiser.SphericalActivation): the invariants are Legendre polynomials of the angles
between a token's vectors, the layer is O(3)-equivariant, it starts as plain GATr, and the whole P1 flow keeps its
turn and flip equivariance with the layer switched on."""

import math

import pytest
import torch

from pose3d.models.gatr_denoiser import SphericalActivation, _VECTOR
from tests.test_dircloud import FLIP, R_Z_MINUS_90, _flip, _rot90


def _legendre(x, lmax):
    p = [torch.ones_like(x), x]
    for l in range(1, lmax):
        p.append(((2 * l + 1) * x * p[l] - l * p[l - 1]) / (l + 1))
    return torch.stack(p[1:lmax + 1], -1)                                           # P_1 .. P_lmax


def test_invariants_are_legendre_polynomials_of_the_angles():
    torch.manual_seed(0)
    act = SphericalActivation(mv_channels=5, s_channels=7, lmax=6, n_vectors=4).double()
    vec = torch.randn(3, 11, 5, 3, dtype=torch.float64)
    feats, n = act.features(vec)
    j, k = act._pairs
    expected = _legendre((n[..., j, :] * n[..., k, :]).sum(-1), 6)                  # (3, 11, 6 pairs, 6 degrees)
    torch.testing.assert_close(feats[..., :36], expected.flatten(-2), atol=1e-12, rtol=1e-10)
    u = act.mix(vec.transpose(-1, -2)).transpose(-1, -2)
    torch.testing.assert_close(feats[..., 36:], torch.log1p(u.norm(dim=-1)))


@pytest.mark.parametrize("reflect", [False, True])
def test_activation_is_o3_equivariant(reflect):
    torch.manual_seed(1)
    act = SphericalActivation(mv_channels=5, s_channels=7, lmax=6, n_vectors=4).double()
    with torch.no_grad():
        act.mlp[-1].weight.normal_(0, 0.3); act.mlp[-1].bias.normal_(0, 0.3)       # switch the zero-init off
    mv = torch.randn(2, 9, 5, 16, dtype=torch.float64); s = torch.randn(2, 9, 7, dtype=torch.float64)
    Q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if reflect != bool(torch.det(Q) < 0):
        Q = Q @ torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    mv_q = mv.clone(); mv_q[..., list(_VECTOR)] = mv[..., list(_VECTOR)] @ Q.T
    out_mv, out_s = act(mv, s)
    q_mv, q_s = act(mv_q, s)
    torch.testing.assert_close(q_s, out_s)
    torch.testing.assert_close(q_mv[..., list(_VECTOR)], out_mv[..., list(_VECTOR)] @ Q.T)
    other = [i for i in range(16) if i not in _VECTOR]
    torch.testing.assert_close(q_mv[..., other], mv[..., other])                    # only the vectors are touched
    assert (out_s - s).abs().max() > 1e-3                                            # the layer does something


def _flow(spherical, active=True):
    from tests.test_dircloud import _flow as base_flow
    import tests.test_dircloud as td
    from pose3d.models.gatr_denoiser import _ensure_xformers_stub
    _ensure_xformers_stub()
    pytest.importorskip("gatr", reason="GATr package not installed")
    from clifford.algebra.cliffordalgebra import CliffordAlgebra
    from pose3d.models.clifford_flow import CliffordFlow
    torch.manual_seed(0)
    model = CliffordFlow(CliffordAlgebra((1, 1, 1)), hidden_dim=[8], pretrained_backbone=False,
                         encoder_type="resnet50", conv_adapter=False, n_time_samples=2,
                         vector_field="gatr", cond_tokens="dircloud", pose_tokens="frame_pin",
                         condition_head="none", dircloud_scalars=8, gatr_spherical_lmax=spherical,
                         gatr=dict(num_blocks=2, mv_channels=4, s_channels=8, num_heads=2)).double()
    with torch.no_grad():
        for p in model.vector_field.out.parameters():
            p.normal_(0, 0.5)
        if spherical and active:
            for act in model.vector_field.spherical:
                act.mlp[-1].weight.normal_(0, 0.3); act.mlp[-1].bias.normal_(0, 0.3)
    return model.eval()


def test_spherical_gatr_starts_as_plain_gatr():
    from tests.test_dircloud import _velocity_from_map
    from pose3d.geometry.rotor import random_rotor
    model = _flow(6, active=False)
    fmap = torch.randn(2, 2048, 7, 7, dtype=torch.float64); t = torch.tensor([0.3, 0.8], dtype=torch.float64)
    rotor = random_rotor(2).double()
    v_sph = _velocity_from_map(model, fmap, rotor, t)
    spherical, model.vector_field.spherical = model.vector_field.spherical, None
    v_plain = _velocity_from_map(model, fmap, rotor, t)
    model.vector_field.spherical = spherical
    torch.testing.assert_close(v_sph, v_plain, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("transform,G,sign", [(_rot90, R_Z_MINUS_90, (1.0, 1.0, 1.0)), (_flip, FLIP, (-1.0, -1.0, 1.0))])
def test_spherical_flow_keeps_turn_and_flip_equivariance(transform, G, sign):
    from tests.test_dircloud import _velocity_from_map
    from pose3d.geometry.rotor import matrix_to_rotor, random_rotor, rotor_to_matrix
    model = _flow(6)
    fmap = torch.randn(2, 2048, 7, 7, dtype=torch.float64)
    R = rotor_to_matrix(random_rotor(2).double(), model.algebra); t = torch.tensor([0.2, 0.9], dtype=torch.float64)
    v = _velocity_from_map(model, fmap, matrix_to_rotor(R), t)
    R_t = G @ R @ G if G is FLIP else G @ R
    v_t = _velocity_from_map(model, transform(fmap), matrix_to_rotor(R_t), t)
    assert v.abs().max() > 1e-3
    torch.testing.assert_close(v_t, torch.tensor(sign, dtype=torch.float64) * v, atol=1e-8, rtol=1e-6)


def test_spherical_flow_trains():
    from pose3d.geometry.rotor import random_rotor, rotor_to_matrix
    model = _flow(6).float().train()
    loss = model.compute_loss(torch.randn(2, 3, 224, 224), rotor_to_matrix(random_rotor(2), model.algebra))
    loss.backward()
    assert math.isfinite(loss.item())
    for act in model.vector_field.spherical:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in act.parameters())


def test_spherical_flag_needs_frame_pin():
    from pose3d.config import parse_args
    from pose3d.models import build_model
    from clifford.algebra.cliffordalgebra import CliffordAlgebra
    with pytest.raises(ValueError, match="frame_pin"):
        build_model(parse_args(["--path_to_datasets=/tmp", "--gatr_spherical_lmax", "6"]), CliffordAlgebra((1, 1, 1)))
