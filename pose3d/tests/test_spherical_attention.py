"""Spherical-attention GATr (models/gatr_denoiser.SphericalAttention): the equivariant variant is exactly
O(3)-equivariant (layer and whole P1 flow), both variants start as plain GATr, both train."""

import math

import pytest
import torch

from pose3d.models.gatr_denoiser import SphericalAttention, _VECTOR
from tests.test_dircloud import FLIP, R_Z_MINUS_90, _flip, _rot90


def _legendre(x, lmax):
    p = [torch.ones_like(x), x]
    for l in range(1, lmax):
        p.append(((2 * l + 1) * x * p[l] - l * p[l - 1]) / (l + 1))
    return torch.stack(p[1:lmax + 1], -1)                                           # P_1 .. P_lmax


def _switch_on(act):
    with torch.no_grad():
        act.mlp[-1].weight.normal_(0, 0.3); act.mlp[-1].bias.normal_(0, 0.3)


@pytest.mark.parametrize("reflect", [False, True])
def test_equivariant_layer_is_o3_equivariant(reflect):
    torch.manual_seed(1)
    act = SphericalAttention(mv_channels=5, s_channels=7, lmax=6, n_vectors=4, num_heads=3).double()
    _switch_on(act)
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
    torch.testing.assert_close(q_mv[..., other], mv[..., other])
    assert (out_s - s).abs().max() > 1e-3


def test_attention_mixes_tokens():
    """Token 0's update depends on the other tokens' directions (the point of the layer); per-token layers would not."""
    torch.manual_seed(2)
    for eq in (True, False):
        act = SphericalAttention(mv_channels=4, s_channels=6, lmax=6, n_vectors=2, equivariant=eq).double()
        _switch_on(act)
        mv = torch.randn(1, 5, 4, 16, dtype=torch.float64); s = torch.randn(1, 5, 6, dtype=torch.float64)
        other = mv.clone(); other[:, 1:, :, list(_VECTOR)] = torch.randn(1, 4, 4, 3, dtype=torch.float64)
        a, b = act(mv, s)[1][:, 0], act(other, s)[1][:, 0]
        assert (a - b).abs().max() > 1e-4, eq


def test_equivariant_scores_are_legendre_sums():
    """With one direction per token, score_ij = sum_l wq_l wk_l P_l(n_i . n_j) / sqrt(L + 1)."""
    torch.manual_seed(3)
    act = SphericalAttention(mv_channels=3, s_channels=4, lmax=6, n_vectors=1, num_heads=1).double()
    y, n = act.harmonics(torch.randn(1, 6, 3, 3, dtype=torch.float64))
    q, k = act._per_degree(act.wq, y), act._per_degree(act.wk, y)
    score = torch.einsum("bhie,bhje->bhij", q, k)[0, 0]
    cos = (n[0, :, 0] @ n[0, :, 0].T).clamp(-1, 1)
    p = torch.cat([torch.ones(6, 6, 1, dtype=torch.float64), _legendre(cos, 6)], -1)        # P_0 .. P_6
    w = (act.wq[0, :, 0] * act.wk[0, :, 0]).detach()
    torch.testing.assert_close(score, (p * w).sum(-1))


def _flow(eq, active=True):
    from pose3d.models.gatr_denoiser import _ensure_xformers_stub
    _ensure_xformers_stub()
    pytest.importorskip("gatr", reason="GATr package not installed")
    from clifford.algebra.cliffordalgebra import CliffordAlgebra
    from pose3d.models.clifford_flow import CliffordFlow
    torch.manual_seed(0)
    model = CliffordFlow(CliffordAlgebra((1, 1, 1)), hidden_dim=[8], pretrained_backbone=False,
                         encoder_type="resnet50", conv_adapter=False, n_time_samples=2,
                         vector_field="gatr", cond_tokens="dircloud", pose_tokens="frame_pin",
                         condition_head="none", dircloud_scalars=8, gatr_spherical_attn_lmax=6,
                         gatr_spherical_attn_equivariant=eq,
                         gatr=dict(num_blocks=2, mv_channels=4, s_channels=8, num_heads=2)).double()
    with torch.no_grad():
        for p in model.vector_field.out.parameters():
            p.normal_(0, 0.5)
        if active:
            for act in model.vector_field.spherical:
                _switch_on(act)
    return model.eval()


@pytest.mark.parametrize("eq", [True, False])
def test_flow_starts_as_plain_gatr(eq):
    from tests.test_dircloud import _velocity_from_map
    from pose3d.geometry.rotor import random_rotor
    model = _flow(eq, active=False)
    fmap = torch.randn(2, 2048, 7, 7, dtype=torch.float64); t = torch.tensor([0.3, 0.8], dtype=torch.float64)
    rotor = random_rotor(2).double()
    v_sph = _velocity_from_map(model, fmap, rotor, t)
    spherical, model.vector_field.spherical = model.vector_field.spherical, None
    v_plain = _velocity_from_map(model, fmap, rotor, t)
    model.vector_field.spherical = spherical
    torch.testing.assert_close(v_sph, v_plain, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("transform,G,sign", [(_rot90, R_Z_MINUS_90, (1.0, 1.0, 1.0)), (_flip, FLIP, (-1.0, -1.0, 1.0))])
def test_equivariant_flow_keeps_turn_and_flip_equivariance(transform, G, sign):
    from tests.test_dircloud import _velocity_from_map
    from pose3d.geometry.rotor import matrix_to_rotor, random_rotor, rotor_to_matrix
    model = _flow(True)
    fmap = torch.randn(2, 2048, 7, 7, dtype=torch.float64)
    R = rotor_to_matrix(random_rotor(2).double(), model.algebra); t = torch.tensor([0.2, 0.9], dtype=torch.float64)
    v = _velocity_from_map(model, fmap, matrix_to_rotor(R), t)
    R_t = G @ R @ G if G is FLIP else G @ R
    v_t = _velocity_from_map(model, transform(fmap), matrix_to_rotor(R_t), t)
    assert v.abs().max() > 1e-3
    torch.testing.assert_close(v_t, torch.tensor(sign, dtype=torch.float64) * v, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("eq", [True, False])
def test_flow_trains(eq):
    from pose3d.geometry.rotor import random_rotor, rotor_to_matrix
    model = _flow(eq).float().train()
    loss = model.compute_loss(torch.randn(2, 3, 224, 224), rotor_to_matrix(random_rotor(2), model.algebra))
    loss.backward()
    assert math.isfinite(loss.item())
    for act in model.vector_field.spherical:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in act.parameters())


def test_flags_reach_the_model():
    from pose3d.config import parse_args
    from pose3d.models import build_model
    from clifford.algebra.cliffordalgebra import CliffordAlgebra
    cfg = parse_args(["--path_to_datasets=/tmp", "--model", "clifford_flow", "--cond_tokens", "spherecloud",
                      "--pose_tokens", "frame_pin", "--condition_head", "none", "--gatr_spherical_attn_lmax", "6",
                      "--no-gatr_spherical_attn_equivariant", "--no-pretrained_backbone"])
    m = build_model(cfg, CliffordAlgebra((1, 1, 1)))
    assert len(m.vector_field.spherical) == 4 and not m.vector_field.spherical[0].equivariant
