"""Spherical GATr (models/gatr_denoiser.SphericalActivation): the features are the spherical harmonics of a token's
vector directions, the layer touches only scalars and vectors, it starts as plain GATr, and the flow trains. (The
layer is not equivariant, so the turn / flip tests of the plain field do not apply to it.)"""

import math

import pytest
import torch

from pose3d.models.gatr_denoiser import SphericalActivation, _VECTOR


def test_features_are_the_spherical_harmonics_of_the_directions():
    from e3nn import o3
    torch.manual_seed(0)
    act = SphericalActivation(mv_channels=5, s_channels=7, lmax=6, n_vectors=4).double()
    vec = torch.randn(3, 11, 5, 3, dtype=torch.float64)
    feats, n = act.features(vec)
    assert feats.shape == (3, 11, 4 * 48 + 4)                                       # (6 + 1)^2 - 1 = 48 per direction
    u = act.mix(vec.transpose(-1, -2)).transpose(-1, -2)
    torch.testing.assert_close(n, u / u.norm(dim=-1, keepdim=True))
    y = o3.spherical_harmonics(list(range(1, 7)), n, normalize=True, normalization="norm")
    torch.testing.assert_close(feats[..., :192], y.flatten(-2))
    torch.testing.assert_close(feats[..., 192:], torch.log1p(u.norm(dim=-1)))


def test_activation_updates_only_scalars_and_vectors():
    torch.manual_seed(1)
    act = SphericalActivation(mv_channels=5, s_channels=7, lmax=6, n_vectors=4).double()
    mv = torch.randn(2, 9, 5, 16, dtype=torch.float64); s = torch.randn(2, 9, 7, dtype=torch.float64)
    out_mv, out_s = act(mv, s)
    torch.testing.assert_close(out_mv, mv); torch.testing.assert_close(out_s, s)    # zero-init: the identity
    with torch.no_grad():
        act.mlp[-1].weight.normal_(0, 0.3); act.mlp[-1].bias.normal_(0, 0.3)
    out_mv, out_s = act(mv, s)
    other = [i for i in range(16) if i not in _VECTOR]
    torch.testing.assert_close(out_mv[..., other], mv[..., other])
    assert (out_s - s).abs().max() > 1e-3 and (out_mv[..., list(_VECTOR)] - mv[..., list(_VECTOR)]).abs().max() > 1e-3


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
