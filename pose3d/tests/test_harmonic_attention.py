"""Harmonics-aware GATr attention (models/harmonic_attention.py)."""
import math

import pytest
import torch

from pose3d.models.gatr_denoiser import GATrCloudField, _ensure_xformers_stub
from pose3d.models.native_stf import NativeSTFCloudField

_ensure_xformers_stub()


def _rot(reflection=False):
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    q = q * torch.sign(torch.linalg.det(q))
    return q @ torch.diag(torch.tensor([-1., 1., 1.], dtype=torch.float64)) if reflection else q


def test_features_give_legendre_polynomials_and_are_o3_invariant():
    from pose3d.models.harmonic_attention import harmonic_degrees, harmonic_features
    torch.manual_seed(0)
    mv = torch.zeros(2, 5, 16, dtype=torch.float64)
    mv[..., 2:5] = torch.nn.functional.normalize(torch.randn(2, 5, 3, dtype=torch.float64), dim=-1) * 50
    a, b = harmonic_features(mv[0], 6, eps=0, lmin=1), harmonic_features(mv[1], 6, eps=0, lmin=1)
    cos = (torch.nn.functional.normalize(mv[0, :, 2:5], dim=-1) * torch.nn.functional.normalize(mv[1, :, 2:5], dim=-1)).sum(-1)
    start = 0
    for l in harmonic_degrees(6, lmin=1):                       # l = 1: P_1 = cos(theta)
        d = 2 * l + 1
        legendre = torch.special.legendre_polynomial_p(cos, l)
        torch.testing.assert_close((a[:, start:start + d] * b[:, start:start + d]).sum(-1), legendre)
        start += d
    for reflection in (False, True):
        q = _rot(reflection)
        moved = mv.clone(); moved[..., 2:5] = mv[..., 2:5] @ q.T
        ra, rb = harmonic_features(moved[0], 6), harmonic_features(moved[1], 6)
        fa, fb = harmonic_features(mv[0], 6), harmonic_features(mv[1], 6)
        start = 0
        for l in harmonic_degrees(6):
            d = 2 * l + 1
            torch.testing.assert_close((ra[:, start:start + d] * rb[:, start:start + d]).sum(-1),
                                       (fa[:, start:start + d] * fb[:, start:start + d]).sum(-1))
            start += d


def test_zero_harmonic_weight_reproduces_gatr_attention():
    from gatr.layers.attention.attention import GeometricAttention
    from gatr.layers.attention.config import SelfAttentionConfig
    from pose3d.models.harmonic_attention import HarmonicGeometricAttention
    torch.manual_seed(1)
    cfg = SelfAttentionConfig(in_mv_channels=4, out_mv_channels=4, in_s_channels=8, out_s_channels=8,
                              num_heads=2, additional_qk_mv_channels=0, additional_qk_s_channels=0)
    base = GeometricAttention(cfg)
    harm = HarmonicGeometricAttention(base, 6)
    c_mv, c_s = cfg.hidden_mv_channels, cfg.hidden_s_channels
    assert base.log_weights.shape == (2, 1, c_mv)
    q_mv, k_mv, v_mv = (torch.randn(2, 2, 7, c_mv, 16) for _ in range(3))
    q_s, k_s, v_s = (torch.randn(2, 2, 7, c_s) for _ in range(3))
    with torch.no_grad():
        harm.harmonic_log_weights.fill_(-float('inf'))          # w = 0: no harmonic term
        a = base(q_mv, k_mv, v_mv, q_s, k_s, v_s)
        b = harm(q_mv, k_mv, v_mv, q_s, k_s, v_s)
        harm.harmonic_log_weights.zero_()                       # w = 1: the term changes the attention
        c = harm(q_mv, k_mv, v_mv, q_s, k_s, v_s)
    torch.testing.assert_close(b[0], a[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(b[1], a[1], atol=1e-5, rtol=1e-5)
    assert (c[0] - a[0]).abs().max() > 1e-3


def _pin_inputs():
    torch.manual_seed(18)
    r, _ = torch.linalg.qr(torch.randn(2, 3, 3))
    pose = torch.zeros(2, 3, 8)
    e1 = r[:, :, 0]
    pose[:, 0, 4:7] = torch.stack([e1[:, 2], -e1[:, 1], e1[:, 0]], -1)
    pose[:, 1, 1:4] = r[:, :, 1]; pose[:, 2, 1:4] = r[:, :, 2]
    cond = torch.randn(2, 7, 8 + 9); cond[..., :8] = 0
    cond[..., 1:4] = torch.nn.functional.normalize(torch.randn(2, 7, 3), dim=-1)
    return r, pose, torch.tensor([.25, .75]), cond


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('reflection', [False, True])
def test_harmonic_fields_stay_physically_covariant(native, reflection):
    kw = dict(mv_channels=4, s_channels=16, num_blocks=2, num_heads=2, harmonic_attn_lmax=6)
    model = (NativeSTFCloudField(9, stf_channels=3, stf_lmax=6, **kw) if native else GATrCloudField(9, **kw)).eval()
    for p in model.parameters():                                # a non-zero read-out (CliffordFlow zeroes it)
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.3) if p.abs().max() == 0 else None
    r, pose, t, cond = _pin_inputs()
    q = _rot(reflection).float()
    rp = q @ r
    if reflection:
        rp = rp @ torch.diag(torch.tensor([-1., 1., 1.]))
    other = pose.clone(); e1 = rp[:, :, 0]
    other[:, 0, 4:7] = torch.stack([e1[:, 2], -e1[:, 1], e1[:, 0]], -1)
    other[:, 1, 1:4] = rp[:, :, 1]; other[:, 2, 1:4] = rp[:, :, 2]
    moved = cond.clone(); moved[..., 1:4] = cond[..., 1:4] @ q.T
    with torch.no_grad():
        a, b = model(pose, t, cond), model(other, t, moved)
    axial = torch.stack([a[:, 2], -a[:, 1], a[:, 0]], -1) @ q.T * torch.linalg.det(q)
    expected = torch.stack([axial[:, 2], -axial[:, 1], axial[:, 0]], -1)
    assert a.abs().max() > 1e-4
    torch.testing.assert_close(b, expected, atol=5e-5, rtol=1e-3)


@pytest.mark.parametrize('native', [False, True])
def test_launch_config_trains(native):
    from pose3d.config import parse_args
    from pose3d.models import build_model
    from pose3d.models.harmonic_attention import HarmonicGeometricAttention
    from pose3d.train import make_algebra
    args = ['--path_to_datasets=/tmp', '--model', 'clifford_flow', '--encoder', 'resnet50', '--dataset', 'symsol',
            '--vector_field', 'gatr', '--condition_head', 'none', '--cond_tokens', 'spherecloud', '--pose_tokens',
            'frame_pin', '--gatr_blocks', '1', '--gatr_mv_channels', '4', '--gatr_s_channels', '8', '--gatr_heads', '2',
            '--no-gatr_zero_readout', '--no-pretrained_backbone', '--n_time_samples', '1']
    if native:
        args += ['--gatr_native_stf', '--gatr_native_stf_lmax', '3', '--gatr_native_stf_channels', '2']
    assert parse_args(args).flow.gatr_harmonic_attn_lmax == 0
    cfg = parse_args(args + ['--gatr_harmonic_attn_lmax', '6'])
    torch.manual_seed(0)
    model = build_model(cfg, make_algebra(cfg.model.algebra_dim))
    attn = [m for m in model.modules() if isinstance(m, HarmonicGeometricAttention)]
    assert len(attn) == 1 and attn[0].harmonic_log_weights.shape == (2, 1, 4, 5)
    assert isinstance(model.vector_field, NativeSTFCloudField) == native
    q, _ = torch.linalg.qr(torch.randn(2, 3, 3))
    rot = q * torch.sign(torch.linalg.det(q))[:, None, None]
    loss = model.compute_loss(torch.randn(2, 3, 224, 224), rot)
    loss.backward()
    assert torch.isfinite(loss) and attn[0].harmonic_log_weights.grad.abs().sum() > 0


def test_lmin1_adds_the_cosine_term_and_lmin2_is_unchanged():
    """--gatr_harmonic_attn_lmin 1 adds <Y_1(u), Y_1(u')> = cos(theta); the default lmin 2 keeps degrees 2..L."""
    from pose3d.config import parse_args
    from pose3d.models import build_model
    from pose3d.models.harmonic_attention import HarmonicGeometricAttention, harmonic_features
    from pose3d.train import make_algebra
    mv = torch.zeros(4, 16, dtype=torch.float64)
    mv[:, 2:5] = torch.nn.functional.normalize(torch.randn(4, 3, dtype=torch.float64), dim=-1) * 50
    y1 = harmonic_features(mv, 1, eps=0, lmin=1)                   # degree 1 alone: the unit direction (e3nn axes)
    torch.testing.assert_close(y1 @ y1.T, torch.nn.functional.normalize(mv[:, 2:5], dim=-1) @
                               torch.nn.functional.normalize(mv[:, 2:5], dim=-1).T)
    args = ['--path_to_datasets=/tmp', '--model', 'clifford_flow', '--encoder', 'resnet50', '--dataset', 'symsol',
            '--vector_field', 'gatr', '--condition_head', 'none', '--cond_tokens', 'spherecloud', '--pose_tokens',
            'frame_pin', '--gatr_blocks', '1', '--gatr_mv_channels', '4', '--gatr_s_channels', '8', '--gatr_heads', '2',
            '--no-gatr_zero_readout', '--no-pretrained_backbone', '--n_time_samples', '1', '--gatr_harmonic_attn_lmax', '6']
    assert parse_args(args).flow.gatr_harmonic_attn_lmin == 2
    for lmin, n in ((2, 5), (1, 6)):
        cfg = parse_args(args + ['--gatr_harmonic_attn_lmin', str(lmin)])
        model = build_model(cfg, make_algebra(cfg.model.algebra_dim))
        attn = [m for m in model.modules() if isinstance(m, HarmonicGeometricAttention)]
        assert len(attn) == 1 and attn[0].lmin == lmin and attn[0].harmonic_log_weights.shape[-1] == n
