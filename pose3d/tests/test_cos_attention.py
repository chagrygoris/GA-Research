"""optimist1938's scaled cosine attention (models/cos_attention.py) behind --gatr_attention (default "dot" = stock GATr)."""
import math

import pytest
import torch

from pose3d.models.gatr_denoiser import GATrCloudField, _ensure_xformers_stub
from pose3d.models.native_stf import NativeSTFCloudField

_ensure_xformers_stub()

BASE = ['--path_to_datasets=/tmp', '--model', 'clifford_flow', '--encoder', 'resnet50', '--dataset', 'symsol',
        '--vector_field', 'gatr', '--condition_head', 'none', '--cond_tokens', 'spherecloud', '--pose_tokens',
        'frame_pin', '--gatr_blocks', '1', '--gatr_mv_channels', '4', '--gatr_s_channels', '8', '--gatr_heads', '2',
        '--no-gatr_zero_readout', '--no-pretrained_backbone', '--n_time_samples', '1']


def _build(args):
    from pose3d.config import parse_args
    from pose3d.models import build_model
    from pose3d.train import make_algebra
    cfg = parse_args(args)
    torch.manual_seed(0)
    return build_model(cfg, make_algebra(cfg.model.algebra_dim))


def _scaled(model):
    from pose3d.models.cos_attention import ScaledAttention
    return [m for m in model.modules() if isinstance(m, ScaledAttention)]


def test_default_is_stock_gatr_attention():
    from pose3d.config import parse_args
    assert parse_args(BASE).flow.gatr_attention == 'dot'
    assert _scaled(_build(BASE)) == []


@pytest.mark.parametrize('mode', ['cosdir', 'cosmv'])
@pytest.mark.parametrize('native', [False, True])
def test_flag_wraps_every_block_and_trains(mode, native):
    extra = ['--gatr_native_stf', '--gatr_native_stf_lmax', '3', '--gatr_native_stf_channels', '2'] if native else []
    model = _build(BASE + extra + ['--gatr_attention', mode])
    attn = _scaled(model)
    assert len(attn) == 1 and attn[0].mode == mode and attn[0].log_scale.shape == (2,)
    assert torch.allclose(attn[0].log_scale.exp(), torch.full((2,), 30.0))
    assert isinstance(model.vector_field, NativeSTFCloudField) == native
    q, _ = torch.linalg.qr(torch.randn(2, 3, 3))
    loss = model.compute_loss(torch.randn(2, 3, 224, 224), q * torch.sign(torch.linalg.det(q))[:, None, None])
    loss.backward()
    assert torch.isfinite(loss) and attn[0].log_scale.grad.abs().sum() > 0


def test_dot_mode_reproduces_stock_gatr():
    torch.manual_seed(0)
    stock = GATrCloudField(9, mv_channels=4, s_channels=16, num_blocks=2, num_heads=2).eval()
    for p in stock.parameters():                                # a non-zero read-out (CliffordFlow zeroes it)
        if p.dim() > 1 and p.abs().max() == 0:
            torch.nn.init.normal_(p, std=.3)
    cos = GATrCloudField(9, mv_channels=4, s_channels=16, num_blocks=2, num_heads=2, attention='cosdir').eval()
    cos.load_state_dict(stock.state_dict(), strict=False)
    pose, t, cond = torch.randn(2, 3, 8), torch.rand(2), torch.randn(2, 7, 8 + 9)
    with torch.no_grad():
        a = stock(pose, t, cond)
        for m in cos.angular_attention:
            m.mode = 'dot'; m.log_scale.zero_()
        b = cos(pose, t, cond)
        for m in cos.angular_attention:
            m.mode = 'cosdir'; m.log_scale.fill_(math.log(30))
        c = cos(pose, t, cond)
    torch.testing.assert_close(b, a, atol=1e-5, rtol=1e-5)
    assert (c - a).abs().max() > 1e-4


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('reflection', [False, True])
def test_cosdir_fields_stay_physically_covariant(native, reflection):
    kw = dict(mv_channels=4, s_channels=16, num_blocks=2, num_heads=2, attention='cosdir')
    model = (NativeSTFCloudField(9, stf_channels=3, stf_lmax=6, **kw) if native else GATrCloudField(9, **kw)).eval()
    for p in model.parameters():
        if p.dim() > 1 and p.abs().max() == 0:
            torch.nn.init.normal_(p, std=.3)
    torch.manual_seed(18)
    r, _ = torch.linalg.qr(torch.randn(2, 3, 3))
    pose = torch.zeros(2, 3, 8); e1 = r[:, :, 0]
    pose[:, 0, 4:7] = torch.stack([e1[:, 2], -e1[:, 1], e1[:, 0]], -1)
    pose[:, 1, 1:4] = r[:, :, 1]; pose[:, 2, 1:4] = r[:, :, 2]
    cond = torch.randn(2, 7, 8 + 9); cond[..., :8] = 0
    cond[..., 1:4] = torch.nn.functional.normalize(torch.randn(2, 7, 3), dim=-1)
    t = torch.tensor([.25, .75])
    q, _ = torch.linalg.qr(torch.randn(3, 3)); q = q * torch.sign(torch.linalg.det(q))
    if reflection:
        q = q @ torch.diag(torch.tensor([-1., 1., 1.]))
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


def test_guards():
    with pytest.raises(ValueError):          # one attention replacement at a time
        _build(BASE + ['--gatr_attention', 'cosdir', '--gatr_harmonic_attn_lmax', '6'])
    rotor_layout = [{'spherecloud': 'pooled', 'frame_pin': 'rotor', 'none': 'clifford'}.get(a, a) for a in BASE]
    _build(rotor_layout)                     # the pooled / rotor GATr field builds without the flag ...
    with pytest.raises(ValueError):          # ... but has no cosine attention: only the frame_pin cloud fields do
        _build(rotor_layout + ['--gatr_attention', 'cosdir'])


def test_champion_recipe_builds():
    """The champion SYMSOL I recipe of pose3d/USAGE.md (model flags only) builds and trains."""
    args = ['--path_to_datasets=/tmp', '--model', 'clifford_flow', '--encoder', 'resnet50', '--dataset', 'symsol',
            '--vector_field', 'gatr', '--cond_tokens', 'spherecloud', '--pose_tokens', 'frame_pin', '--condition_head',
            'none', '--dircloud_scalars', '64', '--gatr_mv_channels', '8', '--gatr_s_channels', '32',
            '--no-gatr_zero_readout', '--gatr_heads', '8', '--gatr_attention', 'cosdir', '--no-pretrained_backbone',
            '--n_time_samples', '1']
    model = _build(args)
    attn = _scaled(model)
    assert len(attn) == 4 and all(a.mode == 'cosdir' and a.log_scale.shape == (8,) for a in attn)
    assert type(model.vector_field) is GATrCloudField
    q, _ = torch.linalg.qr(torch.randn(2, 3, 3))
    loss = model.compute_loss(torch.randn(2, 3, 224, 224), q * torch.sign(torch.linalg.det(q))[:, None, None])
    assert torch.isfinite(loss)
