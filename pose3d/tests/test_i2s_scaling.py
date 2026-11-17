"""--i2s_grid 14 / --i2s_proj_rec_level 3: Image2Sphere with a 14x14 ResNet map and 400 projection points (upstream counts the equator ring: 104 at level 2)."""
import pytest
import torch

pytest.importorskip("image2sphere")
pytest.importorskip("healpy")


def test_i2s_real_with_a_14x14_map_and_denser_projection():
    from pose3d.models.i2s_real import I2SReal

    torch.manual_seed(0)
    base = I2SReal(encoder_type="resnet50", pretrained_backbone=False, lmax=4, rec_level=1, eval_rec_level=1)
    fine = I2SReal(encoder_type="resnet50", pretrained_backbone=False, lmax=4, rec_level=1, eval_rec_level=1,
                   grid=14, proj_rec_level=3)
    assert base.net.projector.n_pts == 104 and tuple(base.net.projector.weight.shape[2:4]) == (7, 7)
    assert fine.net.projector.n_pts == 400 and tuple(fine.net.projector.weight.shape[2:4]) == (14, 14)
    assert tuple(fine.net.encoder.output_shape) == (2048, 14, 14)
    img = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        assert fine.net.encoder(fine._prep(img)).shape == (2, 2048, 14, 14)
    R = torch.eye(3).expand(2, 3, 3).contiguous()
    loss = fine.compute_loss(img, R)
    assert torch.isfinite(loss)
    loss.backward()
