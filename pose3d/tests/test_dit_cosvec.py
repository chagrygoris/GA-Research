"""CosVectorAttention (--dit_cos_vector): the logit is s_h * cos on the first 3 head dimensions + dot product on the rest."""
import math
import torch
from pose3d.models.dit_flow import CosVectorAttention


def test_cos_vector_attention_matches_its_formula():
    torch.manual_seed(0)
    att = CosVectorAttention(dim=28, heads=4, scale=30.0).double()
    x = torch.randn(2, 5, 28, dtype=torch.float64)
    out, _ = att(x, x, x)
    q, k, v = att.qkv(x).reshape(2, 5, 3, 4, 7).permute(2, 0, 3, 1, 4)
    cos = torch.nn.functional.normalize(q[..., :3], dim=-1) @ torch.nn.functional.normalize(k[..., :3], dim=-1).transpose(-1, -2)
    logits = 30.0 * cos + q[..., 3:] @ k[..., 3:].transpose(-1, -2) / math.sqrt(4)
    ref = att.proj((logits.softmax(-1) @ v).transpose(1, 2).reshape(2, 5, 28))
    torch.testing.assert_close(out, ref)


def test_cos_part_is_invariant_to_the_size_of_the_vector_part():
    """Scaling the first 3 dims of q leaves the cosine term unchanged (only the direction matters)."""
    torch.manual_seed(1)
    att = CosVectorAttention(dim=16, heads=2, scale=10.0).double()
    with torch.no_grad():
        att.qkv.weight[:16].zero_(); att.qkv.bias[:16].zero_()
        att.qkv.weight[0:3] = torch.randn(3, 16, dtype=torch.float64)          # head 0 query, vector part only
    x = torch.randn(1, 4, 16, dtype=torch.float64)
    out1, _ = att(x)
    with torch.no_grad():
        att.qkv.weight[0:3] *= 7.0
    out2, _ = att(x)
    torch.testing.assert_close(out1, out2)


def test_dit_flow_builds_with_cos_vector():
    from pose3d.models.dit_flow import DiTVectorField
    vf = DiTVectorField(dim=32, blocks=1, heads=4, cos_vector=True)
    assert isinstance(vf.blocks[0].attn, CosVectorAttention)
    R = torch.linalg.qr(torch.randn(3, 3, 3))[0]
    assert vf(R, torch.rand(3), torch.randn(3, 6, 32)).shape == (3, 3)
