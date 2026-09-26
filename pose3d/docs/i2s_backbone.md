# I2S_Backbone (`--model i2s_backbone`, alias `i2s_resnet`)

Backbone -> conv adapter -> GA head, with the Image2Sphere-style SO(3) Fourier output as one of
several output modes. Implemented in `pose3d/models/i2s_backbone.py`, built by
`pose3d/models/__init__.py`.

## Shape flow (defaults)

1. Input image `[B, 3, H, W]`
2. Backbone feature map: ResNet-50 `[B, 2048, h, w]` (or ConvNeXt-tiny `[B, 768, h, w]`),
   frozen unless `--no-i2s_resnet_freeze_backbone`
3. Conv adapter + adaptive pool (`--i2s_resnet_adapter_output_size`, default 16):
   `[B, mv_per_position * 8, 16, 16]`
4. Flatten to tokens: `[B, 256, 8]` (256 = 16 x 16 x `mv_per_position`)
5. Learned positional embedding added (`--i2s_resnet_use_positional_encoding`)
6. GA head (`--i2s_resnet_ga_head_type`: `tralalero`, `transformer_like`, `reduced`, `residual_gp`)

## Output modes

`--i2s_resnet_output_mode auto` (default) picks the mode from `--loss`:

| Loss | Mode | Output |
|---|---|---|
| `mse`, `mse_ortho`, `geodesic` | `rotation_matrix` | `[B, 3, 3]` |
| `prob` | `fourier` | logits over the SO(3) HEALPix grid |
| `rotor` | `rotor` | unit quaternion `[B, 4]` (needs `--algebra_dim 3`) |
| `mv_rotor` | `multivector_rotor` | multivector `[B, 8]` (needs `--algebra_dim 3`) |

## Examples

```bash
python -m pose3d --path_to_datasets /path/to/data --model i2s_backbone --loss mse_ortho
python -m pose3d --path_to_datasets /path/to/data --model i2s_backbone --loss prob
```
