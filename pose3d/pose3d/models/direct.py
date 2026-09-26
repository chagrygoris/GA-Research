"""Direct-regression baselines: a backbone plus a linear or GA head predicting a 3x3 matrix."""

import torch
import torch.nn as nn

from pose3d.models.encoders import build_encoder
from pose3d.models.ga_layers import TralaleroTralala


def _ga_to_canonical_mv(mv_grid, mv_dim):
    if mv_grid.shape[1] == mv_dim:
        return mv_grid
    e, e123, e1, e2, e13, e23 = torch.unbind(mv_grid, dim=1)
    zeros = torch.zeros_like(e)
    return torch.stack([e, e1, e2, zeros, zeros, e13, e23, e123], dim=1)


class TralaleroCompetitor(nn.Module):
    def __init__(self, algebra, encoder_type: str = "resnet", ga_pool_hw: tuple = (28, 28),
                 pretrained_backbone: bool = False):
        super().__init__()
        self.algebra = algebra
        self._use_ga_backbone = encoder_type in {"ga", "ga_canonical"}
        self._mv_dim = int(2**algebra.dim)

        if self._use_ga_backbone:
            if len(ga_pool_hw) != 2:
                raise ValueError("ga_pool_hw must have exactly two elements: (height, width)")
            self.ga_pool_hw = (int(ga_pool_hw[0]), int(ga_pool_hw[1]))
            if self.ga_pool_hw[0] <= 0 or self.ga_pool_hw[1] <= 0:
                raise ValueError("ga_pool_hw values must be positive")

            self.pre_encode_pool = nn.AdaptiveAvgPool2d(self.ga_pool_hw)
            self.backbone = build_encoder(encoder_type, pretrained=pretrained_backbone)
            self._n_mv = int(self.ga_pool_hw[0] * self.ga_pool_hw[1])
        else:
            self._n_mv = 8
            self.backbone = build_encoder(encoder_type, pretrained=pretrained_backbone)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            enc_channels = getattr(self.backbone, "output_shape", None)[0]
            self.projective_matrix = nn.Linear(enc_channels, self._n_mv * self._mv_dim)

        self.ga_head = TralaleroTralala(algebra, in_features=self._n_mv)

    def forward(self, x):
        if self._use_ga_backbone:
            pooled_x = self.pre_encode_pool(x)
            mv_grid = self.backbone(pooled_x)
            mv_grid = _ga_to_canonical_mv(mv_grid, self._mv_dim)
            b, mv_dim, h, w = mv_grid.shape
            x = mv_grid.permute(0, 2, 3, 1).reshape(b, h * w, mv_dim)
        else:
            x = self.backbone(x)
            x = self.avgpool(x)
            x = x.flatten(1, -1)
            x = self.projective_matrix(x)
            x = x.reshape(x.shape[0], self._n_mv, self._mv_dim)

        x = self.ga_head(x)
        x = x[:, :, 0]
        x = x.reshape(x.shape[0], 3, 3)
        return x


class MLPBaseline(nn.Module):
    def __init__(self, encoder_type: str = "resnet", pretrained_backbone: bool = False):
        super().__init__()
        self.backbone = build_encoder(encoder_type, pretrained=pretrained_backbone)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        enc_channels = getattr(self.backbone, "output_shape", None)[0]
        self.linear_head = nn.Linear(in_features=enc_channels, out_features=9)


    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.flatten(1, -1)
        x = self.linear_head(x)
        x = x.reshape(x.shape[0], 3, 3)
        return x
