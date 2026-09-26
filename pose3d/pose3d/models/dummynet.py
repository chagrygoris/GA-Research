"""DummyNet / PoseNet: small point-cloud regressors for the ModelNet10 debugging dataset."""

import torch
import torch.nn as nn
from clifford.algebra.cliffordalgebra import CliffordAlgebra
from clifford.models.modules.linear import MVLinear

from pose3d.models.ga_layers import TralaleroTralala


class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.algebra = CliffordAlgebra((1, 1, 1))
        self.tralalero = TralaleroTralala(self.algebra, in_features=1, hidden_dim=[8, 16, 32], out_features=64)
        self.head = MVLinear(self.algebra, in_features=64, out_features=9)

    def forward(self, x : torch.tensor):
        '''
        Args:
            x : torch.tensor, point cloud of size (B, N, 3)
        Returns:
            x : rotation matrices of size (B, 3, 3)
        '''
        batch_size = x.shape[0]
        x = self.algebra.embed_grade(x, 1)
        x = x.reshape(-1, 1, 8)
        x = self.tralalero(x)
        x = x.reshape(batch_size, -1, 64, 8)
        x = x.max(dim=1).values
        x = self.head(x)
        x = self.algebra.get_grade(x, 0)
        # x = x.flatten(1, -1)
        # x = self.mlp(x)
        return x.reshape(-1, 3, 3)


class PoseNet(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: point cloud of shape (B, N, 3)
        Returns:
            rotation matrices of shape (B, 3, 3)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3 or x.size(-1) != 3:
            raise ValueError(f"Expected input of shape (B, N, 3), got {tuple(x.shape)}")

        feat = self.encoder(x)
        feat = feat.max(dim=1).values
        R = self.head(feat).view(-1, 3, 3)
        return R
