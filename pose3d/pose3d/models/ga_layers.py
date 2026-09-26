"""Geometric-algebra (Clifford) network stacks used as heads by every GA model.

TralaleroTralala is the CGENN-style MLP over multivectors used by CliffordFlow
(its condition head and vector field). The remaining heads are variants explored
by teammates on the ViT / I2S_Backbone lines.
"""

from typing import List, Union

import torch
import torch.nn as nn
from clifford.models.modules.fcgp import FullyConnectedSteerableGeometricProductLayer
from clifford.models.modules.gp import SteerableGeometricProductLayer
from clifford.models.modules.linear import MVLinear
from clifford.models.modules.mvlayernorm import MVLayerNorm
from clifford.models.modules.mvsilu import MVSiLU

from pose3d.models._utils import move_unregistered_tensors_to_device


class TralaleroTralala(nn.Module):
    def __init__(
        self,
        algebra,
        in_features: int = 512,
        hidden_dim: Union[int, List[int]] = 32,
        out_features: int = 9,
    ):
        super().__init__()

        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = list(hidden_dim)

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dim must be a non-empty int or List[int]")

        self.blocks = nn.ModuleList()
        prev = in_features

        for hd in hidden_dims:
            self.blocks.append(
                nn.ModuleDict({
                    "fc": FullyConnectedSteerableGeometricProductLayer(
                        algebra, in_features=prev, out_features=hd
                    ),
                    "act1": MVSiLU(algebra, hd),
                    "gp": SteerableGeometricProductLayer(algebra, hd),
                    "act2": MVSiLU(algebra, hd),
                })
            )
            prev = hd

        self.out = FullyConnectedSteerableGeometricProductLayer(
            algebra, in_features=prev, out_features=out_features
        )

    def forward(self, x):
        for b in self.blocks:
            x = b["fc"](x)
            x = b["act1"](x)
            x = b["gp"](x)
            x = b["act2"](x)
        x = self.out(x)
        return x


class MVLinearTralala(nn.Module):
    '''Variant of TralaleroTralala built from MVLinear + MVLayerNorm instead of the
    fully-connected geometric-product layers. Used by I2S_Backbone (ga_head_type=
    "tralalero") and the ViT baseline.'''

    def __init__(
        self,
        algebra,
        in_features: int = 512,
        hidden_dim: Union[int, List[int]] = 32,
        out_features: int = 9,
    ):
        super().__init__()

        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = list(hidden_dim)

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dim must be a non-empty int or List[int]")

        self.blocks = nn.ModuleList()
        prev = in_features

        for hd in hidden_dims:
            self.blocks.append(
                nn.ModuleDict({
                    "fc": MVLinear(
                        algebra, in_features=prev, out_features=hd
                    ),
                    "act1": MVSiLU(algebra, hd),
                    "gp": SteerableGeometricProductLayer(algebra, hd),
                    "ln": MVLayerNorm(algebra, hd),
                    "act2": MVSiLU(algebra, hd),
                })
            )
            prev = hd

        self.out = MVLinear(
            algebra, in_features=prev, out_features=out_features
        )


    def forward(self, x):
        for b in self.blocks:
            x = b["fc"](x)
            x = b["act1"](x)
            x = b["gp"](x)
            x = b["ln"](x)
            x = b["act2"](x)
        x = self.out(x)
        return x


class TransformerLikeMVBlock(nn.Module):
    def __init__(self, algebra, n_multivectors: int):
        super().__init__()
        if int(n_multivectors) <= 0:
            raise ValueError("n_multivectors must be positive")
        n_multivectors = int(n_multivectors)

        self.attn_like = FullyConnectedSteerableGeometricProductLayer(
            algebra,
            in_features=n_multivectors,
            out_features=n_multivectors,
        )
        self.up = MVLinear(algebra, n_multivectors, 4 * n_multivectors)
        self.act = MVSiLU(algebra, 4 * n_multivectors)
        self.down = MVLinear(algebra, 4 * n_multivectors, n_multivectors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.attn_like(x)
        x_ffn = self.up(attended)
        x_ffn = self.act(x_ffn)
        x_ffn = self.down(x_ffn)
        return attended + x_ffn


class TransformerLikeMVHead(nn.Module):
    def __init__(
        self,
        algebra,
        in_features: int = 512,
        hidden_dim: Union[int, List[int]] = 32,
        out_features: int = 9,
    ):
        super().__init__()

        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = list(hidden_dim)

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dim must be a non-empty int or List[int]")

        self.blocks = nn.ModuleList(
            [TransformerLikeMVBlock(algebra, n_multivectors=in_features) for _ in hidden_dims]
        )
        self.out = MVLinear(algebra, in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_extra_tensors_device", None) != x.device:
            move_unregistered_tensors_to_device(self, x.device)
            self._extra_tensors_device = x.device

        for block in self.blocks:
            x = block(x)
        return self.out(x)


class ResidualGeometricProductBlock(nn.Module):
    def __init__(
        self,
        algebra,
        n_multivectors: int,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        if int(n_multivectors) <= 0:
            raise ValueError("n_multivectors must be positive")

        self.algebra = algebra
        self.n_multivectors = int(n_multivectors)

        self.norm = nn.LayerNorm(2 ** algebra.dim) if use_layer_norm else nn.Identity()

        self.linear_in = MVLinear(algebra, self.n_multivectors, self.n_multivectors)
        self.act1 = MVSiLU(algebra, self.n_multivectors)
        self.gp = SteerableGeometricProductLayer(algebra, self.n_multivectors)
        self.act2 = MVSiLU(algebra, self.n_multivectors)
        self.linear_out = MVLinear(algebra, self.n_multivectors, self.n_multivectors)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        y = self.norm(x)
        y = self.linear_in(y)
        y = self.act1(y)
        y = self.gp(y)
        y = self.act2(y)
        y = self.linear_out(y)
        y = self.dropout(y)

        return residual + y


class ResidualGeometricProductHead(nn.Module):
    def __init__(
        self,
        algebra,
        in_features: int = 512,
        hidden_dim: Union[int, List[int]] = 32,
        out_features: int = 9,
        num_blocks: int = 2,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        self.algebra = algebra
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_blocks = int(num_blocks)

        if self.in_features <= 0:
            raise ValueError("in_features must be positive")

        if self.out_features <= 0:
            raise ValueError("out_features must be positive")

        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")

        # Keep hidden_dim only for API compatibility with TralaleroTralala.
        self.hidden_dim = hidden_dim

        self.blocks = nn.ModuleList(
            [
                ResidualGeometricProductBlock(
                    algebra=algebra,
                    n_multivectors=self.in_features,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.out = MVLinear(algebra, self.in_features, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_extra_tensors_device", None) != x.device:
            move_unregistered_tensors_to_device(self, x.device)
            self._extra_tensors_device = x.device

        for block in self.blocks:
            x = block(x)

        return self.out(x)


class ReducedGeometricProductHead(nn.Module):
    def __init__(
        self,
        algebra,
        in_features: int = 512,
        hidden_dim: Union[int, List[int]] = 32,
        out_features: int = 9,
        mixing_layer: str = "gp",
    ):
        super().__init__()

        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = list(hidden_dim)

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dim must be a non-empty int or List[int]")

        self.mixing_layer = str(mixing_layer).lower()
        if self.mixing_layer not in {"gp", "mvlinear", "linear"}:
            raise ValueError("mixing_layer must be one of: gp, mvlinear, linear")

        self.blocks = nn.ModuleList()
        prev = in_features

        for hd in hidden_dims:
            self.blocks.append(
                nn.ModuleDict({
                    "fc": FullyConnectedSteerableGeometricProductLayer(
                        algebra, in_features=prev, out_features=hd
                    ),
                    "act1": MVSiLU(algebra, hd),
                    "mix": self._build_mixing_layer(algebra=algebra, channels=hd),
                    "act2": MVSiLU(algebra, hd),
                })
            )
            prev = hd

        self.out = FullyConnectedSteerableGeometricProductLayer(
            algebra, in_features=prev, out_features=out_features
        )

    def _build_mixing_layer(self, algebra, channels: int):
        if self.mixing_layer == "gp":
            return SteerableGeometricProductLayer(algebra, channels)
        if self.mixing_layer == "mvlinear":
            return MVLinear(algebra, channels, channels)
        if self.mixing_layer == "linear":
            return nn.Linear(channels * (2**algebra.dim), channels * (2**algebra.dim))
        raise ValueError(f"Unsupported mixing_layer: {self.mixing_layer}")

    def _apply_mixing(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.mixing_layer == "linear":
            b, n, d = x.shape
            x = x.reshape(b, n * d)
            x = layer(x)
            return x.reshape(b, n, d)
        return layer(x)

    def forward(self, x):
        if getattr(self, "_extra_tensors_device", None) != x.device:
            move_unregistered_tensors_to_device(self, x.device)
            self._extra_tensors_device = x.device

        for b in self.blocks:
            x = b["fc"](x)
            x = b["act1"](x)
            x = self._apply_mixing(b["mix"], x)
            x = b["act2"](x)
        x = self.out(x)
        return x
