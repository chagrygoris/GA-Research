"""I2S_Backbone: ResNet-50 / ConvNeXt-tiny backbone -> conv adapter -> GA head.

Supports Fourier (SO(3) grid logits), rotation-matrix, rotor and multivector-rotor outputs
and several GA head types. `ga_head_type="tralalero"` uses `MVLinearTralala`.
"""

from typing import List

import torch
import torch.nn as nn
import torchvision
from e3nn import o3
from image2sphere.so3_utils import flat_wigner, nearest_rotmat, so3_healpix_grid

from pose3d.geometry.quaternion import multivector_to_rotation_matrix, unit_quaternion_to_matrix
from pose3d.models._utils import move_unregistered_tensors_to_device, so3_num_fourier_coeffs
from pose3d.models.ga_layers import (
    MVLinearTralala,
    ReducedGeometricProductHead,
    ResidualGeometricProductHead,
    TransformerLikeMVHead,
)


class I2S_Backbone(nn.Module):
    def __init__(
        self,
        algebra,
        lmax: int = 6,
        rec_level: int = 3,
        hidden_dim: List = [32],
        temperature: float = 1.0,
        backbone_name: str = "resnet50",
        pretrained_backbone: bool = True,
        freeze_backbone: bool = True,
        use_positional_encoding: bool = True,
        output_mode: str = "auto",
        mv_per_position: int = 1,
        adapter_mid_channels: int = 0,
        adapter_high_channels: int = 0,
        adapter_output_size: int = 16,
        ga_head_type: str = "tralalero",
        ga_head_mixing_layer: str = "gp",
        ga_head_num_blocks: int = 2,
        ga_head_dropout: float = 0.0,
        ga_head_use_layer_norm: bool = False,
    ):
        super().__init__()
        self.algebra = algebra
        self.lmax = int(lmax)
        self.rec_level = int(rec_level)
        self.temperature = float(temperature)
        self.backbone_name = str(backbone_name).lower()
        self.hidden_dim = hidden_dim

        self._mv_dim = int(2**algebra.dim)
        self.mv_per_position = int(mv_per_position)
        if self.mv_per_position <= 0:
            raise ValueError("mv_per_position must be positive")

        self.conv_adapter_output = int(adapter_output_size)
        if self.conv_adapter_output <= 0:
            raise ValueError("adapter_output_size must be positive")
        self._n_mv = self.conv_adapter_output ** 2 * self.mv_per_position
        if self._n_mv > 4096:
            print(
                f"Warning: I2S_Backbone uses {self._n_mv} multivector tokens "
                f"(adapter_output_size={self.conv_adapter_output}, "
                f"mv_per_position={self.mv_per_position}). "
                "This may be memory intensive."
            )
        adapter_out_channels = self.mv_per_position * self._mv_dim
        auto_mid_channels = max(64, adapter_out_channels * 2)
        auto_high_channels = max(256, auto_mid_channels * 2)

        adapter_mid_channels = int(adapter_mid_channels)
        adapter_high_channels = int(adapter_high_channels)

        if adapter_mid_channels < -1:
            raise ValueError("adapter_mid_channels must be -1, 0, or a positive integer")

        if adapter_high_channels < -1:
            raise ValueError("adapter_high_channels must be -1, 0, or a positive integer")

        self.adapter_mid_channels = (
            auto_mid_channels
            if adapter_mid_channels == 0
            else adapter_mid_channels
        )

        self.adapter_high_channels = (
            auto_high_channels
            if adapter_high_channels == 0
            else adapter_high_channels
        )

        if 0 < self.adapter_mid_channels < adapter_out_channels:
            raise ValueError(
                "adapter_mid_channels must be -1, 0, or >= mv_per_position * mv_dim "
                f"({adapter_out_channels}), got {self.adapter_mid_channels}"
            )

        if (
            self.adapter_high_channels > 0
            and self.adapter_mid_channels > 0
            and self.adapter_high_channels < self.adapter_mid_channels
        ):
            raise ValueError(
                "adapter_high_channels must be -1, 0, or >= adapter_mid_channels "
                f"when both blocks are enabled, got high={self.adapter_high_channels}, "
                f"mid={self.adapter_mid_channels}"
            )

        if output_mode not in {"auto", "rotation_matrix", "fourier", "rotor", "multivector_rotor"}:
            raise ValueError("output_mode must be one of: auto, rotation_matrix, fourier, rotor, multivector_rotor")
        self.output_mode = output_mode
        if self.output_mode in {"rotor", "multivector_rotor"} and self._mv_dim != 8:
            raise ValueError(
                "I2S_Backbone rotor/multivector_rotor modes currently require Cl(3,0), "
                f"expected mv_dim=8, got mv_dim={self._mv_dim}."
            )

        self.backbone, backbone_channels = self._build_backbone(
            backbone_name=self.backbone_name,
            pretrained_backbone=pretrained_backbone,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        adapter_layers = []
        adapter_in_channels = backbone_channels

        if self.adapter_high_channels > 0:
            adapter_layers.extend(
                [
                    nn.Conv2d(
                        adapter_in_channels,
                        self.adapter_high_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.adapter_high_channels),
                    nn.SiLU(inplace=True),
                ]
            )
            adapter_in_channels = self.adapter_high_channels
        else:
            adapter_layers.append(nn.Identity())

        if self.adapter_mid_channels > 0:
            adapter_layers.extend(
                [
                    nn.Conv2d(
                        adapter_in_channels,
                        self.adapter_mid_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.adapter_mid_channels),
                    nn.SiLU(inplace=True),
                ]
            )
            adapter_in_channels = self.adapter_mid_channels
        else:
            adapter_layers.append(nn.Identity())

        adapter_layers.extend(
            [
                nn.Conv2d(
                    adapter_in_channels,
                    adapter_out_channels,
                    kernel_size=1,
                    bias=True,
                ),
                nn.AdaptiveAvgPool2d(
                    (self.conv_adapter_output, self.conv_adapter_output)
                ),
            ]
        )

        self.conv_adapter = nn.Sequential(*adapter_layers)

        self.use_positional_encoding = bool(use_positional_encoding)
        if self.use_positional_encoding:
            self.positional_embedding = nn.Parameter(torch.zeros(1, self._n_mv, self._mv_dim))
            nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        self.ga_head_type = str(ga_head_type).lower()
        self.ga_head_mixing_layer = str(ga_head_mixing_layer).lower()
        self.ga_head_num_blocks = int(ga_head_num_blocks)
        self.ga_head_dropout = float(ga_head_dropout)
        self.ga_head_use_layer_norm = bool(ga_head_use_layer_norm)

        if self.ga_head_num_blocks <= 0:
            raise ValueError("ga_head_num_blocks must be positive")

        if self.ga_head_dropout < 0.0 or self.ga_head_dropout >= 1.0:
            raise ValueError("ga_head_dropout must be in [0, 1)")

        self.num_coeffs = so3_num_fourier_coeffs(self.lmax)
        self.ga_head_fourier = self._build_ga_head(out_features=self.num_coeffs)
        self.ga_head_rotation = self._build_ga_head(out_features=9)
        self.ga_head_rotor = self._build_ga_head(out_features=4)
        self.ga_head_mv_rotor = self._build_ga_head(out_features=1)

        xyx = so3_healpix_grid(rec_level=self.rec_level)
        wign = flat_wigner(self.lmax, *xyx)
        self.register_buffer("so3_xyx", xyx, persistent=False)
        self.register_buffer("so3_wigner_T", wign.transpose(0, 1).contiguous(), persistent=False)
        self.register_buffer("so3_rotmats_cache", o3.angles_to_matrix(*self.so3_xyx), persistent=False)

    def _build_ga_head(self, out_features: int):
        if self.ga_head_type == "tralalero":
            return MVLinearTralala(
                algebra=self.algebra,
                in_features=self._n_mv,
                hidden_dim=self.hidden_dim,
                out_features=out_features,
            )
        if self.ga_head_type == "transformer_like":
            return TransformerLikeMVHead(
                algebra=self.algebra,
                in_features=self._n_mv,
                hidden_dim=self.hidden_dim,
                out_features=out_features,
            )
        if self.ga_head_type == "reduced":
            return ReducedGeometricProductHead(
                algebra=self.algebra,
                in_features=self._n_mv,
                hidden_dim=self.hidden_dim,
                out_features=out_features,
                mixing_layer=self.ga_head_mixing_layer,
            )
        if self.ga_head_type == "residual_gp":
            return ResidualGeometricProductHead(
                algebra=self.algebra,
                in_features=self._n_mv,
                hidden_dim=self.hidden_dim,
                out_features=out_features,
                num_blocks=self.ga_head_num_blocks,
                dropout=self.ga_head_dropout,
                use_layer_norm=self.ga_head_use_layer_norm,
            )
        raise ValueError(
            "Unsupported ga_head_type: "
            f"{self.ga_head_type}. Expected one of: "
            "tralalero, transformer_like, reduced, residual_gp"
        )

    def _build_backbone(self, backbone_name: str, pretrained_backbone: bool):
        if backbone_name == "resnet50":
            backbone_weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained_backbone else None
            resnet = torchvision.models.resnet50(weights=backbone_weights)
            backbone = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
            )
            return backbone, 2048

        if backbone_name == "convnext_tiny":
            backbone_weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained_backbone else None
            convnext = torchvision.models.convnext_tiny(weights=backbone_weights)
            return convnext.features, 768

        raise ValueError(f"Unsupported backbone_name: {backbone_name}")

    def _resolve_mode(self):
        if self.output_mode != "auto":
            return self.output_mode
        return "fourier"

    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.backbone(x)
        adapted = self.conv_adapter(fmap)

        b, c, h, w = adapted.shape

        expected_c = self.mv_per_position * self._mv_dim
        expected_h = self.conv_adapter_output
        expected_w = self.conv_adapter_output

        if (c, h, w) != (expected_c, expected_h, expected_w):
            raise RuntimeError(
                f"Expected adapted features [B, {expected_c}, {expected_h}, {expected_w}], "
                f"got [B, {c}, {h}, {w}]"
            )

        tokens = adapted.reshape(
            b,
            self.mv_per_position,
            self._mv_dim,
            h,
            w,
        )

        tokens = tokens.permute(0, 3, 4, 1, 2).reshape(
            b,
            h * w * self.mv_per_position,
            self._mv_dim,
        )

        if tokens.shape[1] != self._n_mv or tokens.shape[2] != self._mv_dim:
            raise RuntimeError(
                f"Expected tokens [B, {self._n_mv}, {self._mv_dim}], "
                f"got {list(tokens.shape)}"
            )

        if self.use_positional_encoding:
            tokens = tokens + self.positional_embedding

        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_extra_tensors_device", None) != x.device:
            move_unregistered_tensors_to_device(self, x.device)
            self._extra_tensors_device = x.device

        mode = self._resolve_mode()
        tokens = self._encode_tokens(x)
        if mode == "fourier":
            coeffs_mv = self.ga_head_fourier(tokens)
            coeffs = coeffs_mv[..., 0]
            logits = self.logits_on_grid(coeffs)
            logits = logits / max(self.temperature, 1e-8)
            return logits
        if mode == "rotation_matrix":
            rot_mv = self.ga_head_rotation(tokens)
            rot = rot_mv[..., 0].reshape(x.shape[0], 3, 3)
            return rot
        if mode == "rotor":
            rotor_mv = self.ga_head_rotor(tokens)
            rotor = rotor_mv[..., 0]
            rotor = rotor / rotor.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            return rotor
        if mode == "multivector_rotor":
            mv = self.ga_head_mv_rotor(tokens)
            mv = mv[:, 0, :]
            return mv
        raise ValueError(f"Unsupported mode: {mode}")

    def logits_on_grid(self, coeffs: torch.Tensor) -> torch.Tensor:
        if coeffs.dim() == 3:
            coeffs = coeffs.squeeze(1)
        return torch.matmul(coeffs, self.so3_wigner_T)

    @torch.no_grad()
    def probs_on_grid(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_rotmat(self, coeffs: torch.Tensor) -> torch.Tensor:
        probs = self.probs_on_grid(coeffs)
        idx = torch.argmax(probs, dim=-1)
        return idx

    @torch.no_grad()
    def predict(self, x):
        out = self.forward(x)
        mode = self._resolve_mode()

        if mode == "fourier":
            idx = self.predict_rotmat(out)
            return self.so3_rotmats_cache[idx]

        if mode == "rotation_matrix":
            return out

        if mode == "rotor":
            return unit_quaternion_to_matrix(out)

        if mode == "multivector_rotor":
            return multivector_to_rotation_matrix(out)

        raise ValueError(f"Unsupported mode: {mode}")

    @torch.no_grad()
    def get_nearest_idx(self, rot_gt: torch.Tensor):
        return nearest_rotmat(rot_gt, self.so3_rotmats_cache)
