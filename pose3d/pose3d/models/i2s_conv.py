"""Earlier I2S-with-GA-head line from the `conv_adapter` branch.

Alternative conv adapters (`conv`, `mlp_block`, `linear`, `geometric`, `inc`), an MLP head
option and an extra `vector_proj` output mode, on a ResNet-50 or ConvNeXt backbone.
Superseded in most respects by `i2s_backbone.I2S_Backbone`; kept because the
ConvNeXt + GA head run reached 10.64 deg on Pascal3D+.
"""

from typing import List, Union

import torch
import torch.nn as nn
import torchvision
from e3nn import o3
from image2sphere.so3_utils import flat_wigner, nearest_rotmat, so3_healpix_grid

from pose3d.geometry.quaternion import multivector_to_rotation_matrix
from pose3d.models._utils import move_unregistered_tensors_to_device, so3_num_fourier_coeffs
from pose3d.models.encoders import ConvNextEncoder
from pose3d.models.ga_layers import TralaleroTralala


class _CliffordInteraction(nn.Module):
    def __init__(self, dim: int, shifts=(1, 2, 4)):
        super().__init__()
        self.shifts = list(shifts)
        self.act = nn.SiLU()
        self.proj = nn.Conv2d(dim * len(self.shifts) * 2, dim, kernel_size=1)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        feats = []
        for s in self.shifts:
            z2_s = torch.roll(z2, shifts=s, dims=1)
            z1_s = torch.roll(z1, shifts=s, dims=1)
            feats.append(z1 * z2_s - z2 * z1_s)       # wedge product
            feats.append(self.act(z1 * z2_s))           # inner product
        return self.proj(torch.cat(feats, dim=1))


class _VectorIncAdapter(nn.Module):
    def __init__(self, in_ch: int, out_hw: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((out_hw, out_hw))
        self.fc = nn.Linear(in_ch, 3)   # project to the 3 grade-1 components

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        B, C, H, W = x.shape
        v = self.fc(x.permute(0, 2, 3, 1).reshape(B * H * W, C))   # [B*HW, 3]
        mv = torch.zeros(B * H * W, 8, device=x.device, dtype=x.dtype)
        mv[:, 1:4] = v                  # embed into e1,e2,e3 slots
        return mv.reshape(B, H, W, 8).permute(0, 3, 1, 2).contiguous()


class _GeometricAdapter(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, out_hw: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((out_hw, out_hw))
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.get_state   = nn.Conv2d(out_ch, out_ch, kernel_size=1)
        self.get_context = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch)
        self.interaction = _CliffordInteraction(out_ch, shifts=(1, 2, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.proj(x)
        x = x + self.interaction(self.get_state(x), self.get_context(x))
        return x


class _MLPBlockAdapter(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, out_hw: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((out_hw, out_hw))
        self.fc1 = nn.Linear(in_ch, 256)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(256, out_ch)
        self._out_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        x = self.fc2(self.act(self.fc1(x)))
        return x.reshape(B, H, W, self._out_ch).permute(0, 3, 1, 2).contiguous()


class _LinearAdapter(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, out_hw: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((out_hw, out_hw))
        self.fc = nn.Linear(in_ch, out_ch)
        self._out_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        x = self.fc(x)
        return x.reshape(B, H, W, self._out_ch).permute(0, 3, 1, 2).contiguous()


class MLPHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        mv_dim: int,
        out_features: int,
        hidden_dim: Union[int, List[int]],
    ):
        super().__init__()
        in_size = in_features * mv_dim
        dims = hidden_dim if isinstance(hidden_dim, list) else [int(hidden_dim)]
        layers = []
        prev = in_size
        for hd in dims:
            layers += [nn.Linear(prev, hd), nn.ReLU()]
            prev = hd
        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        out = self.net(x.reshape(B, -1))  # [B, out_features]
        return out.unsqueeze(-1)           # [B, out_features, 1]


class _I2S_ConvBase(nn.Module):
    """Shared backbone-agnostic logic for I2SConvResNet and I2SConvNext."""

    def __init__(
        self,
        algebra,
        backbone: nn.Module,
        backbone_channels: int,
        lmax: int = 6,
        rec_level: int = 3,
        hidden_dim: List = [32],
        temperature: float = 1.0,
        use_positional_encoding: bool = True,
        output_mode: str = "auto",
        adapter_type: str = "conv",
        head_type: str = "ga",
    ):
        super().__init__()
        self.algebra = algebra
        self.lmax = int(lmax)
        self.rec_level = int(rec_level)
        self.temperature = float(temperature)

        self._mv_dim = int(2**algebra.dim)
        self.conv_adapter_output = 16
        self._n_mv = self.conv_adapter_output**2
        if self._mv_dim != 8:
            raise ValueError(f"Expects mv_dim=8, got {self._mv_dim}")

        if output_mode not in {"auto", "rotation_matrix", "fourier", "rotor", "vector_proj"}:
            raise ValueError("output_mode must be one of: auto, rotation_matrix, fourier, rotor, vector_proj")
        self.output_mode = output_mode

        self.backbone = backbone

        if adapter_type == "mlp_block":
            self.conv_adapter = _MLPBlockAdapter(backbone_channels, self._mv_dim, self.conv_adapter_output)
        elif adapter_type == "linear":
            self.conv_adapter = _LinearAdapter(backbone_channels, self._mv_dim, self.conv_adapter_output)
        elif adapter_type == "geometric":
            self.conv_adapter = _GeometricAdapter(backbone_channels, self._mv_dim, self.conv_adapter_output)
        elif adapter_type == "inc":
            self.conv_adapter = _VectorIncAdapter(backbone_channels, self.conv_adapter_output)
        else:
            self.conv_adapter = nn.Sequential(
                nn.Conv2d(backbone_channels, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.SiLU(inplace=True),
                nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, self._mv_dim, kernel_size=1, bias=True),
                nn.AdaptiveAvgPool2d((self.conv_adapter_output, self.conv_adapter_output)),
            )

        self.use_positional_encoding = bool(use_positional_encoding)
        if self.use_positional_encoding:
            self.positional_embedding = nn.Parameter(torch.zeros(1, self._n_mv, self._mv_dim))
            nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        self.num_coeffs = so3_num_fourier_coeffs(self.lmax)
        if head_type == "mlp":
            self.ga_head_fourier = MLPHead(
                in_features=self._n_mv, mv_dim=self._mv_dim,
                out_features=self.num_coeffs, hidden_dim=hidden_dim,
            )
            self.ga_head_rotation = MLPHead(
                in_features=self._n_mv, mv_dim=self._mv_dim,
                out_features=9, hidden_dim=hidden_dim,
            )
        else:
            self.ga_head_fourier = TralaleroTralala(
                algebra=algebra, in_features=self._n_mv,
                hidden_dim=hidden_dim, out_features=self.num_coeffs,
            )
            self.ga_head_rotation = TralaleroTralala(
                algebra=algebra, in_features=self._n_mv,
                hidden_dim=hidden_dim, out_features=9,
            )

        if output_mode == "rotor":
            self.ga_head_rotor = TralaleroTralala(
                algebra=algebra, in_features=self._n_mv,
                hidden_dim=hidden_dim, out_features=1,
            )

        if output_mode == "vector_proj":
            self.grade1_proj = nn.Linear(self._mv_dim, 3)
            self.ga_head_vector_proj = TralaleroTralala(
                algebra=algebra, in_features=self._n_mv,
                hidden_dim=hidden_dim, out_features=9,
            )
            hd = hidden_dim[0] if isinstance(hidden_dim, list) else int(hidden_dim)
            self.vector_proj_mlp = nn.Sequential(
                nn.Linear(9 * 3, hd),
                nn.ReLU(),
                nn.Linear(hd, 9),
            )

        xyx = so3_healpix_grid(rec_level=self.rec_level)
        wign = flat_wigner(self.lmax, *xyx)
        self.register_buffer("so3_xyx", xyx, persistent=False)
        self.register_buffer("so3_wigner_T", wign.transpose(0, 1).contiguous(), persistent=False)
        self.register_buffer("so3_rotmats_cache", o3.angles_to_matrix(*self.so3_xyx), persistent=False)

    def _resolve_mode(self):
        if self.output_mode != "auto":
            return self.output_mode
        return "fourier"

    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.backbone(x)
        adapted = self.conv_adapter(fmap)
        b, c, h, w = adapted.shape
        if (c, h, w) != (self._mv_dim, self.conv_adapter_output, self.conv_adapter_output):
            raise RuntimeError(
                f"Expected adapted features [B, {self._mv_dim}, {self.conv_adapter_output}, "
                f"{self.conv_adapter_output}], got [B, {c}, {h}, {w}]"
            )
        tokens = adapted.flatten(2).transpose(1, 2)
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
            return logits / max(self.temperature, 1e-8)
        if mode == "rotation_matrix":
            rot_mv = self.ga_head_rotation(tokens)
            return rot_mv[..., 0].reshape(x.shape[0], 3, 3)
        if mode == "rotor":
            return self.ga_head_rotor(tokens).squeeze(1)
        if mode == "vector_proj":
            v = self.grade1_proj(tokens)
            tokens_g1 = torch.zeros_like(tokens)
            tokens_g1[:, :, 1:4] = v
            out_mv = self.ga_head_vector_proj(tokens_g1)
            v_out = out_mv[:, :, 1:4].reshape(x.shape[0], -1)
            return self.vector_proj_mlp(v_out).reshape(x.shape[0], 3, 3)
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
        return torch.argmax(self.probs_on_grid(coeffs), dim=-1)

    @torch.no_grad()
    def predict(self, x):
        out = self.forward(x)
        mode = self._resolve_mode()
        if mode == "fourier":
            return self.so3_rotmats_cache[self.predict_rotmat(out)]
        if mode == "rotor":
            return multivector_to_rotation_matrix(out)
        return out  # rotation_matrix / vector_proj already return (B, 3, 3)

    @torch.no_grad()
    def get_nearest_idx(self, rot_gt: torch.Tensor):
        return nearest_rotmat(rot_gt, self.so3_rotmats_cache)


class I2SConvResNet(_I2S_ConvBase):
    def __init__(
        self,
        algebra,
        lmax: int = 6,
        rec_level: int = 3,
        hidden_dim: List = [32],
        temperature: float = 1.0,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = True,
        use_positional_encoding: bool = True,
        output_mode: str = "auto",
        adapter_type: str = "conv",
        head_type: str = "ga",
    ):
        weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained_backbone else None
        resnet = torchvision.models.resnet50(weights=weights)
        backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False
        super().__init__(
            algebra, backbone, backbone_channels=2048,
            lmax=lmax, rec_level=rec_level, hidden_dim=hidden_dim,
            temperature=temperature, use_positional_encoding=use_positional_encoding,
            output_mode=output_mode, adapter_type=adapter_type, head_type=head_type,
        )


class I2SConvNext(_I2S_ConvBase):
    def __init__(
        self,
        algebra,
        lmax: int = 6,
        rec_level: int = 3,
        hidden_dim: List = [32],
        temperature: float = 1.0,
        variant: str = "tiny",
        pretrained_backbone: bool = True,
        freeze_backbone: bool = True,
        use_positional_encoding: bool = True,
        output_mode: str = "auto",
        adapter_type: str = "conv",
        head_type: str = "ga",
    ):
        encoder = ConvNextEncoder(variant=variant, pretrained=pretrained_backbone, frozen=freeze_backbone)
        super().__init__(
            algebra, encoder.features, backbone_channels=encoder.output_shape[0],
            lmax=lmax, rec_level=rec_level, hidden_dim=hidden_dim,
            temperature=temperature, use_positional_encoding=use_positional_encoding,
            output_mode=output_mode, adapter_type=adapter_type, head_type=head_type,
        )
