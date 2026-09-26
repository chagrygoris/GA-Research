"""ViT / Depth-Anything-V2 hidden-state baseline with mean / attention / transformer / conv / GA pooling."""

import torch
import torch.nn as nn
from clifford.models.modules.fcgp import FullyConnectedSteerableGeometricProductLayer
from clifford.models.modules.mvsilu import MVSiLU

from pose3d.geometry.quaternion import multivector_to_rotation_matrix
from pose3d.models._utils import move_unregistered_tensors_to_device


def hidden_state_to_feature_map(hidden_state: torch.Tensor) -> torch.Tensor:
    if hidden_state.ndim != 3:
        raise ValueError(
            f"Expected hidden_state shape [B, N, C], got {tuple(hidden_state.shape)}"
        )

    tokens = hidden_state[:, 1:]
    batch_size, n_tokens, hidden_size = tokens.shape
    spatial_size = int(n_tokens ** 0.5)

    if spatial_size * spatial_size != n_tokens:
        raise ValueError(f"Expected square number of patch tokens, got {n_tokens}")

    return tokens.transpose(1, 2).reshape(
        batch_size, hidden_size, spatial_size, spatial_size
    )


def _config_get(config, field_name: str):
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(field_name)
    return getattr(config, field_name, None)


def _infer_hidden_size_from_config(config):
    hidden_size = _config_get(config, "hidden_size")
    if hidden_size is not None:
        return int(hidden_size)

    backbone_config = _config_get(config, "backbone_config")
    hidden_size = _config_get(backbone_config, "hidden_size")
    if hidden_size is not None:
        return int(hidden_size)

    return None


class ViTHiddenStatesBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        layers: tuple = (-1, -3, -6, -9),
        freeze: bool = True,
    ):
        super().__init__()
        from transformers import ViTModel

        self.layers = tuple(layers)
        self.model = ViTModel.from_pretrained(
            model_name,
            output_hidden_states=True,
        )
        self.model.config.output_hidden_states = True

        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

        self.hidden_size = _infer_hidden_size_from_config(self.model.config)
        self.output_dim = (
            self.hidden_size * len(self.layers)
            if self.hidden_size is not None
            else None
        )

    def _extract_hidden_states(self, x: torch.Tensor):
        grad_enabled = any(
            parameter.requires_grad for parameter in self.model.parameters()
        )

        with torch.set_grad_enabled(grad_enabled):
            out = self.model(pixel_values=x, output_hidden_states=True)

        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("ViTModel did not return hidden_states")
        return hidden_states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self._extract_hidden_states(x)
        feature_maps = []
        for layer_idx in self.layers:
            try:
                hidden_state = hidden_states[layer_idx]
            except IndexError as exc:
                raise IndexError(
                    f"Requested hidden-state layer {layer_idx}, but backbone returned "
                    f"{len(hidden_states)} hidden states"
                ) from exc
            feature_maps.append(hidden_state_to_feature_map(hidden_state))

        return torch.cat(feature_maps, dim=1)


class DepthAnythingV2HiddenStatesBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        layers: tuple = (-1, -3, -6, -9),
        freeze: bool = True,
    ):
        super().__init__()
        from transformers import AutoModelForDepthEstimation

        self.layers = tuple(layers)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_name,
            output_hidden_states=True,
        )
        self.model.config.output_hidden_states = True

        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

        self.hidden_size = _infer_hidden_size_from_config(self.model.config)
        self.output_dim = (
            self.hidden_size * len(self.layers)
            if self.hidden_size is not None
            else None
        )

    def _extract_hidden_states(self, x: torch.Tensor):
        grad_enabled = any(
            parameter.requires_grad for parameter in self.model.parameters()
        )

        with torch.set_grad_enabled(grad_enabled):
            out = self.model(pixel_values=x, output_hidden_states=True)

        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None:
            raise ValueError(
                "Depth Anything V2 did not return hidden_states. "
                "Ensure the model supports output_hidden_states=True."
            )
        return hidden_states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self._extract_hidden_states(x)
        feature_maps = []
        for layer_idx in self.layers:
            try:
                hidden_state = hidden_states[layer_idx]
            except IndexError as exc:
                raise IndexError(
                    f"Requested hidden-state layer {layer_idx}, but Depth Anything V2 "
                    f"returned {len(hidden_states)} hidden states"
                ) from exc
            feature_maps.append(hidden_state_to_feature_map(hidden_state))

        return torch.cat(feature_maps, dim=1)


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)
        return (x * weights).sum(dim=1)


def _normalize_optional_hidden_dims(hidden_dim):
    if hidden_dim is None:
        return []

    if isinstance(hidden_dim, int):
        if hidden_dim == -1:
            return []
        return [hidden_dim]

    hidden_dims = list(hidden_dim)

    if len(hidden_dims) == 0:
        return []

    if len(hidden_dims) == 1 and int(hidden_dims[0]) == -1:
        return []

    return [int(hd) for hd in hidden_dims]


class SimpleFullGeometricProductPoseHead(nn.Module):
    def __init__(
        self,
        algebra,
        input_dim: int = 512,
        input_features: int = 196,
        hidden_dim: list[int] | int | None = [32],
        out_features: int = 9,
        readout_type: str = "linear",
    ):
        super().__init__()

        if algebra is None:
            raise ValueError("algebra must be provided")

        self.algebra = algebra
        self.input_dim = int(input_dim)
        self.input_features = int(input_features)
        self.mv_dim = 2 ** algebra.dim
        self.out_features = int(out_features)
        self.readout_type = str(readout_type).lower()

        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.input_features <= 0:
            raise ValueError("input_features must be positive")
        valid_readout_types = {"scalar", "mean", "linear", "grade", "rotor"}
        if self.readout_type not in valid_readout_types:
            raise ValueError(
                "readout_type must be one of "
                f"{sorted(valid_readout_types)}, got {readout_type!r}"
            )

        if self.readout_type == "rotor":
            if algebra.dim != 3:
                raise ValueError("vit_ga_readout_type='rotor' requires algebra_dim=3 / Cl(3,0)")
            if self.mv_dim != 8:
                raise ValueError("rotor readout requires mv_dim=8 for Cl(3,0)")
        elif self.out_features != 9:
            raise ValueError("out_features must be 9 to form a 3x3 pose matrix")

        hidden_dims = _normalize_optional_hidden_dims(hidden_dim)

        if any(hd <= 0 for hd in hidden_dims):
            raise ValueError(
                "all hidden_dim values must be positive, "
                "or use -1/None/[] to disable hidden layers"
            )

        self.to_mv = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.mv_dim),
        )

        self.blocks = nn.ModuleList()
        prev_features = self.input_features

        for hd in hidden_dims:
            self.blocks.append(
                nn.ModuleDict({
                    "gp": FullyConnectedSteerableGeometricProductLayer(
                        algebra,
                        in_features=prev_features,
                        out_features=hd,
                    ),
                    "act": MVSiLU(algebra, hd),
                })
            )
            prev_features = hd

        effective_out_features = 1 if self.readout_type == "rotor" else self.out_features
        self.out = FullyConnectedSteerableGeometricProductLayer(
            algebra,
            in_features=prev_features,
            out_features=effective_out_features,
        )

        if self.readout_type == "linear":
            self.mv_to_scalar = nn.Linear(self.mv_dim, 1)
        elif self.readout_type == "grade":
            self.grade_indices = []
            for grade in range(algebra.dim + 1):
                indices = [
                    idx for idx in range(self.mv_dim)
                    if int(idx).bit_count() == grade
                ]
                self.grade_indices.append(indices)
            self.mv_to_scalar = nn.Linear(algebra.dim + 1, 1)
        else:
            self.mv_to_scalar = None

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_extra_tensors_device", None) != patch_embeddings.device:
            move_unregistered_tensors_to_device(self, patch_embeddings.device)
            self._extra_tensors_device = patch_embeddings.device

        if patch_embeddings.ndim != 3:
            raise ValueError(
                f"Expected patch_embeddings shape [B, N, C], got {tuple(patch_embeddings.shape)}"
            )

        batch_size, num_patches, embedding_dim = patch_embeddings.shape

        if num_patches != self.input_features:
            raise RuntimeError(
                f"GA pooling expected {self.input_features} patch tokens from config, got {num_patches}. "
                "Set --vit_ga_input_features to match the actual token count produced by the backbone/token_mlp."
            )

        if embedding_dim != self.input_dim:
            raise RuntimeError(
                f"GA pooling expected embedding_dim={self.input_dim}, got {embedding_dim}"
            )

        x = self.to_mv(patch_embeddings)

        for block in self.blocks:
            x = block["gp"](x)
            x = block["act"](x)

        out_mv = self.out(x)

        if self.readout_type == "rotor":
            mv = out_mv[:, 0, :]
            return multivector_to_rotation_matrix(mv)

        if self.readout_type == "scalar":
            out = out_mv[:, :, 0]
        elif self.readout_type == "mean":
            out = out_mv.mean(dim=-1)
        elif self.readout_type == "linear":
            out = self.mv_to_scalar(out_mv).squeeze(-1)
        elif self.readout_type == "grade":
            grade_values = []
            for indices in self.grade_indices:
                grade_component = out_mv[:, :, indices]
                grade_value = grade_component.norm(dim=-1)
                grade_values.append(grade_value)

            grade_features = torch.stack(grade_values, dim=-1)
            out = self.mv_to_scalar(grade_features).squeeze(-1)
        else:
            raise RuntimeError("Unexpected readout_type")

        return out.view(batch_size, 3, 3)


class ViTMultiLayerPoseBaseline(nn.Module):
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224-in21k",
        backbone_type: str = "vit",
        layers: tuple = (-1, -3, -6, -9),
        freeze_vit: bool = True,
        pooling_type: str = "mean",
        num_transformer_layers: int = 1,
        transformer_nhead: int = 8,
        transformer_ff_dim: int = 1024,
        transformer_dropout: float = 0.1,
        algebra=None,
        ga_input_features: int = 196,
        ga_hidden_dim: list[int] | int = [32],
        ga_readout_type: str = "linear",
    ):
        super().__init__()

        self.model_name = model_name
        self.backbone_type = backbone_type
        self.layers = tuple(layers)
        self.freeze_vit = bool(freeze_vit)
        self.pooling_type = str(pooling_type).lower()

        valid_pooling_types = {
            "mean",
            "attention",
            "transformer_attention",
            "convolution",
            "ga",
        }
        if self.pooling_type not in valid_pooling_types:
            raise ValueError(
                "pooling_type must be one of "
                f"{sorted(valid_pooling_types)}, got {pooling_type!r}"
            )

        if self.pooling_type == "transformer_attention":
            if num_transformer_layers < 1:
                raise ValueError(
                    "num_transformer_layers must be >= 1 for "
                    "transformer_attention pooling"
                )
            if transformer_nhead <= 0:
                raise ValueError("transformer_nhead must be > 0")
            if transformer_ff_dim <= 0:
                raise ValueError("transformer_ff_dim must be > 0")
            if 512 % transformer_nhead != 0:
                raise ValueError(
                    "transformer_nhead must divide the token embedding dimension 512"
                )
            if not 0 <= transformer_dropout < 1:
                raise ValueError("transformer_dropout must satisfy 0 <= dropout < 1")

        if len(self.layers) == 0:
            raise ValueError("layers must contain at least one hidden-state index")

        if self.backbone_type == "vit":
            self.backbone = ViTHiddenStatesBackbone(
                model_name=self.model_name,
                layers=self.layers,
                freeze=self.freeze_vit,
            )
        elif self.backbone_type == "depth_anything_v2":
            self.backbone = DepthAnythingV2HiddenStatesBackbone(
                model_name=self.model_name,
                layers=self.layers,
                freeze=self.freeze_vit,
            )
        else:
            raise ValueError(
                "backbone_type must be one of {'vit', 'depth_anything_v2'}, "
                f"got {self.backbone_type!r}"
            )

        token_mlp_input_dim = self.backbone.output_dim
        first_token_layer = (
            nn.Linear(token_mlp_input_dim, 1024)
            if token_mlp_input_dim is not None
            else nn.LazyLinear(1024)
        )

        self.token_mlp = nn.Sequential(
            first_token_layer,
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.GELU(),
        )

        if self.pooling_type == "mean":
            self.patch_transformer = nn.Identity()
            self.pool = None
        elif self.pooling_type == "attention":
            self.patch_transformer = nn.Identity()
            self.pool = AttentionPool(512)
        elif self.pooling_type == "transformer_attention":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=512,
                nhead=transformer_nhead,
                dim_feedforward=transformer_ff_dim,
                dropout=transformer_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.patch_transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_transformer_layers,
            )
            self.pool = AttentionPool(512)
        elif self.pooling_type == "convolution":
            self.patch_transformer = nn.Identity()
            self.pool = None
            self.conv_pose_head = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.GroupNorm(8, 256),
                nn.GELU(),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.GroupNorm(8, 128),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((3, 3)),
                nn.Conv2d(128, 1, kernel_size=1),
            )
        elif self.pooling_type == "ga":
            self.patch_transformer = nn.Identity()
            self.pool = None

            if algebra is None:
                raise ValueError("pooling_type='ga' requires algebra to be passed")

            self.ga_pose_head = SimpleFullGeometricProductPoseHead(
                algebra=algebra,
                input_dim=512,
                input_features=ga_input_features,
                hidden_dim=ga_hidden_dim,
                out_features=9,
                readout_type=ga_readout_type,
            )

        if self.pooling_type not in {"convolution", "ga"}:
            self.pose_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 256),
                nn.GELU(),
                nn.Linear(256, 9),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(x)
        flattened_spatial = feature_map.flatten(2)
        patch_features = flattened_spatial.transpose(1, 2)

        patch_embeddings = self.token_mlp(patch_features)

        if self.pooling_type == "ga":
            return self.ga_pose_head(patch_embeddings)

        if self.pooling_type == "convolution":
            batch_size, num_patches, embedding_dim = patch_embeddings.shape
            spatial_size = int(num_patches ** 0.5)
            if spatial_size * spatial_size != num_patches:
                raise RuntimeError(
                    "convolution pooling requires a square number of patch tokens, "
                    f"got {num_patches} tokens"
                )
            patch_grid = patch_embeddings.transpose(1, 2).reshape(
                batch_size,
                embedding_dim,
                spatial_size,
                spatial_size,
            )
            out = self.conv_pose_head(patch_grid)
            return out.squeeze(1)

        if self.pooling_type == "mean":
            global_embedding = patch_embeddings.mean(dim=1)
        elif self.pooling_type == "attention":
            global_embedding = self.pool(patch_embeddings)
        elif self.pooling_type == "transformer_attention":
            patch_embeddings = self.patch_transformer(patch_embeddings)
            global_embedding = self.pool(patch_embeddings)
        else:
            raise RuntimeError("Unexpected pooling_type")

        out = self.pose_head(global_embedding)
        return out.view(-1, 3, 3)
