"""Experiment configuration: the single place that decides what a default run is.

The defaults reproduce the best documented Clifford Flow run (W&B `k5sblpo8`,
10.25 deg median rotation error on Pascal3D+): `--model=clifford_flow`, pretrained
ResNet-50, `n_cond_mv=64`, `n_time_samples=8`, `hidden_dim=32`, 100 epochs,
`lr=1e-4`, 32-sample medoid evaluation.

Experiment workflow (see the root README and the idea board in pose3d/README.md):

* A change that is proven better on Pascal3D+ is adopted with its flag set to True in
  `Features` below (or, for numeric options, its value becomes the default).
* A change that is not proven yet lands with its flag False, so `main` stays the
  reference recipe and the experiment stays one `--flag` away.
* A change that turned out worse is not added; it is written up in reports/.

Every field below is also a command-line flag of the same name (`--use_warp`,
`--no-medoid_eval`, `--n_cond_mv 32`, ...). Sections only group them; flag names are
globally unique.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple, Union, get_args, get_origin, get_type_hints

ModelName = Literal[
    "clifford_flow",   # the main line: conditional flow matching on Spin(3) rotors
    "mlp_flow",        # non-equivariant flow-matching baseline
    "i2s_real",        # published Image2Sphere, adapted to this harness
    "i2s",             # in-repo I2S with a GA head (average-pooled features)
    "ga_i2s",          # in-repo I2S with the GA pixel encoder
    "tralalero",       # direct regression with a GA head
    "mlp",             # direct regression with a linear head
    "i2s_backbone",    # ResNet/ConvNeXt + conv adapter + GA head (teammates' line)
    "i2s_resnet",      # alias of i2s_backbone, kept for old command lines
    "i2s_conv_resnet",   # conv_adapter branch: adapters + ResNet-50
    "i2s_conv_convnext",  # conv_adapter branch: adapters + ConvNeXt
    "ipdf_resnet",     # Clifford implicit PDF over rotor-rotated features
    "vit_baseline",    # ViT / Depth-Anything hidden states + pooling head
    "image2pcd",       # Depth-Anything features, direct regression (I2P)
    "image2pcd_pointnet",      # Depth-Anything point cloud + PointNet regression (I2PPointNet)
    "image2pcd_late_fusion",   # Depth-Anything hidden states + point stream, late fusion (I2PLateFusion)
    "image2pcd_ipdf",  # Depth-Anything features, implicit PDF (I2P_IPDF)
    "dummynet",        # point-cloud debugging model (ModelNet10 stub)
]
EncoderName = Literal[
    "resnet", "resnet50", "resnet101", "depth_anything", "ga", "ga_canonical",
    "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
]
LossName = Literal["mse", "mse_ortho", "geodesic", "prob", "rotor", "mv_rotor"]


@dataclass
class RunConfig:
    """Where the data is, how the run is named and logged."""

    path_to_datasets: str = field(default="", metadata={"required": True})
    dataset: Literal["pascal", "dummynet"] = "pascal"
    platform: Literal["kaggle", "colab"] = "kaggle"
    run_name: Optional[str] = None
    path_to_checkpoint: Optional[str] = None   # evaluate this checkpoint before training
    sanity_check: bool = False                 # one tiny batch; no checkpoint saved
    save_checkpoint: bool = True
    wandb_project: str = "3D Pose Estimation"
    wandb_entity: str = "clifforders"
    wandb_group: Optional[str] = None


@dataclass
class TrainConfig:
    n_epochs: int = 100
    warmup_epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-4
    # mse: plain MSE on the matrix; mse_ortho: MSE + orthogonality penalty (teammates' default).
    loss: LossName = "mse"
    label_smoothing: float = 0.0   # only for loss=prob
    # Samples per image for the multi-sample evaluation run after the last epoch
    # (used when Features.medoid_eval is on).
    eval_samples: int = 32


@dataclass
class Features:
    """Switches for modifications of the reference pipeline.

    True  = adopted: measurably better, part of the reference recipe.
    False = experimental: not proven yet, or an ablation kept for comparison.
    """

    # ---- adopted (True) ---------------------------------------------------------
    # ImageNet-initialised, fully fine-tuned backbone (scratch: 56.8 deg vs 10.25 deg).
    pretrained_backbone: bool = True
    # Preload Pascal3D tensors into RAM (infrastructure; same numbers, faster epochs).
    ram_memory: bool = True
    # Draw FlowConfig.n_time_samples (t, r0) pairs per image per step, sharing one
    # backbone pass (8 samples/image is part of the 10.25 deg recipe).
    time_sample_batching: bool = True
    # Score the final model with TrainConfig.eval_samples draws and take the geodesic
    # medoid instead of a single draw (32 samples is what the reported numbers use).
    medoid_eval: bool = True

    # ---- experimental (False) ---------------------------------------------------
    # Pascal3D's own augmentation (flip / up-direction jitter / bbox jitter). Run
    # `lyqxhz1p` reached 9.71 deg with it but its exact recipe is unconfirmed.
    use_warp: bool = False
    # RenderForCNN synthetic training images (needs a separate download; see max_synth).
    use_synth: bool = False
    # With ram_memory, cache the file reads instead of augmented crops so use_warp stays
    # random per access.
    raw_cache: bool = False
    # Freeze the backbone and train only the heads.
    freeze_encoder: bool = False
    # Swap CliffordFlow's two GA heads for parameter-matched plain MLPs (43.7 deg: a
    # comparison showing the GA layers matter, not a candidate improvement).
    mlp_heads: bool = False
    # Draw the flow's source rotor r0 from a pretrained matrix Fisher head instead of the
    # uniform distribution (needs FlowConfig.fisher_checkpoint). Unfinished run only.
    fisher_prior: bool = False


@dataclass
class DataConfig:
    multiprocessing: bool = False   # spawn workers while filling the RAM cache
    max_synth: int = 0              # cap on the synthetic pool with use_synth (0 keeps all)
    cache_draws: int = 1            # augmented passes cached with ram_memory + use_warp
    # Reuse the tensors ram_memory builds instead of decoding every image each session.
    ram_cache_dir: Optional[str] = None        # read pascal_{train,val}.pt from here
    ram_cache_save_dir: Optional[str] = None   # write them here after a normal build


@dataclass
class ModelConfig:
    name: ModelName = field(default="clifford_flow", metadata={"flag": "model"})
    encoder: EncoderName = "resnet50"   # "resnet" is an alias of resnet50
    hidden_dim: List[int] = field(default_factory=lambda: [32])
    algebra_dim: int = 3                # Cl(algebra_dim); most GA paths need 3
    depth_anything_model: str = "depth-anything/Depth-Anything-V2-Base-hf"


@dataclass
class FlowConfig:
    """CliffordFlow options. Defaults are the reference recipe; the rest are ablations."""

    n_cond_mv: int = 64        # conditioning multivectors passed to the vector field
    n_time_samples: int = 8    # (t, r0) pairs per image (see Features.time_sample_batching)
    # ConvAdapter pools to (adapter_grid x adapter_grid) multivectors. 9 -> 10.90 deg,
    # 7 -> 11.95 deg, 11 -> 11.31 deg (n=1 each, before the recipe fix; unconfirmed).
    adapter_grid: int = 16
    # Width of the adapter's first 1x1 conv. 96 -> 10.92 deg (unconfirmed).
    adapter_channels: int = 256
    # Hidden widths of the vector field alone (None: same as hidden_dim). Paired with a
    # smaller adapter_grid this reallocates parameters; 10.49 deg (unconfirmed).
    vector_field_hidden_dim: Optional[List[int]] = None
    # Path to Liu et al.'s Pascal3D+ matrix Fisher checkpoint (state_dict_119.pkl);
    # used only with Features.fisher_prior.
    fisher_checkpoint: Optional[str] = None


@dataclass
class I2SConfig:
    """Image2Sphere-family options (i2s, i2s_real, ga_i2s and the I2S-style heads)."""

    lmax: int = 6
    rec_level: int = 3
    n_mv: int = 8
    ga_pool_hw: Tuple[int, int] = (28, 28)
    temperature: float = 1.0
    # i2s_real: eval grid level (upstream's 5 needs a 4.3 GB Wigner matrix).
    i2s_eval_rec_level: int = 3
    # i2s_real: apply ImageNet normalisation to the [0, 1] inputs.
    i2s_normalize: bool = True


@dataclass
class IPDFConfig:
    """Implicit-PDF options (image2pcd_ipdf, ipdf_resnet)."""

    pe_freqs: int = 4
    n_train_queries: int = 4096
    grad_ascent_steps: int = 100
    grad_ascent_lr: float = 1e-4
    # image2pcd / image2pcd_ipdf: freeze the Depth-Anything backbone (distinct from
    # Features.freeze_encoder, which applies to the flow/I2S encoders).
    freeze_backbone: bool = True
    ipdf_n_queries: int = 511   # ipdf_resnet: negative rotors per image


@dataclass
class PointCloudConfig:
    """Depth-Anything point-cloud regressors (image2pcd_pointnet, image2pcd_late_fusion)."""

    i2p_n_points: int = 2048
    # image2pcd_late_fusion: False zeroes the depth channel (the depth ablation, 10.12 deg,
    # beat the run with depth, 10.74 deg).
    i2p_use_depth: bool = True


@dataclass
class ViTConfig:
    """`--model vit_baseline`."""

    vit_backbone_type: Literal["vit", "depth_anything_v2"] = "vit"
    vit_model_name: str = "google/vit-base-patch16-224-in21k"
    vit_layers: List[int] = field(default_factory=lambda: [-1, -3, -6, -9])
    freeze_vit: bool = True
    vit_pooling_type: Literal["mean", "attention", "transformer_attention", "convolution", "ga"] = "mean"
    vit_num_transformer_layers: int = 1
    vit_transformer_nhead: int = 8
    vit_transformer_ff_dim: int = 1024
    vit_transformer_dropout: float = 0.1
    vit_ga_input_features: int = 196
    vit_ga_hidden_dim: List[int] = field(default_factory=lambda: [32])
    vit_ga_readout_type: Literal["scalar", "mean", "linear", "grade", "rotor"] = "linear"


@dataclass
class I2SBackboneConfig:
    """`--model i2s_backbone` (alias `i2s_resnet`); `ipdf_resnet` reuses the backbone flags."""

    i2s_resnet_output_mode: Literal[
        "auto", "rotation_matrix", "fourier", "rotor", "multivector_rotor"] = "auto"
    i2s_resnet_backbone_name: Literal["resnet50", "convnext_tiny"] = "resnet50"
    i2s_resnet_pretrained_backbone: bool = True
    i2s_resnet_freeze_backbone: bool = True
    i2s_resnet_use_positional_encoding: bool = True
    i2s_resnet_mv_per_position: int = 1
    i2s_resnet_adapter_mid_channels: int = 0
    i2s_resnet_adapter_high_channels: int = 0
    i2s_resnet_adapter_output_size: int = 16
    i2s_resnet_ga_head_type: Literal[
        "tralalero", "transformer_like", "reduced", "residual_gp"] = "tralalero"
    i2s_resnet_ga_head_mixing_layer: Literal["gp", "mvlinear", "linear"] = "gp"
    i2s_resnet_ga_num_blocks: int = 2
    i2s_resnet_ga_head_dropout: float = 0.0
    i2s_resnet_ga_head_use_layer_norm: bool = False


@dataclass
class I2SConvConfig:
    """`--model i2s_conv_resnet | i2s_conv_convnext` (the conv_adapter branch)."""

    i2s_conv_variant: Literal["tiny", "small", "base", "large"] = "tiny"   # ConvNeXt only
    i2s_conv_output_mode: Literal["auto", "rotation_matrix", "fourier", "rotor", "vector_proj"] = "auto"
    i2s_conv_pretrained_backbone: bool = True
    i2s_conv_freeze_backbone: bool = True
    i2s_conv_use_positional_encoding: bool = True
    i2s_conv_adapter_type: Literal["conv", "mlp_block", "linear", "geometric", "inc"] = "conv"
    i2s_conv_head_type: Literal["ga", "mlp"] = "ga"


SECTIONS = (
    "run", "train", "features", "data", "model", "flow",
    "i2s", "ipdf", "pointcloud", "vit", "i2s_backbone", "i2s_conv",
)


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    features: Features = field(default_factory=Features)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    i2s: I2SConfig = field(default_factory=I2SConfig)
    ipdf: IPDFConfig = field(default_factory=IPDFConfig)
    pointcloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    vit: ViTConfig = field(default_factory=ViTConfig)
    i2s_backbone: I2SBackboneConfig = field(default_factory=I2SBackboneConfig)
    i2s_conv: I2SConvConfig = field(default_factory=I2SConvConfig)

    # Set at runtime, not a flag.
    device: Any = None

    @property
    def eval_samples(self) -> int:
        return self.train.eval_samples if self.features.medoid_eval else 1

    @property
    def n_time_samples(self) -> int:
        return self.flow.n_time_samples if self.features.time_sample_batching else 1

    def to_dict(self) -> dict:
        """Nested plain-dict view (for checkpoints and W&B); excludes the runtime device."""
        return {name: dataclasses.asdict(getattr(self, name)) for name in SECTIONS}


def _flag(section_field: dataclasses.Field) -> str:
    return section_field.metadata.get("flag", section_field.name)


def _unwrap_optional(tp):
    if get_origin(tp) is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return tp, False


def _add_flag(parser: argparse.ArgumentParser, fld: dataclasses.Field, hints: dict):
    tp, _ = _unwrap_optional(hints[fld.name])
    name = "--" + _flag(fld)
    default = (fld.default_factory() if fld.default_factory is not dataclasses.MISSING
               else fld.default)
    kwargs: dict = {"default": default, "dest": _flag(fld)}
    if fld.metadata.get("required"):
        kwargs["required"] = True

    origin = get_origin(tp)
    if tp is bool:
        parser.add_argument(name, action=argparse.BooleanOptionalAction, **kwargs)
        return
    if origin is Literal:
        choices = get_args(tp)
        parser.add_argument(name, type=type(choices[0]), choices=choices, **kwargs)
        return
    if origin in (list, List, tuple, Tuple):
        inner = get_args(tp)[0]
        nargs = len(get_args(tp)) if origin in (tuple, Tuple) else "+"
        parser.add_argument(name, type=inner, nargs=nargs, **kwargs)
        return
    parser.add_argument(name, type=tp, **kwargs)


def _summary(cls) -> Optional[str]:
    """First docstring line, ignoring the signature string dataclass generates when there is none."""
    doc = cls.__doc__
    if not doc or doc.startswith(cls.__name__ + "("):
        return None
    return doc.strip().splitlines()[0]


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train / evaluate a 3D pose (SO(3)) estimator. Defaults are the reference recipe.")
    seen: set = set()
    for name, cls in _section_classes().items():
        hints = get_type_hints(cls)
        group = parser.add_argument_group(name, _summary(cls))
        for fld in dataclasses.fields(cls):
            flag = _flag(fld)
            if flag in seen:
                raise ValueError(f"duplicate flag --{flag}")
            seen.add(flag)
            _add_flag(group, fld, hints)
    return parser


def _section_classes() -> dict:
    hints = get_type_hints(Config)
    return {name: hints[name] for name in SECTIONS}


def parse_args(argv: Optional[List[str]] = None) -> Config:
    """Parse command-line flags into a Config."""
    parser = create_argparser()
    ns = vars(parser.parse_args(argv))
    cfg = Config()
    for name, cls in _section_classes().items():
        values = {}
        for fld in dataclasses.fields(cls):
            value = ns[_flag(fld)]
            if get_origin(_unwrap_optional(get_type_hints(cls)[fld.name])[0]) in (tuple, Tuple) and value is not None:
                value = tuple(value)
            values[fld.name] = value
        setattr(cfg, name, cls(**values))
    return cfg
