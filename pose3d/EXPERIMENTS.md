# Experiments

Median rotation error in degrees on Pascal3D+ (lower is better). "final" is the 32-sample medoid
score logged as `final_median_rotation_error`; per-epoch `median_rotation_error` is a single draw and
is not comparable to it. W&B project: `clifforders/3D Pose Estimation`. All numbers below are single
runs unless stated.

## Reference

| Run | Result | Recipe |
|---|---|---|
| `k5sblpo8` make_clifford_flow_great_again | **10.25** | `--model clifford_flow --pretrained_backbone --ram_memory --n_epochs 100 --warmup_epochs 5 --batch_size 32 --lr 1e-4 --hidden_dim 32 --n_cond_mv 64 --n_time_samples 8 --eval_samples 32` (ResNet-50). These are the defaults of `config.py`. |

Same-harness comparison: I2S without synthetic data 10.90 (`bxtqcv3s`, `i2s_real`, ResNet-101);
published I2S 9.6 (with synthetic data), IPDF ~10.2, matrix Fisher 8.9 / 8.1, Rotation Laplace 9.3.

Comparing runs: W&B stores the CLI args in `runInfo.args`; diff them before comparing two runs. The
ablation runs of 2026-09-20..25 (`mlp_heads`, `ga_only`, `drop_linear_skip`, `adapter_grid` 7/9/11,
`adapter_channels`, `vector_field_hidden_dim`, `adapter96+grid9`) omitted
`--n_cond_mv 64 --n_time_samples 8` and ran with 4 and 1, so they compare against a matched baseline
of about 11.1 (`43x1nzh7` 11.14, `nk8rdqxy` 11.06), not against 10.25.

## Adopted (on by default)

| Change | Flag / value | Evidence |
|---|---|---|
| ImageNet-initialised, fine-tuned backbone | `pretrained_backbone` | 11.71 pretrained (`f1rthi7s`, single draw) vs 56.82 from scratch (`nk54is62`) |
| Several (t, r0) samples per image per step | `time_sample_batching`, `n_time_samples=8` | part of the 10.25 recipe |
| More conditioning multivectors | `n_cond_mv=64` (was 4) | part of the 10.25 recipe |
| Geodesic-medoid evaluation with 32 samples | `medoid_eval`, `eval_samples=32` | single draw 11.71 (`f1rthi7s`) vs medoid on later runs |
| RAM-resident dataset | `ram_memory` | infrastructure (faster epochs) |

## Experimental (off by default)

| Change | Flag | Result | Notes |
|---|---|---|---|
| Pascal3D augmentation | `use_warp` (`raw_cache` / `cache_draws` keep it random) | 9.71 (`lyqxhz1p`) | best number seen, but the exact args of that run are unconfirmed; needs a rerun with the flag |
| RenderForCNN synthetic data | `use_synth`, `max_synth` | not run | I2S / matrix Fisher gain from it; needs the extra download |
| ResNet-101 backbone | `--encoder resnet101` | 10.18 (`rde26byv`) | 0.07 below the reference; 70 epochs, single run |
| Depth-Anything backbone | `--encoder depth_anything` | Clifford Flow 12.04 last epoch, run crashed (`ylvhg7i1`); `i2s_real` 10.53 (`96190o1k`) | |
| Smaller condition grid | `adapter_grid` (16) | 4: 10.62 (`5e6lyv7u`); 7: 11.95; 9: 10.90; 11: 11.31 | the 7/9/11 runs used the 4 / 1 recipe, compare with ~11.1 |
| Narrower adapter | `adapter_channels` (256) | 96: 10.92 | same caveat |
| Vector field sized separately | `vector_field_hidden_dim` | 10.49 with `adapter_grid 7`, `[96, 96]` | most promising of the round, unconfirmed |
| Matrix Fisher source distribution | `fisher_prior` + `fisher_checkpoint` | 16.79 at the last epoch, run crashed (`tri30r7e`) | unfinished |
| Frozen backbone | `freeze_encoder` | no comparable run | |
| Parameter-matched MLP heads | `mlp_heads` | 43.65 | comparison showing the GA layers matter; not a candidate |

Four `_cmv64` reruns on the corrected recipe were launched 2026-09-26 (`grid11`, `adapter96+grid9`,
`vf_boost`, and a second baseline seed: `t6vunfp8`, `pf8v440i`, `50nzm900`, `g41l2cbo`); read their
`final_median_rotation_error` before drawing conclusions about the rows above.

## Not adopted

| Change | Result | Branch |
|---|---|---|
| Drop MVLinear skip + MVSiLU from both GA heads (`ga_only`) | 14.71, about +3.6 vs the matched baseline | `ga-only-ablation` |
| Drop only the MVLinear skip (`drop_linear_skip`) | 37.20, likely confounded by a missing normalisation | `mvlinear-skip-ablation` |

## Other model families (selectable, off the main line)

| `--model` | Origin | Result | Notes |
|---|---|---|---|
| `mlp_flow` | `MLPFlow-Baseline` | 11.54 (`fotwj8lx`) | flow matching with plain MLPs on matrices |
| `i2s_real` | `flow_tuning` | 10.90 (R101), 10.53 (Depth-Anything base) | no synthetic data |
| `image2pcd_ipdf` | `Depth_IPDF` | 9.53 (`xai7z8kj`, May 13), scratch backbone 49.3 | Depth-Anything features + Implicit PDF, gradient-ascent refinement |
| `image2pcd_late_fusion` | `img2pcd_exps` | 10.74 with depth (`x35r7abd`), 10.12 without (`k9di1x5s`) | `--no-i2p_use_depth` is the ablation |
| `image2pcd_pointnet` | `img2pcd_exps` | 10.03 (`b52l33ii`) / 9.40 (`adapter_transformerovich`, `j8114i29`), both runs crashed | last logged values of the runs closest in time to this code |
| `i2s_conv_convnext` | `conv_adapter` | 10.64 (`ecba7uve`) | ConvNeXt + GA head |
| `i2s_backbone` | `mvlinear` / `main` (teammates) | no number recorded here | ResNet-50 / ConvNeXt-tiny, several GA heads |
| `vit_baseline` | `mvlinear` / `main` (teammates) | no finished run recorded here | ViT / Depth-Anything hidden states + GA pooling |
| `ipdf_resnet` | `ipdf` | `ipdf_baseline` 133.6 (`xh7cxbkh`, May 8) | early run |
| `tralalero`, `mlp`, `i2s`, `ga_i2s` | `main` | | early baselines |

## Where the rest lives

Consolidated into this folder: `flow_tuning`, `Clifford_Flow`, `MLPFlow-Baseline`, `matrix_fisher`,
`ram-cache`, `flow-grid-ablation`, `conv-adapter-ablation`, `vector-field-boost-ablation`,
`ga-only-ablation`, `mvlinear-skip-ablation`, `mvlinear`, `ipdf`, `conv_adapter`, `Depth_IPDF`,
`img2pcd_exps`, `main` (previous).

Not ported, still on their branches (nothing was deleted):

* `CliffordNet-exps`: CliffordNet backbone with I2S heads, real ModelNet10 and SYMSOL dataloaders.
* `pos_encode_exps`: positional-encoding GA input experiments.
* `Vision-GNN`, `rotMNIST`: 2D / GNN experiments outside 3D pose.
* `GaMLP_for_Pascal`, `I2S`, `I2SProbLoss`, `mlp_baseline_experiment`: early Pascal3D baselines, superseded.
* `revert-12-main` .. `revert-15-main`: reverted multi-GPU changes.
