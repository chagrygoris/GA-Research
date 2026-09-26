#set document(title: "pose3d experiment history up to 2026-09-26", author: "pose3d team")
#set page(paper: "a4", margin: 2cm, numbering: "1")
#set text(size: 10.5pt)
#set heading(numbering: "1.")
#show table: set text(size: 9pt)

#align(center)[#text(size: 18pt, weight: "bold")[pose3d experiment history up to 2026-09-26]]

Median rotation error in degrees on Pascal3D+ (lower is better). "final" is the 32-sample medoid score logged as #raw("final_median_rotation_error"); per-epoch #raw("median_rotation_error") is a single draw and is not comparable to it. W&B project: #raw("clifforders/3D Pose Estimation"). All numbers below are single runs unless stated.

= Reference

#table(
  columns: (1fr, 1fr, 1fr),
  inset: 5pt,
  stroke: 0.4pt + gray,
  table.header([*Run*], [*Result*], [*Recipe*]),
  [#raw("k5sblpo8") make\_clifford\_flow\_great\_again], [#strong[10.25]], [#raw("--model clifford_flow --pretrained_backbone --ram_memory --n_epochs 100 --warmup_epochs 5 --batch_size 32 --lr 1e-4 --hidden_dim 32 --n_cond_mv 64 --n_time_samples 8 --eval_samples 32") (ResNet-50). These are the defaults of #raw("config.py").]
)

Same-harness comparison: I2S without synthetic data 10.90 (#raw("bxtqcv3s"), #raw("i2s_real"), ResNet-101); published I2S 9.6 (with synthetic data), IPDF \~10.2, matrix Fisher 8.9 / 8.1, Rotation Laplace 9.3.

Comparing runs: W&B stores the CLI args in #raw("runInfo.args"); diff them before comparing two runs. The ablation runs of 2026-09-20..25 (#raw("mlp_heads"), #raw("ga_only"), #raw("drop_linear_skip"), #raw("adapter_grid") 7/9/11, #raw("adapter_channels"), #raw("vector_field_hidden_dim"), #raw("adapter96+grid9")) omitted #raw("--n_cond_mv 64 --n_time_samples 8") and ran with 4 and 1, so they compare against a matched baseline of about 11.1 (#raw("43x1nzh7") 11.14, #raw("nk8rdqxy") 11.06), not against 10.25.

= Adopted (on by default)

#table(
  columns: (1fr, 1fr, 1fr),
  inset: 5pt,
  stroke: 0.4pt + gray,
  table.header([*Change*], [*Flag / value*], [*Evidence*]),
  [ImageNet-initialised, fine-tuned backbone], [#raw("pretrained_backbone")], [11.71 pretrained (#raw("f1rthi7s"), single draw) vs 56.82 from scratch (#raw("nk54is62"))],
  [Several (t, r0) samples per image per step], [#raw("time_sample_batching"), #raw("n_time_samples=8")], [part of the 10.25 recipe],
  [More conditioning multivectors], [#raw("n_cond_mv=64") (was 4)], [part of the 10.25 recipe],
  [Geodesic-medoid evaluation with 32 samples], [#raw("medoid_eval"), #raw("eval_samples=32")], [single draw 11.71 (#raw("f1rthi7s")) vs medoid on later runs],
  [RAM-resident dataset], [#raw("ram_memory")], [infrastructure (faster epochs)]
)

= Experimental (off by default)

#table(
  columns: (1fr, 1fr, 1fr, 1fr),
  inset: 5pt,
  stroke: 0.4pt + gray,
  table.header([*Change*], [*Flag*], [*Result*], [*Notes*]),
  [Pascal3D augmentation], [#raw("use_warp") (#raw("raw_cache") / #raw("cache_draws") keep it random)], [9.71 (#raw("lyqxhz1p"))], [best number seen, but the exact args of that run are unconfirmed; needs a rerun with the flag],
  [RenderForCNN synthetic data], [#raw("use_synth"), #raw("max_synth")], [not run], [I2S / matrix Fisher gain from it; needs the extra download],
  [ResNet-101 backbone], [#raw("--encoder resnet101")], [10.18 (#raw("rde26byv"))], [0.07 below the reference; 70 epochs, single run],
  [Depth-Anything backbone], [#raw("--encoder depth_anything")], [Clifford Flow 12.04 last epoch, run crashed (#raw("ylvhg7i1")); #raw("i2s_real") 10.53 (#raw("96190o1k"))], [],
  [Smaller condition grid], [#raw("adapter_grid") (16)], [4: 10.62 (#raw("5e6lyv7u")); 7: 11.95; 9: 10.90; 11: 11.31], [the 7/9/11 runs used the 4 / 1 recipe, compare with \~11.1],
  [Narrower adapter], [#raw("adapter_channels") (256)], [96: 10.92], [same caveat],
  [Vector field sized separately], [#raw("vector_field_hidden_dim")], [10.49 with #raw("adapter_grid 7"), #raw("[96, 96]")], [most promising of the round, unconfirmed],
  [Matrix Fisher source distribution], [#raw("fisher_prior") + #raw("fisher_checkpoint")], [16.79 at the last epoch, run crashed (#raw("tri30r7e"))], [unfinished],
  [Frozen backbone], [#raw("freeze_encoder")], [no comparable run], [],
  [Parameter-matched MLP heads], [#raw("mlp_heads")], [43.65], [comparison showing the GA layers matter; not a candidate]
)

Four #raw("_cmv64") reruns on the corrected recipe were launched 2026-09-26 (#raw("grid11"), #raw("adapter96+grid9"), #raw("vf_boost"), and a second baseline seed: #raw("t6vunfp8"), #raw("pf8v440i"), #raw("50nzm900"), #raw("g41l2cbo")); read their #raw("final_median_rotation_error") before drawing conclusions about the rows above.

= Not adopted

#table(
  columns: (1fr, 1fr, 1fr),
  inset: 5pt,
  stroke: 0.4pt + gray,
  table.header([*Change*], [*Result*], [*Branch*]),
  [Drop MVLinear skip + MVSiLU from both GA heads (#raw("ga_only"))], [14.71, about +3.6 vs the matched baseline], [#raw("ga-only-ablation")],
  [Drop only the MVLinear skip (#raw("drop_linear_skip"))], [37.20, likely confounded by a missing normalisation], [#raw("mvlinear-skip-ablation")]
)

= Other model families (selectable, off the main line)

#table(
  columns: (1fr, 1fr, 1fr, 1fr),
  inset: 5pt,
  stroke: 0.4pt + gray,
  table.header([*#raw("--model")*], [*Origin*], [*Result*], [*Notes*]),
  [#raw("mlp_flow")], [#raw("MLPFlow-Baseline")], [11.54 (#raw("fotwj8lx"))], [flow matching with plain MLPs on matrices],
  [#raw("i2s_real")], [#raw("flow_tuning")], [10.90 (R101), 10.53 (Depth-Anything base)], [no synthetic data],
  [#raw("image2pcd_ipdf")], [#raw("Depth_IPDF")], [9.53 (#raw("xai7z8kj"), May 13), scratch backbone 49.3], [Depth-Anything features + Implicit PDF, gradient-ascent refinement],
  [#raw("image2pcd_late_fusion")], [#raw("img2pcd_exps")], [10.74 with depth (#raw("x35r7abd")), 10.12 without (#raw("k9di1x5s"))], [#raw("--no-i2p_use_depth") is the ablation],
  [#raw("image2pcd_pointnet")], [#raw("img2pcd_exps")], [10.03 (#raw("b52l33ii")) / 9.40 (#raw("adapter_transformerovich"), #raw("j8114i29")), both runs crashed], [last logged values of the runs closest in time to this code],
  [#raw("i2s_conv_convnext")], [#raw("conv_adapter")], [10.64 (#raw("ecba7uve"))], [ConvNeXt + GA head],
  [#raw("i2s_backbone")], [#raw("mvlinear") / #raw("main") (teammates)], [no number recorded here], [ResNet-50 / ConvNeXt-tiny, several GA heads],
  [#raw("vit_baseline")], [#raw("mvlinear") / #raw("main") (teammates)], [no finished run recorded here], [ViT / Depth-Anything hidden states + GA pooling],
  [#raw("ipdf_resnet")], [#raw("ipdf")], [#raw("ipdf_baseline") 133.6 (#raw("xh7cxbkh"), May 8)], [early run],
  [#raw("tralalero"), #raw("mlp"), #raw("i2s"), #raw("ga_i2s")], [#raw("main")], [], [early baselines]
)

= Where the rest lives

Consolidated into this folder: #raw("flow_tuning"), #raw("Clifford_Flow"), #raw("MLPFlow-Baseline"), #raw("matrix_fisher"), #raw("ram-cache"), #raw("flow-grid-ablation"), #raw("conv-adapter-ablation"), #raw("vector-field-boost-ablation"), #raw("ga-only-ablation"), #raw("mvlinear-skip-ablation"), #raw("mvlinear"), #raw("ipdf"), #raw("conv_adapter"), #raw("Depth_IPDF"), #raw("img2pcd_exps"), #raw("main") (previous).

Not ported, still on their branches (nothing was deleted):

- #raw("CliffordNet-exps"): CliffordNet backbone with I2S heads, real ModelNet10 and SYMSOL dataloaders.
- #raw("pos_encode_exps"): positional-encoding GA input experiments.
- #raw("Vision-GNN"), #raw("rotMNIST"): 2D / GNN experiments outside 3D pose.
- #raw("GaMLP_for_Pascal"), #raw("I2S"), #raw("I2SProbLoss"), #raw("mlp_baseline_experiment"): early Pascal3D baselines, superseded.
- #raw("revert-12-main") .. #raw("revert-15-main"): reverted multi-GPU changes.
