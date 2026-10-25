# pose3d

**Best score**: Clifford Flow with a GATr vector field, ResNet-101, trained on Image2Sphere's data (warp + synthetic renders) with EMA weights: **8.95°** mean of per-class median errors, **7.85°** median over all test images, on Pascal3D+ (`clifford_flow_gatr_warp_synth_ema_b64`, Kaggle kernel `syfry5suvzovvakmuj/clifford-gatr-ema-b64-rtx`) — now the default recipe in `pose3d/config.py`.

Setup and how to run: [`USAGE.md`](USAGE.md). Results: [`reports/`](reports/). Reading material:
[`awesome-reference/`](awesome-reference/README.md).

| Status | Idea | Description |
|---|---|---|
| done | Image2Sphere baseline on SYMSOL I (10 %) | `--model i2s_real` under the `symsol1_10pct_*` protocol: **3.2 deg** mean of per-shape nearest-equivalent medians (AdamW/cosine; 4.5 deg with the paper's SGD recipe), log-likelihood 3.48, against 19.9 deg / 3.24 for the best flow run. The paper itself gives no rotation error on SYMSOL. [report](reports/i2s_symsol1_10pct.md) |
