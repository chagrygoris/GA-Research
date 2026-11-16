# Mechanistic probes of the cosine-direction ("cosdir") attention GATr

Scripts behind `reports/cosdir_mechanism.md`. They analyse the SYMSOL I run
`symsol1_10pct_gatr_spherecloud_cosdir_h8_coslr_adamw_l4x4` (Kaggle notebook
`arynabiaroza/symsol-gatr-spherecloud-cosdir-h8`). The attention module of that run,
`pose3d/models/cos_attention.py` (`ScaledAttention`), is **not in this repository yet**: the scripts import
it from the notebook's code snapshot.

```bash
export COSDIR_SNAPSHOT=<notebook output>/GA-Research/pose3d   # holds pose3d/models/cos_attention.py and the .pth
export COSDIR_DATA=<directory with symsol/test/images/*.png and rotations.npz below it>
python probe.py results/spec_all.json 1 0.5      # degree spectrum of every stage; args: output, images per shape, flow times
python attn.py                                   # anatomy of the pose token's attention -> attn.json
python interv.py                                 # temperature rescaling and the untrained field -> interv.json, init.json
python plot.py                                   # figures -> reports/figures/cosdir_*.png
```

| file | what it does |
|---|---|
| `lib.py` | builds the model from the snapshot with the run's flags, loads the checkpoint and test images |
| `probe.py` | turns the pose along random great circles of SO(3), Fourier-transforms every intermediate activation of the pose token, converts frequency power to power per rotation degree (formula in the docstring, self-tested on Wigner-D functions) |
| `attn.py` | splits the attention logits into their angular, scalar and distance terms; fits each head's angular logit as a window around one body-fixed direction |
| `interv.py` | multiplies the inverse temperatures at inference (all blocks, or one block) and measures rotation error and output spectrum; compares an untrained field with cosine attention against GATr's dot-product attention |
| `bottom_diag.py` | bottom-view pyramids (`reports/cosdir_bottom_view.md`): is the turn decodable from the image tokens, is it on the sphere grid, does the velocity contain it |
| `ft.py` | trains only the GATr field of a bottom-view checkpoint on frozen image tokens (CPU, about 3 min per pass over 4,500 images): standard flow matching, nearest coupling, or regression on the exact velocity |
| `snr.py` | gradient noise of the sampled flow-matching target against the exact target at the plateau checkpoint |
| `results/` | the JSON outputs used in the report (CPU, 2026-11-15) |

Everything runs on CPU (4 cores): `probe.py` takes about 2 minutes per image and flow time, `interv.py` about 20 minutes.
