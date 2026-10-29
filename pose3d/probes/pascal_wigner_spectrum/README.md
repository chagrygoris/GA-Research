# Wigner-D degrees on Pascal3D+

Pascal3D+ labels one pose per image, so the SYMSOL degree analysis (`../symsol_wigner_spectrum`) has two label-only
counterparts here (`spectrum.py`; labels: the RAM-cache rotations of `syfry5suvzovvakmuj/pascal3d-ram-cache` and the
classes from the annotation files, 18,371 train / 1,309 val):

1. **Single pose truncated at L** (`single_pose_resolution.png`). The Wigner-D target of a pose R0 scores R by
   f_L(w) = sum_{l<=L} (2l+1) chi_l(w), w = angle(R, R0), chi_l(w) = sin((2l+1) w/2) / sin(w/2). Its argmax is R0 for
   every L >= 1 (D^1 is the rotation matrix itself), so degree 1 identifies a single pose; higher degrees only
   narrow the peak. Half width at half maximum: L = 1: 80 deg, 2: 50, 3: 37, 4: 29, 6: 21, 8: 16, 10: 13, 15: 9.0,
   20: 6.8 deg. Image2Sphere's head stops at L = 6 (21 deg) and reaches ~9 deg medians through its sharp grid argmax.
2. **Class pose prior** (`class_prior_magnitude.png`, `class_prior_truncation.png`): c^l = mean over a class's
   training poses of sqrt(2l+1) D^l, its magnitude per degree, and the val error of the truncated prior's argmax
   (an image-free baseline). Concentrated priors (magnitude still > 0.3 at l = 10): bottle, train, bus, tvmonitor,
   sofa, diningtable; spread priors (< 0.15 at l = 10): aeroplane, bicycle, motorbike. The prior alone predicts
   bottle within 12–13 deg (median), train 16–17, diningtable 18 (7 val images), bus 22–31, tvmonitor 25–28; for
   aeroplane, bicycle, boat, car, motorbike it is 57–107 deg. Truncation degree barely changes these (the prior's
   mode is already set at L = 1–2).

Run from `pose3d/`: `python -m probes.pascal_wigner_spectrum.spectrum --labels pascal_labels.npz`.

## 3. A trained Image2Sphere truncated at L (`i2s_truncation.py`)

**Main checkpoint: `i2s_exact_warp_synth`** (`i2s_exact_truncation.png`, `.json`): upstream Image2Sphere's own training
script (Kaggle `syfry5suvzovvakmuj/i2s-exact-warp-synth-rtx`, 1 Oct 2026, `checkpoint.pt` in its output): ResNet-101,
warp + synthetic renders, SGD, 40 epochs; its final evaluation gave class-mean 9.70 deg, median 8.58 deg. Upstream feeds
the ResNet [0, 1] images without ImageNet normalisation, so it is loaded with `normalize_input=False`. Its Wigner-D
output psi (degrees 0..6) is kept up to degree L and decoded by the argmax over the level-5 SO(3) grid (2.36M
rotations, 1.875 deg) on the 1,309 val images of the RAM cache; the energy |psi_l|^2 is the variance degree l adds to
the predicted log-density over SO(3).

| L | median (deg) | class-mean median | acc@15 | acc@30 |
|---|---|---|---|---|
| 0 | 106.6 | 109.9 | 0.000 | 0.000 |
| 1 | 31.3 | 30.8 | 0.128 | 0.474 |
| 2 | 14.4 | 16.2 | 0.524 | 0.814 |
| 3 | 12.3 | 13.8 | 0.589 | 0.838 |
| 4 | 9.2 | 10.9 | 0.688 | 0.859 |
| 5 | 8.7 | 10.2 | 0.714 | 0.870 |
| 6 | 8.3 | 9.9 | 0.717 | 0.857 |

(L = 6 reproduces the run's own 9.70 / 8.58 deg to 0.2–0.3 deg; the cached val images and the run's loader differ only
in decoding.) Energy share of degrees 1..6: 0.18, 0.33, 0.12, 0.26, 0.06, 0.05.

* The pose is located coarsely by degrees <= 2 (median 14.4 deg, acc@30 0.81 at L = 2) and sharpened by degree 4
  (9.2 deg); degrees 5–6 add about 1 deg (class mean 10.9 -> 9.9). Even degrees carry most of the energy (2 and 4: 59 %).
* Per class (median at L = 2 / 4 / 6): bus 7.4 / 4.0 / 3.2, car 10.5 / 5.6 / 4.6, train 12.4 / 8.3 / 7.4, bottle 17.1 /
  8.9 / 7.4, aeroplane 14.3 / 10.1 / 9.4, diningtable 17.6 / 9.0 / 6.8 (7 images), boat 26.1 / 22.1 / 22.7 (degrees above
  3 do not help it), bicycle 21.3 / 16.4 / 12.5 and sofa 15.9 / 12.2 / 12.9.

**Earlier, weaker checkpoint** (`i2s_truncation.png`, `.json`): `i2s_real_r101_nosynth.pth` (W&B, 6 Sep 2026, repo
wrapper, real images only, median 10.9 deg). It decodes to the wrong side of SO(3) for L <= 2 (median 133–158 deg) and
needs degrees 3–4 (median 14.1 / 10.6 / 10.5 deg at L = 3 / 4 / 6; energy 0.04, 0.17, 0.37, 0.27, 0.08, 0.06), so how
the degrees are used depends on the training recipe, not only on the task.
