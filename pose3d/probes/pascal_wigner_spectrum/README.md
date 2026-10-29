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

## 3. A trained Image2Sphere truncated at L (`i2s_truncation.py`, `i2s_truncation.png`)

Checkpoint `i2s_real_r101_nosynth.pth` (W&B `clifforders/3D Pose Estimation`, run `i2s_real_r101_nosynth`: I2SReal,
ResNet-101, real images only, final median 10.9 deg). Its Wigner-D output psi (degrees 0..6) is kept up to degree L
and decoded by the argmax over the level-5 SO(3) grid (2.36M rotations, 1.875 deg), on the 1,309 val images of the
RAM cache. The energy |psi_l|^2 is the variance degree l adds to the predicted log-density over SO(3).

| L | median (deg) | class-mean median | acc@15 | acc@30 |
|---|---|---|---|---|
| 0 | 106.6 | 109.9 | 0.000 | 0.000 |
| 1 | 132.8 | 133.3 | 0.022 | 0.085 |
| 2 | 157.7 | 123.0 | 0.167 | 0.297 |
| 3 | 14.1 | 26.2 | 0.528 | 0.752 |
| 4 | 10.6 | 18.4 | 0.623 | 0.801 |
| 5 | 10.5 | 15.3 | 0.641 | 0.813 |
| 6 | 10.5 | 14.9 | 0.647 | 0.816 |

Energy share of degrees 1..6 (all val images): 0.04, 0.17, **0.37, 0.27**, 0.08, 0.06.

* I2S does not encode the pose in its low degrees: degrees <= 2 decode to the wrong side of SO(3) (median 133–158
  deg, worse than a random guess's ~126 deg), although degree 1 alone could identify a single pose (section 1). The
  distribution is built from l = 3 and 4 (64 % of the energy), and L = 4 already gives the full model's median
  (10.6 vs 10.5 deg); degrees 5–6 matter for the class mean only through a few classes.
* Per class (median, L = 3 / 4 / 6): bus 6.1 / 4.6 / 4.0, car 8.6 / 6.1 / 5.9, train 11.9 / 7.7 / 6.6, bottle 12.1 / 9.2
  / 8.3 (resolved at L = 3–4); sofa 65.0 / 18.5 / 11.7 and diningtable 82.0 / 75.7 / 46.6 need degrees 5–6 (the two
  classes with the most rotational ambiguity, cf. their concentrated priors in section 2).
