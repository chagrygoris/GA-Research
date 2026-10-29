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
