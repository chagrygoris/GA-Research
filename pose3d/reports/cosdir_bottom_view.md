# Why the cosine-attention GATr guesses the turn of bottom-view pyramids

Investigation, 2026-11-16. Scripts and numbers: `probes/cosdir_interp/` (`bottom_diag.py`, `ft.py`, `snr.py`).
Background: `reports/cosdir_mechanism.md` (how this model reaches high rotation degrees) and
`reports/symmetry_spectrum.md` on branch `claude/symsol-symsol2-differences-2dukos` (SYMSOL-CHILE).

## The task and the failure

* **Bottom view** (SYMSOL-CHILE, `--view bottom`): a pyramid with a regular `n`-sided base, seen from straight
  below. The only thing that varies is the turn `theta` about the optical axis, so the pose is `R = Rz(theta)`,
  and turns that differ by `360/n` degrees are the same image.
* **Model**: the cosine-direction attention GATr of `reports/cosdir_mechanism.md` (a flow model: a ResNet-50 turns
  the image into 89 tokens on the sphere, a GATr predicts the velocity of the pose), trained for 40 epochs with a
  cosine learning-rate schedule, one run per `n` (Kaggle kernels `bottom{4,5,7,10,13}-gatr-spherecloud-cosdir-h8-t4x2`).
* **Result of those runs**: the median rotation error ends at 22.6, 18, 13, 9 and 7 degrees for `n` = 4, 5, 7, 10,
  13. That is `90/n` degrees, the error of a model that knows the pose is a turn about the axis and guesses the
  turn.

All five runs have the same curve: a first plateau at 35 to 45 degrees for 17 to 18 epochs, one drop to `90/n`
between epochs 17 and 19, then nothing.

| n | median error by epoch (every 3rd) |
|---|---|
| 4 | 75 48 46 44 44 44 **24** 23 23 23 23 23 23 23 |
| 5 | 73 45 43 42 41 40 **18** 18 18 18 18 18 18 18 |
| 7 | 72 43 38 38 37 37 **15** 13 13 13 13 13 13 13 |
| 10 | 71 41 37 36 36 36 **17** 9 9 9 9 9 9 9 |
| 13 | 73 41 37 35 35 34 **8** 7 7 7 7 7 7 7 |

## Findings (square, `n` = 4, final checkpoint of the run, 1,000 test images)

**1. The image encoder knows the turn.** A linear read-out of `(cos 4 theta, sin 4 theta)` from the 89 x 64 scalar
features that the GATr receives reaches a held-out R² of 1.00 (0.99 even from their mean over tokens; 1.00 from
the ResNet map).

**2. The turn is on the sphere grid.** On the rings of 16 grid points around the optical axis, 15 to 25 % of the
signal's variance along the ring is at azimuthal frequency 4, and its phase follows the turn (coherence 0.76 to
0.80, where 1 is perfect). The rings of 4 and 8 points near the axis cannot hold frequency 4, as predicted, but
the outer rings can.

**3. The velocity already contains the right component, about 100 times too small.** For a fixed image, the
predicted velocity about the axis as a function of the pose's turn is a clean wave of frequency 4 whose phase
matches the true turn (aligned part 0.0055 of an amplitude 0.0055 at flow time 0.5; 0.0205 of 0.0208 at 0.9). The
correct field has amplitude of order 1.

**4. How far the model is from the best possible field.** With `v*` the exact velocity that flow matching
converges to (the average of the directions to the `n` equivalent poses, weighted by how likely each is given the
current pose; `vstar` in `ft.py`), and "blind" the same for a model that never sees the image:

| mean squared distance | value |
|---|---|
| blind field to `v*` (what knowing the turn is worth) | 0.789 |
| trained model to `v*` | 0.747 |
| trained model to the blind field | 0.078 |

The training loss at the plateau is 2.33, so the turn is worth a third of it, and the model has collected 5 % of
that. It is the blind field.

**5. The architecture can learn it, and so can plain flow matching, given more steps.** Training only the GATr
field from the final checkpoint, on frozen image tokens, AdamW at the run's full learning rate 1.41e-4, batch 64,
one (noise, time) sample per image (so one pass over the 4,500 images is 70 steps):

| passes | exact velocity as the target | standard flow matching (control) |
|---|---|---|
| 0 | 24.0 | 24.0 |
| 3 | 24.1 | 23.9 |
| 5 | 22.3 | 22.8 |
| 6 | 20.1 | 20.8 |
| 7 | 15.4 | 19.1 |
| 8 | 8.0 | 12.1 |
| 9 | 4.2 | 5.3 |
| 10 | 2.4 | 3.4 |
| 11 | 2.6 | 3.7 |
| 14 | 1.8 | (stopped at 11) |

(median error in degrees to the nearest equivalent pose, 200 test images, one sample each, 20 flow steps). Both
leave the plateau after about 350 steps and resolve the turn after about 700. The unchanged model, objective and
data therefore do learn the square's turn.

**6. The noise of the flow-matching target is not the obstacle.** At the plateau checkpoint the sampled target
adds a gradient noise of norm 8.7 per batch of 64, on top of a batch-to-batch scatter of 16 that the exact target
has as well; and the control in 5 follows the exact-target run with a delay of about one pass.

## Conclusion

The failure of these runs is a training-schedule failure, not a limit of the representation:

* Learning proceeds in two steps separated by plateaus. The first step (the pose is a turn about the optical
  axis, which does not depend on the image) takes 17 to 18 of the 40 epochs in every run.
* When the model reaches the second plateau, the cosine schedule has already lowered the learning rate to 60 %,
  and to 19 % by epoch 30. The 22 remaining epochs amount to about 410 steps at the full rate, the number of steps
  after which the continued runs above only begin to leave the plateau.

This also answers the earlier prediction from `reports/cosdir_mechanism.md`: the square and the pentagon are within
the model's reach, as the window argument said. Whether the 7-, 10- and 13-gon follow with more training is open:
for those the sphere grid (at most 16 points per ring) and the window width are real limits.

## Limits

* Field-only training with the image encoder frozen and a fresh optimiser, on the square only. The original runs
  train the encoder too, with batch 512 (8 samples per image); that should help rather than hurt, but it was not
  run.
* Why the first, image-independent step takes 17 epochs was not investigated. It costs half the budget in every
  run and looks like the cheaper thing to fix.
* A run with nearest coupling (each noise sample flows to its nearest equivalent pose) was started from the same
  checkpoint and had not left the plateau after 3 passes; it is not conclusive.

## Next experiments (GPU, need approval)

1. The same bottom-view runs with a constant learning rate, or 120 epochs: `n` = 4 and 5 first. Expected from the
   table above: the turn resolves soon after the first drop.
2. The same for `n` = 7, 10, 13, to see where the grid and window limits start to bite.
3. Find out why the ring of possible poses takes 17 epochs (for instance: start from a checkpoint trained on
   another `n`; the first step should transfer).
