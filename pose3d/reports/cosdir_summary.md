# Cosine-attention GATr on bottom-view pyramids: what was learned, what stalled

Summary of `reports/cosdir_mechanism.md` (how the model represents high rotation degrees), `reports/cosdir_bottom_view.md`
(the bottom views of SYMSOL-CHILE, all runs) and `reports/dit_cos_vector.md` (the DiT control), November 2026. Model: the
cosine-attention GATr (`arynabiaroza/symsol-gatr-spherecloud-cosdir-h8`): image tokens on a hemisphere of directions,
attention logit s_h * cos(angle) between query and key directions, s_h a learnable inverse temperature per head.
Task: an n-sided pyramid seen from straight below, so only its turn about the optical axis varies; a model that does not
resolve the turn guesses and has median error about 90/n degrees (the "guessing level").

## How the model works (mechanism)

* With a cosine logit, each attention head is a soft spotlight on the sphere: the weight on a key falls off as
  exp(s_h (cos - 1)), a von Mises-Fisher window around the query's direction. Its width is set by s_h (concentration
  kappa about 0.4 s_h), and it passes an n-fold pattern with weight a_n = i_n(kappa) / i_0(kappa).
* A model that learns the turn of an n-gon shows an n-lobed pattern on rings of image tokens that turns with the object,
  and a few pose-token heads whose narrow windows sit on that ring and slide along it as the pose hypothesis turns
  (`figures/cosdir_vmf_windows_3d.png`: tokens on the hemisphere, and one ring-reading head's window for four pose hypotheses). Most heads look elsewhere.

## What was learned

| pyramid | what it took | result (final 32-sample median) |
|---|---|---|
| square, pentagon | a constant learning rate instead of the 40-epoch cosine schedule (the plateau outlasted the schedule) | learned (pentagon about 1 degree) |
| 7-gon | 14x14 feature map + start s_h >= 60 (windows pass a_7 about 0.33) | learned in 2 of 4 runs within 120 epochs (0.40, 0.46) |
| 10-gon | 14x14 map + start s_h 120 (a_10 about 0.31) + 240 epochs | learned in 1 of 1 (breakthrough at epoch about 175; 0.40) |
| 13-gon | 28x28 feature map + start s_h 200 (a_13 about 0.37) | learned in 2 of 3 (breakthrough at epochs 84 and 185; 0.46 for seed 0; seed 2 stayed at 6.66) |

## Why some stalled, and what fixed them

1. **Training schedule** (square, pentagon): learning comes in steps separated by plateaus; the cosine schedule had cut
   the learning rate before the second plateau ended.
2. **Window width** (7-gon at s_h 30, 10-gon and 13-gon at 60): the windows were too wide to pass the n-fold pattern
   (a_n 0.07, 0.10, 0.03). Starting s_h so that a_n is about 0.3 fixed it. s_h itself barely moves during training
   (weight decay pulls it down a little), so the start value is what counts.
3. **Feature-map resolution** (7-gon at 7x7, 13-gon at 14x14): the ResNet map samples the silhouette's rim with a fixed
   number of cells, and an n-lobed pattern needs at least 2 cells per lobe. Cells per lobe: 7-gon 1.6 at 7x7 (failed),
   3.2 at 14x14 (learned); 13-gon 1.7 at 14x14 (no pattern formed at all), 3.5 at 28x28 (learned). The 28x28 map needed
   GATr block checkpointing to fit a T4.
4. **Long, random plateaus** (all): even with everything right, the breakthrough comes after a plateau whose length
   varies a lot between identical runs (7-gon: epoch 61 vs not within 120; 13-gon: epochs 84, 185 and never within 240).

## Why the plateau ends suddenly

* Not grokking in the strict sense: training and validation loss stay flat and fall together.
* The turn reaches the output as a product: window pass a_n x strength of the n-lobed pattern x readout weight, with the
  last two near zero on the plateau. Each grows at a rate set by the others, so the product stays tiny for a long time
  and then rises within a few epochs (the deep-linear-network plateau of Saxe, McClelland & Ganguli 2014).
* A toy model of exactly this geometry reproduces it: plateau length about 1/a_n with gradient descent (a_n^-0.55 with
  Adam), sudden drop at the end (`figures/cosdir_toy_plateau.png`).
* In the real runs the n-lobed pattern first lines up with the turn (phase coherence 0.9 or more) while still weak, then
  its amplitude rises faster than exponentially; the window width does not change (10-gon and 13-gon snapshots).
* **s_h is not the cause** of a breakthrough: it jumps only at the early, image-independent transitions, a frozen s_h
  makes the same transitions, and it stays flat through every breakthrough.

## Pascal3D+

* The cosine GATr with the 14x14 map, ResNet-101, warp + synthetic renders, 50 epochs on 4xL4: class-mean median
  8.97 degrees, median 8.00, level with the 100-epoch reference recipe (8.95 / 7.85) and better than the dircloud GATr
  P1 (9.35) and Image2Sphere (9.8); one seed (`reports/pascal_cosdir.md`, `figures/pascal_cosdir_e50.png`).

## What stalled or did not help

* **Spreading the heads' s_h** from 30 to 250: neither the 10-gon nor the 13-gon learned in 180 epochs.
* **Moving s_h 30x faster**: heads spread widely, the 7-gon still did not learn.
* **Image2Sphere with L = 6** (no degree-7 component): stays at 6 to 7 degrees on the 7-gon under every change.
* **DiT with a scaled cosine on 3 dimensions per head** (SYMSOL I): worse than the plain DiT (26.8 vs 19.4 degrees), and
  its s_h fell. The cosine window seems to need equivariant features underneath (vectors that turn with the object).
* **10-gon at 28x28** (control, 4.5 cells per lobe): stayed at the guessing level for 240 epochs (final 9.54 degrees), although it learned at 14x14 (2.3 cells
  per lobe): the finer map does not shorten the plateau; prediction of an earlier breakthrough refuted (`figures/cosdir_g28_curves.png`).
* **13-gon at 28x28, seed 2**: stayed at the guessing level for 240 epochs (final 6.66 degrees), with settings identical to the two seeds
  that learned except the seed.
* **The bottom-view recipe on the 3D SYMSOL-CHILE 7-gon** (poses on all of SO(3); 14x14 map, s_h 100, 180 epochs,
  2 seeds): both stayed at the guessing level (12.74 and 13.11 degrees), as Image2Sphere did (12.67)
  (`figures/cosdir_chile3d_pyr7.png`). Likely: the lobed ring is off-centre and foreshortened for a tilted pyramid, and
  4,500 views cover SO(3) sparsely; untested.
* **The 3D 7-gon with the Pascal3D+ ResNet-101 as the starting encoder** (the backbone that carries some of the turn
  for near-bottom views in CPU read-outs): also at the guessing level, 13.31 degrees.
* **More training views for the 3D 7-gon** (13,500 and 45,000 instead of 4,500; ~20k steps each): guessing level
  (13.37 and 13.16 degrees); against images seen the curves coincide (`figures/cosdir_chile3d_datascale.png`).
* **More training views for the 3D 7-gon** (13,500 and 45,000 instead of 4,500; ~20k steps each): guessing level
  (13.37 and 13.16 degrees); against images seen the curves coincide (`figures/cosdir_chile3d_datascale.png`).
* **A tilt curriculum for the 3D 7-gon** (bottom views first, widening to all tilts by mid-training): 14.13 degrees,
  guessing level; the encoder carries no turn even at small tilts. The easy stage (about 80k images under 30 degrees)
  was much shorter than the about 300k images the bottom view needed to break through.
* **A gated tilt curriculum for the 3D 7-gon** (widen only when the current range is learned): never left the first
  stage, tilt <= 10 degrees, in 750k images (12.69); exact bottom views with the same model broke through at about
  300k (`figures/cosdir_chile3d_gated.png`).
* Not explained: why identical runs break through at such different epochs (the toy model's seeds agree).

## Bugs fixed along the way

* Resume: the RNG state moved to the GPU by `map_location` broke every resume (`engine/resume.py` of the code snapshot;
  patched in the runners, the permanent fix belongs in that snapshot). `--n_epochs` could not change on resume.
* Kaggle: the 28x28 map ran out of memory on a T4 until GATr block checkpointing was switched on.
