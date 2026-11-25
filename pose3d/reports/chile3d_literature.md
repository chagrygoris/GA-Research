# Why the 3D 7-gon's turn is not learned: what the literature suggests

Literature review, 2026-11-25, requested by a human after three 3D SYMSOL-CHILE 7-gon runs stayed at the guessing level
(`reports/cosdir_bottom_view.md`). The arXiv IDs were checked against the arXiv API. None of these papers is in
`literature/papers.yaml`, which covers geometric-algebra machine learning only; they fall outside its scope.

**The problem.** Every model learns the pyramid's axis but not its turn about the axis, so the median error stays at
90/7 = 12.9 degrees. The encoder's features carry the axis direction but not (cos 7 theta, sin 7 theta), where theta is
the turn about the pyramid's own axis. Starting the encoder from the Pascal3D+ backbone did not help. The same models
do learn the turn on the bottom views.

## Findings

1. **No direct precedent.** The SYMSOL II objects (tetX, cylO, sphX) break their symmetry with coloured or textured marks.
   - IPDF: Murphy et al., [2106.05965](https://arxiv.org/abs/2106.05965).
   - Image2Sphere: Klee et al., [2302.13926](https://arxiv.org/abs/2302.13926).
   - SO(2)-consistent encoder: Howell et al., [2307.03704](https://arxiv.org/abs/2307.03704).
   - SpyroPose: [2303.05308](https://arxiv.org/abs/2303.05308).
   - Flows on SO(3): Liu et al., [2304.03937](https://arxiv.org/abs/2304.03937).

   No paper found reports an untextured object of high symmetry order whose only cue is shading or silhouette.
2. **Data scale is the strongest direct evidence.**
   - SYMSOL methods train on 45k-100k views per shape.
   - IPDF's performance plateaus at about 50k. At 10k, IPDF and Image2Sphere lose much of their likelihood even on the
     easy coloured marks of SYMSOL II, and IPDF falls below uniform on SYMSOL I.
   - Our 4,500 views are half of the smallest setting any of these papers reports. The strong models also train much
     longer: Liu et al. use about 115M samples, against our 0.8M.
3. **Hard targets and plateaus.**
   - Spectral bias (Rahaman et al., [1806.08734](https://arxiv.org/abs/1806.08734)): as a function of pose, cos 7 theta
     oscillates 7 times faster than the axis, and such components are learned last.
   - Plateau-then-breakthrough dynamics (Barak et al., [2207.08799](https://arxiv.org/abs/2207.08799); saddle-to-saddle,
     [2106.15933](https://arxiv.org/abs/2106.15933)) match the bottom-view runs.
   - Documented fixes:
     - intermediate targets ("hints", Gülçehre & Bengio, [1301.4083](https://arxiv.org/abs/1301.4083));
     - decomposition (Shalev-Shwartz et al., [1703.07950](https://arxiv.org/abs/1703.07950));
     - easy-first curricula (Abbe et al., [2306.16921](https://arxiv.org/abs/2306.16921)).

     The evidence is theoretical or synthetic, but the qualitative fit is strong.
4. **Equivariant encoders** (G-CNNs [1602.07576](https://arxiv.org/abs/1602.07576), e2cnn
   [1911.08251](https://arxiv.org/abs/1911.08251), harmonic networks [1612.04642](https://arxiv.org/abs/1612.04642)) and
   canonicalisation (Kaba et al., [2211.06489](https://arxiv.org/abs/2211.06489)):
   - In-plane equivariance matches the turn only for bottom views; for tilted views only after the image is rectified
     with the predicted axis.
   - Howell et al.'s equivariant encoder gains almost nothing over Image2Sphere at 10% data.
   - No paper shows these resolving fine discrete symmetries. The evidence is weak to moderate.
5. **Near-symmetric objects outside SYMSOL:**
   - BB8 (see the survey [2006.05927](https://arxiv.org/abs/2006.05927)) treats the object as symmetric and adds a
     separate classifier for the small symmetry-breaking details.
   - Pitteri et al. ([1908.07640](https://arxiv.org/abs/1908.07640)) normalise poses under the symmetry group.
   - SARR ([2604.18208](https://arxiv.org/abs/2604.18208)) builds the symmetry order into the rotation representation.

## Interventions, ranked

1. **More data: render 45k-100k views and retrain unchanged.** Rendering costs nothing (the generator is seeded, and
   CPU or GPU rendering takes minutes). Data scale is the cheapest variable to rule out and the best-supported one.
2. **Auxiliary targets:** regress (cos 7 theta, sin 7 theta) from the encoder. A dense alternative is a heatmap of the 7
   base vertices treated as an unordered set (invariant under the 7 turns), or edge and face maps.
3. **Curriculum from bottom views to tilted views:** start where the model already succeeds and widen the tilt range,
   keeping some easy views throughout.
4. **Factorise as axis → rectify → turn:** a second stage on the image warped with the predicted axis, with a fine
   map and sharp attention as on the bottom views, or a C_N-equivariant encoder.
5. **Higher-resolution features** (14x14 / 28x28 maps throughout); weak literature support.
