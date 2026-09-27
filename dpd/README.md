# dpd

RF power amplifier behavioural modelling with geometric (Clifford) algebra.

**Best score**: Chebyshev memory polynomial, bands A+B, **-24.139 dB** held-out NMSE on band A
of `GeoData_TB` (no-model floor -10.892 dB; the capture's own vendor model reaches -33.451 dB).
The geometric-algebra model is exactly phase-equivariant but not yet competitive at -12.2 dB.

NMSE here is referred to the carrier, so it is negative and **lower is better** --
see [`docs/metrics.md`](docs/metrics.md) before comparing any number.

Setup and how to run: [`USAGE.md`](USAGE.md). Results: [`reports/`](reports/). Reading material:
[`awesome-reference/`](awesome-reference/README.md). A runnable tour of the data, metrics and both
models: [`notebooks/01_explore.ipynb`](notebooks/01_explore.ipynb).

| Status | Idea | Description |
|---|---|---|
| done | Port the supplied MATLAB model | Chebyshev memory polynomial, verified against the original `.m` files under Octave: -24.153 dB vs -24.157 dB, coefficients agreeing to 0.5%. |
| done | Clifford-equivariant gain | Replace the hand-designed envelope basis with a `Cl(2,0)` equivariant network. Exactly phase-equivariant, but -12.2 dB against the reference's -24.1 dB. |
| open | Give the Clifford model the reference delay taps | It currently uses a regular tap grid and 3 linear taps; the reference spends 15 hand-tuned delay triples with `s` in -2..10. The delay structure carries most of the reference's performance, so this is the first thing to try. |
| open | Un-compress the gate invariants | The invariants entering the gate are squared norms of unit-peak envelopes, so they concentrate near zero. Try `\|x\|` rather than `\|x\|^2`, or standardise them. |
| open | Deeper geometric-product trunk | Each block roughly doubles polynomial degree; two blocks reach degree 4 against the reference's degree 8 per dimension. |
| open | Warm-start the Clifford model from the reference | The reference is the special case where the gain is a separable magnitude polynomial. Initialising to reproduce a fitted one separates "can it represent the baseline" from "can it find it". |
| open | Spectrally weighted loss | Plain NMSE removes in-band distortion and leaves the out-of-band shoulders; the residual's ACPR is worse than the distortion it started from. Regulators care about the shoulders. |
