"""Blade-order bookkeeping for the projective algebra Cl(3,0,1).

Three bases for the same 16-dimensional algebra show up in this repo, and they do not agree
on the order of the blades nor on where the degenerate generator sits:

* **CGENN** (`clifford.algebra.CliffordAlgebra((1, 1, 1, 0))`) uses ShortLex order over its
  generators g1..g4 and takes its metric positionally, so the degenerate generator is the
  **last** one, g4. This is the basis the Clifford (CGENN) condition head computes in.
* **GATr** puts the degenerate generator **first** and calls it e0, with the basis
  1, e0, e1, e2, e3, e01, e02, e03, e12, e13, e23, e012, e013, e023, e123, e0123.
* **Cl(3,0)**, the 8-blade algebra the rest of the flow lives in (rotor, time, velocity).

`CL3_IN_CGENN_PGA` and `CGENN_TO_GATR` are the two maps between them. Both are verified to be
algebra homomorphisms in `tests/test_pga_blades.py`; the Cl(3,0) sub-blades of `CGENN_TO_GATR`
reproduce `gatr_denoiser._PGA_INDEX`, which is the same map taken directly from Cl(3,0).
"""

import torch

#: Metric for `CliffordAlgebra(...)`: three Euclidean generators, then the degenerate one.
CGENN_PGA_METRIC = (1, 1, 1, 0)

PGA_MV_DIM = 16
CL3_MV_DIM = 8

#: Cl(3,0) blade k -> its index among the 16 CGENN Cl(3,0,1) blades. Grade-major in both, so
#: this is an inclusion of subalgebras: 1, g1, g2, g3, g1g2, g1g3, g2g3, g1g2g3.
CL3_IN_CGENN_PGA = (0, 1, 2, 3, 5, 6, 8, 11)

#: CGENN Cl(3,0,1) blade k -> (index, sign) of the same blade in GATr's basis. The signs come
#: from reordering a product once the degenerate generator moves from last to first, e.g.
#: g1g4 = e1e0 = -e01.
CGENN_TO_GATR = (0, 2, 3, 4, 1, 8, 9, 5, 10, 6, 7, 14, 11, 12, 13, 15)
CGENN_TO_GATR_SIGN = (1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, 1, 1, -1)

#: Rotation bivectors g1g2, g1g3, g2g3 within the 16 CGENN Cl(3,0,1) blades. Grade 2 also
#: holds the three translation bivectors (g1g4, g2g4, g3g4), which an SO(3) task has no use
#: for, so a rotor read-out has to pick rather than take the whole grade.
CGENN_PGA_ROTATION_BIVECTOR = (5, 6, 8)

#: A rotor's blades (scalar + rotation bivectors) within the 16 CGENN Cl(3,0,1) blades.
CGENN_PGA_ROTOR_BLADES = (0, 5, 6, 8)


def embed_cl3_in_pga(x: torch.Tensor) -> torch.Tensor:
    """(..., 8) Cl(3,0) -> (..., 16) CGENN Cl(3,0,1), zero on every blade touching g4."""
    if x.shape[-1] != CL3_MV_DIM:
        raise ValueError(f"expected Cl(3,0) multivectors of width {CL3_MV_DIM}, got {x.shape[-1]}")
    out = x.new_zeros(*x.shape[:-1], PGA_MV_DIM)
    out[..., list(CL3_IN_CGENN_PGA)] = x
    return out


def project_pga_to_cl3(x: torch.Tensor) -> torch.Tensor:
    """(..., 16) CGENN Cl(3,0,1) -> (..., 8) Cl(3,0), dropping the blades that touch g4."""
    if x.shape[-1] != PGA_MV_DIM:
        raise ValueError(f"expected Cl(3,0,1) multivectors of width {PGA_MV_DIM}, got {x.shape[-1]}")
    return x[..., list(CL3_IN_CGENN_PGA)]


def cgenn_to_gatr(x: torch.Tensor) -> torch.Tensor:
    """(..., 16) in CGENN Cl(3,0,1) blade order -> the same multivectors in GATr's order."""
    if x.shape[-1] != PGA_MV_DIM:
        raise ValueError(f"expected Cl(3,0,1) multivectors of width {PGA_MV_DIM}, got {x.shape[-1]}")
    signs = x.new_tensor(CGENN_TO_GATR_SIGN)
    out = x.new_zeros(x.shape)
    out[..., list(CGENN_TO_GATR)] = x * signs
    return out


def gatr_to_cgenn(x: torch.Tensor) -> torch.Tensor:
    """Inverse of `cgenn_to_gatr`."""
    if x.shape[-1] != PGA_MV_DIM:
        raise ValueError(f"expected Cl(3,0,1) multivectors of width {PGA_MV_DIM}, got {x.shape[-1]}")
    signs = x.new_tensor(CGENN_TO_GATR_SIGN)
    return x[..., list(CGENN_TO_GATR)] * signs
