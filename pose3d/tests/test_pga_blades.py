"""The Cl(3,0) -> Cl(3,0,1) -> GATr blade maps are algebra homomorphisms.

`--cond_algebra pga` hands multivectors across three bases that disagree on blade order and on
where the degenerate generator sits, so the maps are worth pinning down: a wrong sign here is a
silently non-equivariant model, not a crash.
"""

import pytest
import torch
from clifford.algebra.cliffordalgebra import CliffordAlgebra

from pose3d.geometry.pga import (
    CGENN_PGA_METRIC,
    CGENN_PGA_ROTOR_BLADES,
    cgenn_to_gatr,
    embed_cl3_in_pga,
    gatr_to_cgenn,
    project_pga_to_cl3,
)
from pose3d.geometry.rotor import ROTOR_BLADES
from pose3d.models.gatr_denoiser import _ensure_xformers_stub

_ensure_xformers_stub()  # GATr hard-imports xformers; see models/gatr_denoiser.py

CL3 = CliffordAlgebra((1, 1, 1))
PGA = CliffordAlgebra(CGENN_PGA_METRIC)


def test_pga_algebra_shape():
    assert PGA.n_blades == 16
    assert PGA.subspaces.tolist() == [1, 4, 6, 4, 1]
    # 34 of the 125 grade triples carry a nonzero geometric product, against 20 of 64 in Cl(3,0).
    assert int(PGA.geometric_product_paths.sum()) == 34
    assert int(CL3.geometric_product_paths.sum()) == 20


def test_cl3_embeds_in_pga_as_a_subalgebra():
    a, b = torch.randn(32, 8), torch.randn(32, 8)
    lhs = PGA.geometric_product(embed_cl3_in_pga(a), embed_cl3_in_pga(b))
    rhs = embed_cl3_in_pga(CL3.geometric_product(a, b))
    torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)


def test_cl3_pga_roundtrip():
    a = torch.randn(32, 8)
    torch.testing.assert_close(project_pga_to_cl3(embed_cl3_in_pga(a)), a)


def test_cgenn_gatr_roundtrip():
    a = torch.randn(32, 16)
    torch.testing.assert_close(gatr_to_cgenn(cgenn_to_gatr(a)), a)


def test_rotor_blades_agree_across_the_two_algebras():
    """A rotor's blades in Cl(3,0,1) are the images of its blades in Cl(3,0)."""
    marker = torch.zeros(1, 8)
    marker[0, list(ROTOR_BLADES)] = 1.0
    assert embed_cl3_in_pga(marker)[0].nonzero().flatten().tolist() == list(CGENN_PGA_ROTOR_BLADES)


def test_cgenn_to_gatr_is_an_algebra_homomorphism():
    _ensure_xformers_stub()
    gatr_primitives = pytest.importorskip(
        "gatr.primitives", reason="GATr package not installed")
    a, b = torch.randn(32, 16), torch.randn(32, 16)
    lhs = gatr_primitives.geometric_product(cgenn_to_gatr(a), cgenn_to_gatr(b))
    rhs = cgenn_to_gatr(PGA.geometric_product(a, b))
    torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)


def test_composed_map_matches_the_direct_cl3_embedding():
    """Cl(3,0) -> Cl(3,0,1) -> GATr must equal the Cl(3,0) -> GATr map already in the repo."""
    from pose3d.models.gatr_denoiser import _PGA_INDEX

    composed = cgenn_to_gatr(embed_cl3_in_pga(torch.eye(8)))
    direct = torch.zeros(8, 16)
    direct[torch.arange(8), list(_PGA_INDEX)] = 1.0
    torch.testing.assert_close(composed, direct)
