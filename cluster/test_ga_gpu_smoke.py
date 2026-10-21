"""Guards on the smoke test, runnable anywhere -- no GPU and no cluster needed.

The point of most of these is the duplication in :mod:`ga_gpu_smoke`: its
algebra is a copy of :class:`dpd.ga.CliffordAlgebra`, kept so the smoke test
depends on nothing but ``torch``, and a copy silently drifts.  The rest check
that the two correctness gates in the smoke test have teeth, since a gate that
cannot fail would make a broken GPU look fine.

    pytest cluster/test_ga_gpu_smoke.py
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest
import torch

_HERE = pathlib.Path(__file__).parent


def _load(name: str, path: pathlib.Path):
    """Import by path: ``dpd/__init__.py`` pulls in the whole data stack, and
    ``cluster/`` is not a package."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load("ga_gpu_smoke", _HERE / "ga_gpu_smoke.py")
ga = _load("dpd_ga", _HERE.parent / "dpd" / "dpd" / "ga.py")

SIGNATURES = [(2, 0, 0), (3, 0, 0), (0, 2, 0), (1, 3, 0), (3, 0, 1), (4, 1, 0)]
CPU = torch.device("cpu")


@pytest.mark.parametrize("signature", SIGNATURES)
def test_table_is_identical_to_dpd_ga(signature):
    """The copied Cayley table must stay bit-identical to the one in dpd.ga."""
    mine = smoke.flat_cayley_table(*signature, torch.float64)
    theirs = ga.CliffordAlgebra(*signature)._flat_cayley.double()
    assert mine.shape == theirs.shape
    assert torch.equal(mine, theirs)


@pytest.mark.parametrize("signature", SIGNATURES)
def test_product_is_identical_to_dpd_ga(signature):
    algebra = ga.CliffordAlgebra(*signature)
    a = torch.randn(256, algebra.dim, dtype=torch.float64)
    b = torch.randn(256, algebra.dim, dtype=torch.float64)
    table = smoke.flat_cayley_table(*signature, torch.float64)
    assert torch.allclose(smoke.geometric_product(a, b, table), algebra.geometric_product(a, b))


@pytest.mark.parametrize("signature", SIGNATURES)
def test_table_identities_accept_a_sound_table(signature):
    assert smoke.verify_table_identities(*signature) == []


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda t: t.__setitem__((1 * 8 + 2, 3), -t[1 * 8 + 2, 3]), id="flipped-sign"),
        pytest.param(lambda t: t.__setitem__((1 * 8 + 1, 0), 7.0), id="wrong-square"),
        pytest.param(lambda t: t.__setitem__((1 * 8 + 2, 5), 1.0), id="stray-blade"),
        pytest.param(lambda t: t.__setitem__((0 * 8 + 4, 4), 2.0), id="broken-identity"),
    ],
)
def test_table_identities_reject_a_corrupted_table(monkeypatch, mutate):
    """A table that is wrong on *both* sides is invisible to the CPU/GPU
    comparison, so verify_table_identities is the only thing that can catch it."""
    real = smoke.flat_cayley_table

    def corrupted(p, q, r, dtype=torch.float64):
        table = real(p, q, r, dtype).clone()
        mutate(table)
        return table

    monkeypatch.setattr(smoke, "flat_cayley_table", corrupted)
    assert smoke.verify_table_identities(3, 0, 0) != []


@pytest.mark.parametrize("name", sorted(smoke._DTYPES))
def test_device_check_passes_on_correct_hardware(name):
    """Run device-side against the CPU: the tolerances must accept agreement."""
    passed, relative = smoke.check_against_cpu(CPU, 3, 0, 0, smoke._DTYPES[name], batch=1024)
    assert passed, f"{name} failed its own tolerance with CPU on both sides (error {relative:.3e})"


def test_device_check_rejects_a_miscomputing_device(monkeypatch):
    """A device 1% off must fail -- otherwise the gate would pass anything."""
    real = smoke.geometric_product
    calls = {"n": 0}

    def flaky(a, b, table):
        calls["n"] += 1
        out = real(a, b, table)
        return out * 1.01 if calls["n"] == 2 else out  # the 2nd call is the device one

    monkeypatch.setattr(smoke, "geometric_product", flaky)
    passed, relative = smoke.check_against_cpu(CPU, 3, 0, 0, torch.float32, batch=1024)
    assert not passed
    assert relative == pytest.approx(1e-2, rel=0.1)


def test_flops_per_product_counts_the_outer_product_and_the_matmul():
    assert smoke.flops_per_product(8) == 8 * 8 + 2 * 8 * 8 * 8 == 1088


def test_benchmark_reports_consistent_numbers(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 2**20)
    table = smoke.flat_cayley_table(3, 0, 0, torch.float32)
    result = smoke.benchmark(CPU, table, batch=2048, seconds=0.5, iters_per_sync=5)

    assert result["elapsed_s"] >= 0.5
    assert result["products"] > 0
    assert result["products_per_s"] == pytest.approx(result["products"] / result["elapsed_s"])
    expected = result["products"] * smoke.flops_per_product(8) / result["elapsed_s"] / 1e9
    assert result["gflops"] == pytest.approx(expected)
