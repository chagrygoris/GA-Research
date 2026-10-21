r"""GPU smoke test for the HSE cHARISMa cluster: timed random multivector products.

Answers three questions in one short job, and nothing else:

1. does Slurm schedule the job and actually hand us the GPUs we asked for,
2. do those GPUs compute a geometric product *correctly* -- checked against a
   float64 CPU reference, not just ``nvidia-smi`` reporting that a device exists,
3. what throughput does one node sustain, as a baseline to size real runs against.

Deliberately self-contained: the algebra below duplicates
:class:`dpd.ga.CliffordAlgebra` instead of importing it, so the test needs
nothing but ``torch``.  A failure is then unambiguously a cluster problem and
never a broken repo install.  The product is written the same way ``dpd.ga``
writes it -- one outer product contracted against a flattened Cayley table --
so the number it reports is the throughput of the kernel the experiments use.

Exit status is 0 only if every visible GPU computed correctly, so the job log
can be read as a pass/fail answer without interpreting the numbers.
"""

from __future__ import annotations

import argparse
import os
import platform
import socket
import sys
import time

import torch

# Relative tolerance of the GPU product against the float64 CPU reference.
# Loose enough to pass on correct hardware, tight enough that a genuinely
# miscomputing GPU (or a silently downcast matmul) still fails.
_TOLERANCE = {
    torch.float64: 1e-12,
    torch.float32: 1e-5,
    torch.bfloat16: 5e-2,
    torch.float16: 5e-2,
}

_DTYPES = {
    "fp64": torch.float64,
    "fp32": torch.float32,
    "tf32": torch.float32,  # fp32 storage, TF32 matmul -- see --dtype handling
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def _reordering_sign(a: int, b: int) -> int:
    """Sign from sorting the concatenated basis-vector list ``a b`` into order."""
    a >>= 1
    total = 0
    while a:
        total += bin(a & b).count("1")
        a >>= 1
    return -1 if (total & 1) else 1


def flat_cayley_table(p: int, q: int, r: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """``(dim * dim, dim)`` flattened structure constants of ``Cl(p, q, r)``.

    Blade ``i`` is the product of the basis vectors whose bit is set in ``i``,
    so blade 0 is the scalar and ``2**n - 1`` the pseudoscalar.
    """
    n = p + q + r
    dim = 2**n
    metric = [1] * p + [-1] * q + [0] * r
    c = torch.zeros(dim, dim, dim, dtype=dtype)
    for i in range(dim):
        for j in range(dim):
            sign = _reordering_sign(i, j)
            shared = i & j
            for b in range(n):
                if shared >> b & 1:
                    sign *= metric[b]
            if sign:
                c[i, j, i ^ j] = sign
    return c.reshape(dim * dim, dim)


def geometric_product(a: torch.Tensor, b: torch.Tensor, flat_cayley: torch.Tensor) -> torch.Tensor:
    """Geometric product over the last axis: one outer product, one matmul."""
    dim = flat_cayley.shape[-1]
    outer = (a.unsqueeze(-1) * b.unsqueeze(-2)).reshape(*a.shape[:-1], dim * dim)
    return outer @ flat_cayley


def flops_per_product(dim: int) -> int:
    """FLOPs one product costs *as executed* (dense matmul against a sparse table).

    The table has only ``dim**2`` non-zeros out of ``dim**3``, so a sparse
    contraction would do ~``dim`` times less work.  Reported as executed because
    that is what the GPU is asked to do here and in ``dpd.ga``.
    """
    return dim * dim + 2 * dim * dim * dim


def verify_table_identities(p: int, q: int, r: int) -> list[str]:
    """Structural ``Cl`` axioms the Cayley table must satisfy, for any signature.

    Kept separate from :func:`check_against_cpu`, which compares the device
    against the CPU using *one shared table* and so cannot see a table that is
    wrong on both sides -- a mistyped signature, or a sign slip in the build.
    Returns the list of violated axioms, empty when the table is sound.
    """
    n = p + q + r
    dim = 2**n
    metric = [1] * p + [-1] * q + [0] * r
    table = flat_cayley_table(p, q, r).reshape(dim, dim, dim)
    eye = torch.eye(dim, dtype=torch.float64)
    failures: list[str] = []

    if not torch.equal(table[0], eye):
        failures.append("blade 0 is not a left identity")
    if not torch.equal(table[:, 0], eye):
        failures.append("blade 0 is not a right identity")

    for b in range(n):
        blade = 1 << b
        want = torch.zeros(dim, dtype=torch.float64)
        want[0] = metric[b]
        if not torch.equal(table[blade, blade], want):
            failures.append(f"e{b + 1}^2 != {metric[b]}")

    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = 1 << i, 1 << j
            if not torch.equal(table[bi, bj], -table[bj, bi]):
                failures.append(f"e{i + 1} and e{j + 1} do not anticommute")

    # ``e_i e_j`` is a single blade, namely ``i ^ j`` -- degenerate signatures
    # may send it to zero, but never anywhere else.
    for i in range(dim):
        for j in range(dim):
            support = (table[i, j] != 0).nonzero().flatten().tolist()
            if support not in ([], [i ^ j]):
                failures.append(f"blade {i} * blade {j} has support {support}, expected [{i ^ j}]")

    return failures


def check_against_cpu(device: torch.device, p: int, q: int, r: int, dtype: torch.dtype, batch: int = 4096) -> tuple[bool, float]:
    """Compare the device product against a float64 CPU product of the same inputs.

    Both sides contract the same table, so this grades the *hardware*: whether
    the GPU's arithmetic agrees with the CPU's.  The table itself is graded by
    :func:`verify_table_identities`.  Returns ``(passed, relative_error)``.
    """
    dim = 2 ** (p + q + r)
    ref_table = flat_cayley_table(p, q, r, torch.float64)

    generator = torch.Generator().manual_seed(0)
    a = torch.randn(batch, dim, dtype=torch.float64, generator=generator)
    b = torch.randn(batch, dim, dtype=torch.float64, generator=generator)
    # Exactly representable operands ride along with the random ones: a GPU that
    # mangles the easy cases should not hide behind rounding noise.
    eye = torch.eye(dim, dtype=torch.float64)
    a = torch.cat([a, eye, eye])
    b = torch.cat([b, eye, eye.flip(0)])

    expected = geometric_product(a, b, ref_table)
    got = geometric_product(
        a.to(device=device, dtype=dtype),
        b.to(device=device, dtype=dtype),
        ref_table.to(device=device, dtype=dtype),
    )

    error = (got.double().cpu() - expected).abs().max().item()
    scale = expected.abs().max().item()
    relative = error / scale if scale else error
    return relative <= _TOLERANCE[dtype], relative


def benchmark(device: torch.device, flat_cayley: torch.Tensor, batch: int, seconds: float, iters_per_sync: int = 20) -> dict[str, float]:
    """Hammer the device with random multivector products for ``seconds``.

    Operands are drawn once and reused: the point is to time the product, not
    the random number generator.  Nothing consumes the result inside the loop
    because eager PyTorch has no dead-code elimination -- every iteration really
    executes -- but a checksum is read at the end to prove it.
    """
    dim = flat_cayley.shape[-1]
    dtype = flat_cayley.dtype
    a = torch.randn(batch, dim, device=device, dtype=dtype)
    b = torch.randn(batch, dim, device=device, dtype=dtype)

    for _ in range(10):  # warm up: kernel autotuning, allocator, clocks
        out = geometric_product(a, b, flat_cayley)
    torch.cuda.synchronize(device)

    products = 0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        for _ in range(iters_per_sync):
            out = geometric_product(a, b, flat_cayley)
        torch.cuda.synchronize(device)
        products += iters_per_sync * batch
    elapsed = time.perf_counter() - start

    return {
        "elapsed_s": elapsed,
        "products": products,
        "products_per_s": products / elapsed,
        "gflops": products * flops_per_product(dim) / elapsed / 1e9,
        "peak_mem_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "checksum": out.double().sum().item(),
    }


def describe_environment() -> None:
    """Dump everything about the allocation that a later run might want to know."""
    print("=" * 72)
    print("environment")
    print("=" * 72)
    print(f"host                 {socket.gethostname()}")
    print(f"python               {platform.python_version()} ({sys.executable})")
    print(f"torch                {torch.__version__}")
    print(f"torch cuda build     {torch.version.cuda}")
    print(f"cuda available       {torch.cuda.is_available()}")
    for key in (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_ACCOUNT",
        "SLURM_JOB_NODELIST",
        "SLURM_CPUS_ON_NODE",
        "SLURM_JOB_GPUS",
        "SLURM_GPUS_ON_NODE",
        "CUDA_VISIBLE_DEVICES",
    ):
        print(f"{key:<20} {os.environ.get(key, '-')}")

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            print(
                f"gpu {index}                {properties.name}, "
                f"{properties.total_memory / 2**30:.0f} GiB, "
                f"sm_{properties.major}{properties.minor}, "
                f"{properties.multi_processor_count} SMs"
            )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=60.0, help="total GPU time budget, split across visible GPUs (default: 60)")
    parser.add_argument("--signature", default="3,0,0", help="Clifford signature p,q,r (default: 3,0,0; dpd uses 2,0,0)")
    parser.add_argument("--batch", type=int, default=1 << 19, help="multivectors per product (default: 524288)")
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="fp32", help="compute dtype (default: fp32)")
    args = parser.parse_args()

    p, q, r = (int(value) for value in args.signature.split(","))
    dim = 2 ** (p + q + r)
    dtype = _DTYPES[args.dtype]
    # TF32 silently lowers matmul precision, so it is opt-in rather than the
    # torch default -- otherwise the correctness check would be grading a
    # different kernel than the one named on the command line.
    torch.backends.cuda.matmul.allow_tf32 = args.dtype == "tf32"

    describe_environment()

    if not torch.cuda.is_available():
        print("FAIL: no CUDA device visible to torch.")
        print("      Either the job got no GPU (check --gpus and the partition) or")
        print("      this python has a CPU-only torch build (check torch cuda build above).")
        return 1

    print("=" * 72)
    print(f"Cl({p},{q},{r}): {dim} blades, {flops_per_product(dim)} FLOPs per product as executed")
    print(f"batch {args.batch} multivectors, dtype {args.dtype}")
    print("=" * 72)

    violations = verify_table_identities(p, q, r)
    if violations:
        print(f"FAIL: the Cl({p},{q},{r}) Cayley table violates its own axioms:")
        for violation in violations:
            print(f"      {violation}")
        return 1
    print("Cayley table            ok (identity, squares, anticommutation, blade support)")

    count = torch.cuda.device_count()
    per_gpu_seconds = args.seconds / count
    ok = True

    for index in range(count):
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        name = torch.cuda.get_device_properties(index).name

        passed, relative = check_against_cpu(device, p, q, r, dtype)
        ok &= passed
        verdict = "ok" if passed else "WRONG RESULTS"
        print(f"\ngpu {index} ({name})")
        print(f"  correctness vs float64 CPU   {verdict} (relative error {relative:.3e}, tolerance {_TOLERANCE[dtype]:.0e})")

        if not passed:
            continue

        table = flat_cayley_table(p, q, r).to(device=device, dtype=dtype)
        result = benchmark(device, table, args.batch, per_gpu_seconds)
        print(f"  elapsed                      {result['elapsed_s']:.1f} s")
        print(f"  products                     {result['products']:.3e}")
        print(f"  throughput                   {result['products_per_s']:.3e} products/s")
        print(f"  throughput                   {result['gflops']:.1f} GFLOP/s (dense, bandwidth-bound)")
        print(f"  peak memory                  {result['peak_mem_gib']:.2f} GiB")
        print(f"  checksum                     {result['checksum']:.6e}")

    print()
    print("=" * 72)
    if ok:
        print(f"PASS: Slurm scheduled this job and all {count} GPU(s) computed geometric products correctly.")
    else:
        print("FAIL: at least one GPU disagreed with the CPU reference -- see above.")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
