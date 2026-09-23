#!/usr/bin/env python3
"""Verification gate runner. CI fails the build if any active gate regresses.

Each gate is registered against the roadmap stage that introduces it. A stage's
gates must pass before code for the next stage is merged -- that rule is what
keeps scope from running ahead of verification.

Gates needing a binary that was not built are skipped with a notice, so the
suite still runs on a machine without Kokkos, MPI, CGNS or PETSc.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTO = ROOT / "prototype"
FIX = ROOT / "tests" / "fixtures"
BUILD = ROOT / "build" / "tests"
PY = sys.executable
ENV = {**os.environ, "OMP_PROC_BIND": "false"}


def run(cmd, cwd=ROOT):
    return subprocess.run(cmd, cwd=cwd, env=ENV).returncode == 0


def parallel_sweep(ranks=(2, 3, 4), binary_name="mms_parallel", args=()):
    """The answer must not depend on the rank count.

    Catches a wrong halo exchange, which still converges to a plausible-looking
    but different solution -- no other gate sees it. Removing a single exchange
    from the momentum diagonal shifts the answer by 0.45%: large enough to be
    wrong, small enough that an order study would still report second order.
    """
    binary = BUILD / binary_name
    if not binary.exists():
        print("  mms_parallel not built"); return None
    if shutil.which("mpirun") is None:
        print("  mpirun not available"); return None

    r = subprocess.run([str(binary), *args], cwd=ROOT, env=ENV,
                       capture_output=True, text=True)
    m = re.search(r"L2 ([\d.eE+-]+)", r.stdout)
    if r.returncode != 0 or not m:
        print(r.stdout or r.stderr); return False
    ref = m.group(1)
    print(f"  serial reference L2 = {ref}")
    ok = True
    for np in ranks:
        cp = subprocess.run(
            ["mpirun", "--oversubscribe", "--allow-run-as-root", "-n", str(np),
             str(binary), *args, ref],
            cwd=ROOT, env={**ENV, "OMP_NUM_THREADS": "1"},
            capture_output=True, text=True)
        print((cp.stdout or cp.stderr).rstrip())
        ok &= cp.returncode == 0
    print(f"  parallel consistency gate: {'PASS' if ok else 'FAIL'}")
    return ok


GATES = {
    "v0": [
        ("python: MMS diffusion is second order",
         lambda: run([PY, str(PROTO / "mms_diffusion.py"), "8", "16", "32"], PROTO), None),
        ("c++: mesh geometry matches the Python reference",
         lambda: run([str(BUILD / "test_geometry"), str(FIX)]), BUILD / "test_geometry"),
        ("c++: MMS diffusion is second order",
         lambda: run([str(BUILD / "mms_diffusion"), str(FIX)]), BUILD / "mms_diffusion"),
        ("cross-check: python and c++ L2 errors agree",
         lambda: run([PY, str(Path(__file__).parent / "crosscheck.py")]), None),
        ("cgns: external mesh gives identical geometry",
         lambda: run([str(BUILD / "test_cgns"), str(FIX)]), BUILD / "test_cgns"),
        ("vtu: ParaView output round-trips through meshio",
         lambda: run([PY, str(ROOT / "tests" / "unit" / "check_vtu.py")]), BUILD / "test_vtu"),
        ("mpi: the answer is independent of the rank count",
         lambda: parallel_sweep(args=(str(FIX),)), BUILD / "mms_parallel"),
        ("backends: every linear solver gives the same solution",
         lambda: run([str(BUILD / "test_backends"), str(FIX)]), BUILD / "test_backends"),
    ],
    "v1": [
        ("symbolic: the exact solutions really satisfy the equations",
         lambda: run([PY, str(Path(__file__).parent / "verify_exact.py")]), None),
        ("python: convection-diffusion is second order at high Peclet",
         lambda: run([PY, str(PROTO / "mms_convection.py"), "8", "16", "32"], PROTO), None),
        ("python: Navier-Stokes vs the Ethier-Steinman exact solution",
         lambda: run([PY, str(PROTO / "ethier_steinman.py")], PROTO), None),
        ("c++: convection-diffusion is second order at high Peclet",
         lambda: run([str(BUILD / "mms_convection"), "8", "16", "32"]),
         BUILD / "mms_convection"),
        ("c++: Navier-Stokes vs the Ethier-Steinman exact solution",
         lambda: run([str(BUILD / "ethier_steinman"), "8", "16", "32"]),
         BUILD / "ethier_steinman"),
        ("cross-check: python and c++ convection agree",
         lambda: run([PY, str(Path(__file__).parent / "crosscheck_v1.py")]),
         BUILD / "mms_convection"),
        ("cross-check: python and c++ Navier-Stokes agree",
         lambda: run([PY, str(Path(__file__).parent / "crosscheck_ns.py")]),
         BUILD / "ethier_steinman"),
        ("mpi: Navier-Stokes is independent of the rank count",
         lambda: parallel_sweep(binary_name="mms_parallel_ns"),
         BUILD / "mms_parallel_ns"),
        ("open domain: inlet/outlet conserves mass exactly",
         lambda: run([str(BUILD / "open_domain"), "12"]), BUILD / "open_domain"),
        ("benchmark: lid-driven cavity against Ghia et al. (1982)",
         lambda: run([str(BUILD / "cavity"), "64"]), BUILD / "cavity"),
    ],
    # v1 remaining: cylinder_strouhal
    # v2: flat_plate_cf, backward_step, rayleigh_benard
    # v3: sod_shock_tube, naca0012_transonic
    # v4: dam_break, rising_bubble
}


def main():
    stages = sys.argv[1:] or ["v0"]
    failures, skipped = [], []
    for stage in stages:
        for name, fn, needs in GATES.get(stage, []):
            if needs is not None and not needs.exists():
                skipped.append(f"[{stage}] {name}  (not built)")
                continue
            print(f"=== [{stage}] {name}")
            result = fn()
            if result is None:
                skipped.append(f"[{stage}] {name}  (prerequisite missing)")
            elif not result:
                failures.append(f"[{stage}] {name}")
    print()
    for s in skipped:
        print(f"SKIPPED: {s}")
    if skipped:
        print("  build everything with: cmake -S . -B build && cmake --build build -j")
    for f in failures:
        print(f"FAILED: {f}")
    if failures:
        return 1
    print(f"All active gates passed for: {', '.join(stages)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
