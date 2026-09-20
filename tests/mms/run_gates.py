#!/usr/bin/env python3
"""Verification gate runner. CI fails the build if any active gate regresses.

Each gate is registered against the roadmap stage that introduces it. A stage's
gates must pass before code for the next stage is merged -- that rule is what
keeps scope from running ahead of verification.

C++ gates are skipped with a notice when the build tree is absent, so the suite
still runs on a machine without Kokkos.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTO = ROOT / "prototype"
FIX = ROOT / "tests" / "fixtures"
BUILD = ROOT / "build" / "tests"

PY = sys.executable
ENV_OMP = {"OMP_PROC_BIND": "false"}

GATES = {
    "v0": [
        ("python: MMS diffusion, order 2 on orthogonal + skewed mesh",
         [PY, str(PROTO / "mms_diffusion.py"), "8", "16", "32"], PROTO, None),
        ("c++: mesh geometry matches the Python reference",
         [str(BUILD / "test_geometry"), str(FIX)], ROOT, BUILD / "test_geometry"),
        ("c++: MMS diffusion, order 2 on orthogonal + skewed mesh",
         [str(BUILD / "mms_diffusion"), str(FIX)], ROOT, BUILD / "mms_diffusion"),
        ("cross-check: python and c++ L2 errors agree",
         [PY, str(Path(__file__).parent / "crosscheck.py")], ROOT, None),
    ],
    # v1: taylor_green, cavity_ghia, cylinder_strouhal
    # v2: flat_plate_cf, backward_step, rayleigh_benard
    # v3: sod_shock_tube, naca0012_transonic
    # v4: dam_break, rising_bubble
}


def main():
    stages = sys.argv[1:] or ["v0"]
    failures, skipped = [], []
    import os
    env = {**os.environ, **ENV_OMP}
    for stage in stages:
        for name, cmd, cwd, needs in GATES.get(stage, []):
            if needs is not None and not needs.exists():
                skipped.append(f"[{stage}] {name}  (not built)")
                continue
            print(f"=== [{stage}] {name}")
            if subprocess.run(cmd, cwd=cwd, env=env).returncode != 0:
                failures.append(f"[{stage}] {name}")
    print()
    for s in skipped:
        print(f"SKIPPED: {s}")
    if skipped:
        print("  build the C++ gates with: cmake -S . -B build && cmake --build build -j")
    for f in failures:
        print(f"FAILED: {f}")
    if failures:
        return 1
    print(f"All active gates passed for: {', '.join(stages)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
