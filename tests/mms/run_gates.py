#!/usr/bin/env python3
"""Verification gate runner. CI fails the build if any active gate regresses.

Each gate is registered against the roadmap stage that introduces it. A stage's
gates must pass before code for the next stage is merged -- that rule is what
keeps scope from running ahead of verification.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTO = ROOT / "prototype"

GATES = {
    "v0": [
        ("MMS diffusion, order 2 on orthogonal + skewed mesh",
         [sys.executable, str(PROTO / "mms_diffusion.py"), "8", "16", "32"]),
    ],
    # v1: taylor_green.py, cavity_ghia.py, cylinder_strouhal.py
    # v2: flat_plate_cf.py, backward_step.py, rayleigh_benard.py
    # v3: sod_shock_tube.py, naca0012_transonic.py
    # v4: dam_break.py, rising_bubble.py
}


def main():
    stages = sys.argv[1:] or ["v0"]
    failures = []
    for stage in stages:
        for name, cmd in GATES.get(stage, []):
            print(f"=== [{stage}] {name}")
            r = subprocess.run(cmd, cwd=PROTO)
            if r.returncode != 0:
                failures.append(f"[{stage}] {name}")
    print()
    if failures:
        for f in failures:
            print(f"FAILED: {f}")
        return 1
    print(f"All gates passed for: {', '.join(stages)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
