#!/usr/bin/env python3
"""Cross-check the C++ Navier-Stokes solver against the Python reference.

The two share no code and do not even solve their linear systems the same way:
Python uses sparse direct factorisations, C++ uses Jacobi-preconditioned
BiCGStab and CG. Agreement to the tolerance below is evidence one
implementation cannot give on its own.

The tolerance is 1e-4, not the 1e-12 the diffusion cross-check uses, and the
reason is measured rather than assumed. Both sides truncate the same two
iterations, and they truncate them differently:

    outer iterations   worst relative difference
    6 (routine gate)   1.96e-05
    40                 1.49e-06

Tightening the outer loop shrinks the gap by an order of magnitude, which is
what an iteration-truncation difference does and a discretisation mismatch
does not. On top of that Python solves its linear systems with sparse direct
factorisations and C++ with Jacobi-preconditioned BiCGStab and CG, so the
continuity residual bottoms out at machine zero on one side and near 1e-10 on
the other.

1e-4 sits well below the smallest discretisation error being compared
(3.6e-3) and an order above the observed 2e-5, so it still catches a real
discretisation difference. The three the port actually had showed up at
5e-2, 6e-3 and 1e-2.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTO = ROOT / "prototype"
BIN = ROOT / "build" / "tests" / "ethier_steinman"
REL_TOL = 1e-4
import os
OUTER = os.environ.get("VIBEFLOW_OUTER", "6")
ENV = {"OMP_PROC_BIND": "false", "PATH": "/usr/bin:/bin", "VIBEFLOW_OUTER": OUTER}
GRIDS = (6, 12)

ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.eE+-]+)")


def cpp_values():
    r = subprocess.run([str(BIN), *map(str, GRIDS)], cwd=ROOT,
                       capture_output=True, text=True, env=ENV)
    if r.returncode not in (0, 1):
        print(r.stdout, r.stderr); return None
    out, key = {}, None
    for line in r.stdout.splitlines():
        if "spatial order /" in line:
            key = "orthogonal" if "/ orthogonal" in line else "distorted"
            continue
        m = ROW.match(line)
        if m and key:
            out[(key, int(m.group(1)))] = float(m.group(3))
    return out


def py_values():
    sys.path.insert(0, str(PROTO))
    import ethier_steinman as ES
    out = {}
    for key, skew, mode in (("orthogonal", 0.0, "smooth"), ("distorted", 0.25, "smooth")):
        for n in GRIDS:
            out[(key, n)] = ES.run(n, 2e-4, 2, 0.05, skew,
                                   skew_mode=mode, nouter=int(OUTER))["l2"]
    return out


def main():
    if not BIN.exists():
        print(f"C++ solver not built ({BIN}) -- skipping"); return 0
    b = cpp_values()
    if b is None:
        print("C++ run FAILED"); return 1
    a = py_values()

    shared = sorted(set(a) & set(b))
    print(f"{'case':<13}{'N':>4}{'python L2':>22}{'c++ L2':>22}{'rel diff':>12}")
    worst, bad = 0.0, 0
    for k in shared:
        rel = abs(a[k] - b[k]) / max(abs(a[k]), 1e-300)
        worst = max(worst, rel)
        bad += rel > REL_TOL
        print(f"{k[0]:<13}{k[1]:>4}{a[k]:>22.12e}{b[k]:>22.12e}{rel:>12.2e}")
    print(f"\n{len(shared)} rows compared, worst relative difference {worst:.2e} "
          f"(tolerance {REL_TOL:.0e})")
    print("NS cross-check gate: " + ("PASS" if not bad else f"FAIL ({bad} rows over)"))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
