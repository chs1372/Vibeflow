#!/usr/bin/env python3
"""Cross-check the C++ v2a solver against the Python reference (ADR-038, ADR-039).

The two share no code: Python assembles sparse matrices and factorises them,
C++ assembles face by face and iterates. Three comparisons, on meshes both
sides run (6 and 12 cells, orthogonal and smoothly distorted):

  gate 1   L2(T) of the manufactured temperature in the Ethier-Steinman flow
           (dt = 2e-4, two steps, six outer iterations);
  gate 2   L2(u) and L2(T) of the steady manufactured Boussinesq flow, n = 6;
  slip     L2(u) of the Taylor-Green vortex, slip walls and the exact-wall
           control.

Tolerances, fixed before the first comparison:
  * 1e-4 relative for the transient rows (gate 1 and slip), the NS
    cross-check's bound: both sides truncate the same outer iterations
    differently, and crosscheck_ns.py measured what that costs (2e-5);
  * 1e-6 relative for the steady rows. Both sides iterate those to a change
    per step of 1e-12 or less, so nothing is truncated but the linear
    solvers, and the rows should agree to far more digits than this.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTO = ROOT / "prototype"
FIX = ROOT / "tests" / "fixtures"
HEAT = ROOT / "build" / "tests" / "heat_transfer"
TG = ROOT / "build" / "tests" / "taylor_green"
ENV = {"OMP_PROC_BIND": "false", "PATH": "/usr/bin:/bin",
       "VIBEFLOW_BOUSSINESQ_DT": "0.2"}
TRANSIENT, STEADY = 1e-4, 1e-6
GRIDS = (6, 12)
FAMILIES = (("orthogonal", 0.0), ("smooth distortion", 0.25))

G1 = re.compile(r"^\s+(orthogonal|smooth distortion)\s+n=(\d+)\s+L2\(T\) ([\d.eE+-]+)")
G2 = re.compile(r"^\s+(orthogonal|smooth distortion)\s+n=(\d+)\s+steps \d+\s+"
                r"L2\(u\) ([\d.eE+-]+)\s+L2\(T\) ([\d.eE+-]+)")
TGR = re.compile(r"^\s+(orthogonal|smooth distortion)\s+n=(\d+)\s+L2\(u\) slip ([\d.eE+-]+)"
                 r"\s+exact walls ([\d.eE+-]+)")


def cpp(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, **ENV})
    if r.returncode not in (0, 1):
        print(r.stdout, r.stderr)
        return None
    return r.stdout


def cpp_values():
    out = {}
    s = cpp([str(HEAT), str(FIX), "1", *map(str, GRIDS)])
    if s is None:
        return None
    for line in s.splitlines():
        m = G1.match(line)
        if m:
            out[("gate 1 L2(T)", m.group(1), int(m.group(2)))] = float(m.group(3))
    s = cpp([str(HEAT), str(FIX), "2", "6"])
    if s is None:
        return None
    for line in s.splitlines():
        m = G2.match(line)
        if m:
            out[("gate 2 L2(u)", m.group(1), int(m.group(2)))] = float(m.group(3))
            out[("gate 2 L2(T)", m.group(1), int(m.group(2)))] = float(m.group(4))
    s = cpp([str(TG), *map(str, GRIDS)])
    if s is None:
        return None
    for line in s.splitlines():
        m = TGR.match(line)
        if m:
            out[("slip L2(u)", m.group(1), int(m.group(2)))] = float(m.group(3))
            out[("exact-wall L2(u)", m.group(1), int(m.group(2)))] = float(m.group(4))
    return out


def py_values():
    sys.path.insert(0, str(PROTO))
    import boussinesq as B
    import taylor_green as T
    out = {}
    for tag, skew in FAMILIES:
        for n in GRIDS:
            m, s, t_end = B.run_energy_exact_flow(n, 2e-4, 2, 0.05, 0.05, skew)
            out[("gate 1 L2(T)", tag, n)] = B.l2(s.T - B.t_exact(m.cell_centre, t_end),
                                                 m.cell_volume)
            out[("slip L2(u)", tag, n)] = T.run(n, skew)
            out[("exact-wall L2(u)", tag, n)] = T.run(n, skew, boundary="exact")
        m, s, k, ch = B.steady_boussinesq(6, skew, "smooth", 0.2)
        out[("gate 2 L2(u)", tag, 6)] = B.l2(s.u - B.SF.velocity(m.cell_centre), m.cell_volume)
        out[("gate 2 L2(T)", tag, 6)] = B.l2(s.T - B.b_temperature(m.cell_centre),
                                             m.cell_volume)
    return out


def main():
    if not HEAT.exists() or not TG.exists():
        print("C++ v2a gates not built -- skipping")
        return 0
    b = cpp_values()
    if b is None:
        print("C++ run FAILED")
        return 1
    a = py_values()
    keys = sorted(set(a) | set(b))
    print(f"{'quantity':<18}{'mesh':<19}{'N':>3}{'python':>20}{'c++':>20}{'rel diff':>11}"
          f"{'bound':>8}")
    worst, bad = 0.0, 0
    for k in keys:
        if k not in a or k not in b:
            print(f"{k[0]:<18}{k[1]:<19}{k[2]:>3}   MISSING on the "
                  f"{'c++' if k not in b else 'python'} side")
            bad += 1
            continue
        tol = STEADY if k[0].startswith("gate 2") else TRANSIENT
        rel = abs(a[k] - b[k]) / max(abs(a[k]), 1e-300)
        worst = max(worst, rel)
        bad += rel > tol
        print(f"{k[0]:<18}{k[1]:<19}{k[2]:>3}{a[k]:>20.12e}{b[k]:>20.12e}{rel:>11.2e}"
              f"{tol:>8.0e}{'' if rel <= tol else '  OVER'}")
    print(f"\n{len(keys)} rows compared, worst relative difference {worst:.2e}")
    print("v2a cross-check gate: " + ("PASS" if not bad else f"FAIL ({bad} rows)"))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
