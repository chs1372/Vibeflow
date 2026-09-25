#!/usr/bin/env python3
"""Pressure-velocity decoupling gate: the refined cylinder must stay coupled.

The first version of the Rhie-Chow flux in both the Python reference and the
C++ solver put D (grad(p)_f . S - snGrad p_old) into the predicted flux while
re-solving the FULL pressure on every corrector. The checkerboard part of the
pressure equation then reads p_new = -p_old + forcing: an eigenvalue near -1.
A decoupled pressure-velocity mode flips sign on every solve, and once the
velocity coupling pushes its magnitude past one it grows -- adjacent cells
running to pressures of +27 and -39 and one of them flowing backwards, while
face-flux continuity holds to 1e-9 the whole time, because the mode lives in
the cell-centred fields where continuity cannot see it (ADR-026).

Every MMS and benchmark gate passed with that defect in place. The two forms
are both second order, so an order study cannot tell them apart; the coarse
cylinder mesh happened to keep the eigenvalue under one; and the refined mesh
that exposed it was blamed, in turn, on a Courant limit, the deferred
correction, the outer iteration, the diffusion correction and the Choi term.

So this gate runs the case that failed: the 21,811-cell mesh, at a time step
where the defective form blows up within a few steps and the corrected one
is smooth. The verdict is on the fastest cell in the domain, not on a norm --
the mode sat in one cell, and a field-wide norm is how it survived five
wrong explanations.

Verified to fail: CYL_RC_FORM=interpolated in the environment brings the
defective form back, and the gate rejects it.
"""
import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "build" / "tests" / "cylinder"
MESH = ROOT / "cases" / "cylinder" / "cylinder.hex"

DT = 0.1
STEPS = 100
TAIL = 50

# Potential flow past a cylinder peaks at twice the free stream; the viscous
# flow at Re = 100 peaks near 1.5. The decoupled mode reached 12.9 before it
# was stopped. Three sits well clear of both.
U_MAX = 3.0
CD_MAX = 3.0

ROW = re.compile(r"t\s+([\d.]+)\s+Cd\s+(\S+)\s+Cl\s+\S+\s+div\s+\S+\s+Co\s+(\S+)"
                 r"\s+\|u\|max\s+(\S+)")


def num(s):
    try:
        v = float(s)
    except ValueError:
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def main():
    if not BIN.exists():
        print("  cylinder not built")
        return None
    if not MESH.exists():
        print(f"  {MESH.name} missing; generate it with cases/cylinder/make_mesh.py")
        return None

    env = {**os.environ, "OMP_PROC_BIND": "false", "OMP_NUM_THREADS": "2",
           "NSFLOW_PRESSURE": "cg+hypre", "CYL_REPORT": str(STEPS)}
    form = env.get("CYL_RC_FORM", "standard")
    try:
        r = subprocess.run([str(BIN), str(MESH), str(DT), str(DT * STEPS)],
                           cwd=ROOT, env=env, capture_output=True, text=True,
                           timeout=1800)
    except subprocess.TimeoutExpired:
        # A healthy run takes about five minutes. Thirty is a run that has
        # gone wrong in a way the non-finite check did not catch.
        print("  run exceeded 30 minutes -- treated as diverged")
        print("  decoupling gate: FAIL")
        return False
    rows = [(m.group(1), m.group(2), m.group(3), m.group(4)) for m in ROW.finditer(r.stdout)]
    print(f"  refined cylinder, 21,811 cells, dt = {DT}, {STEPS} steps, "
          f"Rhie-Chow form: {form}")
    if not rows:
        print("  no step output parsed")
        return False

    tail = rows[-TAIL:]
    ok = len(rows) >= STEPS - 1
    umax_all = umax_tail = co_tail = 0.0
    cd_last = float("nan")
    for i, (t, cd, co, um) in enumerate(rows):
        c, k, u = num(cd), num(co), num(um)
        if u is None or c is None or k is None:
            ok = False
            continue
        umax_all = max(umax_all, u)
        if (t, cd, co, um) in tail:
            umax_tail = max(umax_tail, u)
            co_tail = max(co_tail, k)
            cd_last = c
            if u > U_MAX or abs(c) > CD_MAX:
                ok = False
    print(f"  fastest cell over the tail: |u| {umax_tail:.2f} (bound {U_MAX})   "
          f"over the whole run: {umax_all:.2f}")
    print(f"  last Cd {cd_last:.4f}   max Courant over the tail {co_tail:.2f}   "
          f"steps completed {len(rows)}")
    print(f"  decoupling gate: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    res = main()
    sys.exit(0 if res else 1)
