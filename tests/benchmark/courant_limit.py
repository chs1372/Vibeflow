#!/usr/bin/env python3
"""Largest convective Courant number the solver survives, by measurement.

The convection matrix is first-order upwind and the second-order accuracy is
carried as a deferred correction on the right-hand side. That correction is
explicit, so an otherwise implicit scheme has a time-step limit -- and where
that limit sits is the number this script produces.

It exists because the limit was previously guessed. ADR-022 put it "above
about 2", reasoning from the smallest cell and the free-stream speed. The case
then ran stably at 7.2. The estimate was wrong by a factor of four and had
been written up as if it were a finding, which is the mistake this file is
here to stop repeating.

Run with no arguments to sweep and report. Run with --gate to also require the
recorded floor, so that making the correction implicit can be shown to raise
the limit and a later change cannot quietly lower it.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = ROOT / "build" / "tests" / "cylinder"
MESH = ROOT / "cases" / "cylinder" / "debug.hex"

# The floor the gate enforces. Raise it when a change earns it; never lower it
# to make a run pass.
STABLE_FLOOR = 2.7

# A FIXED NUMBER OF STEPS per dt, not a fixed end time: every run then does
# the same amount of work and gets the same number of chances to go wrong.
# It has to be a few hundred. Divergence here is not always prompt -- the fine
# mesh sat at Courant 7 looking healthy for 120 steps before drag went from
# 1.56 to 4.0 -- so a short window reports stability that is not there.
STEPS = 200
LADDER = [0.05, 0.1, 0.2, 0.3, 0.4]

# Verdict comes from the TAIL of the run, so the impulsive start does not
# count against it: an abrupt free stream past a cylinder really does produce
# a drag coefficient near 34 on the first step.
TAIL = 50
CD_MAX = 5.0

ROW = re.compile(r"t\s+([\d.]+)\s+Cd\s+(\S+)\s+Cl\s+(\S+)\s+div\s+(\S+)\s+Co\s+(\S+)")


def run(dt):
    """One run of STEPS steps. Returns (max Courant over the tail, stable?, last Cd)."""
    t_end = STEPS * dt
    env = {**os.environ, "OMP_PROC_BIND": "false", "OMP_NUM_THREADS": "2",
           "NSFLOW_PRESSURE": "cg+hypre", "CYL_NONORTH_TOL": "1e-9",
           "CYL_REPORT": str(STEPS)}                # report every step
    r = subprocess.run([str(BIN), str(MESH), str(dt), str(t_end)],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    rows = []
    for m in ROW.finditer(r.stdout):
        rows.append((m.group(2), m.group(5)))
    if not rows:
        return 0.0, False, float("nan")
    tail = rows[-TAIL:]

    def num(s):
        try:
            v = float(s)
        except ValueError:
            return None
        return None if v != v or abs(v) == float("inf") else v

    stable, co, cd = True, 0.0, float("nan")
    for cd_s, co_s in tail:
        c, k = num(cd_s), num(co_s)
        if c is None or k is None or abs(c) > CD_MAX:
            stable = False
            continue
        cd, co = c, max(co, k)
    # The run has to have reached the end at all.
    if len(rows) < STEPS - 1:
        stable = False
    return co, stable, cd


def main():
    gate = "--gate" in sys.argv
    if not BIN.exists():
        print("  cylinder not built")
        return None
    if not MESH.exists():
        print(f"  {MESH.name} missing; generate it with cases/cylinder/make_mesh.py")
        return None

    print(f"  Courant stability sweep, {MESH.name}, {STEPS} steps per dt")
    print(f"  {'dt':>6} {'max Courant':>12} {'last Cd':>10}   verdict")
    best = 0.0
    for dt in LADDER:
        co, stable, cd = run(dt)
        print(f"  {dt:6.3f} {co:12.2f} {cd:10.3f}   {'stable' if stable else 'DIVERGED'}")
        if stable:
            best = max(best, co)
        else:
            break            # past the limit; finer dt above it tells us nothing
    print(f"  largest stable Courant measured: {best:.2f}")

    if not gate:
        return True
    ok = best >= STABLE_FLOOR
    print(f"  floor {STABLE_FLOOR:.2f}: {'PASS' if ok else 'FAIL'}")
    if best > STABLE_FLOOR * 1.5:
        print(f"  (the floor is well below what this measures -- raise "
              f"STABLE_FLOOR to {best * 0.9:.1f} to lock the improvement in)")
    return ok


if __name__ == "__main__":
    r = main()
    sys.exit(0 if r else 1)
