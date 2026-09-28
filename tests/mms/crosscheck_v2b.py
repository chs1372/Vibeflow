#!/usr/bin/env python3
"""Cross-check the C++ SST model against the Python reference (ADR-042, gate 3).

The two share no code but the generated sources: Python assembles sparse
matrices and factorises them, C++ assembles face by face and iterates. Rows,
on meshes both sides run, each side marched until the change per step is
below 1e-12 (relative, of every field):

  2a   L2(k), L2(w), frozen velocity: MS-A and MS-B, both variants, both
       families, n = 6 and 12;
  2b   L2(u), L2(k), L2(w), coupled: MS-A, both variants, both families,
       n = 6.

Bound, fixed before the first comparison: 1e-6 relative (ADR-042), as the
v2a cross-check's steady rows. Nothing is truncated but the linear solvers,
so the rows should agree to far more digits than that.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTO = ROOT / "prototype"
FIX = ROOT / "tests" / "fixtures"
SST = ROOT / "build" / "tests" / "sst_mms"
ENV = {"OMP_PROC_BIND": "false", "OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin",
       "VIBEFLOW_STEADY_TOL": "1e-12"}
BOUND = 1e-6

TITLE = re.compile(r"^2(a|b)\. .*MS-(A|B), SST-(\d{4})")
ROW = re.compile(r"^\s+(orthogonal|smooth distortion)\s+n=(\d+)\s+steps\s+\d+\s+(.*)$")
VAL = re.compile(r"L2\((u|k|w)\) ([\d.eE+-]+)")


def cpp_rows():
    out = {}
    for gate, grids in (("a", ("6", "12")), ("b", ("6",))):
        r = subprocess.run([str(SST), str(FIX), gate, *grids], cwd=ROOT,
                           capture_output=True, text=True, env={**os.environ, **ENV})
        if r.returncode not in (0, 1):
            print(r.stdout, r.stderr)
            return None
        cur = None
        for line in r.stdout.splitlines():
            m = TITLE.match(line)
            if m:
                cur = (m.group(1), m.group(2), m.group(3))
                continue
            m = ROW.match(line)
            if m and cur:
                for q, v in VAL.findall(m.group(3)):
                    out[(f"2{cur[0]} MS-{cur[1]} {cur[2]} L2({q})", m.group(1),
                         int(m.group(2)))] = float(v)
    return out


def python_rows():
    sys.path.insert(0, str(PROTO))
    import sst_gates as G
    out = {}
    fams = ((0.0, "orthogonal"), (0.25, "smooth distortion"))
    for name in ("A", "B"):
        for variant in ("2003", "1994"):
            for skew, tag in fams:
                for n in (6, 12):
                    r = G.run_frozen(name, variant, n, skew)
                    for q in ("k", "w"):
                        out[(f"2a MS-{name} {variant} L2({q})", tag, n)] = r["e" + q]
    for variant in ("2003", "1994"):
        for skew, tag in fams:
            r = G.run_coupled(variant, 6, skew)
            for q in ("u", "k", "w"):
                out[(f"2b MS-A {variant} L2({q})", tag, 6)] = r["e" + q]
    return out


def main():
    c = cpp_rows()
    if c is None:
        print("v2b cross-check gate: FAIL (the C++ gate did not run)")
        return 1
    p = python_rows()
    keys = sorted(set(p) & set(c))
    worst, ok = 0.0, bool(keys) and len(keys) == len(p)
    print(f"{'quantity':<24}{'mesh':<20}{'N':>3}  {'python':>20}  {'c++':>20}  {'rel diff':>9}")
    for key in keys:
        rel = abs(p[key] - c[key]) / abs(p[key])
        worst = max(worst, rel)
        ok &= rel <= BOUND
        print(f"{key[0]:<24}{key[1]:<20}{key[2]:>3}  {p[key]:20.12e}  {c[key]:20.12e}  {rel:9.2e}")
    missing = sorted(set(p) ^ set(c))
    if missing:
        print(f"rows on one side only: {missing}")
    print(f"\n{len(keys)} rows compared, worst relative difference {worst:.2e} (bound {BOUND:g})")
    print("v2b cross-check gate: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
